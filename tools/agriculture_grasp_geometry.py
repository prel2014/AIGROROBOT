"""Plan diagnostic grasp targets from authored USD frames without stepping physics."""
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from dexman.paths import ROOT
from dexman.robot.joints import ARM, FINGERTIPS, JOINT_NAMES, PALM
from dexman.tasks.agriculture.randomization import TABLE_Z


@dataclass(frozen=True)
class GraspCandidate:
    approach: np.ndarray
    pregrasp: np.ndarray
    closed: np.ndarray
    lift: np.ndarray
    direction_deg: float
    ik_error_m: float
    contact_error_m: float


def _frame(position, quaternion):
    from pxr import Gf

    matrix = Gf.Matrix4d(1)
    matrix.SetRotate(Gf.Quatd(quaternion))
    matrix.SetTranslateOnly(Gf.Vec3d(position))
    return np.array(matrix).T


def _transform(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


class _Kinematics:
    def __init__(self, stage, root_path):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.base = f"{root_path}/Arm/base_link"
        self.palm = f"{root_path}/Hand/{PALM}"
        self.tips = [f"{root_path}/Hand/{name}" for name in FINGERTIPS]
        self._parents = {}
        prim = stage.GetPrimAtPath(root_path)
        if not prim:
            raise ValueError(f"Missing arm root {root_path}")
        for joint_prim in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not joint_prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(joint_prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) != 1 or len(b) != 1:
                continue
            name = joint_prim.GetName()
            rotates = joint_prim.IsA(UsdPhysics.RevoluteJoint)
            axis = joint_prim.GetAttribute("physics:axis").Get() or "X"
            basis = np.eye(3)["XYZ".index(axis)]
            self._parents[str(b[0])] = (str(a[0]), JOINT_NAMES.index(name) if rotates else None, basis,
                                        _frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                                        np.linalg.inv(_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())))
        self.arm = self._chain(self.palm, self.base)
        self.fingers = [self._chain(tip, self.palm) for tip in self.tips]
        cache = UsdGeom.XformCache()
        self.capsules, self.boxes = [], []
        pad_indices = {}
        for shape in Usd.PrimRange(stage.GetPrimAtPath(f"{root_path}/Hand"), Usd.TraverseInstanceProxies()):
            if not shape.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = shape.GetParent()
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            # Only the palm and its descendants participate in fingertip planning.
            if not body or body.GetName() in {"robot0_hand_mount", "robot0_forearm", "robot0_wrist"}:
                continue
            body_path = str(body.GetPath())
            chain = self._chain(body_path, self.palm)
            local = np.array(cache.GetLocalToWorldTransform(shape) *
                             cache.GetLocalToWorldTransform(body).GetInverse()).T
            if shape.IsA(UsdGeom.Capsule):
                schema = UsdGeom.Capsule(shape)
                axis = np.eye(3)["XYZ".index(schema.GetAxisAttr().Get())]
                half_length = .5 * schema.GetHeightAttr().Get()
                ends = _transform(local, np.stack((-half_length * axis, half_length * axis)))
                radius = float(schema.GetRadiusAttr().Get())
                filtered = {str(path) for path in UsdPhysics.FilteredPairsAPI(body).GetFilteredPairsRel().GetTargets()}
                pad_indices[body_path] = len(self.capsules)
                self.capsules.append((chain, ends, radius, body_path, filtered))
            if shape.IsA(UsdGeom.Cube):
                size = UsdGeom.Cube(shape).GetSizeAttr().Get()
                corners = np.asarray(list(product((-.5, .5), repeat=3))) * size
                self.boxes.append((chain, _transform(local, corners)))
        self.pad_indices = np.array([pad_indices[tip] for tip in self.tips])
        self.pads = [self.capsules[i][1:3] for i in self.pad_indices]
        self.radii = np.asarray([shape[2] for shape in self.capsules])
        self.enabled_pairs = np.asarray([(a, b) for a, b in combinations(range(len(self.capsules)), 2)
                                         if self.capsules[a][3] not in self.capsules[b][4]
                                         and self.capsules[b][3] not in self.capsules[a][4]], dtype=int)

    def _chain(self, end, start):
        result = []
        while end != start:
            if end not in self._parents:
                raise ValueError(f"No USD joint path from {start} to {end}")
            end, *joint = self._parents[end]
            result.append(joint)
        return tuple(reversed(result))

    @staticmethod
    def fk(chain, q, start=None):
        matrix = np.eye(4) if start is None else start.copy()
        for index, axis, before, after in chain:
            matrix = matrix @ before
            if index is not None:
                matrix[:3, :3] = matrix[:3, :3] @ Rotation.from_rotvec(axis * q[index]).as_matrix()
            matrix = matrix @ after
        return matrix

    def pad_ends(self, finger, q):
        ends, radii = self.pads[finger]
        return _transform(self.fk(self.fingers[finger], q), ends), radii

    def hand_shapes(self, q):
        capsules = np.asarray([_transform(self.fk(chain, q), ends) for chain, ends, *_ in self.capsules])
        boxes = np.concatenate([_transform(self.fk(chain, q), corners) for chain, corners in self.boxes])
        return capsules, boxes


def _fruit_parts(fruit, position, quaternion):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = ROOT / "assets" / "agriculture" / "props" / f"{fruit}.usda"
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    world_to_root = cache.GetLocalToWorldTransform(root).GetInverse()
    rotation = Rotation.from_quat(quaternion)
    parts = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision.GetCollisionEnabledAttr().Get():
            continue
        points = _transform(np.array(cache.GetLocalToWorldTransform(prim) * world_to_root).T,
                            UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if fruit == "banana":
            # Use overlapping local hulls so the concave side stays accessible.
            center = points.mean(axis=0)
            longitudinal = np.linalg.svd(points - center, full_matrices=False)[2][0]
            station = (points - center) @ longitudinal
            bins = np.linspace(station.min(), station.max(), 7)
            segments = (points[(station >= low) & (station <= high)] for low, high in zip(bins[:-2], bins[2:]))
        else:
            segments = (points,)
        for segment in segments:
            if len(segment) >= 4:
                parts.append(rotation.apply(segment) + position)
    if not parts:
        raise ValueError(f"No collision geometry in {path}")
    planes, bounds = [], []
    for points in parts:
        try:
            planes.append(ConvexHull(points).equations)
        except QhullError:
            # Thin stems can be planar in the mesh; they do not supply grasp candidates.
            continue
        bounds.append((points.min(axis=0), points.max(axis=0)))
    if not planes:
        raise ValueError(f"No volumetric collision surface in {path}")
    return planes, np.concatenate(parts), bounds


class _HullDistance:
    """Sample a local signed distance bound to the union of the fruit's convex parts."""

    padding = .027

    def __init__(self, parts, points, bounds):
        self.spacing = .001 if np.ptp(points, axis=0).max() < .03 else .003
        self.origin = points.min(axis=0) - self.padding
        sizes = np.ceil((points.max(axis=0) + self.padding - self.origin) / self.spacing).astype(int) + 1
        self.field = np.full(sizes, self.padding)
        for planes, (low, high) in zip(parts, bounds, strict=True):
            start = np.maximum(0, np.floor((low - self.padding - self.origin) / self.spacing).astype(int))
            stop = np.minimum(sizes, np.ceil((high + self.padding - self.origin) / self.spacing).astype(int) + 1)
            ranges = tuple(slice(a, b) for a, b in zip(start, stop, strict=True))
            grid = np.moveaxis(np.mgrid[ranges], 0, -1)
            samples = self.origin + self.spacing * grid.reshape(-1, 3)
            distance = np.empty(len(samples))
            for offset in range(0, len(samples), 1024):
                batch = samples[offset:offset+1024]
                distance[offset:offset+len(batch)] = (batch @ planes[:, :3].T + planes[:, 3]).max(axis=-1)
            region = self.field[ranges]
            np.minimum(region, distance.reshape(region.shape), out=region)

    def __call__(self, points):
        indices = ((points - self.origin) / self.spacing).reshape(-1, 3)
        signed = map_coordinates(self.field, indices.T, order=1, mode="constant",
                                 cval=self.padding, prefilter=False)
        return signed.reshape(points.shape[:-1])


def _ray_bounds(parts, origin, direction):
    near, far = [], []
    for planes in parts:
        dot = planes[:, :3] @ direction
        gap = -(planes[:, :3] @ origin + planes[:, 3])
        rising, falling = dot > 1.e-9, dot < -1.e-9
        if np.any(gap[~(rising | falling)] < -1.e-7) or not (rising.any() and falling.any()):
            continue
        lower = gap[falling] / dot[falling]
        upper = gap[rising] / dot[rising]
        li, ui = lower.argmax(), upper.argmin()
        lo, hi = lower[li], upper[ui]
        if lo <= hi:
            near.append((lo, planes[falling][li, :3]))
            far.append((hi, planes[rising][ui, :3]))
    return (min(near, key=lambda hit: hit[0]), max(far, key=lambda hit: hit[0])) if near else None


def _support_points(parts, points, x, y, precision):
    base = (points.max(axis=0) + points.min(axis=0)) * .5
    height = np.ptp(points[:, 2])
    cross_width = np.ptp(points @ y)
    fingers = (0, 4) if precision else (0, 1, 4)
    offsets = (0., 0.) if precision else (-.12, .12, 0.)
    best = None
    sections = (.48, .52, .56, .44) if precision else (.44, .50, .38, .56, .62)
    for height_fraction in sections:
        z = points[:, 2].min() + height_fraction * height
        for lateral in (0., -.16, .16, -.29, .29):
            center = base + lateral * cross_width * y
            center[2] = z
            targets = []
            for finger, offset in zip(fingers, offsets, strict=True):
                ray = center + offset * cross_width * y
                bounds = _ray_bounds(parts, ray, x)
                if bounds is None:
                    break
                sign = 1. if finger == 4 else -1.
                along_ray, normal = bounds[int(sign > 0.)]
                targets.append((finger, ray + x * along_ray, normal))
            if len(targets) != len(fingers):
                continue
            width = (targets[-1][1] - targets[0][1]) @ x
            if width < .008 or width > .13:
                continue
            normals = np.asarray([normal for _, _, normal in targets])
            # With outward normal n, a pad pushes in the -n direction. Even with
            # tangential friction, an upward-facing side can push the produce down.
            nz = normals[:, 2]
            vertical_support = .48 * np.sqrt(np.maximum(0.0, 1.0 - nz**2)) - nz
            opposition = np.min(normals[:-1] @ normals[-1])
            score = (abs(lateral) + abs(height_fraction - .50) + abs(width - min(.08, cross_width))
                     + 2.0 * np.maximum(.20 - vertical_support, 0.0).mean()
                     + 1.5 * max(opposition + .50, 0.0))
            if best is None or score < best[0]:
                best = score, targets
    return best[1] if best else None


def _box_distance(points, centers, half_sizes):
    offset = np.abs(points[..., None, :] - centers) - half_sizes
    return (np.linalg.norm(np.maximum(offset, 0.), axis=-1) +
            np.minimum(offset.max(axis=-1), 0.)).min(axis=-1)


def _hand_targets(kin, targets, axes, home, low, high, table_z, distance, *, preload,
                  pad_floor_clearance, support_boxes, opening=None):
    origin = np.mean([target[1] for target in targets], axis=0)
    goals = []
    normals = []
    for finger, position, normal in targets:
        radius = kin.pads[finger][1]
        # A closed target adds preload. An open target moves each pad away from the same surface.
        margin = -preload if opening is None else .010
        goals.append((position + normal * (radius + margin) - origin) @ axes)
        normals.append(normal @ axes)
    goals = np.asarray(goals)
    normals = np.asarray(normals)
    fingers = np.asarray([item[0] for item in targets])
    active = np.flatnonzero((high > low + 1.e-5) & (np.arange(len(home)) >= len(ARM)))
    baseline = home.copy()
    seed = baseline.copy()
    if opening is None:
        for finger, name in enumerate(("FF", "MF", "RF", "LF")):
            angles = (.52, .57, .42) if finger in fingers else (.95, 1.05, .80)
            for joint, angle in zip((2, 1, 0), angles, strict=True):
                i = JOINT_NAMES.index(f"robot0_{name}J{joint}")
                seed[i] = np.clip(angle, low[i], high[i])
        for name, angle in (("THJ4", .46), ("THJ3", .70), ("THJ0", -.62)):
            i = JOINT_NAMES.index(f"robot0_{name}")
            seed[i] = np.clip(angle, low[i], high[i])
        midpoints = np.asarray([kin.pad_ends(finger, seed)[0].mean(axis=0) for finger in fingers])
        translation = (goals - midpoints).mean(axis=0)
        translation_span = np.full(3, .28)
        translation_center = np.zeros(3)
    else:
        translation = translation_center = (opening - origin) @ axes
        translation_span = np.full(3, .005)
        indices = {joint[0] for finger in fingers for joint in kin.fingers[finger]}
        active = np.array([i for i in active if i in indices], dtype=int)
    count = len(fingers)
    pair_a, pair_b = kin.enabled_pairs.T
    pair_radii = kin.radii[pair_a] + kin.radii[pair_b]
    samples = np.linspace(0., 1., 7)[None, :, None]
    fruit_clearance = np.r_[kin.radii + .001, .002]
    fruit_clearance[kin.pad_indices[fingers]] += (-preload - .002 if opening is None else .002)
    floor_clearance = np.full(len(kin.capsules) + 1, .005)
    floor_clearance[kin.pad_indices[fingers]] = pad_floor_clearance

    def pose(values):
        q = baseline.copy()
        q[active] = values[3:3+active.size]
        ends, boxes = kin.hand_shapes(q)
        ends += values[:3]
        boxes += values[:3]
        pads = ends[kin.pad_indices[fingers]]
        contact = pads[:, 0] + values[-count:, None] * (pads[:, 1] - pads[:, 0])
        return q, ends, boxes, pads, contact

    def residual(values):
        q, ends, boxes, pads, contact = pose(values)
        capsule_floor = (ends @ axes[2]).min(axis=1) - kin.radii
        floor = np.r_[capsule_floor, (boxes @ axes[2]).min()] + origin[2] - table_z
        depth = np.einsum("ijk,ik->ij", pads - goals[:, None, :], normals)
        centerlines = ends[:, :1] + samples * (ends[:, 1:] - ends[:, :1])
        pair_distance = np.linalg.norm(centerlines[pair_a, :, None] - centerlines[pair_b, None, :], axis=-1)
        self_clearance = pair_distance.min(axis=(1, 2)) - pair_radii
        world_capsules = centerlines @ axes.T + origin
        world_corners = boxes @ axes.T + origin
        hull_distance = np.r_[distance(world_capsules).min(axis=1), distance(world_corners).min()]
        support_gap = np.empty(0)
        if len(support_boxes):
            centers, half_sizes = support_boxes[:, 0], .5 * support_boxes[:, 1]
            capsules_gap = _box_distance(world_capsules, centers, half_sizes).min(axis=1) - kin.radii
            palm_gap = _box_distance(world_corners, centers, half_sizes).min()
            support_gap = np.r_[capsules_gap, palm_gap] - .002
        # Matching one point on a capsule does not guarantee that each loaded
        # pad actually closes on the fruit hull after the other constraints act.
        pad_gap = hull_distance[kin.pad_indices[fingers]] - kin.radii[kin.pad_indices[fingers]]
        missing_contact = np.maximum(pad_gap + .5 * preload, 0.) if opening is None else np.zeros(count)
        return np.r_[70. * (contact - goals).ravel(), 180. * np.minimum(floor - floor_clearance, 0.),
                     140. * np.minimum(depth + .0005, 0.).ravel(),
                     130. * np.minimum(self_clearance - .001, 0.),
                     160. * np.minimum(hull_distance - fruit_clearance, 0.),
                     180. * np.minimum(support_gap, 0.),
                     220. * missing_contact,
                     .07 * (q[active] - seed[active]), .025 * (values[-count:] - .5)]

    lower = np.r_[translation_center - translation_span, low[active], [0.] * count]
    upper = np.r_[translation_center + translation_span, high[active], [1.] * count]
    initial = np.r_[translation, seed[active], [.5] * count]
    result = least_squares(residual, np.clip(initial, lower, upper), bounds=(lower, upper), max_nfev=65)
    q, _, _, _, contact = pose(result.x)
    error = np.linalg.norm(contact - goals, axis=-1).max()
    return origin + axes @ result.x[:3], q, float(error)


def _solve_arm(kin, base, seed, low, high, position, axes, *, moving=None):
    desired = Rotation.from_matrix(axes)
    joints = np.arange(len(ARM)) if moving is None else np.asarray(moving, dtype=int)

    def residual(values):
        q = seed.copy()
        q[joints] = values
        pose = kin.fk(kin.arm, q, base)
        angle = (Rotation.from_matrix(pose[:3, :3]).inv() * desired).as_rotvec()
        error = np.r_[pose[:3, 3] - position, .28 * angle]
        if len(joints) > 6:
            # Pick the nearest posture when the palm chain has redundant joints.
            error = np.r_[error, .0001 * (values - seed[joints])]
        return error

    bounds = (low[joints], high[joints])
    result = least_squares(residual, np.clip(seed[joints], *bounds), bounds=bounds, max_nfev=75)
    pose_error = residual(result.x)[:6]
    q = seed.copy()
    q[joints] = result.x
    return q, max(np.linalg.norm(pose_error[:3]), np.linalg.norm(pose_error[3:]))


def cartesian_lift_waypoints(stage, arm_path, base_matrix, closed, joint_low, joint_high, lift_height,
                            *, spacing=.002, finger_targets=None):
    """Solve a vertical lift from the closed wrist pose at constant orientation.

    The first row keeps the measured palm pose. Subsequent rows use the previous
    solution as their seed. Closed finger preload can override joints downstream
    of the palm. This leaves the two Shadow wrist joints at their measured pose.
    """
    if not np.isfinite((spacing, lift_height)).all() or min(spacing, lift_height) <= 0.:
        raise ValueError("Lift height and waypoint spacing must be positive and finite")
    base = np.asarray(base_matrix, dtype=float)
    low, high = (np.asarray(bounds, dtype=float) for bounds in (joint_low, joint_high))
    q = np.asarray(closed, dtype=float).copy()
    kin = _Kinematics(stage, arm_path)
    start_pose = kin.fk(kin.arm, q, base)
    palm_joints = {index for index, *_ in kin.arm if index is not None}
    if finger_targets is not None:
        fingers = [index for index in range(len(q)) if index not in palm_joints]
        q[fingers] = np.asarray(finger_targets, dtype=float)[fingers]
    moving = sorted(index for index in palm_joints if high[index] - low[index] > 1.e-6)
    steps = int(np.ceil(lift_height / spacing))
    rows = [q]
    for dz in np.linspace(0., lift_height, steps + 1)[1:]:
        position = start_pose[:3, 3] + (0., 0., dz)
        q, error = _solve_arm(kin, base, q, low, high, position, start_pose[:3, :3], moving=moving)
        if error > .002:
            raise ValueError(f"Vertical lift IK error {error:.4f} m at {dz:.3f} m above grasp")
        rows.append(q)
    return np.stack(rows)


def plan_grasps(stage, arm_path, base_matrix, home, joint_low, joint_high, fruit, fruit_position,
                fruit_quaternion, lift_height, *, candidates=8, table_z=TABLE_Z, preload_m=None,
                support_boxes=()):
    """Return geometric candidates in JOINT_NAMES order.

    The base transform and fruit pose must use the same reference frame. This estimates
    geometric contact only. The caller verifies holds, collisions and release in PhysX.
    """
    home, low, high = (np.asarray(x, dtype=float) for x in (home, joint_low, joint_high))
    support_boxes = np.asarray(support_boxes, dtype=float).reshape(-1, 2, 3)
    base_matrix = np.asarray(base_matrix, dtype=float)
    fruit_position = np.asarray(fruit_position, dtype=float)
    kin = _Kinematics(stage, arm_path)
    parts, points, bounds = _fruit_parts(fruit, fruit_position, fruit_quaternion)
    distance = _HullDistance(parts, points, bounds)
    precision = fruit == "blueberry"
    preload = preload_m if preload_m is not None else (.002 if precision else .0035)
    pad_floor_clearance = .0015 if precision else .005
    plans = []
    tilts = (35., 25., 45., 5.) if fruit == "grapes" else (25., 5.)
    for tilt in tilts:
        for heading in (90., 157.5, 67.5, 22.5, 45., 135., 112.5, 0., 180.,
                        -90., -45., -135., -22.5, -67.5, -112.5, -157.5):
            angle = np.radians(heading)
            x = np.array([np.cos(angle), np.sin(angle), 0.])
            y = np.array([-np.sin(angle), np.cos(angle), 0.])
            # Tilt along a horizontal direction toward the fruit from the arm base.
            toward = np.sign((fruit_position - base_matrix[:3, 3]) @ y) or 1.
            z = toward * np.sin(np.radians(tilt)) * y - np.array([0., 0., np.cos(np.radians(tilt))])
            axes = np.column_stack((x, np.cross(z, x), z))
            targets = _support_points(parts, points, x, y, precision)
            if targets is None:
                continue
            contact_position, closing_hand, contact_error = _hand_targets(
                kin, targets, axes, home, low, high, table_z, distance, preload=preload,
                pad_floor_clearance=pad_floor_clearance, support_boxes=support_boxes)
            closed, error = _solve_arm(kin, base_matrix, closing_hand, low, high, contact_position, axes)
            if error > .025 or contact_error > (.018 if precision else .024):
                continue
            lift_position = contact_position + [0., 0., lift_height]
            lift, lift_error = _solve_arm(kin, base_matrix, closed, low, high, lift_position, axes)
            opening_position, opened, opening_error = _hand_targets(
                kin, targets, axes, closed, low, high, table_z, distance, preload=preload,
                pad_floor_clearance=pad_floor_clearance, support_boxes=support_boxes, opening=contact_position)
            pregrasp, pregrasp_error = _solve_arm(kin, base_matrix, opened, low, high, opening_position, axes)
            approach, approach_error = _solve_arm(
                kin, base_matrix, pregrasp, low, high, opening_position + [0., 0., .085], axes)
            error = float(max(error, lift_error, pregrasp_error, approach_error))
            if error <= .025:
                plans.append(GraspCandidate(approach, pregrasp, closed, lift, heading,
                                            error, max(contact_error, opening_error * .25)))
        if len(plans) >= candidates:
            break
    return sorted(plans, key=lambda plan: 4.0 * plan.contact_error_m + plan.ik_error_m)[:candidates]
