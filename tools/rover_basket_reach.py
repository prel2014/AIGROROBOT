"""Mide en PhysX la llegada simultánea de las palmas a dos zonas de la canasta."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from runpy import run_path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dexman.robot.joints import ARM, JOINT_NAMES

SIDES = ("Left", "Right")
BASE = "/World/Rover/Base"
GOALS = np.array([[.13, .23, .72], [.13, -.23, .72]])
TOLERANCE_M = .02


def plan(stage):
    from agriculture_grasp_geometry import _Kinematics
    from pxr import UsdGeom, UsdPhysics
    from scipy.optimize import least_squares

    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)).GetInverse()
    arms = []
    for side, target in zip(SIDES, GOALS, strict=True):
        path = "/World/Rover/" + side + "Arm"
        mechanism = _Kinematics(stage, path)
        mount = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(mechanism.base)) * inverse).T
        joints = [UsdPhysics.RevoluteJoint.Get(stage, path+"/Arm/joints/"+name) for name in ARM]
        drives = [UsdPhysics.DriveAPI.Get(j.GetPrim(), "angular") for j in joints]
        bounds = np.deg2rad([[j.GetLowerLimitAttr().Get() for j in joints],
                             [j.GetUpperLimitAttr().Get() for j in joints]])
        # Hay una solución de entrega sin una vuelta completa de muñeca.
        low, high = np.maximum(bounds[0], -np.pi), np.minimum(bounds[1], np.pi)
        rest = np.zeros(len(JOINT_NAMES))

        def palm(angles):
            rest[:6] = angles
            return mechanism.fk(mechanism.arm, rest, mount)[:3, 3]

        seed = np.deg2rad([-35., -42., 79., -95., -92., 20.])
        # Usar una rama de IK con la mano aproximándose por encima del borde.
        forward = []
        for height in (.20, 0.):
            point = target + (0., 0., height)
            initial = np.clip(seed, low, high)
            rng = np.random.default_rng(73)
            starts = [initial, *rng.uniform(low, high, size=(12, 6))]
            best = None
            for start in starts:
                def residual(angles):
                    return np.r_[palm(angles)-point, .0005*(angles-initial)]
                result = least_squares(residual, start, bounds=(low, high), max_nfev=100)
                score = np.linalg.norm(result.fun)
                if best is None or score < best[0]:
                    best = score, result.x
                if score < .002:
                    break
            seed = best[1]
            forward.append(seed.copy())
        error = float(np.linalg.norm(palm(seed)-target))
        arms.append((stage.GetPrimAtPath(mechanism.palm), drives, np.rad2deg(forward), error))
    return arms


def simulate(asset):
    import carb
    import omni.physx
    from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.CreateGravityDirectionAttr((0., 0., -1.))
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    scene_api.CreateTimeStepsPerSecondAttr(60)
    scene_api.CreateEnableGPUDynamicsAttr(False)
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.)
    ground.AddTranslateOp().Set((0., 0., -.05))
    ground.AddScaleOp().Set((10., 10., .1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    asset_digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    rover = UsdGeom.Xform.Define(stage, "/World/Rover")
    rover.GetPrim().GetReferences().AddReference(str(asset))
    arms = plan(stage)
    layout = run_path(str(ROOT/"assets/agriculture/robot/rover/layout.py"))
    travel = np.array([layout["ROVER_TRAVEL_DEG"][side] for side in SIDES])
    # El cubo sobresale 5 mm del neumático en la dirección del eje.
    wheel_edges = [(xyz[1] + sign * dims[1] / 2) / 1000
                   for name, (axis, dims), xyz, _ in layout["shapes"]()
                   if axis == "Y" and name.startswith(("Neumatico_", "Llanta_", "Cubo_"))
                   for sign in (-1, 1)]
    wheel_width = (min(wheel_edges), max(wheel_edges))

    # Soporte de cada colisionador en el eje Y del chasis, incluidas manos y dedos.
    def lateral_envelope(cache, inverse):
        limits = {}
        for side in SIDES:
            low, high = np.inf, -np.inf
            arm_root = stage.GetPrimAtPath("/World/Rover/" + side + "Arm")
            for shape in Usd.PrimRange(arm_root, Usd.TraverseInstanceProxies()):
                collision = UsdPhysics.CollisionAPI(shape)
                if not collision or not collision.GetCollisionEnabledAttr().Get():
                    continue
                tf = np.asarray(cache.GetLocalToWorldTransform(shape) * inverse)
                direction, offset = tf[:3, 1], tf[3, 1]
                kind = shape.GetTypeName()
                if kind == "Mesh":
                    ys = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get()) @ direction
                    a, b = ys.min(), ys.max()
                elif kind in ("Cylinder", "Capsule", "Sphere", "Cube"):
                    if kind == "Cube":
                        radius = shape.GetAttribute("size").Get() / 2 * np.abs(direction).sum()
                    elif kind == "Sphere":
                        radius = shape.GetAttribute("radius").Get() * np.linalg.norm(direction)
                    else:
                        axis = "XYZ".index(shape.GetAttribute("axis").Get())
                        transverse = direction if kind == "Capsule" else np.delete(direction, axis)
                        radius = (shape.GetAttribute("radius").Get() * np.linalg.norm(transverse) +
                                  shape.GetAttribute("height").Get() / 2 * abs(direction[axis]))
                    a, b = -radius, radius
                else:
                    raise ValueError(f"Colisionador sin envolvente: {shape.GetPath()} ({kind})")
                low, high = min(low, offset+a), max(high, offset+b)
            if not np.isfinite([low, high]).all():
                raise ValueError(f"No hay superficies de colisión en {arm_root.GetPath()}")
            limits[side] = [float(low), float(high)]
        return limits

    def group(path):
        for side in SIDES:
            root = "/World/Rover/" + side + "Arm/"
            if path.startswith(root):
                return "fixed" if path.startswith(root+"Arm/base_link/") else side
        if path.startswith(BASE+"/") or path.startswith("/World/Rover/Wheel_"):
            return "fixed"
        return None

    for prim in Usd.PrimRange(rover.GetPrim()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    physics = omni.physx.get_physx_simulation_interface()
    cache_id = UsdUtils.StageCache.Get().Insert(stage)
    if not physics.attach_stage(cache_id.ToLongInt()):
        raise RuntimeError("PhysX no pudo abrir el rover")
    steps = 0
    pairs = {}
    same_arm = {}
    ground_contact = False
    observations = {}
    envelopes = {}
    neutral = np.zeros((2, 6))
    approach = np.array([waypoints[0] for _, _, waypoints, _ in arms])
    delivery = np.array([waypoints[1] for _, _, waypoints, _ in arms])
    startup = np.array([[d.GetTargetPositionAttr().Get() for d in drives] for _, drives, _, _ in arms])
    last = startup
    try:
        for label, goal in (("arranque", startup), ("reposo", neutral),
                            ("traslado", travel), ("aproximacion", approach),
                            ("entrega", delivery), ("retirada", approach), ("retorno_traslado", travel)):
            for frame in range(150):
                fraction = min(1., (frame+1)/100)
                fraction = fraction**2 * (3.-2.*fraction)
                command = last + fraction*(goal-last)
                for (_, drives, _, _), angles in zip(arms, command, strict=True):
                    for drive, angle in zip(drives, angles, strict=True):
                        drive.GetTargetPositionAttr().Set(float(angle))
                physics.simulate(1/60., steps/60.)
                physics.fetch_results()
                steps += 1
                headers, points = physics.get_contact_report()
                for header in headers:
                    if not header.num_contact_data:
                        continue
                    paths = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(handle))
                                         for handle in (header.collider0, header.collider1)))
                    a, b = map(group, paths)
                    if "/World/Ground" in paths and "fixed" in (a, b):
                        ground_contact = True
                    if a and b and a != b:
                        pairs.setdefault(paths, (label, frame))
                    elif a in SIDES and a == b:
                        data = same_arm.setdefault(paths, dict(first_phase=label, first_frame=frame,
                                                               min_separation_m=np.inf, max_impulse_ns=0.))
                        first = header.contact_data_offset
                        for index in range(first, first + header.num_contact_data):
                            point = points[index]
                            data["min_separation_m"] = min(data["min_separation_m"], float(point.separation))
                            impulse = float(np.linalg.norm(point.impulse))
                            data["max_impulse_ns"] = max(data["max_impulse_ns"], impulse)
            cache = UsdGeom.XformCache()
            inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE)).GetInverse()
            points = [(cache.GetLocalToWorldTransform(hand)*inverse).ExtractTranslation()
                      for hand, _, _, _ in arms]
            observations[label] = np.asarray(points).tolist()
            if label in ("arranque", "reposo", "traslado", "retorno_traslado"):
                envelopes[label] = lateral_envelope(cache, inverse)
            last = goal
    finally:
        physics.detach_stage()
        UsdUtils.StageCache.Get().Erase(cache_id)
    outcomes = []
    for arm, side, target, measured in zip(arms, SIDES, GOALS, observations["entrega"], strict=True):
        outcomes.append(dict(side=side, target_base_m=target.tolist(), palm_base_m=measured,
                             error_m=float(np.linalg.norm(measured-target)), ik_error_m=arm[3],
                             approach_delivery_deg=arm[2].tolist()))
    contacts = [dict(colliders=paths, phase=phase, frame=frame) for paths, (phase, frame) in pairs.items()]
    internal = [dict(colliders=paths, **metrics) for paths, metrics in same_arm.items()]
    startup_gap = max(np.max(np.abs(np.subtract(envelopes["arranque"][side], envelopes["traslado"][side])))
                      for side in SIDES)
    starts_folded = bool(np.allclose(startup, travel) and startup_gap < .005)
    margin = min(min(low - wheel_width[0], wheel_width[1] - high)
                 for phase in ("arranque", "traslado", "retorno_traslado")
                 for low, high in envelopes[phase].values())
    if not ground_contact:
        raise RuntimeError("No se detectó contacto con el suelo para comprobar el registro de PhysX")
    return dict(asset_sha256=asset_digest, hz=60, steps=steps,
                ground_contact_observed=ground_contact, starts_folded=starts_folded, joint_order=ARM,
                wheel_y_limits_base_m=wheel_width, travel_width_margin_m=margin,
                tolerance_m=TOLERANCE_M, reached=all(row["error_m"] <= TOLERANCE_M for row in outcomes),
                contact_free=not contacts, arms=outcomes, palms_base_m_by_phase=observations,
                collider_y_limits_base_m=envelopes, contacts=contacts, intra_arm_contact_reports=internal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROOT/"assets/agriculture/robot/rover/rover.usdc")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False})
    status = 1
    try:
        report = simulate(opts.asset.resolve(strict=True))
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["arms"]), "contact pairs:", len(report["contacts"]), flush=True)
        status = 0 if (report["reached"] and report["contact_free"] and report["starts_folded"]
                       and report["travel_width_margin_m"] >= 0.) else 2
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=status)


if __name__ == "__main__":
    main()
