"""Check an unassisted physical lift and release before starting agricultural PPO."""
import argparse
import json
import math
from pathlib import Path

from dexman.experiments import write_json


def _arm_frame(group, side, row):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from dexman.tasks.agriculture.interface import ARM_KEYS

    arm = group.scene[ARM_KEYS[side]]
    arm_prim = "/Torso/RightArm" if side else "/Torso/LeftArm"
    path = group.scene.env_prim_paths[row] + arm_prim
    (base_index,), _ = arm.find_bodies("base_link")
    pose = arm.data.body_link_pose_w.torch[row, base_index]
    base = np.eye(4)
    base[:3, :3] = Rotation.from_quat(pose[3:].cpu().numpy()).as_matrix()
    base[:3, 3] = (pose[:3] - group.origins[row]).cpu().numpy()
    return path, base


def _make_plans(env, candidates, vineyard_side, recorded=None):
    from agriculture_grasp_geometry import plan_grasps

    from dexman.tasks.agriculture.interface import JOINTS_PER_ARM
    from dexman.tasks.agriculture.objectives import OBJECTIVES
    from dexman.tasks.agriculture.scene import CROPS

    result = dict(recorded or {})
    for group, produce in zip(env.batch.groups, env.fruits, strict=True):
        if group.crop in result:
            continue
        side = int(vineyard_side == "right") if group.crop == "vineyard" else 1
        arm_path, base = _arm_frame(group, side, 0)
        origin = group.origins[0]
        fruit = produce[0]
        joint_slice = slice(side * JOINTS_PER_ARM, (side + 1) * JOINTS_PER_ARM)
        q = env.home[group.start, joint_slice].cpu().numpy()
        low = env.low[group.start, joint_slice].cpu().numpy()
        high = env.high[group.start, joint_slice].cpu().numpy()
        fruit_position = (fruit.data.root_link_pos_w.torch[0] - origin).cpu().numpy()
        fruit_quaternion = fruit.data.root_link_quat_w.torch[0].cpu().numpy()
        supports = [(asset.init_state.pos, asset.spawn.size) for name, asset in vars(group.scene.cfg).items()
                    if name.startswith("pickup_support_")]
        candidates_found = plan_grasps(
            group.scene.stage, arm_path, base, q, low, high, CROPS[group.crop].fruit,
            fruit_position, fruit_quaternion, OBJECTIVES[group.crop].lift + .008,
            candidates=candidates, support_boxes=supports)
        result[group.crop] = {"side": side, "calibration": {
            "base": base.tolist(), "home": q.tolist(), "low": low.tolist(), "high": high.tolist(),
            "fruit_position": fruit_position.tolist(), "fruit_quaternion": fruit_quaternion.tolist()}, "candidates": [
            {name: getattr(plan, name).tolist() for name in ("approach", "pregrasp", "closed", "lift")}
            | {"direction_deg": float(plan.direction_deg), "ik_error_m": float(plan.ik_error_m),
               "contact_error_m": float(plan.contact_error_m)}
            for plan in candidates_found]}
        print(f"GEOMETRY {group.crop} {len(candidates_found)} candidates", flush=True)
    return result


def _lift_commands(env, plans, phases, settled_joints, valid, ticks, replan):
    import torch
    from agriculture_grasp_geometry import cartesian_lift_waypoints

    from dexman.tasks.agriculture.interface import JOINTS_PER_ARM

    time = torch.linspace(0., 1., ticks + 1, device=env.device)[:, None, None]
    commands = phases["closed"].lerp(phases["lift"], time)
    for group in env.batch.groups:
        crop_plan = plans[group.crop]
        side = crop_plan["side"]
        joint_range = slice(side * JOINTS_PER_ARM, (side + 1) * JOINTS_PER_ARM)
        for local_row, plan in enumerate(crop_plan["candidates"]):
            row = group.start + local_row
            if group.crop in replan and bool(valid[row]):
                path, base = _arm_frame(group, side, local_row)
                try:
                    solved = cartesian_lift_waypoints(
                        group.scene.stage, path, base, settled_joints[row, joint_range].cpu().numpy(),
                        env.low[row, joint_range].cpu().numpy(), env.high[row, joint_range].cpu().numpy(),
                        float(env.lift_height[row]) + .008, finger_targets=plan["closed"])
                except ValueError as error:
                    # Stay at the closed pose when the requested vertical path is unreachable.
                    plan["lift_plan_error"] = str(error)
                    plan.pop("lift_waypoints", None)
                    plan["lift"] = plan["closed"]
                    commands[:, row, joint_range] = phases["closed"][row, joint_range]
                    phases["lift"][row, joint_range] = phases["closed"][row, joint_range]
                    valid[row] = False
                    continue
                plan.pop("lift_plan_error", None)
                plan["lift_waypoints"] = solved.tolist()
                plan["lift"] = solved[-1].tolist()
            if "lift_waypoints" not in plan:
                if "lift_plan_error" in plan:
                    valid[row] = False
                continue
            path = commands.new_tensor(plan["lift_waypoints"])
            if (path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != JOINTS_PER_ARM
                    or not torch.isfinite(path).all()):
                raise ValueError(f"Invalid lift waypoints for {group.crop} candidate {local_row}")
            position = time[:, 0, 0] * (len(path) - 1)
            lower = position.long()
            upper = (lower + 1).clamp_max(len(path) - 1)
            commands[:, row, joint_range] = path[lower].lerp(path[upper], (position - lower)[:, None])
            phases["closed"][row, joint_range] = path[0]
            phases["lift"][row, joint_range] = path[-1]
    return commands


def _contact_snapshot(env, plans):
    import torch

    from dexman.robot.joints import FINGERTIPS
    from dexman.tasks.agriculture.interface import ARM_KEYS

    vectors, points = [], []
    for group in env.batch.groups:
        key = ARM_KEYS[plans[group.crop]["side"]]
        sensors = [group.scene[f"{key}_{tip}"].data for tip in FINGERTIPS]
        vectors.append(torch.stack([sensor.normal_force_matrix_w.torch[:, 0, 0] for sensor in sensors], dim=1))
        points.append(torch.stack([sensor.contact_pos_w.torch[:, 0, 0] for sensor in sensors], dim=1)
                      - group.origins[:, None, :])

    def finite_records(tensor):
        return [[v if all(math.isfinite(x) for x in v) else None for v in row]
                for row in torch.cat(tensor).cpu().tolist()]

    return {"fingertip_normal_vectors_n": finite_records(vectors),
            "fingertip_contact_points_env_m": finite_records(points)}


def _verify(env, plans, *, render=False, replan=()):
    import torch

    from dexman.tasks.agriculture.contact_quality import (MAX_HOLD_LINEAR_SPEED_M_S, MIN_STABLE_HOLD_S,
                                                         steady_contacts)
    from dexman.tasks.agriculture.objectives import FORBIDDEN_IMPACT_N
    from dexman.tasks.agriculture.interface import ACTION_DIM, ARM_KEYS, JOINTS_PER_ARM

    phases = {name: env.home.clone() for name in ("approach", "pregrasp", "closed", "lift")}
    valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    side = torch.zeros_like(env.phase)
    for group in env.batch.groups:
        group_plan = plans[group.crop]
        selected = group_plan["side"]
        for index, plan in enumerate(group_plan["candidates"]):
            row = group.start + index
            valid[row], side[row] = True, selected
            start = selected * JOINTS_PER_ARM
            for name, target in phases.items():
                target[row, start:start + JOINTS_PER_ARM] = target.new_tensor(plan[name])
        # Initialize the arms at the start of this calibration trial.
        # Subsequent motion and every fruit pose are resolved by the simulator.
        for arm_idx, key in enumerate(ARM_KEYS):
            arm = group.scene[key]
            ids = group.joint_ids[arm_idx]
            offset = arm_idx * JOINTS_PER_ARM
            q = phases["approach"][group.start:group.stop, offset:offset + JOINTS_PER_ARM]
            arm.write_joint_position_to_sim_index(position=q, joint_ids=ids)
            arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids)
    for _ in range(24):
        env.batch.step(phases["approach"], render=render)

    initial_measurement = env._measure()
    initial, _, initial_others, *_ = initial_measurement
    initial_clearance = initial_measurement[-1]
    on_support = initial_clearance.abs() < .007
    if env.cfg.raised_support:
        valid &= on_support
    steps_held = torch.zeros_like(env.phase)
    longest_hold = torch.zeros_like(steps_held)
    peak_lift = torch.zeros_like(env.lift_height)
    peak_contact_count = torch.zeros_like(steps_held)
    peak_collision = torch.zeros_like(peak_lift)
    peak_contact_force = torch.zeros_like(peak_lift)
    peak_tracking_error = torch.zeros_like(peak_lift)
    peak_horizontal_shift = torch.zeros_like(peak_lift)
    peak_distractor_shift = torch.zeros_like(peak_lift)
    did_grasp = torch.zeros_like(valid)
    did_oppose = torch.zeros_like(valid)
    rows = torch.arange(env.num_envs, device=env.device)
    snapshots = {}
    hold_samples = math.ceil(MIN_STABLE_HOLD_S / env.batch.control_dt - 1e-9)
    lift_ticks = 48
    stages = (("approach", "pregrasp", 36), ("pregrasp", "closed", 52), ("closed", "closed", 18),
              ("closed", "lift", lift_ticks), ("lift", "lift", 32), ("lift", "closed", lift_ticks),
              ("closed", "pregrasp", 36), ("pregrasp", "pregrasp", 18))
    for start, end, ticks in stages:
        for tick in range(1, ticks + 1):
            if start == "closed" and end == "lift":
                command = lift_commands[tick]
            elif start == "lift" and end == "closed":
                command = lift_commands[lift_ticks - tick]
            else:
                command = phases[start].lerp(phases[end], tick / ticks)
            env.batch.step(command, render=render)
            pose, velocity, others, _, _, contacts, held, collision, _, opposed, force, slip, clearance = env._measure()
            lift = clearance - initial_clearance
            grasp = held[rows, side] & valid
            height_ok = lift >= env.lift_height
            steady = velocity[:, :3].norm(dim=-1) < MAX_HOLD_LINEAR_SPEED_M_S
            no_slip = steady_contacts(slip[rows, side], contacts[rows, side])
            steps_held = torch.where(grasp & height_ok & steady & no_slip, steps_held + 1, 0)
            longest_hold = torch.maximum(longest_hold, steps_held)
            did_grasp |= grasp
            did_oppose |= opposed[rows, side]
            peak_lift = torch.maximum(peak_lift, lift)
            peak_contact_force = torch.maximum(peak_contact_force, force[rows, side].amax(-1))
            joints = env.batch.angular_state()[:, :ACTION_DIM]
            deviation = (joints - command).abs()
            peak_tracking_error = torch.maximum(peak_tracking_error, deviation.amax(-1))
            shift = (pose[:, :2] - initial[:, :2]).norm(dim=-1)
            peak_horizontal_shift = torch.maximum(peak_horizontal_shift, shift)
            shift = (others - initial_others).norm(dim=-1).amax(-1)
            peak_distractor_shift = torch.maximum(peak_distractor_shift, shift)
            peak_contact_count = torch.maximum(peak_contact_count, contacts[rows, side].sum(-1))
            peak_collision = torch.maximum(peak_collision, collision)
            if start == "closed" and end == "lift" and tick == 2:
                snapshots["lift_start"] = {"joint_position_rad": joints.cpu().tolist(),
                                           "fruit_pose": pose.cpu().tolist(),
                                           "fingertip_forces_n": force[rows, side].cpu().tolist(),
                                           **_contact_snapshot(env, plans)}
        if end in {"pregrasp", "closed", "lift"} and (end not in snapshots or (start == end and end != "pregrasp")):
            snapshots[end] = {"joint_position_rad": joints.cpu().tolist(),
                              "fingertip_forces_n": force[rows, side].cpu().tolist(),
                              "fruit_pose": pose.cpu().tolist()}
        if start == end == "closed":
            snapshots[end].update(_contact_snapshot(env, plans))
            lift_commands = _lift_commands(env, plans, phases, joints, valid, lift_ticks, replan)
        print(f"PHYSICS {end} lifted={(longest_hold >= hold_samples).sum().item()}", flush=True)

    stayed_on_table = ((lift.abs() < .012)
                       & ((pose[:, :2] - initial[:, :2]).norm(dim=-1) < .12))
    released = ~contacts.any((1, 2)) & stayed_on_table & steady
    disturbed = peak_distractor_shift > .018
    passed = valid & (longest_hold >= hold_samples) & released & ~disturbed & (peak_collision <= FORBIDDEN_IMPACT_N)
    report = {}
    for group in env.batch.groups:
        count = len(plans[group.crop]["candidates"])
        ids = range(group.start, group.start + count)
        records = [{"candidate": i, "grasp": bool(did_grasp[row]), "opposed": bool(did_oppose[row]),
                    "initial_relative_clearance_m": round(float(initial_clearance[row]), 5),
                    "peak_lift_m": round(float(peak_lift[row]), 5),
                    "hold_s": round(float(longest_hold[row]) * env.batch.control_dt, 3),
                    "peak_contact_force_n": round(float(peak_contact_force[row]), 3),
                    "peak_tracking_error_rad": round(float(peak_tracking_error[row]), 3),
                    "peak_horizontal_shift_m": round(float(peak_horizontal_shift[row]), 4),
                    "max_contact_count": int(peak_contact_count[row]), "released": bool(released[row]),
                    "disturbed_fruit": bool(disturbed[row]), "peak_forbidden_force_n": round(float(peak_collision[row]), 3),
                    "poses": {stage: {name: values[row] for name, values in snapshot.items()}
                              for stage, snapshot in snapshots.items()},
                    "passed": bool(passed[row])} for i, row in enumerate(ids)]
        report[group.crop] = {"attempts": count, "passed": sum(row["passed"] for row in records), "trials": records}
    return report


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=8, help="Number of simultaneous attempts per crop")
    parser.add_argument("--physics-substeps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--raised-support", action="store_true", help="Set loose fruit on narrow physical supports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vineyard-side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Replay the joint trajectories stored in a previous report")
    parser.add_argument("--cartesian-lift", action="store_true",
                        help="Replan a vertical lift from the measured closed pose for replayed candidates")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.candidates < 1 or args.seed < 0:
        parser.error("Expected positive candidates and a nonnegative seed")
    replay = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else None
    source = replay["plans"] if replay else None
    raised_support = args.raised_support or bool(replay and replay.get("raised_support"))
    count = max((len(plan["candidates"]) for plan in (source or {}).values()), default=0)
    candidates = max(count, 1 if source and set(source) == set(OBJECTIVES) else args.candidates)
    cfg = make_batch_config(candidates * len(OBJECTIVES), contacts=True,
                            nominal=True, physics_substeps=args.physics_substeps, raised_support=raised_support)
    with launch_simulation(cfg, args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        sim = SimulationContext(cfg.sim)
        sim.set_camera_view((2.8, -3.8, 2.6), (.35, 0., 1.1))
        env = AgriculturalPPOEnv(sim, cfg, seed=args.seed)
        with torch.inference_mode():
            for _ in range(30):
                env.batch.step(env.home)
            plans = _make_plans(env, candidates, args.vineyard_side, source)
            replan = set(plans) if args.cartesian_lift else set(plans).difference(source or {})
            crops = _verify(env, plans, render="kit" in (args.visualizer or ()), replan=replan)
        passed = all(result["passed"] for result in crops.values())
        write_json(args.output, {"passed": passed, "physics_substeps": args.physics_substeps,
                                 "raised_support": raised_support,
                                 "seed": args.seed, "plans": plans, "crops": crops})
        print(f"GRASP_PHYSICS {json.dumps({crop: record['passed'] for crop, record in crops.items()})}", flush=True)
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
