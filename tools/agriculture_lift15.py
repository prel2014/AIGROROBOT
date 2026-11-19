"""Identify and replay fingertip grasps against measured loose-fruit dynamics."""
import argparse
import json
from pathlib import Path

import numpy as np

from dexman.experiments import write_json
from dexman.robot.joints import JOINT_NAMES
from dexman.tasks.agriculture.interface import ARM_KEYS, JOINTS_PER_ARM


HEIGHT = .15
CROPS = ("citrus", "banana", "vineyard")


def palm_paths(env, group, plans):
    """Compute smooth vertical palm trajectories once per initial grasp geometry."""
    from agriculture_grasp import _arm_frame
    from agriculture_grasp_geometry import _Kinematics, _solve_arm

    path, base = _arm_frame(group, plans["side"], 0)
    kin = _Kinematics(group.scene.stage, path)
    lower, upper = (np.array(plans["calibration"][key]) for key in ("low", "high"))
    heights = np.arange(-.04, .265, .01)
    paths = []
    for plan in plans["candidates"]:
        closed = np.array(plan["closed"])
        pose = kin.fk(kin.arm, closed, base)
        rows = []
        seed = closed
        for z in heights:
            seed, error = _solve_arm(kin, base, seed, lower, upper, pose[:3, 3] + (0, 0, z), pose[:3, :3])
            if error > .004:
                break
            rows.append(seed.copy())
        if len(rows) >= 21:
            paths.append((plan, np.stack(rows), heights[:len(rows)]))
    if not paths:
        raise ValueError(f"No hay trayectoria vertical alcanzable para {group.crop}")
    return paths


# Depth of palm approach, extra closure, individual index/middle flexion, thumb opposition and flexion.
SCALE = np.array([.011, .65, .27, .32, .18, .24, .24, .20])
CENTER = np.array([-.014, .85, .14, .24, 0., .03, .05, -.14])


def pose_candidate(path, params, low, high, travel):
    plan, palm, heights = path
    dz, closure, ff, mf, th4, th3, th2, th0 = params
    closed = np.array(plan["closed"])
    opened = np.array(plan["pregrasp"])
    digits = closed.copy()
    digits[8:] += closure * (closed - opened)[8:]
    for finger, change in (("FF", ff), ("MF", mf)):
        for j in (0, 1, 2):
            digits[JOINT_NAMES.index(f"robot0_{finger}J{j}")] += change
    for j, change in zip((4, 3, 2, 0), (th4, th3, th2, th0), strict=True):
        digits[JOINT_NAMES.index(f"robot0_THJ{j}")] += change
    def at(height):
        row = digits.copy()
        row[:6] = [np.interp(height, heights, axis) for axis in palm[:, :6].T]
        return np.clip(row, low, high)
    dz = np.clip(dz, -.035, .012)
    closed = at(dz)
    opened[:6] += closed[:6] - np.asarray(plan["closed"][:6])
    approach = opened.copy()
    approach[:6] += at(dz + .05)[:6] - closed[:6]
    return np.stack([np.clip(approach, low, high), np.clip(opened, low, high), closed,
                     *(at(dz + h) for h in np.linspace(travel / 9, travel, 9))])


def trial(env, poses, side, capture=None, exercise="lift15", *, settle=False):
    """Only the simulated drives receive targets after the initial robot reset."""
    import torch
    from dexman.tasks.agriculture.contact_quality import MAX_HOLD_LINEAR_SPEED_M_S, steady_contacts
    from dexman.tasks.agriculture.objectives import FORBIDDEN_IMPACT_N, REACH, LIFT, PLACE, SETTLE

    batch = env.batch
    group, = batch.groups
    batch.reset()
    targets = env.home[:, None, :].repeat(1, poses.shape[1], 1)
    section = slice(side * JOINTS_PER_ARM, (side + 1) * JOINTS_PER_ARM)
    targets[:, :, section] = torch.as_tensor(poses, device=env.device, dtype=targets.dtype)
    arm = group.scene[ARM_KEYS[side]]
    ids = group.joint_ids[side]
    initial = targets[:, 0, section]
    arm.write_joint_position_to_sim_index(position=initial, joint_ids=ids)
    arm.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial), joint_ids=ids)
    for _ in range(30):
        batch.step(targets[:, 0])
    if settle:
        for _ in range(24):
            speeds = [fruit.data.root_com_vel_w.torch for fruit in env.fruits[0]]
            if all((v[:, :3].norm(dim=-1) < .005).all() and (v[:, 3:].norm(dim=-1) < .08).all()
                   for v in speeds):
                break
            for _ in range(8):
                batch.step(targets[:, 0])
    if capture:
        batch.sim.render()
    start = env._measure()
    support = start[-1].clone()
    if (support.abs() >= .007).all():
        raise RuntimeError(f"La fruta no comenzó sobre el apoyo físico: {support.median().item():.4f} m")
    safe = support.abs() < .007
    zero = torch.zeros_like(support)
    best_height, peak, max_force = zero.clone(), zero.clone(), zero.clone()
    max_impact, other_shift = zero.clone(), zero.clone()
    stable_ticks, max_ticks, grip_ticks, acquisition_ticks = (zero.clone() for _ in range(4))
    pregrasp = torch.zeros_like(safe)
    grip_fraction = zero.clone()
    final_hold = torch.zeros_like(safe)
    phase = REACH
    if capture:
        capture.start(env, support)
    stages = [(0, 1, 36), (1, 2, 52), (2, 2, 24)]
    if exercise != "pinch":
        lift_ticks = 14 if group.crop == "vineyard" else 8
        stages.extend((i, i + 1, lift_ticks) for i in range(2, 11))
        stages.append((11, 11, 60 if group.crop == "vineyard" else 28))
    if exercise == "return15":
        stages.extend((i, i - 1, 8) for i in range(11, 2, -1))
        stages.extend(((2, 2, 18), (2, 1, 52), (1, 1, 18)))
    step = 0
    for stage_index, (a, b, ticks) in enumerate(stages):
        if stage_index == 3:
            phase = LIFT
        if stage_index == 13:
            phase = PLACE
        if exercise == "return15" and stage_index == len(stages) - 1:
            phase = SETTLE
        for tick in range(1, ticks + 1):
            u = tick / ticks
            target = targets[:, a].lerp(targets[:, b], u)
            if capture:
                capture.observe(env)
            batch.step(target, render=bool(capture))
            m = env._measure()
            held = m[6][:, side]
            quiet = m[1][:, :3].norm(dim=-1) < MAX_HOLD_LINEAR_SPEED_M_S
            steady = held & steady_contacts(m[11][:, side], m[5][:, side]) & quiet
            lift = m[-1] - support
            on_support = (lift.abs() < .004) & ((m[0][:, :2] - start[0][:, :2]).norm(dim=-1) < .012)
            max_impact = torch.maximum(max_impact, m[7])
            other_shift = torch.maximum(other_shift, (m[2] - start[2]).norm(dim=-1).amax(-1))
            safe &= (max_impact <= FORBIDDEN_IMPACT_N) & (other_shift < .018)
            safe &= torch.isfinite(m[0]).all(-1)
            lifted = steady & (lift >= HEIGHT) & (m[-1] >= HEIGHT) & pregrasp
            if exercise == "pinch":
                safe &= on_support
            seated = on_support if exercise == "pinch" else (lift.abs() < .006)
            grip_ticks = torch.where(steady & seated, grip_ticks + 1, 0.)
            acquisition_ticks = torch.where(held & (lift < .03), acquisition_ticks + 1, 0.)
            pregrasp |= acquisition_ticks >= 3
            stable_ticks = torch.where(lifted, stable_ticks + 1, 0.)
            max_ticks = torch.maximum(max_ticks, stable_ticks)
            peak = torch.maximum(peak, lift)
            best_height = torch.maximum(best_height, torch.where(held, lift, 0.))
            if phase == LIFT:
                grip_fraction += held.float()
            max_force = torch.maximum(max_force, m[10][:, side].amax(-1))
            if phase == PLACE:
                safe &= on_support | held
            if a == b == 11 and tick == ticks:
                final_hold = stable_ticks >= 15
            released = ~m[5].any((1, 2)) & quiet & on_support
            if phase == SETTLE:
                safe &= released
            if capture:
                capture.append(step, phase, target, m, stage_index == len(stages)-1 and tick == ticks)
            step += 1
    success = safe & (grip_ticks >= 15 if exercise == "pinch" else final_hold)
    if capture:
        capture.finish(success)
    return {name: tensor.cpu().tolist() for name, tensor in {
        "success": success, "safe": safe, "pregrasp": pregrasp, "held_height": best_height,
        "peak": peak, "hold_ticks": max_ticks, "loaded_ticks": grip_fraction, "peak_force_n": max_force,
        "forbidden_force_n": max_impact, "distractor_shift_m": other_shift, "initial_clearance_m": support}.items()}


def search(env, plans, seed, rounds, output, *, settle, refine=None, record_dir=None):
    group, = env.batch.groups
    if refine:
        reference = json.loads(refine.read_text(encoding="utf-8"))
        if reference["crop"] != group.crop or reference["side"] != plans["side"]:
            raise ValueError("La trayectoria inicial debe corresponder al mismo cultivo y mano")
        q = reference["waypoints_rad"]
        plans = {**plans, "candidates": [{"pregrasp": q[1], "closed": q[2]}] + plans["candidates"]}
    paths = palm_paths(env, group, plans)
    low, high = (np.asarray(plans["calibration"][k]) for k in ("low", "high"))
    rng = np.random.default_rng(seed)
    parents = []
    high_score = -np.inf
    for generation in range(rounds):
        candidates = []
        for i in range(env.num_envs):
            source = i % len(paths)
            center, scale = CENTER, SCALE
            if refine and i % 4:
                source = 0
                center, scale = np.zeros_like(CENTER), .45 * SCALE
            if parents and i % 5:
                _, source, center = parents[rng.integers(len(parents))]
                scale = SCALE * (.4 if i % 3 else .85)
            params = center + rng.normal(size=len(center)) * scale
            if refine and i == 0:
                source, params = 0, np.zeros_like(CENTER)
            candidates.append((source, params))
        travel = .22 if group.crop == "vineyard" else .18
        poses = np.stack([pose_candidate(paths[k], params, low, high, travel) for k, params in candidates])
        writer = None
        if record_dir:
            from dexman.data.lift_trials import LiftTrialCapture
            # Save the commands actually evaluated, including the candidate index of each replica.
            attempt = record_dir / f"round_{generation:03d}"
            attempt.mkdir(parents=True, exist_ok=False)
            trajectory = attempt / "candidates.json"
            write_json(trajectory, {"side": plans["side"], "waypoints_rad": poses.tolist()})
            writer = LiftTrialCapture(env, attempt/"lift15", trajectory, "lift15")
        try:
            results = trial(env, poses, plans["side"], writer, settle=settle)
        finally:
            if writer:
                writer.close()
        ranking = []
        for i, (source, params) in enumerate(candidates):
            score = (results["held_height"][i] + .002 * results["loaded_ticks"][i] +
                     .003 * results["pregrasp"][i] + .25 * results["hold_ticks"][i] / 30)
            if not results["safe"][i]:
                score -= .035 + .04 * np.log1p(results["forbidden_force_n"][i]) + results["distractor_shift_m"][i]
            ranking.append((score, source, params))
        parents = sorted(parents + ranking, key=lambda item: item[0], reverse=True)[:5]
        best = max(range(len(poses)), key=lambda i: (results["success"][i], ranking[i][0]))
        metrics = {k: v[best] for k, v in results.items()}
        print(f"SEARCH {group.crop} {generation} {sum(results['success'])}/{env.num_envs} {metrics}", flush=True)
        record = {"crop": group.crop, "side": plans["side"], "seed": seed, "generation": generation,
                  "settled_reset": settle, "waypoints_rad": poses[best].tolist(), "metrics": metrics}
        if any(results["success"]):
            write_json(output, record)
            return True
        if ranking[best][0] > high_score:
            high_score = ranking[best][0]
            write_json(output.with_suffix(".diagnostic.json"), record)
    return False


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import make_batch_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=CROPS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--refine", type=Path, help="Usar también las posturas de un ensayo físico anterior")
    parser.add_argument("--record-search", type=Path, help="Capturar el ensayo físico con dos ojos durante la calibración")
    parser.add_argument("--capture", action="store_true", help="Repetir la trayectoria aprobada y guardar HDF5")
    parser.add_argument("--exercises", nargs="+", choices=("pinch", "lift15", "return15"), default=["lift15"])
    parser.add_argument("--seed", type=int, default=512)
    parser.add_argument("--narrow-support", action="store_true", help="Apoyo de tres puntos para acceso de los dedos")
    parser.add_argument("--settled-reset", action="store_true", help="Dejar que las frutas se asienten antes de medir")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("El destino ya existe")
    if args.record_search and (args.capture or args.record_search.exists()):
        parser.error("--record-search requiere una búsqueda y una carpeta nueva")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    narrow = plan.get("narrow_support", False) if args.capture else args.narrow_support
    with_eyes = bool(args.capture or args.record_search)
    cfg = make_batch_config(args.num_envs, crop=args.crop, contacts=True, nominal=True, raised_support=True,
                            rgbd=with_eyes)
    if narrow:
        from dexman.tasks.agriculture.scene import configure_open_support
        configure_open_support(getattr(cfg.scenes, args.crop), args.crop)
    with launch_simulation(cfg, args):
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
        env = AgriculturalPPOEnv(SimulationContext(cfg.sim), cfg, seed=args.seed, capture=with_eyes)
        with torch.inference_mode():
            if args.capture:
                from dexman.data.lift_trials import LiftTrialCapture
                if plan["crop"] != args.crop or plan["metrics"]["success"] is not True:
                    raise ValueError("Se requiere una calibración física aprobada del mismo cultivo")
                poses = np.repeat([plan["waypoints_rad"]], env.num_envs, axis=0)
                passed = True
                for name in args.exercises:
                    env.batch.episode_ids[:] = [-1] * env.num_envs
                    directory = args.output / name
                    writer = LiftTrialCapture(env, directory, args.plan, name)
                    try:
                        result = trial(env, poses, plan["side"], writer, name, settle=plan.get("settled_reset", False))
                    finally:
                        writer.close()
                    write_json(directory / "summary.json", result)
                    passed &= all(result["success"])
                    print(f"CAPTURE {args.crop} {name}: {sum(result['success'])}/{env.num_envs}", flush=True)
                return 0 if passed else 2
            if not search(env, plan["plans"][args.crop], args.seed, args.rounds, args.output,
                          settle=args.settled_reset, refine=args.refine, record_dir=args.record_search):
                return 2
            result = json.loads(args.output.read_text(encoding="utf-8"))
            write_json(args.output, {**result, "narrow_support": narrow})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
