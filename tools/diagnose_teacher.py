"""Measure PPO reward, grasp, action clipping and joint tracking without changing training."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--probe-hand", action="store_true", help="Also command open/close/reopen at HOME")
    parser.add_argument("--no-self-collisions", action="store_true", help="Diagnostic counterfactual only")
    parser.add_argument("--native-physx-drives", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists() or args.num_envs < 1 or args.steps < 1:
        parser.error("Use a new output file and positive num_envs/steps")
    from dexman.tasks.lift.config import LiftEnvCfg
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.native_physx_drives:
        cfg.sim.use_newton_actuators = False
    if args.no_self_collisions:
        for overrides in cfg.scene.robot.spawn.articulation_props.values():
            for override in overrides:
                override.enabled_self_collisions = False
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import torch
        import dexman.tasks  # noqa: F401
        from dexman.agents.ppo import LiftPPORunnerCfg
        from dexman.robot.joints import ACTION_NAMES, JOINT_NAMES
        from dexman.tasks.lift.rewards import grasp_metrics
        from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, check_rsl_rl_version,
            create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)

        env = gym.make("Dexman-UR10e-Shadow-Lift-v0", cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            agent = handle_deprecated_rsl_rl_cfg(LiftPPORunnerCfg(), check_rsl_rl_version())
            agent.device = cfg.sim.device
            runner = create_rsl_rl_runner(wrapped, agent)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=cfg.sim.device)
            obs, _ = wrapped.reset()
            raw = env.unwrapped
            samples = []
            episodes = []
            with torch.inference_mode():
                for step in range(args.steps):
                    action = policy(obs)
                    obs, reward, dones, extras = wrapped.step(action)
                    pos, palm, tips = raw._geometry()
                    pd, fd, grasp = grasp_metrics(pos, palm, tips)
                    local = pos - raw.scene.env_origins
                    height = local[:, 2]
                    lifted = ((height - cfg.object_initial_height) / cfg.lift_height).clamp(0, 1) * grasp
                    goal_dist = (raw.goal - pos).norm(dim=-1)
                    target_error = (raw.targets - raw.robot.data.joint_pos.torch[:, raw.joint_ids]).abs()
                    samples.append({
                        "palm_distance": pd.cpu(), "tip_distances": (tips - pos[:, None]).norm(dim=-1).cpu(),
                        "grasp": grasp.cpu(), "lift_m": (height - cfg.object_initial_height).cpu(),
                        "goal_distance": goal_dist.cpu(), "xy_from_goal": (local[:, :2] -
                            (raw.goal - raw.scene.env_origins)[:, :2]).norm(dim=-1).cpu(),
                        "approach_reward": (1.5 * (1 - torch.tanh(4 * pd))).cpu(),
                        "fingers_reward": (1 - torch.tanh(8 * fd)).cpu(),
                        "lift_reward": (5 * lifted).cpu(),
                        "goal_reward": (4 * (1 - torch.tanh(8 * goal_dist)) * grasp * lifted).cpu(),
                        "tracking_error_rad": target_error.cpu(), "action_raw": action.cpu().clone(),
                        "hold_steps": raw.hold_steps.cpu().clone(), "done": dones.cpu().clone(),
                        "recorded_reward": reward.cpu().clone()})
                    for i in dones.nonzero(as_tuple=False).flatten().cpu().tolist():
                        episodes.append({"robot": i + 1, "step": step + 1,
                                         "success": bool(extras["success"][i].item())})
                    policy.reset(dones)
            stacked = {key: torch.stack([row[key] for row in samples]) for key in samples[0]}
            # Same-Step autoreset replaces terminal geometry. Exclude reset rows for geometry summaries.
            valid = ~stacked["done"].bool()
            report = {"checkpoint": str(checkpoint), "robots": args.num_envs, "steps_per_robot": args.steps,
                "self_collisions_disabled_for_diagnostic": args.no_self_collisions,
                "episodes": episodes, "success_count": sum(int(e["success"]) for e in episodes),
                "mean": {key: float(stacked[key][valid].float().mean()) for key in (
                    "palm_distance", "goal_distance", "xy_from_goal", "approach_reward", "fingers_reward",
                    "lift_reward", "goal_reward", "recorded_reward")},
                "grasp_fraction": float(stacked["grasp"][valid].float().mean()),
                "thumb_close_fraction": float((stacked["tip_distances"][..., -1][valid] < .09).float().mean()),
                "two_other_fingers_close_fraction": float(((stacked["tip_distances"][..., :-1] < .09)
                    .sum(-1)[valid] >= 2).float().mean()),
                "minimum_tip_distance_m": stacked["tip_distances"].reshape(-1, 5).min(0).values.tolist(),
                "max_lift_m": float(stacked["lift_m"][valid].max()),
                "max_hold_steps": int(stacked["hold_steps"].max()),
                "action_clipped_fraction": float((stacked["action_raw"].abs() > 1).float().mean()),
                "clipped_fraction_by_action": dict(zip(ACTION_NAMES,
                    (stacked["action_raw"].abs() > 1).float().mean((0, 1)).tolist())),
                "mean_absolute_raw_action": dict(zip(ACTION_NAMES,
                    stacked["action_raw"].abs().mean((0, 1)).tolist())),
                "mean_tracking_error_rad_by_joint": dict(zip(JOINT_NAMES,
                    stacked["tracking_error_rad"][valid].mean(0).tolist())),
                "notes": "Deterministic checkpoint replay, 4 independent replicas; no modification to active training."}
            if args.probe_hand:
                with torch.inference_mode():
                    wrapped.reset()
                base = raw.home_action.repeat(args.num_envs, 1)
                close = base.clone()
                desired_angles = {f"robot0_{finger}J2": 1.0 for finger in ("FF", "MF", "RF", "LF")}
                desired_angles.update(robot0_THJ3=0.7, robot0_THJ0=-0.8, robot0_THJ4=0.4)
                for action_id, name in enumerate(ACTION_NAMES[:22]):
                    if name in desired_angles:
                        joint_id = JOINT_NAMES.index(name)
                        low, high = raw.limits[joint_id]
                        close[:, action_id] = 2 * (desired_angles[name] - low) / (high - low) - 1
                close[:, 22:] = 2 * 0.8 / (torch.pi / 2) - 1
                phases = []
                with torch.inference_mode():
                    for name, command in (("open", base), ("close", close), ("reopen", base)):
                        errors = []
                        for t in range(90):
                            wrapped.step(command)
                            if t >= 70:
                                errors.append((raw.targets - raw.robot.data.joint_pos.torch[:, raw.joint_ids]).abs())
                        q = raw.robot.data.joint_pos.torch[:, raw.joint_ids]
                        pos, palm, tips = raw._geometry()
                        phases.append({"phase": name,
                            "mean_tracking_error_rad_by_joint": dict(zip(JOINT_NAMES,
                                torch.stack(errors).mean((0, 1)).cpu().tolist())),
                            "mean_q_rad_by_joint": dict(zip(JOINT_NAMES, q.mean(0).cpu().tolist())),
                            "minimum_fingertip_height_above_table_m": float((tips[..., 2] -
                                raw.scene.env_origins[:, None, 2] - cfg.table_height).min().item())})
                report["hand_probe"] = phases
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2), flush=True)
            print(f"Saved diagnostics to {output}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
