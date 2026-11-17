"""Evaluate teacher and visual student on identical held-out seeds at the final 15 cm goal."""
import argparse
from pathlib import Path

from dexman.experiments import provenance, seed_everything, sha256, write_json
from dexman.evaluation.metrics import summarize


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New evaluation directory")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=200, help="Episodes PER seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--max-steps", type=int, default=100000, help="Vector steps per seed")
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--min-success-rate", type=float, default=.9)
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero unless every seed meets the target")
    from dexman.tasks.manipulation.scenarios import add_scenario_argument, configure_scenario
    add_scenario_argument(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    if (min(args.num_envs, args.episodes, args.max_steps, args.execute_steps) < 1
            or not 0 <= args.min_success_rate <= 1 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Invalid budgets, threshold or repeated seeds")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Evaluation output already exists")
    from dexman.tasks.lift.config import CorrectedLiftEnvCfg, JointAngleVisualLiftEnvCfg
    from dexman.agents.checkpoint import validate_teacher
    from dexman.evaluation.policies import TeacherPolicy, VisualPolicy
    from dexman.evaluation.rollout import evaluate_rollout
    cfg = CorrectedLiftEnvCfg() if args.policy == "teacher" else JointAngleVisualLiftEnvCfg()
    configure_scenario(cfg, args.scenario)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seeds[0]
    if args.num_envs == 1 and "kit" in (args.visualizer or ()):
        from isaaclab_visualizers.kit import KitVisualizerCfg
        cfg.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(1.8, 1.4, 1.6), lookat=(.05, 0., .8),
            origin_type="env", origin_env_index=0, focal_length=18.)
    if args.policy == "teacher":
        validate_teacher(checkpoint, cfg)
    seed_everything(args.seeds[0])
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import torch
        import dexman.tasks  # noqa: F401
        task = "Dexman-UR10e-Shadow-Lift-v1" if args.policy == "teacher" else "Dexman-UR10e-Shadow-Lift-VLA-v2"
        env = gym.make(task, cfg=cfg)
        output.mkdir(parents=True)
        write_json(output / "run.json", {"arguments": vars(args), "provenance": provenance(),
            "checkpoint_sha256": sha256(checkpoint), "task": task, "target_lift_m": .15,
            "success_criterion": cfg.success_criterion, "minimum_lift_required_m": .15,
            "success_lift_overshoot_m": cfg.success_lift_overshoot_m,
            "reset": "calibrated_pregrasp", "hold_seconds": cfg.success_hold_steps * cfg.sim.dt * cfg.decimation})
        try:
            policy = (TeacherPolicy(checkpoint, cfg.sim.device) if args.policy == "teacher" else
                      VisualPolicy(checkpoint, cfg.sim.device, args.num_envs, args.execute_steps))
            if args.policy == "student":
                from dexman.vla.contract import check_environment_contract
                check_environment_contract(policy.predictor.robot_contract, cfg)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            results = []
            for seed in args.seeds:
                seed_everything(seed)
                result = evaluate_rollout(env, policy, args.episodes, args.max_steps, seed)
                results.append(result)
                write_json(output / f"seed_{seed}.json", result)
                print(f"Seed {seed}: {result['completed']}/{args.episodes}, success={result['success_rate']}", flush=True)
            all_records = [row for result in results for row in result["episodes"]]
            summary = summarize(all_records, args.episodes * len(args.seeds))
            summary.update(target_lift_m=.15, reset="calibrated_pregrasp", policy=args.policy,
                success_criterion=cfg.success_criterion, minimum_lift_required_m=.15,
                success_lift_overshoot_m=cfg.success_lift_overshoot_m,
                hold_seconds=cfg.success_hold_steps * cfg.sim.dt * cfg.decimation,
                checkpoint_sha256=sha256(checkpoint), seeds=args.seeds, episodes_per_seed=args.episodes,
                min_success_rate=args.min_success_rate,
                passed=all(r["complete"] and r["success_rate"] >= args.min_success_rate for r in results),
                per_seed=[{key: result[key] for key in ("seed", "completed", "success_rate", "inference_ms", "elapsed_s")}
                          for result in results],
                torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            summary["elapsed_s"] = sum(result["elapsed_s"] for result in results)
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                summary["device_used_bytes_after"] = total - free
                summary["device_memory_scope"] = "All processes and simulator allocations; end sample, not peak"
            write_json(output / "evaluation.json", summary)
            if not summary["complete"] or (args.require_pass and not summary["passed"]):
                raise RuntimeError("Evaluation budget/acceptance criterion not met; see evaluation.json")
        finally:
            env.close()
