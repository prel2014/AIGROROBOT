"""Train the corrected teacher with resumable curriculum and immutable run metadata."""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256, write_json


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Dexman-UR10e-Shadow-Lift-v1", choices=["Dexman-UR10e-Shadow-Lift-v1"])
    parser.add_argument("--rl_library", default="rsl_rl", choices=["rsl_rl"])
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=3000, help="Additional PPO updates, including on resume")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Teacher checkpoint file or run directory")
    source.add_argument("--resume", nargs="?", const="latest", help="Resume a file/directory, or latest saved teacher")
    parser.add_argument("--run_timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--bootstrap-updates", type=int, default=600,
                        help="Fresh-teacher behavior-cloning updates; 0 explicitly disables bootstrap. Ignored on resume.")
    parser.add_argument("--bootstrap-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=5e-4)
    parser.add_argument("--start-lift-cm", type=float, default=1.5, choices=[1.5, 3., 6., 10., 15.],
                        help="Fresh-run initial curriculum height; ignored when resuming saved curriculum")
    parser.add_argument("--exploration-std-scale", type=float, default=1.,
                        help="Fresh-run post-imitation standard-deviation multiplier before closed-loop gates, in (0,1]")
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.save_interval) < 1:
        parser.error("Environment, update and save counts must be positive")
    if args.bootstrap_updates < 0 or args.bootstrap_batch_size < 1 or not math.isfinite(args.bootstrap_learning_rate) or args.bootstrap_learning_rate <= 0:
        parser.error("Bootstrap updates must be nonnegative, batch size and learning rate positive")
    if not math.isfinite(args.exploration_std_scale) or not 0 < args.exploration_std_scale <= 1:
        parser.error("--exploration-std-scale must be finite and in (0,1]")
    if args.exploration_std_scale != 1. and (args.checkpoint or args.resume or args.bootstrap_updates == 0):
        parser.error("A non-unit --exploration-std-scale requires a fresh run with bootstrap enabled")
    if Path(args.run_timestamp).name != args.run_timestamp or any(c in args.run_timestamp for c in "/\\:"):
        parser.error("run_timestamp must be a directory name")
    from dexman.paths import ROOT
    from dexman.tasks.lift.config import CorrectedLiftEnvCfg, TeacherCameraEnvCfg
    from dexman.agents.ppo import CorrectedLiftPPORunnerCfg
    from dexman.agents.checkpoint import validate_teacher, validate_teacher_runtime
    from dexman.agents.teacher import make_teacher_runner
    show_camera = "kit" in (args.visualizer or ())
    validate_teacher_runtime()
    cfg = TeacherCameraEnvCfg() if show_camera else CorrectedLiftEnvCfg()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    if not (args.checkpoint or args.resume):
        cfg.curriculum_initial_level = cfg.curriculum_heights.index(args.start_lift_cm / 100)
    runs_root = ROOT / "logs/rsl_rl/ur10e_shadow_lift_v1"
    checkpoint = None
    if args.checkpoint or args.resume:
        try:
            checkpoint = resolve_checkpoint(args.checkpoint or args.resume, runs_root, "model_*.pt")
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if checkpoint:
        try:
            validate_teacher(checkpoint, cfg)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    output = runs_root / args.run_timestamp
    if output.exists():
        parser.error("Run directory already exists; choose another run_timestamp")
    seed_everything(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import dexman.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
        from isaaclab.utils.io import dump_yaml
        env = gym.make(args.task, cfg=cfg)
        wrapped = None
        try:
            if show_camera and not getattr(args, "headless", False):
                from dexman.sensors.teacher_camera_panel import TeacherCameraVecEnv
                wrapped = TeacherCameraVecEnv(env)
            else:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
            agent = CorrectedLiftPPORunnerCfg()
            # At least 2048 transitions and 512 samples per minibatch, including with four replicas.
            agent.num_steps_per_env = max(agent.num_steps_per_env, (2048 + args.num_envs - 1) // args.num_envs)
            agent.device, agent.seed = cfg.sim.device, args.seed
            agent.max_iterations, agent.save_interval = args.max_iterations, args.save_interval
            agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
            runner = make_teacher_runner(wrapped, agent.to_dict(), str(output), cfg.sim.device)
            if checkpoint:
                runner.resume(checkpoint)
                print("Teacher bootstrap skipped: resuming complete PPO state", flush=True)
                print(f"Teacher curriculum restored: level {env.unwrapped.curriculum_level}; "
                      "--start-lift-cm ignored and saved exploration preserved", flush=True)
            elif args.bootstrap_updates:
                print(f"Teacher fresh initial lift: {args.start_lift_cm:g} cm "
                      f"(curriculum level {cfg.curriculum_initial_level})", flush=True)
                from dexman.agents.bootstrap import bootstrap_teacher
                bootstrap_teacher(runner, output, args.bootstrap_updates, args.bootstrap_batch_size,
                                  args.bootstrap_learning_rate, args.seed, args.exploration_std_scale)
            else:
                print(f"Teacher fresh initial lift: {args.start_lift_cm:g} cm "
                      f"(curriculum level {cfg.curriculum_initial_level}); bootstrap disabled", flush=True)
                runner.bootstrap_report = {"source": "scripted_calibration", "status": "explicitly_disabled",
                                           "updates": 0, "seed": args.seed}
                write_json(output / "bootstrap.json", runner.bootstrap_report)
            dump_yaml(str(output / "params/env.yaml"), cfg)
            dump_yaml(str(output / "params/agent.yaml"), agent)
            write_json(output / "run.json", {"version": 2, "task": args.task, "library": "rsl_rl",
                "arguments": vars(args), "provenance": provenance(), "control_contract": runner.contract,
                "teacher_architecture": runner.architecture,
                "bootstrap": runner.bootstrap_report,
                "resume": None if checkpoint is None else {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                "initial_learning_state": env.unwrapped.learning_state(),
                "resume_semantics": "weights, optimizer, curriculum and RNG; fresh simulated episodes"})
            print(f"Teacher rollout: {agent.num_steps_per_env * args.num_envs} transitions/update; "
                  f"source: {checkpoint or 'new policy'}", flush=True)
            runner.learn(args.max_iterations, init_at_random_ep_len=False)
            write_json(output / "training_state.json", env.unwrapped.learning_state())
        finally:
            (wrapped if wrapped is not None else env).close()
