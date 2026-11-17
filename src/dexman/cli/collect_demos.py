"""Run a PPO teacher with RGB-D rendering to record successful VLA demonstrations."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--output", default="datasets/demos")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1, help="Robot replicas on this GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default=None, help="Text paired with every recorded RGB/state frame")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_attempts < args.episodes or args.num_envs < 1:
        parser.error("episodes debe ser positivo y max-attempts >= episodes")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    from dexman.tasks.lift.config import CorrectedVisualLiftEnvCfg
    cfg = CorrectedVisualLiftEnvCfg()
    from dexman.agents.checkpoint import validate_teacher
    validate_teacher(checkpoint, cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.instruction is not None:
        if not args.instruction.strip():
            parser.error("instruction must contain text")
        cfg.instruction = args.instruction
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import dexman.tasks  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from dexman.agents.teacher import load_teacher_actor
        from dexman.data.capture import collect_parallel
        from dexman.data.writer import EpisodeWriter
        from dexman.data.metadata import teacher_metadata
        from dexman.data.validation import audit_dataset
        from dexman.experiments import provenance, seed_everything, write_json

        run_dir = Path(args.output) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata = teacher_metadata(checkpoint, cfg)
        seed_everything(args.seed)
        write_json(run_dir / "run.json", {"arguments": vars(args), "provenance": provenance(), "metadata": metadata})
        writers = [EpisodeWriter(run_dir / f"env_{i:03d}", {**metadata, "replica": i},
                                 success_only=not args.keep_failures) for i in range(args.num_envs)]
        env = gym.make("Dexman-UR10e-Shadow-Lift-RGBD-v1", cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            policy = load_teacher_actor(checkpoint, cfg.sim.device)
            wrapped.reset()
            collect_parallel(wrapped, policy, writers, args.episodes, args.max_attempts)
        finally:
            for writer in writers:
                writer.abort()
            env.close()
        write_json(run_dir / "audit.json", audit_dataset(run_dir))
