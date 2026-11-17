"""Train or evaluate multitask RGB + text + angles reinforcement learning with GRU memory."""
import argparse
import json
from pathlib import Path


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True, help="A NEW run directory")
    parser.add_argument("--encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--instruction", help="Evaluation only: exact instruction from the task catalog")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.episodes, args.max_steps) < 1:
        parser.error("Counts must be positive")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint")
    if args.mode == "train" and args.instruction:
        parser.error("Training samples all tasks; --instruction is for evaluation")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output already exists; choose a new run directory")

    import torch
    from dexman.tasks.manipulation.catalog import resolve_instruction
    from dexman.tasks.manipulation.config import MemoryTaskEnvCfg
    from dexman.tasks.manipulation.environment import MemoryTaskEnv
    from dexman.robot.commands import JointCommands
    from dexman.memory_rl.observations import VisualFeatures
    from dexman.memory_rl.vec_env import VisualVecEnv
    from dexman.memory_rl.checkpoint import checkpoint_contract, load_checkpoint
    from dexman.memory_rl.config import training_config
    if args.instruction:
        resolve_instruction(args.instruction)
    cfg = MemoryTaskEnvCfg()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.instruction_override = args.instruction
    checkpoint = Path(args.checkpoint).resolve(strict=True) if args.checkpoint else None
    saved_contract = None
    if checkpoint:
        saved, saved_contract = load_checkpoint(checkpoint)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg, args):
        env = MemoryTaskEnv(cfg)
        try:
            commands = JointCommands(env.limits, cfg.action_center, env.device)
            encoder_source = checkpoint.parent / "encoder" if checkpoint else args.encoder
            encoder = VisualFeatures(encoder_source, commands, env.device, local_only=bool(checkpoint))
            config = training_config()
            contract = checkpoint_contract(
                joint_limits=env.limits, joint_center=cfg.action_center, smoothing=cfg.action_smoothing,
                step_dt=env.step_dt, episode_length_s=cfg.episode_length_s,
                object_xy_range=cfg.object_xy_range, feature_dimension=encoder.dimension, training=config)
            if saved_contract is not None and contract != saved_contract:
                raise ValueError("Checkpoint task/control/encoder contract differs from this environment")
            output.mkdir(parents=True)
            (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
            (output / "run.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
            if args.mode == "train":
                encoder.save(output / "encoder")
            wrapped = VisualVecEnv(env, encoder, commands, output / "episodes.jsonl")
            if args.mode == "train":
                from dexman.memory_rl.training import MemoryRunner
                runner = MemoryRunner(wrapped, config, contract, str(output), env.device)
                if checkpoint:
                    runner.resume(checkpoint)
                runner.learn(args.iterations, init_at_random_ep_len=False)
            else:
                from dexman.memory_rl.evaluation import evaluate, load_actor
                actor = load_actor(saved, contract, env.device)
                summary = {"checkpoint": str(checkpoint), "seed": args.seed,
                           **evaluate(wrapped, actor, args.episodes, args.max_steps)}
                (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(summary, indent=2), flush=True)
                if summary["completed"] < args.episodes:
                    raise RuntimeError("max-steps reached before evaluation completed")
        finally:
            env.close()
