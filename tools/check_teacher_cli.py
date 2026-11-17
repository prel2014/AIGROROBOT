"""Resolve the actual PPO CLI/configuration without launching simulation or training.

Run with Isaac Sim's Python and the project src on PYTHONPATH. Arguments are the
native RSL-RL arguments, e.g. --task Dexman-UR10e-Shadow-Lift-v0 --num_envs 16 --viz none.
"""
import sys


def main():
    import dexman.tasks  # noqa: F401
    from isaaclab.app import validate_launch
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import _parse_args
    from isaaclab_rl.entrypoints.common import apply_env_overrides
    from isaaclab_tasks.utils import resolve_task_config

    args = _parse_args(sys.argv[1:])
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    apply_env_overrides(args, env_cfg)
    scan, _ = validate_launch(env_cfg, args)
    print(f"CLI/config OK: task={args.task}, robots={env_cfg.scene.num_envs}, "
          f"visualizers={scan.visualizer_intent}, algorithm={agent_cfg.algorithm.class_name}")
    print("No simulation launched; no training started.")


if __name__ == "__main__":
    main()
