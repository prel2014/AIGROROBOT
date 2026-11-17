"""Evaluate a trained PPO checkpoint through Isaac Lab's official entrypoint."""
import dexman.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_play_cli

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
