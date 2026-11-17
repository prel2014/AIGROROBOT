"""Launch only when explicitly invoked by the user."""
import argparse
import dexman.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_train_cli

if __name__ == "__main__":
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument("--task", default="Dexman-UR10e-Shadow-Lift-v1")
    preflight.add_argument("--checkpoint")
    options, _ = preflight.parse_known_args()
    if options.task == "Dexman-UR10e-Shadow-Lift-v1":
        from dexman.cli.train_teacher import main
        raise SystemExit(main())
    raise SystemExit(run_train_cli())
