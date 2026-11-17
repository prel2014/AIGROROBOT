"""Optional zero-action scene preview; this starts simulation when invoked."""
import dexman.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_zero_agent_cli

if __name__ == "__main__":
    raise SystemExit(run_zero_agent_cli())
