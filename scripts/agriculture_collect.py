"""Collect synchronized demonstrations with a saved agricultural PPO teacher."""
import argparse
from pathlib import Path

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256
from dexman.paths import ROOT


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Bimanual checkpoint, run directory, or latest")
    parser.add_argument("--output", required=True, type=Path, help="New HDF5 dataset directory")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=640)
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--keep-failures", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < len(OBJECTIVES) or not 0 < args.episodes <= args.max_attempts or args.seed < 0:
        parser.error("Configura al menos cuatro réplicas, 0 < episodes <= max-attempts y semilla no negativa")
    if args.output.exists():
        parser.error(f"La carpeta de salida ya existe: {args.output}")
    try:
        checkpoint = resolve_checkpoint(args.checkpoint, ROOT / "logs/rsl_rl/agriculture_bimanual", "model_*.pt")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    cfg = make_batch_config(args.num_envs, rgbd=True, contacts=True)
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from dexman.agents.agriculture import actor_from_checkpoint, contract, read_checkpoint
        from dexman.agents.checkpoint import validate_teacher_runtime
        from dexman.data.agriculture import collect_demonstrations
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        validate_teacher_runtime()
        seed_everything(args.seed)
        saved = read_checkpoint(checkpoint)
        sim = SimulationContext(cfg.sim)
        actor = actor_from_checkpoint(saved, sim.device)
        env = AgriculturalPPOEnv(sim, cfg, seed=args.seed, capture=True)
        metadata = {"teacher": {"path": str(checkpoint), "sha256": sha256(checkpoint), "iteration": saved["iter"],
                                "contract": contract()}, "seed": args.seed, "replicas": args.num_envs,
                    "provenance": provenance()}
        count = collect_demonstrations(env, actor, args.output, metadata, episodes=args.episodes,
                                       max_attempts=args.max_attempts, keep_failures=args.keep_failures)
        if count < args.episodes:
            print(f"Se agotaron los intentos. Se guardaron {count}/{args.episodes} episodios.", flush=True)
            raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
