"""Collect synchronized demonstrations with a saved agricultural PPO teacher."""
import argparse
from dataclasses import asdict
from pathlib import Path

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256
from dexman.paths import ROOT


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.grasp_curriculum import read_grasp_calibration
    from dexman.tasks.agriculture.objectives import BERRY_EXERCISES, OBJECTIVES, PLACE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Bimanual checkpoint, run directory, or latest")
    parser.add_argument("--output", required=True, type=Path, help="New HDF5 dataset directory")
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=640)
    parser.add_argument("--seed", type=int, default=712)
    parser.add_argument("--rover", action="store_true", help="Registrar las acciones de brazos y cabezal del rover")
    parser.add_argument("--crop", choices=tuple(OBJECTIVES), help="Seleccionar un cultivo para todas las réplicas")
    parser.add_argument("--grasp-calibration", type=Path, help="Capturar exclusivamente la habilidad de agarre calibrado")
    parser.add_argument("--grasp-crop", choices=tuple(OBJECTIVES), default="berries")
    parser.add_argument("--grasp-start", choices=("closed", "pregrasp"), default="pregrasp")
    parser.add_argument("--exercise", choices=tuple(BERRY_EXERCISES),
                        help="Demostración compuesta con el teacher de arándano y una regla de cierre o retorno")
    parser.add_argument("--keep-failures", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.rover and args.grasp_calibration:
        parser.error("La postura calibrada del torso fijo no corresponde al montaje de brazos del rover")
    minimum = 1 if args.crop or args.grasp_calibration else len(OBJECTIVES)
    if args.num_envs < minimum or not 0 < args.episodes <= args.max_attempts or args.seed < 0:
        parser.error(f"Configura al menos {minimum} réplicas, 0 < episodes <= max-attempts y semilla no negativa")
    if args.output.exists():
        parser.error(f"La carpeta de salida ya existe: {args.output}")
    try:
        checkpoint = resolve_checkpoint(args.checkpoint, ROOT / "logs/rsl_rl/agriculture_bimanual", "model_*.pt")
        grasp = (read_grasp_calibration(args.grasp_calibration, args.grasp_crop, args.grasp_start)
                 if args.grasp_calibration else None)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    cfg = make_batch_config(args.num_envs, rgbd=True, contacts=True, grasp=grasp, exercise=args.exercise,
                            crop=args.crop, rover=args.rover)
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from dexman.agents.agriculture import actor_from_checkpoint, contract, read_checkpoint
        from dexman.agents.checkpoint import validate_teacher_runtime
        from dexman.data.agriculture import collect_demonstrations
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        validate_teacher_runtime()
        seed_everything(args.seed)
        saved = read_checkpoint(checkpoint)
        profile = saved["infos"]["simulation_profile"]
        expected = contract(grasp=grasp is not None, rover=args.rover)
        if saved["infos"]["agriculture"] != expected or profile.get("grasp") != (asdict(grasp) if grasp else None):
            raise ValueError("El checkpoint debe coincidir con el robot (--rover), la tarea y la calibración de captura")
        cfg.nominal = profile["nominal_scene"]
        cfg.physics_variation = profile["physics_variation"]
        sim = SimulationContext(cfg.sim)
        actor = actor_from_checkpoint(saved, sim.device)
        env = AgriculturalPPOEnv(sim, cfg, seed=args.seed, capture=True)
        if args.rover and profile.get("rover_asset_sha256") != env.rover_asset_sha256:
            raise ValueError("El modelo USD del rover es distinto al usado por el teacher")

        def demonstrate(observations):
            rates = actor(observations)
            if args.exercise == "pinch":
                rates[:, 1] = 0.
            elif args.exercise == "return":
                lowering = env.phase >= PLACE
                rates[lowering, 1] = -1.
                rates[lowering, 0] = (env.grasp_motion.progress[lowering, 1] > 0.).float() * 2. - 1.
            return rates

        metadata = {"teacher": {"path": str(checkpoint), "sha256": sha256(checkpoint), "iteration": saved["iter"],
                                "contract": expected}, "seed": args.seed, "replicas": args.num_envs,
                    "task_scope": "calibrated_grasp" if grasp else "full_task",
                    "exercise": args.exercise or "default", "demonstrator": "ppo_with_phase_rules" if args.exercise else "ppo",
                    "simulation_profile": profile, "provenance": provenance()}
        count = collect_demonstrations(env, demonstrate, args.output, metadata, episodes=args.episodes,
                                       max_attempts=args.max_attempts, keep_failures=args.keep_failures)
        if count < args.episodes:
            print(f"Se agotaron los intentos. Se guardaron {count}/{args.episodes} episodios.", flush=True)
            raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
