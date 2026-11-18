"""Train or validate the agricultural bimanual Transformer teacher with RSL-RL PPO."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256, write_json
from dexman.paths import ROOT


RUNS = ROOT / "logs/rsl_rl/agriculture_bimanual"


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--max_iterations", type=int, default=3000, help="Additional PPO updates")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Bimanual checkpoint, run directory, or latest")
    parser.add_argument("--validate", action="store_true", help="Check rollouts with no PPO updates or checkpoint writes")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < len(OBJECTIVES) or min(args.max_iterations, args.save_interval) < 1 or args.seed < 0:
        parser.error("Configura al menos cuatro réplicas, iteraciones y guardado positivos y semilla no negativa")
    if (Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}
            or any(c in args.run_name for c in "/\\:") or args.run_name.strip(" .") != args.run_name):
        parser.error("--run-name debe ser un nombre de carpeta")
    output = RUNS / args.run_name
    if output.exists() and not args.validate:
        parser.error(f"La corrida ya existe: {output}")
    try:
        checkpoint = resolve_checkpoint(args.resume, RUNS, "model_*.pt") if args.resume else None
    except (ValueError, OSError) as error:
        parser.error(str(error))
    show_window = "kit" in (args.visualizer or ())
    cfg = make_batch_config(args.num_envs, contacts=True, monitor=show_window)
    with launch_simulation(cfg, args), ExitStack() as ui_lifetime:
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.agents.agriculture import AgriculturalRunner, contract, ppo_config, read_checkpoint
        from dexman.agents.checkpoint import validate_teacher_runtime
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        validate_teacher_runtime()
        seed_everything(args.seed)
        saved = read_checkpoint(checkpoint) if checkpoint else None
        sim = SimulationContext(cfg.sim)
        sim.set_camera_view((23., -25., 36.), (15., 14., 0.))
        env = AgriculturalPPOEnv(sim, cfg, seed=args.seed, render=show_window)
        if show_window:
            from dexman.sensors.teacher_camera_panel import AgriculturalCameraPanel

            panel = AgriculturalCameraPanel(env.batch)
            ui_lifetime.callback(panel.close)
            env.on_step = panel.refresh
            panel.refresh()
        agent = ppo_config(args.save_interval)
        if not args.validate:
            output.mkdir(parents=True, exist_ok=False)
        runner = AgriculturalRunner(env, agent, log_dir=None if args.validate else str(output), device=sim.device)
        if saved:
            runner.resume(saved)
        if args.validate:
            env.episode_length_buf[-1] = env.max_episode_length - 1
            episode_ids = list(env.batch.episode_ids)
            with torch.inference_mode():
                obs = env.get_observations()
                actor = runner.get_inference_policy()
                actions = actor(obs)
                value = runner.alg.critic(obs)
                updated, reward, done, extras = env.step(actions)
            if (actions.shape != (env.num_envs, env.num_actions) or value.shape != (env.num_envs, 1)
                    or not torch.isfinite(updated["policy"]).all() or not torch.isfinite(reward).all()
                    or bool(done[:-1].any()) or not bool(extras["time_outs"][-1])
                    or env.batch.episode_ids != [*episode_ids[:-1], episode_ids[-1] + 1]):
                raise RuntimeError("Falló la validación de observaciones, PPO o reinicio indexado")
            print(f"AGRICULTURE_PPO_VALIDATED {env.num_envs} envs; "
                  f"obs={tuple(obs['policy'].shape)}; actions={tuple(actions.shape)}; no updates", flush=True)
            return 0
        write_json(output / "run.json", {"contract": contract(), "args": vars(args), "agent": agent,
                                        "provenance": provenance(), "resume": None if checkpoint is None else
                                        {"path": str(checkpoint), "sha256": sha256(checkpoint)}})
        print(f"PPO agrícola: {env.num_envs} réplicas, {agent['num_steps_per_env'] * env.num_envs} "
              f"transiciones/actualización, salida {output}", flush=True)
        runner.learn(args.max_iterations, init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
