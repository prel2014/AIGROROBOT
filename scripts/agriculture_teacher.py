"""Train or validate the agricultural bimanual Transformer teacher with RSL-RL PPO."""
import argparse
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from shutil import which

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256, write_json
from dexman.paths import ROOT


RUNS = ROOT / "logs/rsl_rl/agriculture_bimanual"


def validate_grasp_controls(env):
    import torch

    with torch.inference_mode():
        ids = torch.arange(env.num_envs, device=env.device)
        settled = env.home.new_zeros((env.num_envs, env.num_actions))
        for _ in range(env.max_episode_length):
            _, _, _, extras = env.step(settled)
            if extras["success"].any():
                raise RuntimeError("Una acción nula activó un éxito físico inesperado")
        env._reset(ids)
        requested = torch.ones_like(settled)
        completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(completed)
        for _ in range(env.max_episode_length):
            _, _, done, extras = env.step(requested)
            success |= ~completed & extras["success"]
            completed |= done.bool()
            if completed.all():
                break
        count = int(success.sum())
        print(f"AGRICULTURE_GRASP_CONTROLS {env.cfg.grasp.crop} {count}/{env.num_envs}; zero-action=0", flush=True)
        if not count:
            raise RuntimeError("La trayectoria calibrada no consiguió un agarre sostenido con el controlador PPO")


def evaluate_teacher(env, actor, episodes):
    import torch
    from dexman.tasks.agriculture.objectives import LIFT

    groups = env.batch.groups
    order = [group.start + offset for offset in range(max(group.stop - group.start for group in groups))
             for group in groups if group.start + offset < group.stop]
    uniform, remainder = divmod(episodes, env.num_envs)
    pending = torch.full((env.num_envs,), uniform, device=env.device, dtype=torch.long)
    pending[order[:remainder]] += 1
    tallies = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.long)
    with torch.inference_mode():
        obs = env.get_observations()
        for _ in range(env.max_episode_length * (uniform + bool(remainder))):
            obs, _, done, details = env.step(actor(obs))
            finished = done.bool() & (pending > 0)
            outcomes = torch.stack((finished, details["success"], details["phase_at_step"] >= LIFT,
                                    details["dropped"], details["invalid_pose"], details["time_outs"]), dim=1)
            tallies += (outcomes & finished[:, None]).long()
            pending -= finished.long()
            if not pending.any():
                break
    if pending.any():
        raise RuntimeError("La evaluación no completó los episodios solicitados")
    names = ("completed", "success", "reached_lift", "drops", "invalid", "timeouts")
    return {group.crop: dict(zip(names, tallies[group.start:group.stop].sum(0).tolist(), strict=True))
            for group in groups}


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.grasp_curriculum import read_grasp_calibration
    from dexman.tasks.agriculture.objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--max_iterations", type=int, default=3000, help="Additional PPO updates")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Bimanual checkpoint, run directory, or latest")
    parser.add_argument("--record-video", action="store_true", help="Save a separate RGB-D MP4 per robot and PPO iteration")
    parser.add_argument("--grasp-calibration", type=Path, help="Passing raised-support calibration for grasp curriculum")
    parser.add_argument("--grasp-crop", choices=tuple(OBJECTIVES), default="berries")
    parser.add_argument("--grasp-start", choices=("pregrasp", "closed"), default="pregrasp")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--validate", action="store_true", help="Check rollouts with no PPO updates or checkpoint writes")
    behavior.add_argument("--evaluate-episodes", type=int, help="Evaluate deterministic episodes without training")
    parser.add_argument("--nominal-scene", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep the scene free of changing obstacles")
    parser.add_argument("--physics-variation", type=float,
                        help="Strength of Gaussian mass, friction and command variation in [0, 1]")
    add_launcher_args(parser)
    args = parser.parse_args()
    check_only = args.validate or args.evaluate_episodes is not None
    if args.evaluate_episodes is not None and args.evaluate_episodes < 1:
        parser.error("--evaluate-episodes requiere al menos un episodio")
    if args.num_envs < len(OBJECTIVES) or min(args.max_iterations, args.save_interval) < 1 or args.seed < 0:
        parser.error("Configura al menos cuatro réplicas, iteraciones y guardado positivos y semilla no negativa")
    if args.physics_variation is not None and not 0. <= args.physics_variation <= 1.:
        parser.error("--physics-variation debe estar entre 0 y 1")
    if (Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}
            or any(c in args.run_name for c in "/\\:") or args.run_name.strip(" .") != args.run_name):
        parser.error("--run-name debe ser un nombre de carpeta")
    output = RUNS / args.run_name
    if output.exists() and not check_only:
        parser.error(f"La corrida ya existe: {output}")
    try:
        checkpoint = resolve_checkpoint(args.resume, RUNS, "model_*.pt") if args.resume else None
        grasp = (read_grasp_calibration(args.grasp_calibration, args.grasp_crop, args.grasp_start)
                 if args.grasp_calibration else None)
    except (ValueError, OSError, KeyError, IndexError, TypeError) as error:
        parser.error(str(error))
    show_window = "kit" in (args.visualizer or ())
    if args.record_video and (show_window or (grasp is None and args.num_envs % len(OBJECTIVES)) or check_only):
        parser.error("--record-video requiere modo headless, réplicas iguales por cultivo y entrenamiento activo")
    ffmpeg = which("ffmpeg") if args.record_video else None
    if args.record_video and ffmpeg is None:
        parser.error("--record-video requiere ffmpeg con libx264 en PATH")
    if args.record_video:
        from dexman.data.epoch_video import VIDEO_EYE_SIZE, EpochVideoRecorder
        from dexman.tasks.agriculture.interface import EYE_KEYS

        cfg = make_batch_config(args.num_envs, rgbd=True, rgbd_size=VIDEO_EYE_SIZE, contacts=True, grasp=grasp)
        for scene in vars(cfg.scenes).values():
            for eye in EYE_KEYS:
                getattr(scene, eye).update_period = 0.
    else:
        cfg = make_batch_config(args.num_envs, contacts=True, monitor=show_window, grasp=grasp)
    with launch_simulation(cfg, args), ExitStack() as ui_lifetime:
        import torch
        from isaaclab.sim import SimulationContext
        from dexman.agents.agriculture import AgriculturalRunner, contract, ppo_config, read_checkpoint
        from dexman.agents.checkpoint import validate_teacher_runtime
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        validate_teacher_runtime()
        seed_everything(args.seed)
        saved = read_checkpoint(checkpoint) if checkpoint else None
        profile = (saved["infos"]["simulation_profile"] if saved else
                   {"nominal_scene": grasp is not None, "physics_variation": 0.})
        if args.nominal_scene is None:
            args.nominal_scene = profile["nominal_scene"]
        if args.physics_variation is None:
            args.physics_variation = profile["physics_variation"]
        cfg.nominal, cfg.physics_variation = args.nominal_scene, args.physics_variation
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
        if not check_only:
            output.mkdir(parents=True, exist_ok=False)
        runner = AgriculturalRunner(env, agent, log_dir=None if check_only else str(output), device=sim.device)
        if saved:
            runner.resume(saved)
        if args.record_video:
            runner.record_epochs(EpochVideoRecorder(env.batch, output / "videos", ffmpeg,
                                                     rollout_steps=agent["num_steps_per_env"]))
        if args.evaluate_episodes is not None:
            scores = evaluate_teacher(env, runner.get_inference_policy(), args.evaluate_episodes)
            print(f"AGRICULTURE_EVALUATION {json.dumps(scores)}", flush=True)
            return 0
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
            if grasp is not None:
                validate_grasp_controls(env)
                env._reset(torch.arange(env.num_envs, device=env.device))
                env._update_observations()
                baseline = evaluate_teacher(env, actor, env.num_envs)
                print(f"AGRICULTURE_EVALUATION {json.dumps(baseline)}", flush=True)
            print(f"AGRICULTURE_PPO_VALIDATED {env.num_envs} envs; "
                  f"obs={tuple(obs['policy'].shape)}; actions={tuple(actions.shape)}; no updates", flush=True)
            return 0
        write_json(output / "run.json", {"contract": contract(grasp=grasp is not None), "args": vars(args), "agent": agent,
                                        "simulation_profile": runner.simulation_profile(),
                                        "provenance": provenance(), "resume": None if checkpoint is None else
                                        {"path": str(checkpoint), "sha256": sha256(checkpoint)}})
        print(f"PPO agrícola: {env.num_envs} réplicas, {agent['num_steps_per_env'] * env.num_envs} "
              f"transiciones/actualización, salida {output}", flush=True)
        runner.learn(args.max_iterations, init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
