"""Train or validate the agricultural bimanual Transformer teacher with RSL-RL PPO."""
import argparse
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile, which

from dexman.experiments import provenance, resolve_checkpoint, seed_everything, sha256, write_json
from dexman.paths import ROOT


RUNS = ROOT / "logs/rsl_rl/agriculture_bimanual"


def validate_rover_gaze(env):
    import torch
    from isaaclab.sensors import SensorBase
    from dexman.tasks.agriculture.interface import EYE_KEYS

    eyes = [group.scene[key] for group in env.batch.groups for key in EYE_KEYS]

    def capture():
        for _ in range(4):
            env.batch.sim.render()
            SensorBase.update_batch(eyes, 0.)
            rgb, depth = env.batch.rgbd()
            valid = torch.isfinite(depth) & (depth > 0.)
            varied = rgb.flatten(2).amax(-1) > rgb.flatten(2).amin(-1)
            if valid.flatten(2).any(-1).all() and varied.all():
                rotations, consistent = [], True
                for group in env.batch.groups:
                    left, right = (group.scene[key].data for key in EYE_KEYS)
                    baseline = (left.pos_w.torch - right.pos_w.torch).norm(dim=-1)
                    expected = (group.rover.eye_offsets[0] - group.rover.eye_offsets[1]).norm()
                    consistent &= bool(((baseline - expected).abs() < .003).all())
                    rotations.append(torch.stack((left.quat_w_ros.torch, right.quat_w_ros.torch), dim=1))
                if consistent:
                    return rgb.clone(), torch.cat(rotations)
        raise RuntimeError("RTX no produjo ambas imágenes RGB-D con la separación física de los ojos del rover")

    # Send opposite targets to alternate replicas to catch state leakage between robots.
    with torch.inference_mode():
        rgb0, rotation0 = capture()
        actions = env.home.new_zeros((env.num_envs, env.num_actions))
        signs = 1. - 2. * (torch.arange(env.num_envs, device=env.device) % 2)
        actions[:, -2] = signs * .20
        actions[:, -1] = -signs * .25
        for _ in range(round(1.5 / env.dt)):
            _, _, _, info = env.step(actions)
        target = info["joint_command"][:, -2:]
        measured = env.vla_state[:, -2:]
        if ((target.abs() < .08).any() or ((target - measured).abs().rad2deg() > 2.).any()
                or not torch.isfinite(measured).all()):
            raise RuntimeError(f"Los encoders del rover no siguieron los dos ejes: {measured}, consigna {target}")
        rgb, rotation = capture()
        rotation_deg = 2. * (rotation0 * rotation).sum(-1).abs().clamp(0., 1.).acos().rad2deg()
        if not (rotation_deg > 5.).all() or not (rgb != rgb0).flatten(2).any(-1).all():
            raise RuntimeError("Una cámara no giró o sus píxeles no cambiaron con el cabezal")
        print(f"ROVER_GAZE_VALIDATED error_max_deg={(target-measured).abs().rad2deg().max().item():.3f} "
              f"eye_rotation_min_deg={rotation_deg.min().item():.3f} rgbd={tuple(rgb.shape)}",
              flush=True)


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


def initialize_grasp_from_demonstrations(env, actor):
    """Supervise only on completed physical pinches, then hand the actor to PPO."""
    import torch
    from dexman.tasks.agriculture.objectives import LIFT

    states, commands, valid = [], [], []
    with torch.no_grad():
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        succeeded = torch.zeros_like(active)
        for _ in range(env.max_episode_length):
            action = torch.stack((torch.ones_like(env.phase), env.phase == LIFT), dim=1).float()
            states.append(env.get_observations()["policy"].clone())
            commands.append(action)
            valid.append(active.clone())
            _, _, done, info = env.step(action)
            succeeded |= active & info["success"]
            active &= ~done.bool()
            if not active.any():
                break
        if not succeeded.any():
            raise RuntimeError("Ningún episodio guiado alcanzó el criterio físico para inicializar el actor")
        mask = torch.stack(valid) & succeeded
        examples, labels = torch.stack(states)[mask], torch.stack(commands)[mask]
        actor.update_normalization({"policy": examples})

    optimizer = torch.optim.Adam(actor.parameters(), lr=3.e-4)
    actor.train()
    for update in range(160):
        idx = torch.randint(len(examples), (128,), device=env.device)
        predicted = actor({"policy": examples[idx]})
        loss = torch.nn.functional.mse_loss(predicted, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if update >= 50 and loss.item() < .0005:
            break
    env._reset(torch.arange(env.num_envs, device=env.device))
    env._update_observations()
    return {"successful_demonstrations": int(succeeded.sum()), "attempted": env.num_envs,
            "frames": len(examples), "supervised_updates": update + 1, "last_batch_mse": loss.item()}


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.grasp_curriculum import read_grasp_calibration
    from dexman.tasks.agriculture.interface import EYE_KEYS
    from dexman.tasks.agriculture.objectives import OBJECTIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--max_iterations", type=int, default=3000, help="Additional PPO updates")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rover", action="store_true", help="Teacher del rover con giro y elevación aprendidos")
    parser.add_argument("--crop", choices=tuple(OBJECTIVES), help="Seleccionar un cultivo para todas las réplicas")
    parser.add_argument("--resume", help="Bimanual checkpoint, run directory, or latest")
    parser.add_argument("--record-video", action="store_true", help="Save a separate RGB-D MP4 per robot and PPO iteration")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Ruta al ejecutable FFmpeg para grabar videos")
    parser.add_argument("--grasp-calibration", type=Path, help="Passing raised-support calibration for grasp curriculum")
    parser.add_argument("--grasp-crop", choices=tuple(OBJECTIVES), default="berries")
    parser.add_argument("--grasp-start", choices=("pregrasp", "closed"), default="pregrasp")
    parser.add_argument("--initialize-grasp-from-demos", action="store_true",
                        help="Inicializar el actor desde ensayos físicos exitosos antes de PPO")
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
    if args.initialize_grasp_from_demos and (not args.grasp_calibration or args.resume or check_only):
        parser.error("La inicialización supervisada requiere un teacher nuevo con calibración")
    if args.evaluate_episodes is not None and args.evaluate_episodes < 1:
        parser.error("--evaluate-episodes requiere al menos un episodio")
    if args.rover and args.grasp_calibration:
        parser.error("La postura calibrada del torso fijo no corresponde al montaje de brazos del rover")
    minimum = 1 if args.crop or args.grasp_calibration else len(OBJECTIVES)
    if args.num_envs < minimum or min(args.max_iterations, args.save_interval) < 1 or args.seed < 0:
        parser.error(f"Configura al menos {minimum} réplicas, iteraciones y guardado positivos y semilla no negativa")
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
    if args.record_video and (show_window or args.num_envs % minimum or check_only):
        parser.error("--record-video requiere modo headless, réplicas iguales por cultivo y entrenamiento activo")
    ffmpeg = which(args.ffmpeg) if args.record_video else None
    if args.record_video:
        import subprocess
        from dexman.data.epoch_video import VIDEO_EYE_SIZE, EpochVideoRecorder

        codecs = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True) if ffmpeg else None
        if codecs is None or codecs.returncode or b" libx264 " not in codecs.stdout:
            parser.error("Indica el ejecutable FFmpeg con libx264 mediante --ffmpeg antes de grabar")
        cfg = make_batch_config(args.num_envs, rgbd=True, rgbd_size=VIDEO_EYE_SIZE, contacts=True,
                                grasp=grasp, crop=args.crop, rover=args.rover)
    else:
        cfg = make_batch_config(args.num_envs, contacts=True, monitor=show_window, grasp=grasp,
                                rgbd=args.rover and args.validate,
                                crop=args.crop, rover=args.rover)
    if args.record_video or (args.rover and args.validate):
        for scene in vars(cfg.scenes).values():
            for eye in EYE_KEYS:
                getattr(scene, eye).update_period = 0.
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
        env = AgriculturalPPOEnv(sim, cfg, seed=args.seed, render=show_window)
        if show_window:
            from dexman.sensors.teacher_camera_panel import AgriculturalCameraPanel

            if args.rover:
                x, y, z = env.batch.groups[0].origins[0].tolist()
                sim.set_camera_view((x+2.5, y-3.1, z+2.2), (x-.2, y, z+.95))
            else:
                sim.set_camera_view((23., -25., 36.), (15., 14., 0.))
            panel = AgriculturalCameraPanel(env.batch)
            ui_lifetime.callback(panel.close)
            env.on_step = panel.refresh
            panel.refresh()
        agent = ppo_config(args.save_interval)
        if not check_only:
            output.mkdir(parents=True, exist_ok=False)
            if args.grasp_calibration:
                copyfile(args.grasp_calibration, output / "calibration.json")
        runner = AgriculturalRunner(env, agent, log_dir=None if check_only else str(output), device=sim.device)
        if saved:
            runner.resume(saved)
        if args.initialize_grasp_from_demos:
            warmup = initialize_grasp_from_demonstrations(env, runner.alg.actor)
            warmup["deterministic_before_ppo"] = evaluate_teacher(env, runner.get_inference_policy(), env.num_envs)
            write_json(output / "grasp_initialization.json", warmup)
            print(f"AGRICULTURE_INITIALIZATION {json.dumps(warmup)}", flush=True)
            env._reset(torch.arange(env.num_envs, device=env.device))
            env._update_observations()
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
            if args.rover:
                validate_rover_gaze(env)
            if grasp is not None:
                validate_grasp_controls(env)
                env._reset(torch.arange(env.num_envs, device=env.device))
                env._update_observations()
                baseline = evaluate_teacher(env, actor, env.num_envs)
                print(f"AGRICULTURE_EVALUATION {json.dumps(baseline)}", flush=True)
            print(f"AGRICULTURE_PPO_VALIDATED {env.num_envs} envs; "
                  f"obs={tuple(obs['policy'].shape)}; actions={tuple(actions.shape)}; no updates", flush=True)
            return 0
        run_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        write_json(output / "run.json", {"contract": contract(grasp=grasp is not None, rover=args.rover),
                                        "args": run_args, "agent": agent,
                                        "simulation_profile": runner.simulation_profile(),
                                        "provenance": provenance(), "resume": None if checkpoint is None else
                                        {"path": str(checkpoint), "sha256": sha256(checkpoint)}})
        print(f"PPO agrícola: {env.num_envs} réplicas, {agent['num_steps_per_env'] * env.num_envs} "
              f"transiciones/actualización, salida {output}", flush=True)
        runner.learn(args.max_iterations, init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
