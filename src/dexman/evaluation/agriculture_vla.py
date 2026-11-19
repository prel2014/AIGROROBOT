"""Closed-loop evaluation of Spanish instructions in a selected agricultural scene."""
import torch

from dexman.experiments import seed_everything, sha256, write_json
from dexman.tasks.agriculture.objectives import exercise_text


@torch.inference_mode()
def rollout(env, predictor, crop, instruction, *, phase_hints, episodes, max_steps):
    from dexman.data.agriculture import live_robot_contract

    if live_robot_contract(env) != predictor.robot_contract:
        raise ValueError("El contrato del checkpoint difiere de las articulaciones, cámaras o frecuencia de Isaac Sim")
    model = predictor.model
    goals = [instruction] * env.num_envs if instruction is not None else env.instructions
    phase_text = exercise_text(crop, env.cfg.exercise)[1]
    memory = env.home.new_zeros((env.num_envs, model.memory_dim))
    peak = env.home.new_full((env.num_envs,), float("-inf"))
    records = []
    for _ in range(max_steps):
        rgb, depth = env.batch.rgbd()
        subtasks = [phase_text[phase] for phase in env.phase.tolist()] if phase_hints else None
        angles, memory = model.predict(predictor.processor, predictor.tokenizer, rgb, depth, goals,
                                       env.angular_state, memory, subtasks=subtasks)
        if not torch.isfinite(angles).all() or not torch.isfinite(memory).all():
            raise ValueError("El estudiante devolvió ángulos o memoria no finitos")
        lengths = env.episode_length_buf + 1
        _, _, done, details = env.step_joint_targets(angles[:, 0])
        peak = torch.maximum(peak, details["fruit_clearance_m"])
        finished = done.bool()
        for index in finished.nonzero(as_tuple=True)[0].tolist():
            success = bool(details["success"][index])
            outcome = next((name for name, field in (("success", "success"), ("drop", "dropped"),
                            ("invalid_pose", "invalid_pose"), ("timeout", "time_outs")) if details[field][index]),
                           "task_constraint" if env.cfg.exercise else "disturbed_fruit")
            height = float(peak[index])
            records.append({"replica": index, "episode": env.batch.episode_ids[index] - 1,
                            "steps": int(lengths[index]), "success": success, "outcome": outcome,
                            "final_phase": int(details["phase_at_step"][index]),
                            "peak_clearance_cm": 100 * height if torch.isfinite(peak[index]) else None})
            if len(records) == episodes:
                return records
        memory[finished] = 0
        peak[finished] = float("-inf")
    return records


def play(args, checkpoint, output):
    from isaaclab.app import launch_simulation
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.grasp_curriculum import read_grasp_calibration
    from dexman.vla.inference import BimanualVLAPredictor

    predictor = BimanualVLAPredictor(checkpoint, device="cpu")
    grasp = read_grasp_calibration(args.grasp_calibration, args.crop, "pregrasp") if args.grasp_calibration else None
    cfg = make_batch_config(args.num_envs, crop=args.crop, rgbd=True, contacts=True, grasp=grasp,
                            nominal=bool(grasp), exercise=args.exercise)
    with launch_simulation(cfg, args):
        from isaaclab.sim import SimulationContext
        from dexman.tasks.agriculture.environment import AgriculturalPPOEnv

        seed_everything(args.seed)
        sim = SimulationContext(cfg.sim)
        sim.set_camera_view((2., -2., 1.7), (.35, 0., .75))
        env = AgriculturalPPOEnv(sim, cfg, capture=True, seed=args.seed)
        predictor.model.to(env.device)
        records = rollout(env, predictor, args.crop, args.instruction, phase_hints=args.phase_hints,
                          episodes=args.episodes, max_steps=args.max_steps)
        score = sum(row["success"] for row in records) / len(records) if records else None
        write_json(output, {"crop": args.crop, "exercise": args.exercise or "default", "scene_objective": env.instructions[0],
                            "instruction": args.instruction or env.instructions[0], "phase_hints": args.phase_hints,
                            "start": "calibrated_pregrasp" if grasp else "full_task",
                            "calibration_sha256": sha256(args.grasp_calibration) if grasp else None,
                            "seed": args.seed, "checkpoint_sha256": sha256(checkpoint), "requested": args.episodes,
                            "completed": len(records), "success_rate": score, "episodes": records})
        print(f"Evaluación VLA agrícola: {len(records)}/{args.episodes} episodios, éxito={score}; {output}", flush=True)
        if len(records) < args.episodes:
            raise RuntimeError("Se agotó el presupuesto de pasos. Resultados parciales en el informe.")
