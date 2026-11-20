"""Stream synchronized RGB-D and bimanual teacher targets to per-episode HDF5 files."""
from dataclasses import asdict
from itertools import zip_longest
from json import dumps

import numpy as np
import torch

from dexman.experiments import write_json
from dexman.tasks.agriculture.interface import EYE_KEYS, JOINT_ORDER
from dexman.tasks.agriculture.objectives import PHASE_TEXT, exercise_text
from dexman.tasks.agriculture.randomization import OBSTACLE_RADIUS
from dexman.vla.bimanual_contract import DEMO_SCHEMA, OUTPUT_SPACE, bimanual_contract, mobile_metadata, state_field
from .writer import EpisodeWriter


def _eye_poses(group, index):
    eyes = [group.scene[name].data for name in EYE_KEYS]
    return (torch.stack([eye.pos_w.torch[index] - group.origins[index] for eye in eyes], dim=-2),
            torch.stack([eye.quat_w_ros.torch[index] for eye in eyes], dim=-2))


def live_robot_contract(env):
    limits = env.action_limits.detach()
    if not torch.allclose(limits, limits[0].expand_as(limits), rtol=0., atol=1e-6):
        raise ValueError("Las réplicas deben compartir los límites articulares")
    fields = mobile_metadata() if env.cfg.rover else {
        "schema": DEMO_SCHEMA, "joint_order": list(JOINT_ORDER), "eye_order": list(EYE_KEYS),
        "state_layout": "q_left,q_right,dq_left,dq_right", "joint_command_space": OUTPUT_SPACE, "depth_unit": "m"}
    return bimanual_contract({**fields, "joint_limits_rad": limits[0].cpu().tolist(), "sample_hz": round(1 / env.dt)})


@torch.inference_mode()
def collect_demonstrations(env, actor, output, metadata, *, episodes, max_attempts, keep_failures):
    """Record observations at tick t with the physical commands used for t → t+1."""
    contract = live_robot_contract(env)
    state_key, _ = state_field(contract["schema"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**metadata, **contract,
                "instruction_language": "es", "phase_text": list(PHASE_TEXT),
                "obstacle_order": list(OBSTACLE_RADIUS),
                "lighting_fields": ["intensity", "color_temperature_k"],
                "attempt_limit": max_attempts, "requested_episodes": episodes, "keep_failures": keep_failures}
    write_json(output / "dataset.json", metadata)
    counters = {group.crop: {"attempts": 0, "successes": 0, "saved": 0} for group in env.batch.groups}
    slots = [range(group.start, group.stop) for group in env.batch.groups]
    order = [slot for row in zip_longest(*slots) for slot in row if slot is not None]
    group_by_env = [group for group in env.batch.groups for _ in range(group.scene.num_envs)]
    writers, active = {}, set()
    launched = saved = 0

    def start(index):
        nonlocal launched
        group = group_by_env[index]
        if index not in writers:
            writers[index] = EpisodeWriter(output / group.crop / f"replica_{index:03d}",
                                           metadata, success_only=not keep_failures)
        local = index - group.start
        calibration = {"camera_intrinsics": torch.stack(
            [group.scene[eye].data.intrinsic_matrices.torch[local] for eye in EYE_KEYS]).cpu().numpy()}
        if not env.cfg.rover:
            position, rotation = _eye_poses(group, local)
            calibration.update(camera_pos_env_m=position.cpu().numpy(), camera_quat_ros_xyzw=rotation.cpu().numpy())
        episode_id = env.batch.episode_ids[index]
        calibration.update(crop=group.crop, replica_id=index, episode_id=episode_id,
                           subtasks_by_phase=dumps(exercise_text(group.crop, env.cfg.exercise)[1], ensure_ascii=False),
                           initial_layout=dumps(asdict(env.batch.dynamics[index].initial)))
        writers[index].start(env.instructions[index], calibration)
        active.add(index)
        launched += 1

    try:
        for index in order[:min(env.num_envs, max_attempts)]:
            start(index)
        while active and saved < episodes:
            indices = sorted(active)
            rgb, depth = env.batch.rgbd()
            rgb = rgb[indices].cpu().numpy()
            depth = depth[indices].cpu().numpy().astype(np.float32, copy=False)
            depth_valid = np.isfinite(depth) & (depth > 0.)
            depth = np.where(depth_valid, depth, 0.)
            state = env.vla_state[indices].cpu().numpy()
            camera_fields = {}
            if env.cfg.rover:
                poses = [_eye_poses(group, slice(None)) for group in env.batch.groups]
                for name, values in zip(("camera_pos_env_m", "camera_quat_ros_xyzw"), zip(*poses), strict=True):
                    camera_fields[name] = torch.cat(values)[indices].cpu().numpy()
            phase = env.phase[indices].cpu().numpy()
            time_s = env.episode_length_buf[indices].cpu().numpy() * env.dt
            obstacles = env.obstacle_features[indices].cpu().numpy()
            layouts = [env.batch.dynamics[index].layout for index in indices]
            lighting = np.asarray([(layout.light_intensity, layout.color_temperature_k) for layout in layouts],
                                  dtype=np.float32)
            actions = actor(env.get_observations())
            _, rewards, done, extras = env.step(actions)
            commands = extras["joint_command"][indices].cpu().numpy()
            contacts = extras["fingertip_contacts"][indices].cpu().numpy()
            clearance = extras["fruit_clearance_m"][indices].cpu().numpy()
            held = extras["held"][indices].cpu().numpy()
            fruit = extras["target_pose"][indices].cpu().numpy()
            rewards = rewards[indices].cpu().numpy()
            finished = done[indices].cpu().tolist()
            success = extras["success"][indices].cpu().tolist()
            timeout = extras["time_outs"][indices].cpu().tolist()
            dropped = extras["dropped"][indices].cpu().tolist()
            invalid = extras["invalid_pose"][indices].cpu().tolist()
            collisions = extras["obstacle_contact"][indices].cpu().numpy()
            restart = []
            for row, index in enumerate(indices):
                writer = writers[index]
                if time_s[row] == 0. and (not depth_valid[row].any(axis=(1, 2)).all()
                                          or any(view.min() == view.max() for view in rgb[row])):
                    raise RuntimeError(f"Fotograma inicial RGB-D inválido en la réplica {index}")
                writer.append(rgb=rgb[row], depth_m=depth[row], depth_valid=depth_valid[row],
                              **{state_key: state[row]}, joint_command=commands[row], phase=np.uint8(phase[row]),
                              **{key: values[row] for key, values in camera_fields.items()},
                              time_s=np.float32(time_s[row]), obstacles=obstacles[row], lighting=lighting[row],
                              reward=rewards[row], obstacle_contact=collisions[row],
                              next_contact=contacts[row], next_held=held[row], next_clearance_m=clearance[row],
                              next_target_pose=fruit[row], done=np.bool_(finished[row]))
                if not finished[row]:
                    continue
                crop = group_by_env[index].crop
                counts = counters[crop]
                counts["attempts"] += 1
                counts["successes"] += int(success[row])
                if saved < episodes and (success[row] or keep_failures):
                    result = ("success" if success[row] else "timeout" if timeout[row] else "drop" if dropped[row]
                              else "invalid_pose" if invalid[row] else "task_constraint" if env.cfg.exercise else "disturbed_fruit")
                    writer.finish(success[row], {"outcome": result})
                    counts["saved"] += 1
                    saved += 1
                else:
                    writer.abort(discard=True)
                active.remove(index)
                restart.append(index)
            if restart:
                attempts = sum(counts["attempts"] for counts in counters.values())
                print(f"Captura: {saved}/{episodes} episodios guardados, {attempts}/{max_attempts} intentos",
                      flush=True)
                if saved < episodes:
                    for index in restart[:max_attempts - launched]:
                        start(index)
        # Partial concurrent episodes are irrelevant once the requested quota is reached.
        for index in active:
            writers[index].abort(discard=True)
    finally:
        for writer in writers.values():
            writer.abort()
        write_json(output / "summary.json", {"saved": saved, "requested": episodes,
                                             "attempts": sum(c["attempts"] for c in counters.values()),
                                             "per_crop": counters})
    return saved
