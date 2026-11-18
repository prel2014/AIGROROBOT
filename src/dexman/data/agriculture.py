"""Stream synchronized RGB-D and bimanual teacher targets to per-episode HDF5 files."""
from dataclasses import asdict
from itertools import zip_longest
from json import dumps

import numpy as np
import torch

from dexman.experiments import write_json
from dexman.tasks.agriculture.interface import EYE_KEYS, JOINT_ORDER
from dexman.tasks.agriculture.objectives import PHASE_TEXT
from dexman.tasks.agriculture.randomization import OBSTACLE_RADIUS
from .writer import EpisodeWriter


def _calibration(group, index):
    data = [group.scene[eye].data for eye in EYE_KEYS]
    origin = group.origins[index]
    return {
        "camera_intrinsics": torch.stack([d.intrinsic_matrices.torch[index] for d in data]).cpu().numpy(),
        "camera_pos_env_m": torch.stack([d.pos_w.torch[index] - origin for d in data]).cpu().numpy(),
        "camera_quat_ros_xyzw": torch.stack([d.quat_w_ros.torch[index] for d in data]).cpu().numpy(),
    }


@torch.inference_mode()
def collect_demonstrations(env, actor, output, metadata, *, episodes, max_attempts, keep_failures):
    """Record observations at tick t with the physical commands used for t → t+1."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**metadata, "schema": "dexman.agriculture.demo.v1", "joint_order": list(JOINT_ORDER),
                "state_layout": "q_left,q_right,dq_left,dq_right", "eye_order": list(EYE_KEYS),
                "phase_text": list(PHASE_TEXT), "obstacle_order": list(OBSTACLE_RADIUS),
                "sample_hz": round(1 / env.dt), "depth_unit": "m",
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
        calibration = _calibration(group, index - group.start)
        episode_id = env.batch.episode_ids[index]
        calibration.update(crop=group.crop, replica_id=index, episode_id=episode_id,
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
            state = env.angular_state[indices].cpu().numpy()
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
                              angular_state=state[row], joint_command=commands[row], phase=np.uint8(phase[row]),
                              time_s=np.float32(time_s[row]), obstacles=obstacles[row], lighting=lighting[row],
                              reward=rewards[row], obstacle_contact=collisions[row],
                              next_contact=contacts[row], next_target_pose=fruit[row], done=np.bool_(finished[row]))
                if not finished[row]:
                    continue
                crop = group_by_env[index].crop
                counts = counters[crop]
                counts["attempts"] += 1
                counts["successes"] += int(success[row])
                if saved < episodes and (success[row] or keep_failures):
                    result = ("success" if success[row] else "timeout" if timeout[row] else "drop" if dropped[row]
                              else "invalid_pose" if invalid[row] else "disturbed_fruit")
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
