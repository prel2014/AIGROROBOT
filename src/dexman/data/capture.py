"""Align observation(t), instruction, and command(t), before Same-Step autoreset."""
import torch
from dexman.robot.control import decode
from dexman.robot.joints import FINGERTIPS


def calibration(env, index=0):
    camera = env.scene["camera"]
    return {"camera_intrinsics": camera.data.intrinsic_matrices.torch[index].cpu().numpy(),
            "camera_position_world": camera.data.pos_w.torch[index].cpu().numpy(),
            "camera_quaternion_ros_xyzw": camera.data.quat_w_ros.torch[index].cpu().numpy()}


def observation_frame(env, index=0):
    camera = env.scene["camera"]
    depth = camera.data.output["distance_to_image_plane"].torch[index]
    frame = {
        "rgb": camera.data.output["rgb"].torch[index, ..., :3].to(torch.uint8).cpu().numpy().copy(),
        "depth_m": torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
        "qpos": env.robot.data.joint_pos.torch[index, env.joint_ids].cpu().numpy().copy(),
        "qvel": env.robot.data.joint_vel.torch[index, env.joint_ids].cpu().numpy().copy(),
    }
    sensors = getattr(env.scene, "sensors", {})
    if FINGERTIPS[0] in sensors:
        forces = torch.stack([
            sensors[tip].data.normal_force_matrix_w.torch[index, 0, 0]
            for tip in FINGERTIPS
        ])
        frame["fingertip_normal_force_w_n"] = forces.cpu().numpy()
    return frame


def collect_parallel(env, policy, writers, episodes, max_attempts):
    """Independent per-replica episodes; quota/attempt budget apply to the entire run."""
    if len(writers) != env.num_envs:
        raise ValueError("One writer per environment is required")
    raw = env.unwrapped
    obs = env.get_observations()
    steps = [0] * env.num_envs
    saved = attempts = 0
    for index, writer in enumerate(writers):
        writer.start(raw.cfg.instruction, calibration(raw, index))
    while saved < episodes and attempts < max_attempts:
        frames = [observation_frame(raw, i) for i in range(env.num_envs)]
        with torch.inference_mode():
            action = policy(obs).clamp(-1, 1)
            command = decode(action, raw.limits, getattr(raw, "action_center", None))
            target = torch.lerp(raw.targets, command, raw.cfg.action_smoothing)
            obs, reward, dones, extras = env.step(action)
        for i, writer in enumerate(writers):
            done = bool(dones[i].item())
            writer.append(**frames[i], action=action[i].cpu().numpy(),
                          joint_command=command[i].cpu().numpy(), joint_target=target[i].cpu().numpy(),
                          reward=float(reward[i].item()),
                          done=done, timestamp_s=steps[i] * raw.step_dt)
            steps[i] += 1
            if done:
                success = bool(extras["success"][i].item())
                writer.finish(success)
                saved += int(success or not writer.success_only)
                attempts += 1
                print(f"Replica {i}; attempt {attempts}; saved {saved}/{episodes}; success={success}", flush=True)
                if saved >= episodes or attempts >= max_attempts:
                    break
                steps[i] = 0
                writer.start(raw.cfg.instruction, calibration(raw, i))
        policy.reset(dones)
    if saved < episodes:
        raise RuntimeError(f"Only collected {saved}/{episodes} episodes after {max_attempts} attempts")
