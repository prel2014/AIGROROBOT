"""Reset only selected replicas; sample object positions on the tabletop."""
import torch


def reset_state(env, env_ids):
    robot, obj = env.robot, env.object
    q = robot.data.default_joint_pos.torch[env_ids].clone()
    qd = robot.data.default_joint_vel.torch[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
    robot.set_joint_position_target_index(target=q, env_ids=env_ids)
    pose = obj.data.default_root_pose.torch[env_ids].clone()
    noise = (torch.rand((len(env_ids), 2), device=env.device) - 0.5) * (2 * env.cfg.object_xy_range)
    pose[:, :2] += noise
    pose[:, :3] += env.scene.env_origins[env_ids]
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), device=env.device),
                                         env_ids=env_ids)
    for name in ("distractor_0", "distractor_1"):
        if name in env.scene.rigid_objects:
            other = env.scene[name]
            other_pose = other.data.default_root_pose.torch[env_ids].clone()
            other_pose[:, :3] += env.scene.env_origins[env_ids]
            other.write_root_pose_to_sim_index(root_pose=other_pose, env_ids=env_ids)
            other.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
    env.goal[env_ids] = pose[:, :3]
    lift = env.cfg.lift_height if env.cfg.reward_version == 1 else env.cfg.curriculum_heights[env.curriculum_level]
    env.goal[env_ids, 2] += lift
    env.targets[env_ids] = q[:, env.joint_ids]
    env.actions[env_ids] = env.home_action
    env.previous_actions[env_ids] = env.home_action
    env.hold_steps[env_ids] = 0
    env.invalid_action[env_ids] = False
    env.invalid_pose[env_ids] = False
    env.episode_max_lift[env_ids] = 0
    if env.cfg.reward_version == 4:
        env.potential_valid[env_ids] = False
