"""DAgger labels the teacher action at the student's current, pre-action observation."""
import torch

from dexman.data.capture import calibration, observation_frame
from dexman.robot.control import decode


@torch.inference_mode()
def collect_corrections(env, teacher, student, writers, episodes, max_attempts, beta=.2):
    if not 0 <= beta <= 1 or min(episodes, max_attempts) < 1 or max_attempts < episodes:
        raise ValueError("Invalid DAgger mixture or episode budget")
    raw = env.unwrapped
    if len(writers) != raw.num_envs or any(writer.success_only for writer in writers):
        raise ValueError("DAgger requires one failure-preserving writer per replica")
    observations, _ = env.reset()
    student.reset(torch.ones(raw.num_envs, dtype=torch.bool).numpy())
    steps = [0] * raw.num_envs
    completed = 0
    for index, writer in enumerate(writers):
        writer.start(raw.cfg.instruction, calibration(raw, index))
    while completed < episodes and completed < max_attempts:
        frames = [observation_frame(raw, index) for index in range(raw.num_envs)]
        labels = teacher.act(raw, raw.teacher_observations()).clamp(-1, 1)
        teacher_targets = decode(labels, raw.limits, raw.action_center)
        student_targets = student.act(raw, observations)
        use_teacher = torch.rand(raw.num_envs, device=raw.device) < beta
        executed = torch.where(use_teacher[:, None], teacher_targets, student_targets)
        target = torch.lerp(raw.targets, executed, raw.cfg.action_smoothing)
        observations, rewards, terminated, truncated, extras = env.step(executed)
        dones = terminated | truncated
        for index, writer in enumerate(writers):
            writer.append(**frames[index], action=labels[index].cpu().numpy(),
                joint_command=teacher_targets[index].cpu().numpy(), joint_target=target[index].cpu().numpy(),
                executed_joint_command=executed[index].cpu().numpy(), teacher_applied=bool(use_teacher[index]),
                reward=float(rewards[index]), done=bool(dones[index]), timestamp_s=steps[index] * raw.step_dt)
            steps[index] += 1
            if dones[index]:
                writer.finish(bool(extras["success"][index]))
                completed += 1
                if completed >= episodes or completed >= max_attempts:
                    break
                steps[index] = 0
                writer.start(raw.cfg.instruction, calibration(raw, index))
        teacher.reset(dones.cpu().numpy())
        student.reset(dones.cpu().numpy())
    return {"completed": completed, "beta": beta, "label_horizon": 1,
            "failures_retained": True, "teacher_actions_are_labels": True}
