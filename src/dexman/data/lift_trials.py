"""Store synchronized, physically verified calibration demonstrations in agricultural HDF5."""
from dataclasses import asdict
import json

import numpy as np

from dexman.data.agriculture import _calibration, live_robot_contract
from dexman.data.writer import EpisodeWriter
from dexman.experiments import provenance, sha256, write_json
from dexman.tasks.agriculture.objectives import exercise_text


class LiftTrialCapture:
    def __init__(self, env, path, plan, exercise):
        group, = env.batch.groups
        self.group = group
        self.exercise = exercise
        self.order, self.subtasks = exercise_text(group.crop, exercise)
        self.order += " Si ves obstáculos, esquívalos."
        self.metadata = {**live_robot_contract(env), "seed": env.batch.seed, "exercise": exercise,
                         "instruction_language": "es", "demonstrator": "calibrated_joint_trajectory",
                         "calibration_sha256": sha256(plan), "provenance": provenance(),
                         "simulation_profile": {"physics_hz": round(1 / env.batch.sim.get_physics_dt()),
                                                "raised_support": True, "nominal_scene": env.cfg.nominal}}
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "dataset.json", self.metadata)
        self.writers = [EpisodeWriter(path / group.crop / f"replica_{i:03d}", self.metadata)
                        for i in range(env.num_envs)]

    def start(self, env, support):
        for i, writer in enumerate(self.writers):
            info = _calibration(self.group, i)
            info.update(crop=self.group.crop, replica_id=i, episode_id=env.batch.episode_ids[i],
                        initial_clearance_m=float(support[i]), initial_layout=json.dumps(asdict(env.batch.dynamics[i].initial)),
                        subtasks_by_phase=json.dumps(self.subtasks, ensure_ascii=False))
            writer.start(self.order, info)
        self.first = True

    def observe(self, env):
        rgb, depth = env.batch.rgbd()
        self.rgb = rgb.cpu().numpy()
        depth = depth.cpu().numpy()
        self.valid = np.isfinite(depth) & (depth > 0.)
        self.depth = np.where(self.valid, depth, 0.)
        self.state = env.batch.angular_state().cpu().numpy()
        if self.first and (not self.valid.any(axis=(2, 3)).all() or
                           any(eye.min() == eye.max() for pair in self.rgb for eye in pair)):
            raise ValueError("Un ojo RGB-D no produjo una observación inicial válida")
        self.first = False

    def append(self, step, phase, target, measured, last):
        pose, contact, held, impact, slip, height = (measured[i].cpu().numpy() for i in (0, 5, 6, 7, 11, 12))
        angles = target.cpu().numpy()
        for i, writer in enumerate(self.writers):
            writer.append(rgb=self.rgb[i], depth_m=self.depth[i], depth_valid=self.valid[i], angular_state=self.state[i],
                          joint_command=angles[i], phase=np.uint8(phase), time_s=np.float32(step / 30),
                          next_target_pose=pose[i], next_contact=contact[i], next_held=held[i],
                          next_clearance_m=height[i], next_slip_m_s=slip[i], obstacle_contact=impact[i],
                          done=np.bool_(last))

    def finish(self, success):
        for writer, passed in zip(self.writers, success.tolist(), strict=True):
            writer.finish(passed, {"outcome": "success" if passed else "failed_physics"})

    def close(self):
        for writer in self.writers:
            writer.abort()
