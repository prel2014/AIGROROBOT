"""Acquisition at the rover control boundary during a scripted field harvest."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from dexman.tasks.agriculture.interface import EYE_KEYS
from dexman.tasks.agriculture.harvest_route import FIELD_PHASE_INSTRUCTIONS
from dexman.tasks.agriculture.mobile_state import WHEEL_JOINTS
from dexman.tasks.agriculture.rover_basket_task import complete_field_cycle
from .agriculture import _eye_poses, live_robot_contract
from .writer import EpisodeWriter


class FieldCycleWriter:
    def __init__(self, root, evidence, env, instruction, versions, source_hashes, field_seed, route, *, situation="base"):
        self.env = env
        self.group, = env.batch.groups
        self.writers = []
        self.tick = 0
        self.evidence = evidence
        self.root = Path(root) / self.group.crop / evidence.name
        if self.root.exists():
            raise FileExistsError(self.root)
        staging = evidence/"adquisicion"
        contract = live_robot_contract(env)
        metadata = {**contract, "seed": env.batch.seed, "field_seed": field_seed, "situation": situation,
                    "exercise": "cosecha_en_surco", "instruction_language": "es",
                    "wheel_order": list(WHEEL_JOINTS), "physics_dt_s": env.batch.sim.get_physics_dt(),
                    "source_sha256": source_hashes, "simulator": "Isaac Sim 6.2 / PhysX"}
        self.phase_indices = {name: index for index, name in enumerate(FIELD_PHASE_INSTRUCTIONS)}
        try:
            for index, (name, side, _, _) in enumerate(versions):
                writer = EpisodeWriter(staging / f"robot_{index:03d}", metadata)
                self.writers.append(writer)
                intrinsic = torch.stack([self.group.scene[key].data.intrinsic_matrices.torch[index]
                                         for key in EYE_KEYS])
                writer.start(instruction, dict(camera_intrinsics=intrinsic.cpu().numpy(),
                    crop=self.group.crop, replica_id=index, episode_id=0, trajectory=name, hand=int(side),
                    situation=situation,
                    route_xy_m=np.asarray(route), subtasks_by_phase=json.dumps(
                        dict(enumerate(FIELD_PHASE_INSTRUCTIONS.values())), ensure_ascii=False)))
        except BaseException:
            self.close()
            raise

    def sample(self, phase, arms, head, wheels):
        batch = self.env.batch
        rgb, depth = batch.rgbd()
        for _ in range(8 if self.tick == 0 else 0):
            if (depth.isfinite() & (depth > 0)).flatten(2).any(-1).all() and (rgb.flatten(2).amax(-1) > rgb.flatten(2).amin(-1)).all():
                break
            batch.sim.render()
            rgb, depth = batch.rgbd()
        valid = torch.isfinite(depth) & (depth > 0)
        if not valid.flatten(2).any(-1).all():
            raise ValueError(f"Falta profundidad en uno de los ojos en el paso {self.tick}")
        if self.tick == 0 and (rgb.flatten(2).amax(-1) == rgb.flatten(2).amin(-1)).any():
            raise ValueError("Falta imagen en uno de los ojos al inicio")
        if rgb.shape[2:4] != (224, 224):
            raise ValueError("La captura del VLA requiere ojos de 224 × 224")
        arms = arms.clamp(*batch.joint_limits.unbind(-1))
        head = head.clamp(*self.group.rover.head_limits.unbind(-1))
        pos, quat = _eye_poses(self.group, slice(None))
        fruit = self.group.scene["fruit_0"].data.root_link_pose_w.torch.clone()
        fruit[:, :3] -= self.group.origins
        base = self.group.rover.robot.data.root_link_pose_w.torch.clone()
        base[:, :3] -= self.group.origins
        arrays = {key: value.cpu().numpy() for key, value in {
            "rgb": rgb, "depth_m": torch.where(valid, depth, 0.), "depth_valid": valid,
            "mobile_state": batch.mobile_state(batch.angular_state()),
            "joint_command": torch.cat((arms, head), dim=-1), "wheel_command_rad_s": wheels,
            "camera_pos_env_m": pos, "camera_quat_ros_xyzw": quat,
            "fruit_pose_env_xyzw": fruit, "base_pose_env_xyzw": base}.items()}
        for i, writer in enumerate(self.writers):
            writer.append(**{key: values[i] for key, values in arrays.items()},
                          phase=np.uint8(self.phase_indices[phase]), done=np.bool_(False),
                          time_s=np.float32(self.tick * self.env.dt))
        self.tick += 1
        if self.tick % 90 == 0:
            for writer in self.writers:
                writer.file.flush()

    def finish(self, robots):
        from .field_release import verify_detachment
        from dexman.vla.bimanual_data import inspect_bimanual_episode
        report = json.loads((self.evidence/"resultado.json").read_text(encoding="utf-8"))
        for writer, robot in zip(self.writers, robots, strict=True):
            accepted = complete_field_cycle(robot)
            if accepted:
                try:
                    verify_detachment(self.evidence,robot["robot"],report)
                except ValueError as error:
                    print(f"Demostración excluida: {error}",flush=True)
                    accepted = False
            if accepted:
                writer.file["done"][-1] = True
                writer.file.attrs["success"] = True
                writer.metadata["source_sha256"] = report["source_sha256"]
                writer.file.attrs["metadata"] = json.dumps(writer.metadata)
                writer.file.flush()
                # Validate before publishing so an unusable recording stays partial.
                inspect_bimanual_episode(writer.path, writer.file, writer.metadata)
            writer.finish(accepted, {"physical_result": json.dumps(robot), "outcome": "success" if accepted else "failed"})
            if accepted:
                src = writer.path.with_suffix("")
                dest = self.root/src.parent.name/src.name
                dest.parent.mkdir(parents=True,exist_ok=False)
                if src.stat().st_dev == dest.parent.stat().st_dev:
                    src.rename(dest)
                else:
                    partial = dest.with_suffix(".h5.partial")
                    shutil.copy2(src,partial)
                    os.replace(partial,dest)

    def close(self):
        for writer in self.writers:
            writer.abort()
