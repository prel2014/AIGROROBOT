"""Show RGB and metric depth from a selected teacher replica in Kit."""
from time import monotonic

import numpy as np
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _upload(provider, image):
    height, width = image.shape[:2]
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., :3] = image[..., None] if image.ndim == 2 else image
    provider.set_bytes_data(rgba.ravel().data, [width, height])


def _refresh_rgbd(camera, replica, rgb_provider, depth_provider, near, far):
    outputs = camera.data.output
    rgb = outputs["rgb"].torch[replica, ..., :3].detach().cpu().numpy()
    depth = outputs["distance_to_image_plane"].torch[replica].squeeze(-1).detach().cpu().numpy()
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    gray[valid] = (255 * np.clip((far - depth[valid]) / (far - near), 0, 1)).astype(np.uint8)
    _upload(rgb_provider, rgb)
    _upload(depth_provider, gray)


class TeacherCameraPanel:
    def __init__(self, camera, num_envs, instruction, title="Cámaras RGB-D"):
        import omni.ui as ui

        self.camera = camera
        self._rgb = ui.ByteImageProvider()
        self._depth = ui.ByteImageProvider()
        self._near = 0.05
        self._far = 1.5
        self._last_replica = None
        self._next_update = 0.0
        self.window = ui.Window(title, width=490, height=368, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"Orden de la tarea · {instruction}", word_wrap=True, height=34)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Réplica", width=65)
                    self._selector = ui.ComboBox(0, *(f"Robot {index} (env_{index})" for index in range(num_envs)))
                with ui.HStack(spacing=8):
                    with ui.VStack(width=224):
                        ui.Label("RGB", height=20)
                        ui.ImageWithProvider(self._rgb, width=224, height=224)
                    with ui.VStack(width=224):
                        ui.Label(f"Profundidad · {self._near:.2f}–{self._far:.2f} m", height=20)
                        ui.ImageWithProvider(self._depth, width=224, height=224)
                ui.Label("Blanco cerca · Negro lejos o sin lectura", height=20)

    def refresh(self):
        if not self.window.visible:
            return
        replica = self._selector.model.get_item_value_model().as_int
        now = monotonic()
        if replica == self._last_replica and now < self._next_update:
            return
        _refresh_rgbd(self.camera, replica, self._rgb, self._depth, self._near, self._far)
        self._last_replica, self._next_update = replica, now + 0.1

    def close(self):
        self.window.destroy()


class AgriculturalCameraPanel:
    """Display both eyes of the selected robot, including a moving camera head."""

    def __init__(self, batch):
        import omni.ui as ui
        import torch

        self._batch = batch
        self._sim = batch.sim
        self.cameras = batch.monitor_eyes
        if self.cameras:
            self._origins = torch.cat([group.origins for group in batch.groups])
            self._offsets = tuple(self._origins.new_tensor(camera.cfg.offset.pos) for camera in self.cameras)
        self._providers = tuple((ui.ByteImageProvider(), ui.ByteImageProvider()) for _ in range(2))
        self._last_replica = None
        self._last_physics_step = self._sim.get_physics_step_count()
        self._next_update = 0.
        crop_names = {"citrus": "Cítricos", "banana": "Plátano", "vineyard": "Uvas", "berries": "Arándanos"}
        labels = [f"Robot {index:02d} · {crop_names[group.crop]}" for group in batch.groups
                  for index in range(group.start, group.stop)]

        self.window = ui.Window("Ojos RGB-D · Teacher agrícola", width=490, height=565, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=6):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Robot", width=60)
                    self._selector = ui.ComboBox(0, *labels)
                with ui.HStack(spacing=10):
                    for side, providers in zip(("Izquierdo", "Derecho"), self._providers, strict=True):
                        with ui.VStack(width=224, spacing=4):
                            for title, provider in zip(("RGB", "Profundidad"), providers, strict=True):
                                ui.Label(f"{side} · {title}", height=18)
                                ui.ImageWithProvider(provider, width=224, height=224)
                ui.Label("Profundidad 0,055–2,5 m · Blanco cerca · Negro lejos o sin lectura", height=20)

    def refresh(self):
        from isaaclab.sensors import Camera

        if not self.window.visible:
            return
        replica = self._selector.model.get_item_value_model().as_int
        now = monotonic()
        if replica == self._last_replica and now < self._next_update:
            return
        step = self._sim.get_physics_step_count()
        if self.cameras:
            if replica != self._last_replica:
                origin = self._origins[replica:replica + 1]
                for camera, offset in zip(self.cameras, self._offsets, strict=True):
                    camera.set_world_poses(positions=origin + offset)
                self._sim.render()
            Camera.update_batch(self.cameras, (step - self._last_physics_step) * self._sim.get_physics_dt())
            cameras, local_id = self.cameras, 0
        else:
            cameras, local_id = self._batch.eye_cameras(replica)
        for camera, (rgb, depth) in zip(cameras, self._providers, strict=True):
            _refresh_rgbd(camera, local_id, rgb, depth, .055, 2.5)
        self._last_replica, self._last_physics_step, self._next_update = replica, step, now + .1

    def close(self):
        self.window.destroy()


class TeacherCameraVecEnv(RslRlVecEnvWrapper):
    """Refresh the Kit panel after each completed environment step."""
    def __init__(self, env):
        super().__init__(env, clip_actions=1.)
        raw = self.unwrapped
        self._panel = TeacherCameraPanel(raw.scene["camera"], raw.num_envs, raw.cfg.instruction, "RGB-D · vista frontal")
        self._secondary_panel = TeacherCameraPanel(raw.scene["camera_secondary"], raw.num_envs, raw.cfg.instruction, "RGB-D · vista opuesta")

    def step(self, actions):
        result = super().step(actions)
        self._panel.refresh()
        self._secondary_panel.refresh()
        return result

    def close(self):
        try:
            self._panel.close()
            self._secondary_panel.close()
        finally:
            super().close()
