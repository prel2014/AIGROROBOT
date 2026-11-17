"""Show RGB and metric depth from a selected teacher replica in Kit."""
from time import monotonic

import numpy as np
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


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

    @staticmethod
    def _upload(provider, image):
        height, width = image.shape[:2]
        rgba = np.full((height, width, 4), 255, dtype=np.uint8)
        rgba[..., :3] = image[..., None] if image.ndim == 2 else image
        provider.set_bytes_data(rgba.ravel().data, [width, height])

    def refresh(self):
        if not self.window.visible:
            return
        replica = self._selector.model.get_item_value_model().as_int
        now = monotonic()
        if replica == self._last_replica and now < self._next_update:
            return
        outputs = self.camera.data.output
        rgb = outputs["rgb"].torch[replica, ..., :3].detach().cpu().numpy()
        depth = outputs["distance_to_image_plane"].torch[replica].squeeze(-1).detach().cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = (255 * np.clip((self._far - depth[valid]) / (self._far - self._near), 0, 1)).astype(np.uint8)
        self._upload(self._rgb, rgb)
        self._upload(self._depth, gray)
        self._last_replica, self._next_update = replica, now + 0.1

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
