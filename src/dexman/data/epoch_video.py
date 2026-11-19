"""Record each agricultural replica as a separate MP4 per PPO rollout."""
from itertools import groupby
from math import ceil, isqrt
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from dexman.tasks.agriculture.interface import EYE_KEYS


VIDEO_EYE_SIZE = 224
VIDEO_SAMPLE_STEPS = 4
VIDEO_FPS = 4


class EpochVideoRecorder:
    def __init__(self, batch, directory, ffmpeg, *, rollout_steps):
        from PIL import Image, ImageDraw

        if rollout_steps % VIDEO_SAMPLE_STEPS:
            raise ValueError("El rollout debe tener un múltiplo de cuatro pasos para grabar video")
        self.batch = batch
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg
        self.expected_frames = rollout_steps // VIDEO_SAMPLE_STEPS
        self.frames = []
        self.eyes = [group.scene[key] for group in batch.groups for key in EYE_KEYS]

        columns = isqrt(batch.num_envs - 1) + 1
        rows = ceil(batch.num_envs / columns)
        size = VIDEO_EYE_SIZE
        title_height, label_height = 28, 22
        self.tile_width = 2 * size
        self.tile_height = 2 * (size + label_height) + title_height
        self.rgb_top = title_height + label_height
        self.depth_top = self.rgb_top + size + label_height
        canvas = Image.new("RGB", (columns * self.tile_width, rows * self.tile_height), (16, 20, 28))
        labels = ImageDraw.Draw(canvas)
        self.tiles = []
        for group in batch.groups:
            for replica in range(group.start, group.stop):
                row, column = divmod(replica, columns)
                x, y = column * self.tile_width, row * self.tile_height
                title = f"Robot {replica:03d} | {group.crop} | profundidad 0,05–1,50 m"
                labels.text((x + 6, y + 6), title, fill="white", font_size=15)
                for eye, side in enumerate(("izquierda", "derecha")):
                    left = x + eye * size + 6
                    labels.text((left, y + title_height + 3), f"RGB · {side}", fill="white", font_size=13)
                    labels.text((left, y + self.depth_top - label_height + 3), f"Profundidad · {side}",
                                fill="white", font_size=13)
                self.tiles.append((replica, x, y))
        self.canvas = np.asarray(canvas)

    def capture_step(self):
        if self.batch.control_tick % VIDEO_SAMPLE_STEPS:
            return
        from isaaclab.sensors import SensorBase

        SensorBase.update_batch(self.eyes, 0.)
        rgb, depth = (data.cpu().numpy() for data in self.batch.rgbd())
        valid_depth = np.isfinite(depth) & (depth > 0.)
        if not self.frames and (not rgb.any(axis=(2, 3, 4)).all() or not valid_depth.any(axis=(2, 3)).all()):
            # RTX can return empty annotator buffers on the first render.
            self.batch.sim.render()
            SensorBase.update_batch(self.eyes, 0.)
            rgb, depth = (data.cpu().numpy() for data in self.batch.rgbd())
            valid_depth = np.isfinite(depth) & (depth > 0.)
            if not rgb.any(axis=(2, 3, 4)).all() or not valid_depth.any(axis=(2, 3)).all():
                raise RuntimeError("RTX no devolvió RGB-D para todos los robots del video")
        gray = (255 * np.clip((1.5 - np.where(valid_depth, depth, 1.5)) / 1.45, 0., 1.)).astype(np.uint8)
        frame = self.canvas.copy()
        size = VIDEO_EYE_SIZE
        for replica, x, y in self.tiles:
            rgb_top, depth_top = y + self.rgb_top, y + self.depth_top
            for eye in range(len(EYE_KEYS)):
                left = x + eye * size
                frame[rgb_top:rgb_top + size, left:left + size] = rgb[replica, eye]
                frame[depth_top:depth_top + size, left:left + size] = gray[replica, eye, ..., None]
        self.frames.append(frame)

    def write_epoch(self, iteration):
        if len(self.frames) != self.expected_frames:
            raise RuntimeError(f"Video PPO {iteration}: {len(self.frames)} fotogramas de {self.expected_frames}")
        output = self.directory / f"epoch_{iteration:06d}"
        with TemporaryDirectory(dir=self.directory, prefix=f".{output.name}-") as temporary:
            self._encode_robots(temporary)
            Path(temporary).replace(output)
        self.frames.clear()

    def _encode_robots(self, directory):
        """Stream one row at a time to bound encoder memory and command-line length."""
        for y, tiles in groupby(self.tiles, key=lambda tile: tile[2]):
            self._encode_row(directory, y, list(tiles))

    def _encode_row(self, directory, y, tiles):
        height, width = self.tile_height, self.canvas.shape[1]
        layers = "".join(f"[tile{i}]" for i, _, _ in tiles)
        crops = [f"[tile{i}]crop={self.tile_width}:{height}:{x}:0[robot{i}]" for i, x, _ in tiles]
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "2",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(VIDEO_FPS), "-i", "pipe:0",
                   "-filter_complex", ";".join([f"[0:v]split={len(tiles)}{layers}", *crops])]
        for i, _, _ in tiles:
            command.extend(["-map", f"[robot{i}]", "-an", "-c:v", "libx264", "-threads:v", "1",
                            "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", f"robot_{i:03d}.mp4"])
        with subprocess.Popen(command, cwd=directory, stdin=subprocess.PIPE,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as encoder:
            try:
                with encoder.stdin as stream:
                    for frame in self.frames:
                        stream.write(memoryview(frame[y:y + height]).cast("B"))
            except BrokenPipeError as error:
                raise subprocess.CalledProcessError(encoder.wait(), command) from error
            status = encoder.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
