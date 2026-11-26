"""Monitor en vivo de las mismas imágenes que recibe el controlador de marcha."""
from time import perf_counter

import numpy as np
import omni.ui as ui


class MonitorCamaras:
    def __init__(self, orden):
        self.orden = orden
        self.window = ui.Window("Cámaras en vivo · Xavier", width=450, height=510, visible=True)
        self.providers = [ui.ByteImageProvider() for _ in orden]
        self.etiquetas = []
        with self.window.frame:
            with ui.VStack(spacing=4):
                for fila in range(4):
                    with ui.HStack(spacing=3):
                        for col in range(4):
                            i = fila * 4 + col
                            with ui.VStack():
                                self.etiquetas.append(ui.Label(f"Xavier {i:02}", height=19, alignment=ui.Alignment.CENTER))
                                ui.ImageWithProvider(self.providers[i])
                self.status = ui.Label("Esperando imágenes", height=23, alignment=ui.Alignment.CENTER)
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        self.frames = 0
        self.previous = None
        self.last = perf_counter()
        self.start_frame = 0

    def actualizar(self, rgb, periodo, cargas):
        if not self.window.visible:
            return
        rgba = np.empty((*rgb.shape[:3], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        h, w = rgb.shape[1:3]
        for provider, idx in zip(self.providers, self.orden):
            provider.set_data_array(rgba[idx].reshape(-1), [w, h], strict=True)
        self.frames += 1
        now = perf_counter()
        elapsed = now - self.last
        if elapsed >= 1.:
            for i, idx in enumerate(self.orden):
                izq, der = cargas[idx].sum(axis=1)
                self.etiquetas[i].text = f"{i:02}   I {izq:.0f}  D {der:.0f} N"
                filas = [f"{lado}: " + "   ".join(f"{v:.1f}" for v in pie)
                         for lado, pie in zip(("Izq.", "Der."), cargas[idx])]
                self.etiquetas[i].tooltip = "Carga en N · talón int./ext., antepié int./ext.\n" + "\n".join(filas)
            hz = (self.frames - self.start_frame) / elapsed
            self.status.text = f"{w} × {h}   ·   Imagen {self.frames}   ·   {hz:.1f} FPS/cámara   ·   {hz * periodo:.2f}×"
            self.last, self.start_frame = now, self.frames
        if self.frames == 1:
            self.previous = rgb.copy()
        elif self.frames == 16:
            distintas = np.any(self.previous != rgb, axis=(1, 2, 3)).sum()
            print(f"MARCHA monitor: {distintas}/16 vistas cambiaron entre imágenes 1 y 16.", flush=True)
            self.previous = None
