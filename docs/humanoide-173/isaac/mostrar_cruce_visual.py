"""Grabación de la misma ejecución física que produce el informe de evaluación."""
from pathlib import Path
import shutil
import subprocess

import numpy as np


class GrabadoraCruce:
    def __init__(self, env, salida, robot, semilla):
        import omni.usd
        from PIL import ImageFont
        from pxr import UsdGeom, Gf
        from isaacsim.sensors.experimental.rtx import TiledCameraSensor
        self.env, self.salida, self.robot, self.semilla = env, salida, robot, semilla
        self.indice = int(np.where(env.numeros == robot)[0][0])
        vista = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), "/World/VistaDelCruce")
        self.matriz_vista = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(9.2, -.4, 9.8), Gf.Vec3d(0, 0, .6), Gf.Vec3d(0, 0, 1))
        vista.AddTransformOp().Set(self.matriz_vista.GetInverse())
        vista.CreateFocalLengthAttr(19.)
        vista.CreateHorizontalApertureAttr(26.)
        self.exterior = TiledCameraSensor([str(vista.GetPath())], resolution=(640, 864), annotators=["rgb"])
        self.fuente = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
        self.titulo = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 23)
        encoder = Path.home() / "scoop/apps/ffmpeg/current/bin/ffmpeg.exe"
        ejecutable = str(encoder) if encoder.exists() else shutil.which("ffmpeg")
        if not ejecutable:
            raise FileNotFoundError("Se necesita ffmpeg para grabar el cruce.")
        comando = [ejecutable, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", "1136x640", "-framerate", "10", "-i", "-",
                   "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(salida)]
        self.proceso = subprocess.Popen(comando, stdin=subprocess.PIPE)
        self.k = 0

    def capturar(self, fin=False):
        from PIL import Image, ImageDraw
        from pxr import Gf
        k = self.k
        self.k += 1
        if k % 2 and not fin:
            return
        env, i = self.env, self.indice
        rgb, _ = self.exterior.get_data("rgb")
        frame = Image.fromarray(rgb.numpy()[0])
        if frame.size != (864, 640) or np.asarray(frame).std() < 2:
            raise RuntimeError("La cámara exterior no entregó una vista válida del campus.")
        canvas = Image.new("RGB", (1136, 640), "#18252b")
        canvas.paste(frame)
        canvas.paste(Image.fromarray(env.rgb[i]), (872, 75))
        d = ImageDraw.Draw(canvas)
        pelvis = env.robot.get_root_transforms()[i, :3].astype(float)
        punto = self.matriz_vista.Transform(Gf.Vec3d(*map(float, pelvis)) + Gf.Vec3d(0, 0, .64))
        if punto[2] < 0:
            escala = -864 * 19 / (26*punto[2])
            x, y = int(432 + escala*punto[0]), int(320 - escala*punto[1])
            d.polygon([(x-5, y-5), (x+5, y-5), (x, y+2)], fill="#76eec7", outline="#142c25")
            d.text((x, y-7), f"{self.robot:02}", anchor="mb", font=self.fuente, fill="white",
                   stroke_width=2, stroke_fill="#142c25")
        estado = ("Caída" if env.caido[i] else "Contacto" if env.chocado[i] else "En meta"
                  if env.en_meta[i] else "Cruzando" if env.en_travesia[i] else "Esperando")
        d.text((876, 10), f"Cámara de Xavier {self.robot:02}", font=self.titulo, fill="white")
        d.text((876, 42), "RGB   256 × 256", font=self.fuente, fill="white")
        d.text((876, 342), estado, font=self.titulo, fill="#a2efca")
        valores = [f"Campus   ·   {env.time:04.1f} s", f"En meta   {env.arrivals} / 16",
                   f"Sin iniciar   {np.isnan(env.salida_s).sum()}", f"Caídas   {env.falls}",
                   f"Robots con choque   {env.hits}", f"Semilla   {self.semilla}"]
        for fila, valor in enumerate(valores):
            d.text((876, 402 + 34*fila), valor, font=self.fuente, fill="white")
        if k == 80:
            canvas.save(self.salida.with_suffix(".jpg"), quality=91)
            mosaico = Image.new("RGB", (4*256, 4*286), "#18252b")
            etiquetas = ImageDraw.Draw(mosaico)
            for j in np.argsort(env.numeros):
                numero = int(env.numeros[j])
                x, y = (numero % 4)*256, (numero // 4)*286
                mosaico.paste(Image.fromarray(env.rgb[j]), (x, y))
                accion = ("Caída" if env.caido[j] else "Contacto" if env.chocado[j] else "En meta"
                          if env.en_meta[j] else "Cruzando" if env.en_travesia[j] else "Esperando")
                etiquetas.text((x+6, y+256), f"Xavier {numero:02}   ·   {accion}", font=self.fuente, fill="white")
            mosaico.save(self.salida.with_name(self.salida.stem + "_camaras.jpg"), quality=94)
        self.proceso.stdin.write(canvas.tobytes())

    def cerrar(self):
        self.proceso.stdin.close()
        if self.proceso.wait():
            raise RuntimeError("Falló la codificación del vídeo.")
