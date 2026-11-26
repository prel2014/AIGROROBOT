"""Registra una ejecución determinista en tercera persona de las cuatro tareas de Xavier."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITULOS = ("Flexionar y extender", "Alternar brazos", "Flexión bilateral", "Caminar, parar y señalar")


def capturar(app, args):
    import numpy as np
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, UsdGeom
    from isaacsim.sensors.experimental.rtx import TiledCameraSensor
    from entrenar_marcha import Pista
    from politica_marcha import Politica, restaurar, tensores

    env = Pista(app, args)
    obs = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    politica = Politica(obs[1].shape[1], env.prev_action.shape[1]).to(device)
    restaurar(torch.load(args.modelo, weights_only=True, map_location="cpu"), politica)
    indices = [np.flatnonzero(env.secuencia.tipo == tarea)[0] for tarea in range(4)]
    stage = omni.usd.get_context().get_stage()
    rutas = []
    for tarea, i in enumerate(indices):
        x, y = env.robot.get_root_transforms()[i, :2].tolist()
        direccion = -1 if x < 0 else 1
        ruta = f"/World/VistaDeTarea_{tarea}"
        cam = UsdGeom.Camera.Define(stage, ruta)
        ojo = Gf.Vec3d(x + direccion * 1.10, y + 2.85, 1.43)
        centro = Gf.Vec3d(x, y + .20, .93)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(ojo, centro, Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.CreateFocalLengthAttr(29.)
        cam.CreateHorizontalApertureAttr(22.)
        rutas.append(ruta)
    sensor = TiledCameraSensor(rutas, resolution=(320, 320), annotators=["rgb"])
    orden = [sensor.camera.paths.index(ruta) for ruta in rutas]
    try:
        fuente = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        fuente = ImageFont.load_default()
    for _ in range(15):
        env.sim.render()
    cuadros = []
    for paso in range(int(env.horizonte / env.dt_control)):
        if paso % 4 == 0:
            batch, _ = sensor.get_data("rgb")
            if batch is None:
                raise RuntimeError("Las cámaras de demostración no entregaron imagen.")
            pixels = batch.numpy()[orden]
            if np.any(pixels.reshape(4, -1).std(axis=1) < 2):
                raise RuntimeError("Una cámara de demostración está vacía.")
            lienzo = Image.new("RGB", (640, 704), "#16191d")
            texto = ImageDraw.Draw(lienzo)
            for tarea, i in enumerate(indices):
                x, y = (tarea % 2) * 320, (tarea // 2) * 352
                lienzo.paste(Image.fromarray(pixels[tarea]), (x, y))
                estado = "meta" if env.en_meta[i] else "caída" if env.caido[i] else f"{env.time:.1f} s"
                texto.text((x + 7, y + 328), f"{TITULOS[tarea]}  ·  {estado}", font=fuente, fill="white")
            cuadros.append(lienzo)
        if env.terminado():
            break
        with torch.inference_mode():
            dist, _ = politica(*tensores(obs, device))
        obs, _ = env.step(dist.mean.tanh().cpu().numpy())
    cuadros[0].save(args.salida, save_all=True, append_images=cuadros[1:], duration=200, loop=0)
    print(f"Secuencia de {len(cuadros)} fotogramas en {args.salida}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escenario", choices=("campus", "vivienda", "calle"), default="campus")
    parser.add_argument("--semilla", type=int, default=197)
    parser.add_argument("--modelo", type=Path, default=HERE / "situaciones.pt")
    parser.add_argument("--salida", type=Path, default=HERE / "escenarios" / "situaciones.gif")
    args = parser.parse_args()
    args.etapa, args.visible, args.fisica = "situaciones", False, "cpu"
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, multi_gpu=False, hide_ui=True,
                             open_usd=str(HERE / "escenarios" / (args.escenario + ".usda"))))
    estado = 0
    try:
        capturar(app, args)
    except Exception:
        import traceback
        traceback.print_exc()
        estado = 1
    finally:
        app.close(exit_code=estado)
    raise SystemExit(estado)
