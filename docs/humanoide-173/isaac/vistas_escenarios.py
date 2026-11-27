"""Capturas RTX de los escenarios con un Xavier de la pista como referencia de escala."""
import argparse
from pathlib import Path
import traceback

from isaacsim import SimulationApp

DIRECTORIO = Path(__file__).resolve().parent / "escenarios"
VISTAS = {
    "vivienda": ("Vivienda", 9),
    "calle": ("Calle comercial", 12),
    "campus": ("Campus", 9),
    "residencial": ("Calle residencial", 12),
    "acopio": ("Acopio de cítricos", 12),
    "naranjal": ("Naranjal de cosecha", None),
}


CAMARAS_CABEZA = {
    "calle": "/World/Xavier_00/cabeza/Camara",
    "naranjal": "/World/Xavier/cabeza/Camara",
}


def capturar(app, nombres, desde_cabeza=False):
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom

    (DIRECTORIO / "renders").mkdir(exist_ok=True)
    contexto = omni.usd.get_context()
    ancho, alto = (1024, 1024) if desde_cabeza else (1440, 900)
    for nombre in nombres:
        if not contexto.open_stage(str(DIRECTORIO / (nombre + ".usda"))):
            raise RuntimeError(f"No se pudo abrir {nombre}")
        while contexto.get_stage_loading_status()[2]:
            app.update()
        stage = contexto.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        if not desde_cabeza and VISTAS[nombre][1] is not None:
            for n in range(16):
                if n != VISTAS[nombre][1]:
                    UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Xavier_{n:02}")).MakeInvisible()
        ruta_camara = CAMARAS_CABEZA[nombre] if desde_cabeza else "/World/Escenario/Vista"
        if not stage.GetPrimAtPath(ruta_camara).IsA(UsdGeom.Camera):
            raise RuntimeError(f"Falta la cámara {ruta_camara} en {nombre}")
        producto = rep.create.render_product(ruta_camara, (ancho, alto))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(producto)
        try:
            for _ in range(4):
                rep.orchestrator.step(rt_subframes=12, delta_time=0.)
            pixeles = np.asarray(rgb.get_data())[:, :, :3]
            if pixeles.shape != (alto, ancho, 3) or pixeles.std() < 2:
                raise RuntimeError(f"Captura RTX vacía en {nombre}")
            sufijo = "_cabeza.jpg" if desde_cabeza else ".jpg"
            Image.fromarray(pixeles).save(DIRECTORIO / "renders" / (nombre + sufijo),
                                         quality=95, subsampling=0)
            print(f"Vista RTX {nombre}", flush=True)
        finally:
            rgb.detach(producto)
            producto.destroy()


def galeria():
    from PIL import Image, ImageDraw, ImageFont

    w, h, margen = 576, 360, 14
    fondo = Image.new("RGB", (3*w + 4*margen, 2*(h+40) + 3*margen), "#eeeae1")
    dibujo = ImageDraw.Draw(fondo)
    try:
        fuente = ImageFont.truetype("arial.ttf", 21)
    except OSError:
        fuente = ImageFont.load_default()
    for i, (nombre, (titulo, _)) in enumerate(VISTAS.items()):
        x = margen + (i % 3) * (w + margen)
        y = margen + (i // 3) * (h + 40 + margen)
        with Image.open(DIRECTORIO / "renders" / (nombre + ".jpg")) as imagen:
            fondo.paste(imagen.resize((w, h), Image.Resampling.LANCZOS), (x, y))
        dibujo.text((x, y + h + 6), titulo, font=fuente, fill="#26332c")
    fondo.save(DIRECTORIO / "vistas.jpg", quality=94, subsampling=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escenas", nargs="+", choices=tuple(VISTAS))
    parser.add_argument("--desde-cabeza", action="store_true", help="Capturar el sensor frontal de Xavier")
    args = parser.parse_args()
    nombres = args.escenas or list(CAMARAS_CABEZA if args.desde_cabeza else VISTAS)
    if args.desde_cabeza and set(nombres) - CAMARAS_CABEZA.keys():
        parser.error("Las vistas de cabeza disponibles son calle y naranjal.")
    app = SimulationApp({"headless": True, "multi_gpu": False, "hide_ui": True,
                         "anti_aliasing": 4})
    estado = 0
    try:
        capturar(app, nombres, args.desde_cabeza)
        if not args.desde_cabeza and all((DIRECTORIO / "renders" / (n + ".jpg")).exists() for n in VISTAS):
            galeria()
    except Exception:
        traceback.print_exc()
        estado = 1
    finally:
        app.close(exit_code=estado)
