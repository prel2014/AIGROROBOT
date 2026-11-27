"""Renderiza con RTX una lámina de las tres cámaras arquitectónicas de los escenarios."""
from pathlib import Path
import traceback

from isaacsim import SimulationApp

DIRECTORIO = Path(__file__).resolve().parent / "escenarios"


def capturar(app):
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdGeom

    w, h = 800, 500
    margen = 16
    lamina = Image.new("RGB", (3 * w + 4 * margen, h + 76), (242, 239, 231))
    dibujar = ImageDraw.Draw(lamina)
    try:
        fuente = ImageFont.truetype("arial.ttf", 29)
    except OSError:
        fuente = ImageFont.truetype("DejaVuSans.ttf", 29)
    contexto = omni.usd.get_context()
    for i, (nombre, titulo) in enumerate(
        (("vivienda", "Vivienda"), ("calle", "Calle de barrio"), ("campus", "Campus universitario"))):
        if not contexto.open_stage(str(DIRECTORIO / (nombre + ".usda"))):
            raise RuntimeError(f"No se pudo abrir {nombre}")
        while contexto.get_stage_loading_status()[2]:
            app.update()
        stage = contexto.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        # Vista de arquitectura. Los mosaicos camaras_*.png se toman con los 16 robots activos.
        for n in range(16):
            UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Xavier_{n:02}")).MakeInvisible()
        producto = rep.create.render_product("/World/Escenario/Vista", (w, h))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(producto)
        try:
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=12, delta_time=0.)
            pixeles = np.asarray(rgb.get_data())[:, :, :3]
            if pixeles.shape != (h, w, 3) or pixeles.std() < 2:
                raise RuntimeError(f"Captura RTX vacía en {nombre}")
            x = margen + i * (w + margen)
            lamina.paste(Image.fromarray(pixeles), (x, margen))
            dibujar.text((x, h + margen + 11), titulo, font=fuente, fill=(42, 47, 43))
            print(f"Vista RTX {nombre}", flush=True)
        finally:
            rgb.detach(producto)
            producto.destroy()
    lamina.save(DIRECTORIO / "vistas.jpg", quality=94, subsampling=0)


if __name__ == "__main__":
    app = SimulationApp({"headless": True, "multi_gpu": False, "hide_ui": True})
    estado = 0
    try:
        capturar(app)
    except Exception:
        traceback.print_exc()
        estado = 1
    finally:
        app.close(exit_code=estado)
