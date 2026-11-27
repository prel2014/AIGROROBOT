"""Renderiza en RTX la lámina de marcha o la de los ocho escenarios de servicio."""
from pathlib import Path
import sys
import traceback

from isaacsim import SimulationApp

DIRECTORIO = Path(__file__).resolve().parent / "escenarios"


def capturar(app, servicio=False, detalle=None):
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdGeom
    from escenarios_servicio import PRIMEROS_PLANOS

    w, h = 800, 500
    margen = 16
    if detalle in PRIMEROS_PLANOS:
        w, h = 1280, 800
        escenas = [(detalle, titulo) for _, titulo, *_ in PRIMEROS_PLANOS[detalle]]
        columnas = 2
    elif detalle == "materiales":
        escenas = [("invernadero", "Cultivos vistos a través del vidrio"),
                   ("ascensor", "Espejo e inox de la cabina")]
        columnas = 2
    elif detalle:
        escenas = [("mercado", "Puestos y productos"), ("estacion", "Banca y señalización del andén")]
        columnas = 2
    elif servicio:
        from escenarios_servicio import ESCENAS
        escenas = [(nombre, ficha[0]) for nombre, ficha in ESCENAS.items()]
        columnas = 2
    else:
        escenas = [("vivienda", "Vivienda"), ("calle", "Calle de barrio"),
                   ("campus", "Campus universitario")]
        columnas = 3
    filas = (len(escenas) + columnas - 1) // columnas
    lamina = Image.new("RGB", (columnas * (w + margen) + margen, filas * (h + 76)), (242, 239, 231))
    dibujar = ImageDraw.Draw(lamina)
    try:
        fuente = ImageFont.truetype("arial.ttf", 29)
    except OSError:
        fuente = ImageFont.truetype("DejaVuSans.ttf", 29)
    contexto = omni.usd.get_context()
    anterior = None
    for i, (nombre, titulo) in enumerate(escenas):
        if anterior != nombre:
            if not contexto.open_stage(str(DIRECTORIO / (nombre + ".usda"))):
                raise RuntimeError(f"No se pudo abrir {nombre}")
            while contexto.get_stage_loading_status()[2]:
                app.update()
            anterior = nombre
        stage = contexto.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        # Vista de arquitectura. Los mosaicos camaras_*.png se toman con los 16 robots activos.
        if not servicio:
            for n in range(16):
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Xavier_{n:02}")).MakeInvisible()
        camara = {"materiales": "VistaMaterial", "entorno": "VistaDetalle"}.get(detalle, "Vista")
        if detalle in PRIMEROS_PLANOS:
            camara = PRIMEROS_PLANOS[detalle][i][0]
        producto = rep.create.render_product("/World/Escenario/" + camara, (w, h))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(producto)
        try:
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=12, delta_time=0.)
            pixeles = np.asarray(rgb.get_data())[:, :, :3]
            if pixeles.shape != (h, w, 3) or pixeles.std() < 2:
                raise RuntimeError(f"Captura RTX vacía en {nombre}")
            x = margen + (i % columnas) * (w + margen)
            y = margen + (i // columnas) * (h + 76)
            lamina.paste(Image.fromarray(pixeles), (x, y))
            dibujar.text((x, y + h + 11), titulo, font=fuente, fill=(42, 47, 43))
            print(f"Vista RTX {nombre} / {camara}", flush=True)
        finally:
            rgb.detach(producto)
            producto.destroy()
    archivo = "vistas_servicio.jpg" if servicio else "vistas.jpg"
    archivo = {
        "materiales": "detalle_vidrio_reflejos.jpg",
        "entorno": "detalle_mercado_estacion.jpg"
    }.get(detalle, archivo)
    if detalle in PRIMEROS_PLANOS:
        archivo = f"primeros_planos_{detalle}.jpg"
    lamina.save(DIRECTORIO / archivo, quality=94, subsampling=0)


if __name__ == "__main__":
    servicio = "--servicio" in sys.argv
    if servicio:
        sys.argv.remove("--servicio")
    solo_primeros = "--primeros-planos" in sys.argv
    if solo_primeros:
        sys.argv.remove("--primeros-planos")
        servicio = True
    opciones = {"headless": True, "multi_gpu": False, "hide_ui": True}
    if servicio:
        opciones.update(max_bounces=8, max_specular_transmission_bounces=10)
    app = SimulationApp(opciones)
    estado = 0
    try:
        if not solo_primeros:
            capturar(app, servicio)
            if servicio:
                capturar(app, servicio, detalle="entorno")
                capturar(app, servicio, detalle="materiales")
        if servicio:
            from escenarios_servicio import PRIMEROS_PLANOS
            for nombre in PRIMEROS_PLANOS:
                capturar(app, servicio, detalle=nombre)
    except Exception:
        traceback.print_exc()
        estado = 1
    finally:
        app.close(exit_code=estado)
