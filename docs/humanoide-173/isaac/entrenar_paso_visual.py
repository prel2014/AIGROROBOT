"""Entrenamiento por imitación con etiquetas geométricas exclusivas del generador."""
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

HERE = Path(__file__).resolve().parent


def guardar_lote(ruta, imagenes, etiquetas, escenas):
    with NamedTemporaryFile(dir=ruta.parent, delete=False) as archivo:
        temporal = Path(archivo.name)
    try:
        with temporal.open("wb") as archivo:
            np.savez(archivo, x=imagenes, y=etiquetas, escena=escenas)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def muestra(rng, numeros, caravanas=False, contraste=None):
    from paso_visual import itinerarios_cruce
    inicio, frente, meta, puesto = itinerarios_cruce(rng, numeros)
    grupo, slot = puesto // 4, puesto % 4
    cabecera = slot < 2
    centro = inicio + frente * np.where(cabecera, .8, 2.)[:, None]
    fase = rng.integers(0, 6, (4, 2))[grupo, slot % 2]
    s = np.where(cabecera, -.8, -2.) + rng.uniform(-.17, .17, 16)
    espera = (fase == 0) | (~cabecera & (fase <= 3))
    delantera = [(-.95, -.65),(-.62, .02),(.02, .80),(.80, 2.02),(1.35, 2.10),(1.91, 2.14)]
    for k in range(1, 6):
        filas = cabecera & (fase == k)
        s[filas] = rng.uniform(*delantera[k], filas.sum())
    filas = ~cabecera & (fase == 4)
    s[filas] = rng.uniform(-1.83, .73, filas.sum())
    filas = ~cabecera & (fase == 5)
    s[filas] = rng.uniform(.77, 1.06, filas.sum())
    if caravanas:
        # Una fila atraviesa el cruce como en los recorridos físicos: cuando la
        # cabecera acaba de pasar, el segundo todavía se acerca por la derecha.
        eje_movil = rng.integers(0, 2, 4)
        movil = slot % 2 == eje_movil[grupo]
        avance = rng.uniform(-1.35, 3.05, 4)[grupo]
        base = np.where(cabecera, -.8, -2.) + rng.uniform(-.14, .14, 16)
        salida = np.where(cabecera, 2.05, .95) + rng.uniform(-.07, .07, 16)
        objetivo = np.where(cabecera, avance, avance-rng.uniform(1.58, 1.96, 4)[grupo])
        s = np.where(movil, np.clip(objetivo, base, salida), base)
        espera = ~movil | (objetivo <= base)
        # El observador también puede ser el segundo de su fila mientras su
        # predecesor avanza al otro lado o espera cerca de la intersección.
        precedente_avanzado = rng.random(4) < .36
        adelantar = ~movil & cabecera & precedente_avanzado[grupo]
        s[adelantar] = rng.uniform(-.32, 2.09, adelantar.sum())
        espera[adelantar] = False
    if contraste is not None:
        # Dos fotos del mismo observador y el mismo fondo. Sólo cambia la
        # posición del transversal, incluso con su compañero ya estacionado.
        observada = slot % 2 == rng.integers(0, 2, 4)[grupo]
        s = np.where(cabecera, -.8, -2.) + rng.uniform(-.14, .14, 16)
        primero_fuera = observada & cabecera & (rng.random(4)[grupo] < .26)
        s[primero_fuera] = rng.uniform(.0, 2.10, primero_fuera.sum())
        espera = observada & ~primero_fuera
        estacionados = np.where(cabecera, 2.04, .98) + rng.uniform(-.1, .1, 16)
        acercandose = rng.uniform(-1.72, .58, 16)
        s[~observada] = estacionados[~observada]
        if contraste:
            s[~observada & ~cabecera] = acercandose[~observada & ~cabecera]
    der = frente[:, ::-1] * [1., -1.]
    lateral = rng.normal(0., np.where(espera, .018, .05))
    xy = centro + frente * s[:, None] + der * lateral[:, None]
    return xy, frente, espera, etiquetar(xy, frente, grupo, s)


def etiquetar(xy, frente, grupo, s):
    # Etiquetas de conducta local: distancia al predecesor y cesión al tráfico
    # transversal que se aproxima por la derecha o que ya ocupa la intersección.
    der = frente[:, ::-1] * [1., -1.]
    delta = xy[None, :, :] - xy[:, None, :]
    delante = (delta * frente[:, None, :]).sum(axis=2)
    costado = (delta * der[:, None, :]).sum(axis=2)
    alineacion = frente @ frente.T
    misma = grupo[:, None] == grupo[None, :]
    cerca = (alineacion > .9) & (delante > .1) & (delante < 1.69) & (abs(costado) < .38)
    derecha = (der @ frente.T) < -.9
    transversal = (abs(alineacion) < .1) & ((derecha & (s[None, :] < .69)) | (abs(s[None, :]) < .48))
    etiquetas = np.stack(((misma & cerca).any(axis=1), (misma & transversal).any(axis=1)), axis=1)
    return etiquetas


def recoger(args):
    import sys
    sys.argv[1:] = []
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, multi_gpu=False, hide_ui=True,
        open_usd=str(HERE / "escenarios/campus.usda"), extra_args=["--/app/hydraEngine/waitIdle=true"]))
    estado = 0
    try:
        from scipy.spatial.transform import Rotation
        from entrenar_marcha import Pista
        import torch
        torch.set_num_threads(2)
        args.escenario, args.etapa, args.fisica, args.visible = "campus", "cruce", "cpu", False
        args.modelo_paso = args.modelo
        env = Pista(app, args)
        if args.contrafactual:
            contrastar(args, env)
            return
        if args.semillas_recorrido:
            recoger_recorridos(args, env)
            return
        env.reset()
        rng = np.random.default_rng(args.semilla)
        root = env.robot.get_root_transforms().copy()
        imagenes, clases, escenas = [], [], []
        for k in range(args.escenas):
            if args.pares_entrenamiento:
                rng = np.random.default_rng([args.semilla, k//2])
            contraste = bool(k % 2) if args.pares_entrenamiento else None
            xy, frente, espera, etiquetas = muestra(rng, env.numeros, args.caravanas, contraste)
            yaw = -np.arctan2(frente[:, 0], frente[:, 1]) + rng.normal(0., .018, 16)
            root[:, :2] = xy
            root[:, 2] = 1.173
            ang = np.column_stack((yaw, rng.normal(0., .009, 16), rng.normal(0., .012, 16)))
            root[:, 3:] = Rotation.from_euler("zyx", ang).as_quat()
            env.robot.set_root_transforms(root, env.ids)
            env.robot.set_root_velocities(np.zeros((16, 6), np.float32), env.ids)
            pose = env.neutral.copy()
            pose[:, env.joints[:6]] += (~espera[:, None]) * np.deg2rad(
                rng.uniform([2, 8, 3]*2, [13, 28, 11]*2, (16, 6)))
            env.robot.set_dof_positions(pose, env.ids)
            env.robot.set_dof_velocities(np.zeros_like(pose), env.ids)
            env.robot.set_dof_position_targets(pose, env.ids)
            # Tras cambiar todos los cuerpos RTX puede seguir entregando píxeles
            # de la disposición anterior. Vaciar la cola antes de etiquetar el RGB.
            env.avanzar(2)
            for _ in range(12):
                env.sim.render()
            rgb = env.imagen()[espera]
            if len(rgb):
                mini = torch.nn.functional.avg_pool2d(torch.from_numpy(rgb).permute(0,3,1,2).float(), 2)
                imagenes.append(mini.round().byte().numpy())
                clases.append(etiquetas[espera])
                escena = 200000 + k//2 if args.pares_entrenamiento else k + 100000*args.caravanas
                escenas.append(np.full(len(rgb), escena))
            if (k+1) % 50 == 0 or k+1 == args.escenas:
                prefijo = "lote_pareado" if args.pares_entrenamiento else "lote_caravana" if args.caravanas else "lote"
                guardar_lote(args.datos / f"{prefijo}_{k:05d}.npz", np.concatenate(imagenes),
                             np.concatenate(clases), np.concatenate(escenas))
                imagenes.clear()
                clases.clear()
                escenas.clear()
                print(f"VISIÓN {k+1}/{args.escenas} disposiciones de tráfico renderizadas", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        estado = 1
    finally:
        app.close(exit_code=estado)
    raise SystemExit(estado)


def contrastar(args, env):
    """Conserva al observador en su sitio y cambia el tráfico por su derecha."""
    import json
    from scipy.spatial.transform import Rotation
    from paso_visual import PasoVisual, itinerarios_cruce
    env.reset()
    red = PasoVisual(args.modelo)
    rng = np.random.default_rng(args.semilla)
    root = env.robot.get_root_transforms().copy()
    resultados = []
    for _ in range(args.pares):
        inicio, frente, _, puestos = itinerarios_cruce(rng, env.numeros)
        grupo, slot = puestos // 4, puestos % 4
        centros = inicio + frente*np.where(slot < 2, .8, 2.)[:, None]
        derecha = frente[:, ::-1]*[1., -1.]
        cede = ((derecha @ frente.T < -.9) & (grupo[:, None] == grupo)).any(1)
        observador = np.flatnonzero(cede & (slot < 2))
        posiciones = np.where(cede, np.where(slot < 2, -.8, -2.), np.where(slot < 2, 1.96, .91))
        posiciones += rng.uniform(-.05, .05, 16)
        predicciones = []
        for bloqueado in (False, True):
            s = posiciones.copy()
            if bloqueado:
                s[~cede & (slot >= 2)] -= 1.8
            root[:, :2] = centros + frente*s[:, None]
            root[:, 2] = 1.173
            root[:, 3:] = Rotation.from_rotvec(np.column_stack((np.zeros((16, 2)),
                                                    -np.arctan2(frente[:, 0], frente[:, 1])))).as_quat()
            env.robot.set_root_transforms(root, env.ids)
            env.robot.set_root_velocities(np.zeros((16, 6), np.float32), env.ids)
            env.robot.set_dof_positions(env.neutral, env.ids)
            env.robot.set_dof_velocities(np.zeros_like(env.neutral), env.ids)
            env.avanzar(2)
            for _ in range(12):
                env.sim.render()
            predicciones.append(red.riesgo(env.imagen()[observador]).max(1))
        for i, libre, ocupado in zip(observador, *predicciones):
            resultados.append(dict(robot=int(env.numeros[i]), riesgo_libre=float(libre),
                riesgo_ocupado=float(ocupado), pasa_libre=bool(libre < red.umbral),
                espera_ocupado=bool(ocupado >= red.umbral)))
    resumen = dict(semilla=args.semilla, pares=len(resultados), umbral=red.umbral,
        ambas_correctas=sum(r["pasa_libre"] and r["espera_ocupado"] for r in resultados),
        pares_observados=resultados)
    args.contrafactual.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"PRUEBA PAREADA {resumen['ambas_correctas']}/{resumen['pares']} pares correctos", flush=True)


def recoger_recorridos(args, env):
    """Guarda ejemplos de espera que la política encuentra al caminar en el simulador."""
    import torch
    from politica_marcha import Politica, restaurar, tensores
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for semilla in args.semillas_recorrido:
        env.rng = np.random.default_rng(semilla)
        obs = env.reset()
        marcha = Politica(obs[1].shape[1], env.prev_action.shape[1]).to(dev).eval()
        restaurar(torch.load(args.modelo_marcha, weights_only=True, map_location="cpu"), marcha)
        # La geometría se consulta para etiquetar la copia de los fotogramas.
        # env.step usa exclusivamente su red visual para la decisión de entrar.
        centro = np.where(env.start > 0, 3.2, -3.2)
        grupo = (centro[:, 0] > 0).astype(int) + 2*(centro[:, 1] > 0)
        x, y = [], []
        k = 0
        decisiones_erroneas = 0
        while not env.terminado():
            grupos_con_incidente = grupo[env.chocado | env.caido]
            espera = env.active & ~env.en_travesia & ~np.isin(grupo, grupos_con_incidente)
            if espera.any() and (k % 8 == 0 if env.time < 38. else k % 32 == 0):
                xy = env.robot.get_root_transforms()[:, :2]
                s = ((xy - centro)*env.heading).sum(axis=1)
                etiquetas = etiquetar(xy, env.heading, grupo, s)[espera]
                imagen = torch.from_numpy(env.rgb[espera]).permute(0, 3, 1, 2).float()
                x.append(torch.nn.functional.avg_pool2d(imagen, 2).round().byte().numpy())
                y.append(etiquetas)
            with torch.inference_mode():
                dist, _ = marcha(*tensores(obs, dev))
            antes = env.en_travesia.copy()
            xy = env.robot.get_root_transforms()[:, :2]
            etiquetas_antes = etiquetar(xy, env.heading, grupo, ((xy-centro)*env.heading).sum(axis=1))
            obs, _ = env.step(dist.mean.tanh().cpu().numpy())
            decisiones_erroneas += int((~antes & env.en_travesia & etiquetas_antes.any(1)).sum())
            k += 1
        imagenes = np.concatenate(x)
        # Reservar episodios completos cuando su semilla termina en cero.
        guardar_lote(args.datos / f"lote_recorrido_{semilla}.npz", imagenes, np.concatenate(y),
                     np.full(len(imagenes), 1000000 + semilla))
        print(f"RECORRIDO {semilla}: aprobados {env.resultado().sum()}/16, caídas {env.falls}, "
              f"robots con choque {env.hits}, sin iniciar {np.isnan(env.salida_s).sum()}, "
              f"entradas con conflicto {decisiones_erroneas}, imágenes {len(imagenes)}", flush=True)


def entrenar(args):
    import torch
    from paso_visual import RedDePaso
    torch.set_num_threads(3)
    torch.manual_seed(args.semilla)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lotes = sorted(args.datos.glob("lote_*.npz"))
    if args.datos_caravanas:
        lotes.extend(sorted(args.datos_caravanas.glob("lote_caravana_*.npz")))
    if args.datos_pareados:
        lotes.extend(sorted(args.datos_pareados.glob("lote_pareado_*.npz")))
    if args.datos_recorridos:
        lotes.extend(sorted(args.datos_recorridos.glob("lote_recorrido_*.npz")))
    lotes = sorted(set(lotes))
    if not lotes:
        raise ValueError("Primero recoge imágenes del campus con --recoger.")
    def leer(clave):
        partes = []
        for ruta in lotes:
            with np.load(ruta) as datos:
                partes.append(datos[clave])
        return torch.from_numpy(np.concatenate(partes))
    x, y, escena = leer("x"), leer("y"), leer("escena")
    ent = torch.where(escena % 10 != 0)[0]
    val = torch.where(escena % 10 == 0)[0]
    fondos = None
    if args.fondos:
        from sklearn.cluster import KMeans
        claves = torch.cat([RedDePaso.describir(x[ids]) for ids in ent.split(128)]).numpy()
        grupos = KMeans(n_clusters=32, n_init=2, random_state=args.semilla).fit_predict(claves)
        # La mediana de imágenes del mismo fondo suprime el tráfico variable.
        # Las referencias se ajustan sólo con imágenes de entrenamiento.
        fondos = torch.stack([x[ent[grupos == k]].median(0).values for k in range(32)])
    net = RedDePaso(fondos).to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=.0007)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epocas, eta_min=.00006)
    mejor = float("inf")
    for epoch in range(args.epocas):
        net.train()
        for ids in ent[torch.randperm(len(ent))].split(96):
            pix = x[ids].to(dev).float()
            pix = pix * torch.empty((len(ids),3,1,1),device=dev).uniform_(.91,1.09)
            pix = pix + torch.empty((len(ids),1,1,1),device=dev).uniform_(-7,7)
            logits = net.desde_miniatura(pix)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[ids].to(dev).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()
        net.eval()
        with torch.inference_mode():
            logits = torch.cat([net.desde_miniatura(x[ids].to(dev).float()).cpu() for ids in val.split(128)])
            error = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[val].float()).item()
            pred = (logits.sigmoid() >= args.umbral).any(1)
            verdad = y[val].any(1)
            print(f"VISIÓN época {epoch+1} · pérdida validación {error:.4f} · pasos indebidos {(~pred & verdad).sum().item()}/{verdad.sum().item()} · esperas indebidas {(pred & ~verdad).sum().item()}/{(~verdad).sum().item()}", flush=True)
        if error < mejor:
            mejor = error
            temporal = args.modelo.with_suffix(".tmp")
            torch.save(dict(contrato="xavier-rgb-paso-v1", red=net.state_dict(), umbral=args.umbral,
                confirmacion_s=args.confirmacion, epoca=epoch+1, semilla=args.semilla, ejemplos_entrenamiento=len(ent),
                ejemplos_validacion=len(val), perdida_validacion=error, referencias_visuales=args.fondos,
                umbral_movimiento=args.umbral_movimiento,
                entrenamiento="imitación de riesgo frontal y transversal con imágenes renderizadas",
                imagenes_caravanas=int(((escena >= 100000) & (escena < 200000)).sum()),
                imagenes_pareadas=int(((escena >= 200000) & (escena < 1000000)).sum()),
                semillas_recorridos=[int(r.stem.rsplit("_", 1)[1]) for r in lotes
                                     if r.stem.startswith("lote_recorrido_")]), temporal)
            temporal.replace(args.modelo)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recoger", action="store_true")
    p.add_argument("--caravanas", action="store_true", help="Muestrear dos robots consecutivos en la fila que cruza.")
    p.add_argument("--pares-entrenamiento", action="store_true", help="Alternar imágenes de tráfico libre y ocupado manteniendo el observador.")
    p.add_argument("--semillas-recorrido", type=int, nargs="+", help="Recoger imágenes físicas durante cruces autónomos.")
    p.add_argument("--contrafactual", type=Path, help="Informe con pares de vistas de paso libre y ocupado.")
    p.add_argument("--pares", type=int, default=12, help="Disposiciones para la prueba pareada de cuatro observadores.")
    p.add_argument("--modelo-marcha", type=Path, default=HERE/"marcha_campus_validada.pt")
    p.add_argument("--datos", type=Path, required=True)
    p.add_argument("--datos-recorridos", type=Path, help="Ejemplos adicionales de recorridos autónomos.")
    p.add_argument("--datos-caravanas", type=Path, help="Ejemplos de cabecera y segundo robot cruzando consecutivamente.")
    p.add_argument("--datos-pareados", type=Path, help="Pares de vistas del mismo observador con distinta ocupación.")
    p.add_argument("--fondos", action="store_true", help="Aprender referencias de fondo con los RGB de entrenamiento.")
    p.add_argument("--escenas", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=2761)
    p.add_argument("--epocas", type=int, default=24)
    p.add_argument("--umbral", type=float, default=.35)
    p.add_argument("--confirmacion", type=float, default=1.5, help="Segundos continuos de vía libre antes de entrar.")
    p.add_argument("--umbral-movimiento", type=float, help="Fracción mínima de cambios locales para mantener la espera tras ver tráfico por la derecha.")
    p.add_argument("--modelo", type=Path, default=HERE/"paso_visual_campus.pt")
    args = p.parse_args()
    if not 0. < args.umbral < 1. or args.confirmacion <= 0.:
        p.error("El umbral debe estar entre 0 y 1 y la confirmación debe ser positiva.")
    if args.umbral_movimiento is not None and not 0. < args.umbral_movimiento <= 1.:
        p.error("El umbral de movimiento debe estar en el intervalo (0, 1].")
    args.datos = args.datos.resolve()
    args.datos.mkdir(parents=True, exist_ok=True)
    if args.recoger or args.semillas_recorrido or args.contrafactual:
        recoger(args)
    else:
        entrenar(args)
