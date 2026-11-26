"""Política compartida PPO. Cada fila contiene exclusivamente los sensores de un Xavier."""
import os
from itertools import count
from pathlib import Path
import tempfile
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from torch.distributions import Normal

from cargas_plantares import LADOS, ZONAS
from situaciones import ACCIONES_ADICIONALES, CONTRATO, OBS_ADICIONALES, TAREAS

ENTRADAS_PLANTARES = [f"{lado}/{zona}" for lado in LADOS for zona in ZONAS]

class Politica(nn.Module):
    def __init__(self, estado, acciones=7):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 12, 5, stride=3), nn.SiLU(),
                                    nn.Conv2d(12, 16, 3, stride=2), nn.SiLU(),
                                    nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten())
        self.fusion = nn.Sequential(nn.Linear(16 * 9 + estado, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh())
        self.media = nn.Linear(64, acciones)
        self.valor = nn.Linear(64, 1)
        nn.init.orthogonal_(self.media.weight, gain=.02)
        nn.init.zeros_(self.media.bias)
        self.log_sigma = nn.Parameter(torch.full((acciones,), -.55))

    def forward(self, rgb, estado):
        vision = self.vision(rgb.permute(0, 3, 1, 2).float() / 255.)
        x = self.fusion(torch.cat((vision, estado), dim=1))
        return Normal(self.media(x), self.log_sigma.clamp(-3, 1).exp()), self.valor(x).squeeze(-1)


def tensores(obs, device):
    return tuple(torch.as_tensor(x, device=device) for x in obs)


def guardar(path, model, optim, iteration, etapa, racha, escenario="pista", perfil=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        torch.save(dict(politica=model.state_dict(), optimizador=optim.state_dict(), iteracion=iteration,
                        entradas_plantares=ENTRADAS_PLANTARES,
                        etapa=etapa, racha=racha, escenario=escenario, perfil=perfil), name)
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def actualizar(model, optim, buffer, bootstrap):
    rgb, estado, acciones, old_logp, valores, premios, continuan, activos = (
        torch.stack(items) for items in zip(*buffer))
    # La máscara corta la ventaja en caídas, contactos, metas y reinicios de la pista.
    ventajas = torch.zeros_like(premios)
    gae = torch.zeros_like(bootstrap)
    siguiente = bootstrap
    for t in reversed(range(len(buffer))):
        delta = premios[t] + .99 * continuan[t] * siguiente - valores[t]
        gae = delta + .99 * .95 * continuan[t] * gae
        ventajas[t] = gae
        siguiente = valores[t]
    retorno = (ventajas + valores)[activos]
    ventajas = ventajas[activos]
    ventajas = (ventajas - ventajas.mean()) / (ventajas.std(unbiased=False) + 1e-6)
    rgb, estado, acciones, old_logp = (x[activos] for x in (rgb, estado, acciones, old_logp))
    for _ in range(4):
        for idx in torch.randperm(len(retorno), device=retorno.device).split(128):
            dist, value = model(rgb[idx], estado[idx])
            # PPO opera sobre la normal antes de tanh. El jacobiano se cancela en el cociente.
            ratio = (dist.log_prob(acciones[idx]).sum(-1) - old_logp[idx]).exp()
            gain = torch.minimum(ratio * ventajas[idx], ratio.clamp(.8, 1.2) * ventajas[idx])
            loss = -gain.mean() + .5 * (value - retorno[idx]).square().mean() - .003 * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Pérdida PPO no finita.")
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optim.step()
    return len(retorno)


def restaurar(ckpt, model, optim=None):
    pesos = ckpt["politica"].copy()
    clave = "fusion.0.weight"
    capa = model.fusion[0].weight
    adicionales = capa.shape[1] - pesos[clave].shape[1]
    if adicionales == len(ENTRADAS_PLANTARES) and "entradas_plantares" not in ckpt:
        # Una política anterior conserva exactamente su función al recibir las ocho cargas.
        pesos[clave] = pad(pesos[clave], (0, adicionales))
    elif adicionales or ckpt.get("entradas_plantares") != ENTRADAS_PLANTARES:
        raise ValueError("Los canales del modelo no coinciden con el diseño de las plantas.")
    model.load_state_dict(pesos)
    if optim is not None:
        optim.load_state_dict(ckpt["optimizador"])
        if adicionales:
            for nombre, valor in optim.state[capa].items():
                if valor.shape == ckpt["politica"][clave].shape:
                    optim.state[capa][nombre] = pad(valor, (0, adicionales))


def aprender(args, env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = None
    if args.reanudar or args.evaluar:
        ckpt = torch.load(args.modelo, map_location="cpu", weights_only=True)
        env.etapa = args.etapa or ckpt.get("etapa", "equilibrio")
    obs = tensores(env.reset(), device)
    if args.captura:
        from PIL import Image
        ordered = [env.rgb[env.contact_index[p]] for p in env.paths]
        Image.fromarray(np.vstack([np.hstack(ordered[i:i+4]) for i in range(0, 16, 4)])).save(args.captura)
    model = Politica(obs[1].shape[1], env.prev_action.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    inicio = 0
    racha = 0
    if args.inicial:
        base = Politica(obs[1].shape[1] - len(OBS_ADICIONALES))
        restaurar(torch.load(args.inicial, map_location="cpu", weights_only=True), base)
        pesos = model.state_dict()
        # Los nuevos canales empiezan con residuo cero y un optimizador propio.
        for nombre, origen in base.state_dict().items():
            destino = pesos[nombre]
            if nombre != "log_sigma":
                destino.zero_()
            destino[tuple(slice(0, n) for n in origen.shape)].copy_(origen)
        model.load_state_dict(pesos)
    if ckpt is not None:
        if env.secuencia and ckpt.get("perfil"):
            perfil_guardado = ckpt["perfil"]
            if (perfil_guardado.get("contrato") != CONTRATO or
                tuple(perfil_guardado.get("observaciones_adicionales", ())) != OBS_ADICIONALES or
                tuple(perfil_guardado.get("acciones_adicionales", ())) != ACCIONES_ADICIONALES):
                raise ValueError("El contrato o el orden de canales de situaciones no coincide con la política.")
        restaurar(ckpt, model, None if args.evaluar else optim)
        inicio = ckpt["iteracion"]
        if ckpt.get("etapa") == env.etapa and ckpt.get("escenario", "pista") == env.escenario:
            racha = ckpt.get("racha", 0)
    print(f"MARCHA escenario {env.escenario}, política {device}, iteración inicial {inicio}", flush=True)
    print(f"MARCHA sensores: RGB {tuple(obs[0].shape)}, IMU {env.imu.shape}, "
          f"cargas N {env.cargas.shape}, estado {tuple(obs[1].shape)}, "
          f"aceleración media {9.81 * env.imu[:, 2].mean():.2f} m/s²", flush=True)
    episode = 0
    perfil = dict(semilla=args.semilla, escenario=env.escenario, solver=args.fisica, rgb=list(obs[0].shape[1:]),
                  periodo_control_s=env.dt_control, contrato=CONTRATO if env.secuencia else "xavier-marcha-plantar",
                  observaciones_adicionales=OBS_ADICIONALES if env.secuencia else (),
                  acciones_adicionales=ACCIONES_ADICIONALES if env.secuencia else ())
    evaluaciones = []
    iterations = count(inicio + 1) if args.evaluar else range(inicio + 1, inicio + args.iteraciones + 1)
    for it in iterations:
        buffer = []
        start = perf_counter()
        t_fisica, t_imagen = env.seg_fisica, env.seg_imagen
        for _ in range(args.pasos):
            while env.app.is_running() and not env.sim.is_playing() and not env.sim.is_stopped():
                env.app.update()
            if not env.app.is_running() or env.sim.is_stopped():
                return
            with torch.no_grad():
                dist, value = model(*obs)
                raw = dist.mean if args.evaluar else dist.sample()
                logp = dist.log_prob(raw).sum(-1)
            active = torch.as_tensor(env.active.copy(), device=device)
            new_obs, reward = env.step(raw.tanh().cpu().numpy())
            ended = env.terminado()
            cont = torch.as_tensor(env.active.copy(), device=device) & (not ended)
            if not args.evaluar:
                buffer.append((*obs, raw, logp, value, torch.as_tensor(reward, device=device), cont, active))
            if ended:
                episode += 1
                aprobados = env.resultado()
                if args.evaluar:
                    evaluaciones.append(dict(etapa=env.etapa, episodio=episode, segundos=round(env.time, 2),
                        aprobados=int(aprobados.sum()), caidas=env.falls, robots_con_choque=env.hits,
                        choques_con_entorno=int(env.chocado_entorno.sum()), llegadas=env.arrivals, eventos=env.eventos,
                        robots=[dict(id=int(env.numeros[i]), aprobado=bool(aprobados[i]),
                        **(dict(tarea=TAREAS[env.secuencia.tipo[i]], fases=int(env.secuencia.fase[i]),
                                distancia_objetivo_m=round(float(env.secuencia.recorrido[i]), 3),
                                angulo_objetivo_grados=round(float(np.rad2deg(env.secuencia.angulo[i])), 2),
                                lado_inicial=LADOS[env.secuencia.lado[i]]) if env.secuencia else {}),
                        avance_m=round(float(env.avance[i]),3), alternancias=int(env.steps[i]),
                        maximos_plantares_N=np.round(env.cargas_maximas[i], 2).tolist(),
                        distancia_minima_entre_pelvis_m=round(float(env.separacion_minima[i]),3),
                        caida=bool(env.caido[i]), choque=bool(env.chocado[i]), meta=bool(env.en_meta[i]))
                        for i in np.argsort(env.numeros)]))
                print(f"MARCHA {env.etapa} episodio {episode}: aprobados {aprobados.sum()}/16, "
                      f"avance medio {env.avance.mean():.3f} m, "
                      f"alternancias {env.steps.sum()}, robots con choque {env.hits}/16, "
                      f"caídas {env.falls}/16, llegadas {env.arrivals}/16", flush=True)
                if args.evaluar and args.informe:
                    args.informe.write_text(json.dumps(dict(semilla=args.semilla, iteracion=inicio, escenario=env.escenario,
                        pies=LADOS, zonas_por_pie=ZONAS, perfil=perfil,
                        episodios=evaluaciones), ensure_ascii=False, indent=2), encoding="utf8")
                if args.evaluar and episode >= args.evaluar:
                    return
                if not args.evaluar:
                    racha = racha + 1 if aprobados.all() else 0
                    if racha >= 2 and env.etapa in ("equilibrio", "marcha"):
                        env.etapa = "marcha" if env.etapa == "equilibrio" else "cruce"
                        racha = 0
                        print(f"MARCHA siguiente etapa: {env.etapa}", flush=True)
                new_obs = env.reset()
            obs = tensores(new_obs, device)
        if not args.evaluar:
            with torch.no_grad():
                _, bootstrap = model(*obs)
            samples = actualizar(model, optim, buffer, bootstrap)
            guardar(args.modelo, model, optim, it, env.etapa, racha, env.escenario, perfil)
            wall = perf_counter() - start
            fps = args.pasos / wall
            print(f"MARCHA PPO {it}: {samples} muestras activas, {fps:.2f} FPS/cámara, "
                  f"{fps * env.dt_control:.3f}x tiempo real, física {env.seg_fisica - t_fisica:.2f} s, "
                  f"imagen {env.seg_imagen - t_imagen:.2f} s, total {wall:.2f} s, modelo {args.modelo}", flush=True)
