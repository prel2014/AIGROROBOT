"""Punto de entrada manual al currículo de Xavier. Sin argumentos muestra los planes."""
import argparse
import json
from pathlib import Path
import sys
from curriculo_xavier import ACCIONES, DT, PLANES, VERSION, diseno

HERE = Path(__file__).resolve().parent
ESTADO = ("imu_acel_local_g:3","imu_gyro_local_rad_s:3","vertical_local:3","cargas_por_pie:8",
          "angulos_rad","velocidades_rad_s","accion_anterior","meta_local_m:2","fase:6")


def configurar(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("modo", choices=("plan","comprobar","entrenar","evaluar"), default="plan", nargs="?")
    p.add_argument("--tarea", choices=PLANES)
    p.add_argument("--entorno", choices=("vivienda","calle","campus"))
    p.add_argument("--nivel", type=int, choices=range(3), default=0)
    p.add_argument("--robots", type=int, choices=range(1,17), default=16)
    p.add_argument("--semilla", type=int, default=17)
    p.add_argument("--actualizaciones", type=int, default=300)
    p.add_argument("--pasos", type=int, default=48)
    p.add_argument("--episodios", type=int, default=5)
    p.add_argument("--salida", type=Path, default=Path("runs/habilidades"))
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--reanudar", action="store_true")
    grupo.add_argument("--inicial", type=Path, help="Transferir pesos de otra tarea o nivel del mismo contrato.")
    p.add_argument("--visible", action="store_true")
    p.add_argument("--json", action="store_true", help="Emitir la matriz de planes para el lanzador por lotes.")
    a = p.parse_args(argv)
    if a.modo in ("entrenar","evaluar") and not a.tarea:
        p.error("Selecciona una tarea")
    if a.tarea:
        a.entorno = a.entorno or PLANES[a.tarea].entornos[0]
        if a.entorno not in PLANES[a.tarea].entornos:
            p.error("La tarea no está configurada para ese entorno")
    if min(a.actualizaciones,a.pasos,a.episodios)<1:
        p.error("Los tamaños de ejecución deben ser positivos")
    return a


def verificar():
    for nombre, plan in PLANES.items():
        assert plan.fases and plan.segundos>0
        for entorno in plan.entornos:
            if not (HERE/"escenarios"/(entorno+".usda")).is_file():
                raise FileNotFoundError(entorno)
        for nivel in range(3):
            diseno(nombre,nivel)
    if not (HERE/"humanoide_xavier_isaac.usd").is_file():
        raise FileNotFoundError("Falta el USD de Xavier")
    if len(set(ACCIONES))!=len(ACCIONES):
        raise ValueError("Acciones repetidas")


def ejecutar(a, env):
    import numpy as np
    import torch
    from politica_marcha import Politica, actualizar, guardar, tensores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.semilla)
    torch.set_num_threads(2)
    carpeta = a.salida / a.tarea / a.entorno / f"nivel{a.nivel}" / f"semilla{a.semilla}"
    carpeta.mkdir(parents=True,exist_ok=True)
    ruta = carpeta/"politica.pt"
    evaluando = a.modo=="evaluar"
    if not evaluando and ruta.exists() and not a.reanudar:
        raise FileExistsError(f"{ruta} ya existe. Usa --reanudar u otra carpeta.")
    obs = tensores(env.reset(), device)
    modelo = Politica(obs[1].shape[1],len(ACCIONES)).to(device)
    optim = torch.optim.Adam(modelo.parameters(),lr=3e-4)
    perfil = dict(contrato=VERSION,acciones=list(ACCIONES),estado=list(ESTADO),rgb=[256,256,3],
                  nivel=a.nivel,entorno=a.entorno,tarea=a.tarea,semilla=a.semilla,robots=a.robots,
                  politica=str(device),fisica="PhysX CPU",dt_control=DT,dt_fisica=DT/6)
    inicio = 0
    if a.inicial or a.reanudar or evaluando:
        ck = torch.load(a.inicial or ruta,map_location=device,weights_only=True)
        anterior = ck["perfil"]
        for clave in ("contrato","acciones","estado","rgb"):
            if perfil[clave]!=anterior[clave]:
                raise ValueError(f"Checkpoint incompatible en {clave}")
        if a.reanudar or evaluando:
            for clave in ("nivel","entorno","tarea"):
                if perfil[clave]!=anterior[clave]:
                    raise ValueError(f"El perfil solicitado difiere en {clave}")
        modelo.load_state_dict(ck["politica"])
        if a.reanudar:
            optim.load_state_dict(ck["optimizador"])
        if a.reanudar or evaluando:
            inicio = ck["iteracion"]
    modelo.train(not evaluando)
    registro = carpeta/("evaluacion.jsonl" if evaluando else "entrenamiento.jsonl")
    episodio = 0
    it = inicio
    while evaluando or it < inicio+a.actualizaciones:
        if not evaluando:
            it += 1
        buffer = []
        for _ in range(a.pasos):
            with torch.no_grad():
                dist,valor = modelo(*obs)
                accion = dist.mean if evaluando else dist.sample()
                logp = dist.log_prob(accion).sum(-1)
            activos = torch.as_tensor([not e.hecho for e in env.episodios],device=device)
            nuevo,premio = env.step(accion.tanh().cpu().numpy())
            siguen = torch.as_tensor([not e.hecho for e in env.episodios],device=device)
            if not evaluando:
                buffer.append((*obs,accion,logp,valor,torch.as_tensor(premio,device=device),siguen,activos))
            if not siguen.any():
                episodio += 1
                resultados = [dict(estacion=env.rutas[i],exito=e.exito,fases=e.indice,segundos=round(e.ticks*DT,3))
                              for i,e in enumerate(env.episodios)]
                with registro.open("a",encoding="utf8") as f:
                    f.write(json.dumps(dict(perfil=perfil,episodio=episodio,iteracion=it,robots=resultados))+"\n")
                print(f"{a.tarea} · episodio {episodio} · éxitos {sum(e.exito for e in env.episodios)}/{env.n}",flush=True)
                if evaluando and episodio >= a.episodios:
                    return
                nuevo = env.reset()
            obs = tensores(nuevo,device)
        if not evaluando:
            with torch.no_grad():
                _,bootstrap = modelo(*obs)
            actualizar(modelo,optim,buffer,bootstrap)
            guardar(ruta,modelo,optim,it,a.tarea,0,a.entorno,perfil)


def main():
    a = configurar()
    if a.modo=="plan":
        if a.json:
            print(json.dumps({n:p.entornos for n,p in PLANES.items()}))
            return
        for nombre,p in PLANES.items():
            print(f"{nombre:12} {' / '.join(p.entornos):22} {' → '.join(p.fases)}")
        return
    verificar()
    if a.modo=="comprobar":
        print(f"{len(PLANES)} planes · {len(ACCIONES)} acciones · archivos presentes")
        return
    sys.argv[1:] = []
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=not a.visible,multi_gpu=False,hide_ui=not a.visible,
                             extra_args=["--/app/hydraEngine/waitIdle=true"]))
    status = 0
    try:
        from adaptador_habilidades import Estaciones
        ejecutar(a, Estaciones(app,a.tarea,a.nivel,a.entorno,a.robots,a.semilla))
    except Exception:
        import traceback
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    raise SystemExit(status)


if __name__=="__main__":
    main()
