"""Optimiza una política de frenado con poblaciones de Xavier de base libre en PhysX."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from calibrar_marcha import Banco
from control_marcha import GANANCIAS_MARCHA, GANANCIAS_REPOSO, MARCHA, referencias, corregir_apoyos
from frenado import INICIAL, NOMBRES, consignas, ventana_parada

DT = 1/60
ARCHIVO = Path(__file__).with_name("frenado_entrenado.json")


def probar(b, params, rng, fase_objetivo=None):
    r = b.robot
    n = r.count
    root = b.inicio.copy()
    root[:, :2] = np.column_stack((np.arange(n)%4 * 3., np.arange(n)//4 * 3.)) - 4.5
    root[:, 2] = 1.173
    root[:, 3:] = (0,0,0,1)
    b.ganancias(*GANANCIAS_REPOSO)
    r.set_root_transforms(root,b.ids)
    r.set_root_velocities(np.zeros((n,6),np.float32),b.ids)
    r.set_dof_positions(b.neutro,b.ids)
    r.set_dof_velocities(np.zeros_like(b.neutro),b.ids)
    orden = b.neutro.copy()
    r.set_dof_position_targets(orden,b.ids)
    b.avanzar(84)
    b.ganancias(*GANANCIAS_MARCHA)
    meta = root[:,1] + rng.uniform(.38,.72,n)
    lado = rng.integers(0,2,n)
    angulo = rng.uniform(.49,.84,n)
    codos=[r.shared_metatype.dof_names.index('codo_'+s) for s in ('izq','der')]
    reloj=np.zeros(n)
    inicial=orden[:, b.juntas].copy()
    parar=np.zeros(n,bool)
    vivo=np.ones(n,bool)
    apoyo=np.zeros(n)
    acierto=np.zeros(n)
    duracion=np.zeros(n)
    scores=np.zeros(n)
    for tick in range(620):
        pose=r.get_root_transforms()
        vel=r.get_root_velocities()
        rot=Rotation.from_quat(pose[:,3:])
        arriba=rot.as_matrix()[:,2,:]
        pitch=np.arctan2(arriba[:,1],arriba[:,2])
        q=r.get_dof_positions()[:,b.juntas]
        dq=r.get_dof_velocities()[:,b.juntas]
        pies=r.get_link_transforms()[:,b.pies,:3].mean(axis=1)
        sep=pose[:,1]-pies[:,1]
        fase=np.full(n,2*np.pi*MARCHA[0]*DT*(tick+1))
        amp=np.full(n,min(tick*DT/.7,1.))
        desired=referencias(np.tile(MARCHA,(n,1)),fase,amp,arriba,rot.inv().apply(vel[:,3:]),q,dq)
        corregir_apoyos(desired,fase,amp,sep,vel[:,1])
        nuevo=(pies[:,1] >= meta-.06) & ~parar
        if fase_objetivo is not None:
            nuevo=ventana_parada(meta-pies[:,1],fase,fase_objetivo) & ~parar
        if nuevo.any():
            inicial[nuevo]=q[nuevo]
            parar |= nuevo
            for matriz, base, set_attr in ((b.kp,GANANCIAS_REPOSO[0],r.set_dof_stiffnesses),
                                           (b.kd,GANANCIAS_REPOSO[1],r.set_dof_dampings)):
                matriz[np.ix_(nuevo,b.juntas)]=base*2
                set_attr(matriz,b.ids)
        reloj += parar*DT
        freno=consignas(params,inicial,q,sep,vel[:,1],pitch,reloj)
        desired[parar]=freno[parar]
        orden[:,b.juntas]+=np.clip(np.clip(desired,b.limites[:,0],b.limites[:,1])-orden[:,b.juntas],
                                     -b.velocidad*DT,b.velocidad*DT)
        # Dos ciclos de levantar y bajar el antebrazo durante cuatro segundos.
        brazos=np.zeros((n,2))
        brazos[np.arange(n),lado]=angulo*np.clip(np.sin(reloj*np.pi/2)*2,0,1)
        orden[:,codos]+=np.clip(brazos-orden[:,codos],-1.45*DT,1.45*DT)
        r.set_dof_position_targets(orden,b.ids)
        vivo &= (pose[:,2]>.85)&(arriba[:,2]>.85)&(np.abs(pies[:,1]-meta)<1.)
        estable=(np.abs(vel[:,1])<.16)&(arriba[:,2]>.96)&(abs(pies[:,1]-meta)<.22)
        alcanzado=np.max(abs(r.get_dof_positions()[:,codos]-brazos),axis=1)<.14
        ventana=parar & vivo & (reloj<=5.)
        scores+=DT*ventana*(1.+2.*estable+alcanzado*.3-np.minimum(np.abs(vel[:,1]),1.)*.3)
        duracion+=DT*ventana
        apoyo+=DT*(ventana&estable)
        acierto+=DT*(ventana&estable&alcanzado)
        if np.all(~vivo | (reloj>=5.)): break
        b.avanzar(2)
    ok=(duracion>4.95)&(apoyo>3.5)&(acierto>3.3)
    return scores+4*ok, ok


def entrenar(b,args):
    rng=np.random.default_rng(args.semilla)
    media=INICIAL.copy()
    des=np.array([.15,.22,.12,1.6,.60,2.,.8,1.3,.09,.045])
    lower=np.array([-.10,.0,-.4,-4,-2,-6,-4,-6,.025,-.12])
    upper=np.array([.61,1.15,.4,8,4,8,5,7,.55,.14])
    mejor=media.copy()
    if args.desde:
        mejor=np.asarray(json.loads(args.desde.read_text())['parametros'])
        media=mejor.copy()
    if args.fases:
        fases=np.arange(16)*np.pi/16
        exitos=np.zeros(16,int)
        for ep in range(12):
            _,ok=probar(b,np.tile(mejor,(16,1)),rng,fases)
            exitos+=ok
        args.salida.write_text(json.dumps(dict(contrato="ventana-freno-v1",semilla=args.semilla,
            intentos_por_fase=12,fases_rad=fases.tolist(),exitos=exitos.tolist(),
            fase_rad=float(fases[np.argmax(exitos)])),indent=2))
        print('FASES',exitos.tolist(),flush=True)
        return
    historial=[]
    fase=None if args.ventana is None else json.loads(args.ventana.read_text())['fase_rad']
    for gen in range(args.generaciones):
        lote=np.clip(rng.normal(media,des,(16,len(media))),lower,upper)
        lote[0]=mejor
        lote[1]=media
        notas=np.zeros(16)
        total=0
        for _ in range(2):
            score,ok=probar(b,lote,rng,fase)
            notas+=score
            total+=int(ok.sum())
        elites=lote[np.argsort(notas)[-4:]]
        mejor=elites[-1].copy()
        media=elites.mean(axis=0)
        des=np.maximum([.035,.04,.025,.35,.11,.33,.15,.23,.025,.013],.45*des+.55*elites.std(axis=0))
        historial.append(dict(generacion=gen+1,aprobados=total,intentos=32,nota=float(notas.max())))
        args.salida.write_text(json.dumps(dict(contrato='frenado-sagital-1',semilla=args.semilla,solver='PhysX CPU 120 Hz',
            nombres=NOMBRES,parametros=mejor.tolist(),fase_rad=fase,historial=historial),indent=2))
        print('FRENO',gen+1,total,notas.max(),mejor.round(3).tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generaciones',type=int,default=20)
    parser.add_argument('--semilla',type=int,default=426)
    parser.add_argument('--salida',type=Path,default=ARCHIVO)
    parser.add_argument('--desde',type=Path)
    parser.add_argument('--fases',action='store_true')
    parser.add_argument('--ventana',type=Path)
    args=parser.parse_args()
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,disable_viewport_updates=True))
    try:
        b=Banco()
        try:
            entrenar(b,args)
        finally:
            b.cerrar()
    finally:
        app.close()
