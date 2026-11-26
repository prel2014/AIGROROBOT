"""Ensayo integral en PhysX. El fruto siempre responde a gravedad y contactos."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from control_marcha import GANANCIAS_MARCHA, GANANCIAS_REPOSO, MARCHA, corregir_apoyos, referencias
from entrenar_marcha import ARTICULACIONES
from frenado import consignas
from criterio_cosecha import Sosten
from escena_naranjal import ESCENA, FRUTO

DT=1/120
CONTROL=1/60
HERE=Path(__file__).resolve().parent


def ejecutar(stage, parametros, semilla, salida_traza=None, fase_objetivo=None, centro_pinza=(-.030,.052,-.055),
             repeticion=None):
    import omni.physics.tensors as tensors
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import UsdUtils, PhysicsSchemaTools, Sdf
    origen_fruta=np.array(stage.GetPrimAtPath('/World/Naranja').GetAttribute('xformOp:translate').Get(),copy=True)
    cache=UsdUtils.StageCache.Get()
    key=cache.Insert(stage)
    fx=get_physx_simulation_interface()
    fx.attach_stage(key.ToLongInt())
    clock=0.
    contactos=set()
    apoyos=set()
    impulsos={}
    impulso_mano=np.zeros(3)
    centro_fruta=origen_fruta.copy()
    def evento(headers,data):
        for h in headers:
            if h.type==ContactEventType.CONTACT_LOST or not h.num_contact_data:
                continue
            if not any(np.linalg.norm(data[k].impulse)>1e-7 for k in
                       range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data)):
                continue
            a,b=(str(PhysicsSchemaTools.intToSdfPath(k)) for k in (h.actor0,h.actor1))
            if '/World/Naranja' in (a,b):
                otro=b if a=='/World/Naranja' else a
                contactos.add(otro)
                if '/World/Xavier/' in otro:
                    dedo=otro.split('/')[-1].split('_')[0]
                    impulsos[dedo]=impulsos.get(dedo,0.)+sum(np.linalg.norm(data[k].impulse) for k in
                        range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data))
                    if dedo in ('pulgar','indice','medio','anular','menique','mano'):
                        for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                            impulso=np.asarray(data[k].impulse)
                            hacia_centro=centro_fruta-np.asarray(data[k].position)
                            impulso_mano[:] += impulso if impulso@hacia_centro>=0. else -impulso
            for pie,otro in ((a,b),(b,a)):
                if '/pie_' in pie and ('Suelo' in otro or otro==''):
                    apoyos.add(pie)
    sub=fx.subscribe_contact_report_events(evento)
    def avanzar(n,dt=DT):
        nonlocal clock
        for _ in range(n):
            fx.simulate(dt,clock)
            fx.fetch_results()
            clock+=dt
    try:
        avanzar(1)
        sim=tensors.create_simulation_view('numpy',stage_id=key.ToLongInt(),backend='physx')
        r=sim.create_articulation_view('/World/Xavier/pelvis')
        fruta=sim.create_rigid_body_view('/World/Naranja')
        assert r.count==fruta.count==1 and not r.shared_metatype.fixed_base
        ids=np.arange(1,dtype=np.uint32)
        names=list(r.shared_metatype.dof_names)
        links=list(r.shared_metatype.link_names)
        juntas=[names.index(k) for k in ARTICULACIONES]
        arm=[names.index(k+'_der') for k in ('hombro','codo','muneca')]
        pie=[links.index('pie_'+k) for k in ('izq','der')]
        mano=links.index('mano_der')
        brazo=links.index('brazo_der')
        antebrazo=links.index('antebrazo_der')
        dedos=[j for j,name in enumerate(names) if '_der_m' in name]
        assert len(dedos)==10
        limites=r.get_dof_limits()[0]
        maxvel=r.get_dof_max_velocities()[0]
        cierre=np.where(abs(limites[dedos,0])>abs(limites[dedos,1]),limites[dedos,0],limites[dedos,1])
        kp=r.get_dof_stiffnesses()
        kd=r.get_dof_dampings()
        def ganancias(perfil):
            for j,m,send in ((0,kp,r.set_dof_stiffnesses),(1,kd,r.set_dof_dampings)):
                m[:,juntas]=perfil[j]*2
                send(m,ids)
        ganancias(GANANCIAS_REPOSO)
        rng=np.random.default_rng(semilla)
        root=r.get_root_transforms()
        root[0,1]+=rng.uniform(-.035,.035)
        r.set_root_transforms(root,ids)
        orden=r.get_dof_position_targets().copy()
        pose_inicio=root[0,:3].copy()
        avanzar(83)
        inicial_fruta=fruta.get_transforms()[0,:3].copy()
        if np.linalg.norm(inicial_fruta-origen_fruta)>.02:
            raise RuntimeError(f'La naranja se desprendió antes del ensayo: {inicial_fruta}')
        ganancias(GANANCIAS_MARCHA)
        fase=0.
        frenado=None
        quieto=0.
        etapa='marcha'
        principio=clock
        ref_freno=orden[:,juntas].copy()
        eventos=[]
        registros=[]
        cuadros=[]
        mejor_altura=0.
        sosten=Sosten()
        dedos_vistos=set()
        cierre_actual=np.zeros(len(dedos))
        tocados=set()
        tiempo_pinza=0.
        liberado=False
        # El pulgar en reposo se opone a los dedos largos cuando estos se flexionan.
        agarre_local=np.array(centro_pinza)
        es_pulgar=np.array([names[j].startswith('pulgar') for j in dedos])
        precarga=0.
        pose_inicial_pies=r.get_link_transforms()[0,pie,:2].mean(axis=0)
        minimo=1.
        for tick in range(940):
            trans=r.get_link_transforms()[0]
            if repeticion and tick%6==0:
                cuadros.append(np.vstack((trans,fruta.get_transforms()[0])))
            def punto(i,local):
                return trans[i,:3]+Rotation.from_quat(trans[i,3:]).apply(local)
            pose=r.get_root_transforms()[0]
            rot=Rotation.from_quat(pose[3:])
            vertical=rot.as_matrix()[2]
            vel=r.get_root_velocities()[0]
            q=r.get_dof_positions()[0]
            dq=r.get_dof_velocities()[0]
            pies=trans[pie,:3].mean(axis=0)
            obj=fruta.get_transforms()[0,:3].copy()
            centro_fruta[:]=obj
            pinza=punto(mano,agarre_local)
            dist=np.linalg.norm(obj-pinza)
            avance=pies[1]-pose_inicial_pies[1]
            if pose[2]<.85 or vertical[2]<.69 or np.linalg.norm(vel[:3])>3.:
                etapa='caida'
                break
            entrar= pies[1]>=FRUTO[1]-.40
            if fase_objetivo is not None:
                from frenado import ventana_parada
                entrar=ventana_parada(FRUTO[1]-.34-pies[1],fase,fase_objetivo)
            if etapa=='marcha' and entrar:
                etapa='freno'
                frenado=clock
                ref_freno=q[None,juntas].copy()
                ganancias(GANANCIAS_REPOSO)
                eventos.append([round(clock,3),etapa])
            if etapa=='marcha':
                t=tick*CONTROL
                fase+=2*np.pi*MARCHA[0]*CONTROL
                amp=np.array([min(t/.7,1.)])
                leg=referencias(MARCHA[None,:],np.array([fase]),amp,vertical[None,:],
                                rot.inv().apply(vel[3:])[None,:],q[None,juntas],dq[None,juntas])
                corregir_apoyos(leg,np.array([fase]),amp,np.array([pose[1]-pies[1]]),np.array([vel[1]]))
            else:
                leg=consignas(parametros[None,:],ref_freno,q[None,juntas],np.array([pose[1]-pies[1]]),
                              np.array([vel[1]]),np.array([np.arctan2(vertical[1],vertical[2])]),np.array([clock-frenado]))
            def aplicar(idx,meta):
                tope=maxvel[idx]*CONTROL
                orden[0,idx]+=np.clip(np.clip(meta,limites[idx,0],limites[idx,1])-orden[0,idx],-tope,tope)
            aplicar(juntas,leg[0])
            if etapa=='freno':
                quieto=quieto+CONTROL if np.linalg.norm(vel[:2])<.10 and vertical[2]>.96 and apoyos else 0.
                if quieto>=.55:
                    etapa='alcance'
                    principio=clock
                    eventos.append([round(clock,3),etapa])
            if etapa in ('alcance','cierre','elevar'):
                minimo=min(minimo,float(dist))
                target=inicial_fruta.copy()
                if etapa=='alcance':
                    aproximacion=np.clip((clock-principio-.75)/.90,0.,1.)
                    target[1]-=.075*(1.-aproximacion)
                    target[2]+=.065*(1.-aproximacion)
                if etapa=='elevar':
                    target[2]+=precarga+min(.080,.035*(clock-principio))
                if etapa=='cierre':
                    if clock-principio>.65:
                        precarga=np.clip(precarga+CONTROL*np.clip((1.12-impulso_mano[2]/CONTROL)*.005,-.006,.012),0.,.022)
                    target[2]+=precarga
                # Jacobiano sagital en el centro de prensión de la palma.
                pivotes=[punto(brazo,[0,.003,.15]),punto(antebrazo,[0,0,.125]),punto(mano,[0,-.003,.078])]
                axes=[Rotation.from_quat(trans[i,3:]).apply([1,0,0]) for i in (brazo,antebrazo,mano)]
                J=np.stack([np.cross(ax,pinza-piv) for ax,piv in zip(axes,pivotes)],axis=1)
                delta=target-pinza
                ajuste=J.T@np.linalg.solve(J@J.T+np.eye(3)*.003,delta)
                aplicar(arm,orden[0,arm]+np.clip(ajuste*.12,-.018,.018))
                if etapa=='alcance' and clock-principio>1.3 and dist<.024:
                    etapa='cierre'
                    principio=clock
                    eventos.append([round(clock,3),etapa])
                if etapa in ('cierre','elevar'):
                    # Al tocar la fruta, el dedo mantiene una pequeña precarga sobre la posición medida.
                    fuerza=np.array([impulsos.get(names[j].split('_')[0],0.)/CONTROL for j in dedos])
                    tasa=np.clip((1.6-fuerza)*.35,-.40,.50)
                    tasa[es_pulgar]=np.clip((fuerza[es_pulgar]-2.2)*.18,-.18,.18)
                    cierre_actual=np.clip(cierre_actual+CONTROL*tasa,0.,1.)
                    meta_dedos=cierre*cierre_actual
                    for k,j in enumerate(dedos):
                        if names[j].split('_')[0] in tocados and cierre[k]!=0:
                            if abs(meta_dedos[k])>abs(q[j])+.025:
                                meta_dedos[k]=q[j]+np.sign(cierre[k])*.025
                    aplicar(dedos,meta_dedos)
                if etapa=='cierre':
                    opuesto=bool({'mano','pulgar'} & tocados) and bool({'indice','medio','anular','menique'} & tocados)
                    carga=impulso_mano/CONTROL
                    soportado=carga[2]>.72 and np.linalg.norm(carga[:2])<2.6
                    tiempo_pinza=tiempo_pinza+CONTROL if opuesto and dist<.045 and soportado else 0.
                if etapa=='cierre' and clock-principio>1.1 and tiempo_pinza>=.10:
                    # Retirar únicamente el soporte del árbol. A partir de aquí actúan gravedad y contactos.
                    stage.GetPrimAtPath('/World/ArbolObjetivo/UnionPedunculo').GetAttribute('physics:jointEnabled').Set(False)
                    liberado=True
                    etapa='elevar'
                    principio=clock
                    eventos.append([round(clock,3),etapa])
            contactos.clear()
            impulsos.clear()
            impulso_mano.fill(0.)
            apoyos.clear()
            r.set_dof_position_targets(orden,ids)
            if etapa in ('alcance','cierre','elevar'):
                avanzar(4,CONTROL/4)
            else:
                avanzar(2)
            tocados={p.split('/')[-1].split('_')[0] for p in contactos if '/World/Xavier/' in p}
            dedos_vistos.update(tocados & {'mano','pulgar','indice','medio','anular','menique'})
            altura=float(fruta.get_transforms()[0,2]-inicial_fruta[2])
            mejor_altura=max(mejor_altura,altura)
            libre=liberado and not any('ArbolObjetivo' in p or 'Suelo' in p for p in contactos)
            palma_despues=r.get_link_transforms()[0,mano]
            centro_despues=palma_despues[:3]+Rotation.from_quat(palma_despues[3:]).apply(agarre_local)
            raiz_despues=r.get_root_transforms()[0]
            erguido=Rotation.from_quat(raiz_despues[3:]).as_matrix()[2,2]
            aprobado=sosten.medir(CONTROL,etapa=='elevar',altura,
                np.linalg.norm(fruta.get_transforms()[0,:3]-centro_despues),tocados,libre,erguido,
                bool(apoyos) and raiz_despues[2]>.85)
            if tick%6==0:
                registros.append([round(clock,3),etapa,round(float(avance),3),np.round(pose[:3],3).tolist(),
                                  np.round(vel[:2],3).tolist(),np.round(pinza,3).tolist(),np.round(obj,3).tolist(),
                                  sorted(tocados),round(float(vertical[2]),3),len(apoyos),
                                  Rotation.from_quat(trans[mano,3:]).inv().apply(obj-trans[mano,:3]).round(3).tolist(),
                                  np.rad2deg(q[arm]).round(1).tolist(),
                                  np.rad2deg(q[dedos]).round(1).tolist(),
                                  {k:round(float(v/CONTROL),3) for k,v in impulsos.items()},
                                  (impulso_mano/CONTROL).round(2).tolist()])
            if aprobado:
                etapa='agarre_verificado'
                break
        if salida_traza:
            salida_traza.write_text(json.dumps(registros))
        if repeticion:
            from repeticion_cosecha import guardar
            guardar(repeticion,ESCENA,links,cuadros)
        return dict(semilla=semilla,resultado=etapa,eventos=eventos,avance_m=round(float(avance),3),
                    distancia_minima_pinza_m=round(minimo,4) if minimo<1 else None,
                    altura_max_m=round(mejor_altura,4),sosten_continuo_s=round(sosten.maximo,3),
                    contactos_mano=sorted(dedos_vistos),soporte_retirado=liberado,caida=etapa=='caida')
    finally:
        del sub
        fx.detach_stage()
        cache.Erase(key)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo',type=Path,default=HERE/'freno_sincronizado.json')
    parser.add_argument('--semilla',type=int,default=192)
    parser.add_argument('--episodios',type=int,default=8)
    parser.add_argument('--informe',type=Path,default=HERE/'escenarios/ensayo_naranjal.json')
    parser.add_argument('--traza',type=Path)
    parser.add_argument('--repeticion',type=Path,help="Reproduce las poses medidas del primer episodio en una capa USDA.")
    parser.add_argument('--fase-apoyo',type=float,default=5*np.pi/8,
                        help="Fase del último apoyo calibrada con los dedos articulados, en radianes.")
    args=parser.parse_args()
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True,multi_gpu=False,disable_viewport_updates=True))
    try:
        from pxr import Usd
        modelo=json.loads(args.modelo.read_text())
        resultados=[]
        for i in range(args.episodios):
            stage=Usd.Stage.Open(str(ESCENA))
            stage.SetEditTarget(stage.GetSessionLayer())
            result=ejecutar(stage,np.array(modelo['parametros']),args.semilla+i,args.traza if i==0 else None,
                            args.fase_apoyo,repeticion=args.repeticion if i==0 else None)
            resultados.append(result)
            args.informe.write_text(json.dumps(dict(contrato='naranjal-1',fisica_marcha_hz=120,fisica_mano_hz=240,control_hz=60,
                parametros=modelo['parametros'],fase_apoyo_rad=args.fase_apoyo,episodios=resultados),indent=2))
            print('NARANJAL',result,flush=True)
    finally:
        app.close()
