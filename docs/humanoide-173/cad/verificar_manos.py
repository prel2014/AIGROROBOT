"""Comprobaciones dimensionales, de postura y de cierre del mecanismo."""
from itertools import combinations
from math import radians
from build123d import Compound, Pos, Rot
from manos_servo import (OPEN,THUMB_REST,WRIST_REST_DEG,cover,finger,library,loc,
    module_positions,palm,segment_poses,thumb_screws,wrist_parts,wrist_pose,yoke)
from interior_xavier import original,normal


def bound_gap(a,b):
    a,b=a.bounding_box(),b.bounding_box()
    return sum(max(0,getattr(a.min,d)-getattr(b.max,d),getattr(b.min,d)-getattr(a.max,d))**2 for d in 'XYZ')**.5


def overlap(a,b):
    if bound_gap(a,b)>.001:
        return 0.
    x=a.intersect(b)
    return sum(s.volume for s in x.solids()) if x is not None else 0.


def main():
    robot=original()
    f=finger()
    maximum=0
    for k in range(33):
        t=k/32
        for angles in [(-32+57*t,32-57*t),(-32-10*t,32-10*t)]:
            _,_,err=f.solve(angles)
            maximum=max(maximum,err)
    print(f'Cierre de lazos: {maximum:.6f} mm en 66 pares de mando',flush=True)
    for side,leg in [(1,'derecha'),(-1,'izquierda')]:
        limb=next(g for g in robot.children if 'pierna '+leg in normal(g.label))
        thigh=next(g for g in limb.children if 'muslo' in normal(g.label))
        cuff=loc(wrist_pose(side))
        tilt=Pos(236*side,0,845)*Rot(WRIST_REST_DEG,0,0)*Pos(-236*side,0,-845)
        shell=palm(side)
        covers=[cover(side,False),cover(side,True)]
        thumb=loc(module_positions(side)['Pulgar'])
        assert shell.is_valid and len(shell.solids())==1
        for key,m in f.fixed:
            if key.startswith('finger_frame'):
                frame=loc(m)*library()[key]
                support=thumb.inverse()*shell
                area=overlap(Pos(0,0,.1)*frame,support)/.1
                gap=frame.distance_to(support)
                assert gap<.001 and area>1 and overlap(frame,support)<.005
                print(leg,key,'distancia al asiento',round(gap,4),'mm',flush=True)
        anchor=thumb_screws(side)
        assert all(overlap(anchor,item)<.005 for item in [shell,yoke(side),*covers])
        assert overlap(yoke(side),shell)<.005
        assert all(overlap(s,yoke(side))<.005 for s in covers)
        assert all(overlap(s,shell)<.005 for s in covers)
        assert overlap(*covers)<.005
        forearm=next(g for b in robot.children if ('brazo derecho' if side==1 else 'brazo izquierdo') in normal(b.label)
                     for g in b.children if 'antebrazo' in normal(g.label))
        assert overlap(tilt*cuff*yoke(side),forearm)<.005
        print('Pasador y asiento',leg,round(overlap(yoke(side),wrist_parts(side)[0]),4),flush=True)
        check=['proximal_shell','distal_shell']
        for mode in ('abierta','menique','lateral','cierre','reposo'):
            surfaces={}
            for name,tf in module_positions(side).items():
                q=OPEN
                if mode=='reposo' and name=='Pulgar':q=THUMB_REST
                if mode=='cierre':q=(2,-2) if name=='Pulgar' else (25,-25)
                if name=='Menique' and mode=='menique':q=(25,-25)
                if name=='Menique' and mode=='lateral':q=(-42,22) if side==1 else (-22,42)
                frames,_,_=f.solve(q)
                exposed=[loc(tf@m)*library()[key] for key,m in segment_poses(frames) if key in check]
                surfaces[name]=Compound(children=exposed)
                if name in ('Menique','Pulgar') and mode in ('abierta','lateral','cierre','reposo'):
                    keys=('proximal','proximal_shell','gimbal','rotule_lever','link')
                    for key,m in segment_poses(frames):
                        if key in keys:
                            part=loc(tf@m)*library()[key]
                            assert overlap(part,covers[0])<.02,(leg,name,key,q)
                if mode=='abierta':
                    for key,m in f.fixed:
                        if key=='scs0009':assert overlap(loc(tf@m)*library()[key],shell)<.005
            volumes=[(a,b,overlap(surfaces[a],surfaces[b])) for a,b in combinations(surfaces,2)]
            print(leg,mode,'Cruce entre dedos',round(max(v for a,b,v in volumes),3),flush=True)
            assert max(v for a,b,v in volumes)<.02,volumes
            if mode=='reposo':
                # All five digits and fixed palm: list parts whose bounding boxes could beat the best result.
                objects=[tilt*cuff*item for item in [*surfaces.values(),shell,*covers,yoke(side)]]
                min_gap=1e3
                for item in sorted(objects,key=lambda x:bound_gap(x,thigh)):
                    if bound_gap(item,thigh)<min_gap:
                        min_gap=min(min_gap,item.distance_to(thigh))
                print(leg,'distancia a muslo',round(min_gap,4),'mm',flush=True)
                assert min_gap>1
    print('Verificación completada.')


if __name__=='__main__':
    main()

