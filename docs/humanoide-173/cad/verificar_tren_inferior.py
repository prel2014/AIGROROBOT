"""Barrido de interferencia entre eslabones con las transformaciones del visor."""
from itertools import product
from math import cos,radians,sin

import numpy as np
from build123d import Axis, Compound, Location
from interior_xavier import find_torso, normal, original
from tren_inferior import ANKLE, HIP, KNEE, Leg
from equipo_rodilla import pelvis_with_ports


def leaves(shape,parent=Location()):
    if shape.children:
        for child in shape.children:
            yield from leaves(child,parent*shape.location)
    else:
        shape=parent*shape
        box=shape.bounding_box()
        yield shape,np.array(list(product(*zip(tuple(box.min),tuple(box.max)))))


def sweep(name,fixed,mobile,pivot,angles):
    stationary=[(s,c.min(axis=0),c.max(axis=0)) for s,c in fixed]
    axis=Axis((0,0,pivot),(1,0,0))
    failures=[]
    for angle in angles:
        a=radians(angle)
        matrix=np.array([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]])
        for moving,points in mobile:
            pts=(points-[0,0,pivot])@matrix.T+[0,0,pivot]
            low,high=pts.min(axis=0),pts.max(axis=0)
            candidates=[shape for shape,lo,hi in stationary
                        if np.all(np.minimum(hi,high)-np.maximum(lo,low)>1e-5)]
            if not candidates:
                continue
            posed=moving.rotate(axis,angle) if angle else moving
            for shape in candidates:
                # Exact B-rep common. Face contact at seats and journals is intentional.
                common=posed.intersect(shape)
                volume=sum(s.volume for s in common.solids()) if common else 0
                if volume>.01:
                    failures.append((angle,shape.label,moving.label,round(volume,3)))
        if failures:
            break
    print(name, list(angles),'solapes >0.01 mm3:',failures[:15],flush=True)
    return not failures


def run():
    original_robot=original()
    torso=find_torso(original_robot)
    pelvic_parts=[]
    for p in torso.children:
        if 'pelvi' in normal(p.label):
            placed=torso.location*p
            placed.label=normal(p.label)
            pelvic_parts.append(placed)
    assert pelvic_parts
    pelvis=Compound(children=pelvic_parts)
    sockets=pelvis_with_ports(pelvis,[Leg(-1),Leg(1)])
    ok=True
    for side in (1,-1):
        design=Leg(side)
        old=next(p for p in original_robot.children if normal(p.label)==
                 ('pierna derecha' if side==1 else 'pierna izquierda'))
        leg=design.assemble(old)
        motor=design.motor()
        foot,ankle,shin,knee,thigh,hip=leg.children
        fork,kfix,kdrive=knee.children
        case,neck,rotor=motor.children
        assert all(p.is_valid for p,_ in leaves(leg))
        assert all(len(p.solids())==1 for p,_ in leaves(leg)), 'Componente discontinuo'
        for segment in (thigh,shin):
            cover=segment.children[34]
            assert cover.label=='Carcasa exterior desmontable' and len(cover.children)==4
        assert thigh.children[35].label=='Arnés CAN0 y 48 V'
        assert thigh.children[36].label=='Fundas exteriores hasta tapa'
        assert case.children[1].label.startswith('Controlador integrado CubeMars')
        assert case.children[2].label=='Tapa exterior ventilada'
        def parts(*nodes):
            return [item for node in nodes for item in leaves(node)]
        proximal=parts(fork,kfix,case,neck,thigh,hip)
        tibial=parts(kdrive,rotor,shin,ankle)
        plantar=parts(foot)
        for segment in (shin,thigh):
            panels=parts(segment.children[34])
            frame=parts(*[p for i,p in enumerate(segment.children) if i!=34])
            ok=sweep(f'{side} ajuste {segment.label}',frame,panels,0,[0]) and ok
        wire=thigh.children[35]
        cowling=case.children[2]
        ok=sweep(f'{side} tarjeta y clavija contra cuerpo',parts(case.children[0]),parts(case.children[1],case.children[3]),0,[0]) and ok
        ok=sweep(f'{side} mazo y tarjeta contra tapa',parts(cowling),parts(case.children[1],case.children[3],wire,thigh.children[36]),0,[0]) and ok
        ok=sweep(f'{side} mazo contra estructura',parts(hip,fork,case.children[0],*thigh.children[:34]),parts(*wire.children[:4]),0,[0]) and ok
        ok=sweep(f'{side} rodilla',proximal,tibial,KNEE,range(0,-81,-5)) and ok
        ok=sweep(f'{side} tobillo',tibial,plantar,ANKLE,range(-12,13,2)) and ok
        ok=sweep(f'{side} cadera',parts(pelvis,sockets),proximal,HIP,range(-15,36,5)) and ok
    return ok


if __name__=='__main__':
    raise SystemExit(0 if run() else 1)
