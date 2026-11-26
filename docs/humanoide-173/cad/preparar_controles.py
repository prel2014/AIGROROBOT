"""Precalcula las poses de AmazingHand y las líneas centrales desde las fuentes CAD."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from arnes_brazos import external_points, rounded_path
from equipo_brazos import traction_points
from cinematica_manos import finger, module_positions, rotation, translation, wrist_pose
from instalacion_control import cable_axes, hand_loom_path
from manos_servo import OPEN, THUMB_REST, WRIST_REST_DEG, library, loc, part_key

HERE=Path(__file__).resolve().parent

def packed(matrix):
    return np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat()].round(8).tolist()

def rod_motion(base,pose,tip):
    # Las dos rótulas fijan los centros pero dejan libre el giro axial de la
    # varilla. Elegir el giro mínimo evita saltos al cambiar de solución.
    axis=tip/np.linalg.norm(tip)
    a,b=base[:3,:3]@axis,pose[:3,:3]@axis
    v=np.cross(a,b)
    r=Rotation.from_quat([*v,1+np.dot(a,b)]).as_matrix()
    delta=np.eye(4)
    delta[:3,:3]=r
    delta[:3,3]=pose[:3,3]-r@base[:3,3]
    return delta

def finger_rig():
    f=finger()
    groups={}
    index=len(f.fixed)
    for joint,raw,_ in f.geometry:
        if part_key(raw) in library():
            index+=1
            groups.setdefault(joint,[]).append(index)
    moves={}
    for role,start,end in [('dedos',OPEN,(25,-25)),('pulgar',THUMB_REST,(2,-2))]:
        rest,seed,_=f.solve(start)
        samples=[]
        tips={1:f.joints[2][1][:3,3],7:next(p for j,_,p in f.sites if j==7)}
        for t in np.linspace(0,1,21):
            frames,seed,_=f.solve(np.array(start)*(1-t)+np.array(end)*t,seed)
            samples.append([packed(rod_motion(rest[j],frames[j],tips[j]) if j in tips
                                   else frames[j]@np.linalg.inv(rest[j])) for j in groups])
        moves[role]=samples
    return list(groups.values()),moves

def route(path,anchors,weights,spacing=5):
    # Muestrear por distancia de arco y conservar los terminales exactos.
    ts=np.linspace(0,1,max(24,int(path.length/spacing)+1))
    pts=np.array([tuple(path.position_at(t)) for t in ts])
    locations=[0,*[int(np.argmin(np.linalg.norm(pts-p,axis=1))) for p in anchors[1:-1]],len(pts)-1]
    assert np.all(np.diff(locations)>=0)
    blends=[np.interp(np.arange(len(pts)),locations,axis) for axis in np.array(weights).T]
    return np.column_stack([pts,*blends]).round(5).tolist()

def build():
    groups,moves=finger_rig()
    sides={}
    for side,key in [(-1,'izq'),(1,'der')]:
        pivot=translation(side*236,0,845)
        hand=pivot@rotation([1,0,0],np.radians(WRIST_REST_DEG))@np.linalg.inv(pivot)@wrist_pose(side)
        frames=[packed(hand@m) for m in module_positions(side).values()]
        curves=[]
        for power in (False,True):
            anchors=external_points(side,power)
            weights=[(0,0),(0,0),(.15,0),(.35,0),(.7,0),(.95,0),(1,0),(1,0),
                     (1,.15),(1,.35),(1,.65),(1,.85),(1,1),(1,1),(1,1)]
            samples=route(rounded_path(anchors,22),anchors,weights)
            if power:
                # The 12 V loop has the outer lane. Carry its free span round
                # the shoulder sooner to preserve separation from the CAN1 loom.
                for row in samples:
                    u=row[3]
                    row[3]=round(u+.25*u*(1-u),5)
            curves.append(samples)
        wrist=hand_loom_path(side,loc(hand))
        pts=np.array([tuple(wrist.position_at(t)) for t in np.linspace(0,1,50)])
        outer=np.array(cable_axes(side)[0][1])
        crown=(hand@np.array([34*side,-78,40,1]))[:3]
        w=np.interp(np.arange(len(pts)),[np.argmin(np.linalg.norm(pts-outer,axis=1)),
                                         np.argmin(np.linalg.norm(pts-crown,axis=1))],[0,1])
        curves.append(np.column_stack([pts,w]).round(5).tolist())
        for power in (False,True):
            anchors=traction_points(side,power)
            blend=[0,0,.12,.70,.96,1,1,1,1,1,1,1,1]
            curves.append(route(rounded_path(anchors,12),anchors,[(v,0) for v in blend],
                                spacing=2 if side<0 else 5))
        sides[key]={'digits':frames,'curves':curves}
    return {'groups':groups,'moves':moves,'sides':sides}

if __name__=='__main__':
    data=json.dumps(build(),ensure_ascii=False,separators=(',',':'))
    template=(HERE/'controles_pose.js').read_text(encoding='utf8')
    (HERE/'humanoide_xavier.params.js').write_text(
        '// Generado por preparar_controles.py a partir del CAD y el MJCF.\nconst rig = '+data+';\n'+template,
        encoding='utf8')
