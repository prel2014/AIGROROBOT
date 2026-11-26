"""Comprueba el ajuste de los paneles y el paso de los mazos en poses del visor."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from build123d import Compound, Pos, Rot
from arnes_brazos import upper_with_pilots
from equipo_brazos import upper_panels, traction_branch, elbow_motor, elbow_mount
from interior_xavier import original, normal
from manos_servo import equipped_forearm
from verificar_instalacion import check_arm, overlap, bare


def surface(shape):
    vs,fs=shape.tessellate(.12,.13)
    pts=vtkPoints()
    pts.SetData(numpy_to_vtk(np.array([tuple(v) for v in vs]),deep=True))
    cells=vtkCellArray()
    indices=np.ascontiguousarray(fs,dtype=np.int64).ravel()
    cells.SetData(3,numpy_to_vtkIdTypeArray(indices,deep=True))
    poly=vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    dist=vtkImplicitPolyDataDistance()
    dist.SetInput(poly)
    return dist


def rotate(points,z,degrees):
    a=np.deg2rad(degrees)
    mat=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    return (points-[0,0,z])@mat.T+[0,0,z]


def cable_points(curve, shoulder, elbow):
    return np.asarray([rotate(rotate(p[:3],1100,elbow*p[4]),1400,shoulder*p[3]) for p in curve])


def run():
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/'humanoide_xavier.params.js').read_text(encoding='utf8').split('const rig = ',1)[1].split(';',1)[0])
    robot=original()
    failures=[]
    for side,key in ((1,'der'),(-1,'izq')):
        arm=next(c for c in robot.children if normal(c.label)==('brazo derecho' if side==1 else 'brazo izquierdo'))
        upper=upper_panels(side,upper_with_pilots(arm.children[4],side))
        fore=equipped_forearm(arm.children[2],side)
        check_arm(fore,side)
        motor=elbow_motor(side)
        elbow=elbow_mount(arm.children[3],side)
        branch=traction_branch(side)
        def clear(a,b,label):
            v=overlap(bare(a),bare(b))
            if v>.04: failures.append((key,label,round(v,2),'mm3'))
        closed=Compound(children=[bare(p) for p in upper.children[-2:]])
        for p in [*upper.children[:-2],branch.children[3],branch.children[1],branch.children[2]]:
            clear(closed,p,'Panel / '+p.label)
        for a in (-20,0,30,65):
            move=Pos(0,0,1400)*Rot(a,0,0)*Pos(0,0,-1400)
            clear(move*bare(closed),branch.children[0],f'Panel a {a} / conectores del hombro')
        for a in (0,30,60,90):
            move=Pos(0,0,1100)*Rot(a,0,0)*Pos(0,0,-1100)
            for fixed in (closed,elbow.children[-1],motor):
                clear(fixed,move*bare(fore.children[0]),f'Carcasa antebrazo a {a} / '+fixed.label)
            clear(elbow.children[-1],move*bare(elbow.children[3]),f'Eje a {a} / soporte')
        distances={k:surface(s) for k,s in [('panel',closed),('soporte',elbow.children[-1]),
                   ('cubierta motor',motor.location*motor.children[0].children[2].children[0]),
                   ('tapa antebrazo',fore.children[3]),('hombro',arm.children[5])]}
        curves=[np.asarray(cfg['sides'][key]['curves'][i]) for i in (0,1,3,4)]
        radii=(1.55,1.95,2.,2.6)
        lowest={}
        for shoulder,angle in product((-20,0,30,65),(0,30,60,90)):
            ctr=[cable_points(c,shoulder,angle) for c in curves]
            dense=[]
            for i,pts in enumerate(ctr):
                # Sample line segments every ~1 mm, including ends.
                fine=np.vstack([((1-t)*pts[:-1]+t*pts[1:]) for t in np.arange(5)/5]+[pts[-1:]])
                dense.append(fine)
                upper_pts=rotate(fine,1400,-shoulder)
                for label,s in distances.items():
                    local=rotate(upper_pts,1100,-angle) if label=='tapa antebrazo' else upper_pts
                    gap=min(s.EvaluateFunction(p) for p in local)-radii[i]
                    k=(label,i)
                    lowest[k]=min(lowest.get(k,(1e6,0,0)),(round(gap,2),shoulder,angle))
            for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                gap=cKDTree(dense[i]).query(dense[j])[0].min()-radii[i]-radii[j]
                k=('mazos',i,j)
                lowest[k]=min(lowest.get(k,(1e6,0,0)),(round(float(gap),2),shoulder,angle))
        for k,(gap,sh,el) in lowest.items():
            if gap<0: failures.append((key,k,gap,sh,el,'mm'))
        print(key,'holguras por grupo:',lowest,flush=True)
    print('Interferencias:',failures,flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': run()
