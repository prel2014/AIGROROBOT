"""Holguras del brazo izquierdo en el STEP y con la cinemática de CAD Viewer (mm)."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from arnes_brazos import saddle, upper_with_pilots
from equipo_brazos import LANES, elbow_mount, elbow_motor, traction_branch, upper_panels
from interior_xavier import original, normal
from manos_servo import equipped_forearm, loc, wrist_parts, wrist_pose
from verificar_instalacion import bare, overlap
from verificar_brazos import rotate, surface

ROOT=Path(__file__).resolve().parent
TARGET=5.0


def centreline(controls, shoulder):
    # Same centripetal Catmull-Rom coefficients, 200-entry arc-length lookup
    # and ring count as THREE.CatmullRomCurve3 / TubeGeometry in controles_pose.js.
    t=np.radians(shoulder*controls[:,3])
    p=controls[:,:3].copy()
    y,z=p[:,1].copy(),p[:,2]-1400
    p[:,1]=np.cos(t)*y-np.sin(t)*z
    p[:,2]=1400+np.sin(t)*y+np.cos(t)*z
    ext=np.vstack((2*p[0]-p[1],p,2*p[-1]-p[-2]))
    diff=np.diff(ext,axis=0)
    dt=np.sqrt(np.linalg.norm(diff,axis=1))[:,None]
    slopes=diff/dt
    mid=(ext[2:]-ext[:-2])/(dt[1:]+dt[:-1])
    tangents=slopes[:-1]+slopes[1:]-mid
    a,b=tangents[:-1]*dt[1:-1],tangents[1:]*dt[1:-1]
    h=np.diff(p,axis=0)
    def point(f):
        u=f*(len(p)-1)
        i=np.minimum(u.astype(int),len(h)-1)
        u=(u-i)[:,None]
        return p[i]+u*(a[i]+u*((3*h-2*a-b)[i]+u*(-2*h+a+b)[i]))
    f=np.linspace(0,1,201)
    lens=np.r_[0,np.cumsum(np.linalg.norm(np.diff(point(f),axis=0),axis=1))]
    return point(np.interp(np.linspace(0,lens[-1],2*len(p)+1),lens,f))


def ring_samples(line):
    # Bound point sampling error for a continuous chord by half the longest
    # sample interval. The swept discs enclose the viewer's octagonal tubes.
    delta=np.diff(line,axis=0)
    n=max(1,int(np.ceil(np.linalg.norm(delta,axis=1).max()/.20)))
    cloud=(line[:-1,None,:]+delta[:,None,:]*np.arange(n)[None,:,None]/n).reshape(-1,3)
    return np.vstack((cloud,line[-1])),np.linalg.norm(delta,axis=1).max()/(2*n)


def ensure_free(a,b,label):
    v=overlap(bare(a),bare(b))
    if v>.01: raise ValueError(f'{label}: interferencia de {v:.3f} mm3')


def run():
    conf=json.loads((ROOT/'humanoide_xavier.params.js').read_text(encoding='utf8').split('const rig = ',1)[1].split(';',1)[0])
    controls=[np.asarray(c) for c in conf['sides']['izq']['curves'][3:5]]
    assert all(np.all(p[:,4]==0) for p in controls), 'El ramal ya no está fijo al estator'
    radii=[r for _,_,r,_ in LANES]
    arm=next(p for p in original().children if normal(p.label)=='brazo izquierdo')
    upper=arm.children[4]
    panels=upper_panels(-1,upper_with_pilots(upper,-1))
    branch=traction_branch(-1)
    jackets=list(map(bare,branch.children[1:3]))
    retain=branch.children[3].children
    sleeves=list(map(bare,retain[:2]))
    boot=retain[-1]
    near=lambda a,b:a.distance_to(b)
    fixed=[near(jackets[0],jackets[1]),near(sleeves[0],sleeves[1]),
           near(sleeves[0],jackets[1]),near(jackets[0],sleeves[1])]
    print('STEP: mazos, protectores y dos pares mixtos (mm):',np.round(fixed,3),flush=True)
    assert min(fixed)>=TARGET
    for idx,z in ((2,1185),(4,1280)):
        guide=retain[idx]
        for item in [saddle(-1,z,upper),upper,*panels.children[-2:],*jackets,retain[idx+1]]:
            ensure_free(guide,item,guide.label+' / '+item.label)
        for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3)]:
            ensure_free(guide.children[a],guide.children[b],'Cunas y tornillos')
    for j in jackets:
        ensure_free(boot,j,'Entrada al manguito')
    ensure_free(boot,elbow_motor(-1),'Manguito / motor equipado')
    for core in boot.children[1:]:
        ensure_free(core,boot.children[0],'Conductos en el manguito')
    ensure_free(*boot.children[1:],'Conductores dentro del manguito')
    # The reinforced sleeves and guide are rigid on the humerus. Test their
    # actual solids against the contralateral cable outside its assigned seat.
    obstacles=[surface(p) for p in sleeves]
    boxes=[p.bounding_box() for p in sleeves]
    seated=[(height,surface(bare(retain[ix].children[0]).fuse(bare(retain[ix].children[1]))))
            for ix,height in ((2,1185),(4,1280))]
    seat_gap=float('inf')
    closest=(float('inf'),None)
    bound=float('inf')
    # Local upper-arm boxes around the distal tongue, forearm body and wrist.
    # Each is transformed at every elbow detent. An expanded AABB miss proves
    # clearance; test B-rep triangulations only if such boxes overlap.
    fore=equipped_forearm(arm.children[2],-1)
    elbow=elbow_mount(arm.children[3],-1)
    wrist=loc(wrist_pose(-1))
    distal=[fore.children[0],fore.children[3],elbow.children[2],elbow.children[3],
            *[wrist*p for p in wrist_parts(-1)]]
    corners=[]
    for part in distal:
        bb=part.bounding_box()
        corners.append(np.array(list(product(*zip(tuple(bb.min),tuple(bb.max))))))
    distal_boxes=[(np.min([rotate(c,1100,e).min(0) for c in corners],axis=0),
                   np.max([rotate(c,1100,e).max(0) for c in corners],axis=0)) for e in range(91)]
    distal_sdf=None
    elbow_bound=float('inf')
    for sh in range(-20,66):
        poses=[ring_samples(centreline(p,sh)) for p in controls]
        (a,ea),(b,eb)=poses
        gap=cKDTree(a).query(b)[0].min()-sum(radii)
        if gap<closest[0]: closest=(gap,sh)
        bound=min(bound,gap-ea-eb)
        for i,(world,eps) in enumerate(poses):
            local=rotate(world,1400,-sh)
            radius=radii[i]
            for height,guide in seated:
                in_guide=local[np.abs(local[:,2]-height)<7]
                clear=min(guide.EvaluateFunction(p) for p in in_guide)-radius
                seat_gap=min(seat_gap,clear-eps-.015)
            bb=boxes[1-i]
            roi=np.all((local>=np.array(tuple(bb.min))-12)&(local<=np.array(tuple(bb.max))+12),axis=1)
            if roi.any():
                ds=min(obstacles[1-i].EvaluateFunction(p) for p in local[roi])
                bound=min(bound,ds-radius-eps-.15)
            # A nearest box distance > 10 mm suffices. Retain all cable samples
            # that can reach any box, and rotate back only when close to a link.
            roi=local[:,2]<max(high[2] for _,high in distal_boxes)+14
            points=local[roi]
            for el,(lo,hi) in enumerate(distal_boxes):
                if not len(points): continue
                offset=np.maximum(np.maximum(lo-points,points-hi),0)
                near_box=np.linalg.norm(offset,axis=1)<14
                if not near_box.any(): continue
                if distal_sdf is None: distal_sdf=[surface(p) for p in distal]
                q=rotate(points[near_box],1100,-el)
                distance=min(s.EvaluateFunction(p) for p in q for s in distal_sdf)
                elbow_bound=min(elbow_bound,distance-radius-eps-.15)
        if bound<TARGET or elbow_bound<=0 or seat_gap<=0:
            raise ValueError((sh,bound,elbow_bound,seat_gap))
    print('Hombro -20..65 y codo 0..90, paso 1 grado: 7826 combinaciones.')
    print(f'Mínimo muestreado {closest[0]:.3f} mm con hombro a {closest[1]}°; cota conservadora con protectores {bound:.3f} mm.')
    print(f'Holgura radial comprobada en los pasos de las mordazas: >= {seat_gap:.3f} mm.')
    print('Sin interferencia con las piezas móviles de codo/antebrazo/muñeca en el barrido.')


if __name__=='__main__': run()
