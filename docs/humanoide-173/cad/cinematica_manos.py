"""Cinemática de un módulo de AmazingHand. Matrices homogéneas en milímetros.
El MJCF de Pollen Robotics define las articulaciones y los puntos de cierre.
Las dos rotaciones activas se imponen y se resuelven las articulaciones pasivas.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REF=Path(__file__).resolve().parents[1]/'referencias'/'AmazingHand'
WRIST_V=-121.


def translation(x=0,y=0,z=0):
    m=np.eye(4)
    m[:3,3]=(x,y,z)
    return m


def rotation(axis,angle):
    m=np.eye(4)
    m[:3,:3]=Rotation.from_rotvec(np.array(axis)*angle).as_matrix()
    return m


def xml_pose(node):
    m=translation(*(1000*np.fromstring(node.get('pos','0 0 0'),sep=' ')))
    quat=np.fromstring(node.get('quat','1 0 0 0'),sep=' ')
    m[:3,:3]=Rotation.from_quat(np.roll(quat,-1)).as_matrix()
    return m


class Finger:
    def __init__(self):
        doc=ET.parse(REF/'robot.xml')
        root=doc.find('worldbody/body')
        fixed=root.findall('geom')
        self.basis=np.linalg.inv(xml_pose(next(g for g in fixed if g.get('mesh')=='finger_frame_1')))
        self.fixed=[]
        for g in fixed:
            name=g.get('mesh','')
            m=self.basis@xml_pose(g)
            if (name in ('finger_frame_1','finger_frame_2','bushing_0608_04') and abs(m[0,3])<1 and abs(m[1,3])<1
                or name=='scs0009' and abs(abs(m[0,3])-6.05)<.05 and abs(m[1,3]+32.4)<.1):
                self.fixed.append((name,m))
        self.joints=[]
        self.sites=[]
        self.geometry=[]
        self.npassive=0
        def walk(body,parent):
            j=body.find('joint')
            name=j.get('name')
            active=name in ('finger1_motor1','finger1_motor2')
            count=3 if j.get('type')=='ball' else 1
            start=0 if name=='finger1_motor1' else 1 if active else self.npassive
            idx=len(self.joints)
            self.joints.append((parent,xml_pose(body),active,start,count))
            if not active:
                self.npassive+=count
            for g in body.findall('geom'):
                self.geometry.append((idx,g.get('mesh'),xml_pose(g)))
            for site in body.findall('site'):
                self.sites.append((idx,site.get('name'),xml_pose(site)[:3,3]))
            for child in body.findall('body'):
                walk(child,idx)
        for body in root.findall('body')[:2]:
            walk(body,-1)
        self.sites.extend((-1,g.get('name'),xml_pose(g)[:3,3]) for g in root.findall('site'))
        self.pairs=[(c.get('site1'),c.get('site2')) for c in doc.findall('equality/connect')
                    if '(1)' in c.get('site1') or c.get('site1')=='closing_ball1_1']

    def frames(self,active,passive):
        mats=[]
        for parent,base,driven,start,count in self.joints:
            q=active if driven else passive
            vec=q[start:start+3] if count==3 else [0,0,q[start]]
            turn=np.eye(4)
            turn[:3,:3]=Rotation.from_rotvec(vec).as_matrix()
            mats.append((self.basis if parent==-1 else mats[parent])@base@turn)
        return mats

    def residual(self,active,passive):
        mats=self.frames(active,passive)
        locations={}
        for idx,name,p in self.sites:
            frame=self.basis if idx==-1 else mats[idx]
            locations[name]=frame[:3,:3]@p+frame[:3,3]
        return np.concatenate([locations[a]-locations[b] for a,b in self.pairs])

    def solve(self,angles,seed=None):
        active=np.radians(angles)
        opt=least_squares(lambda q:self.residual(active,q),np.zeros(self.npassive) if seed is None else seed,
                          method='trf',ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=70)
        err=np.linalg.norm(opt.fun.reshape(-1,3),axis=1).max()
        if err>.055:
            raise ValueError(f'Cierre de bieletas fuera de tolerancia: {err:.3f} mm para {angles}')
        return self.frames(active,opt.x),opt.x,float(err)


@lru_cache()
def finger():
    return Finger()


@lru_cache()
def module_positions(side):
    # Both thumb mounts are proper rotations. The commercial servo is identical on both sides.
    doc=ET.parse(REF/('robot.xml' if side==1 else 'robot_left.xml'))
    gs=doc.findall('worldbody/body/geom')
    ref=next(g for g in gs if 'hand_plate' in g.get('mesh',''))
    shift=translation(100 if side==-1 else 0)@np.linalg.inv(xml_pose(ref))
    frames=[shift@xml_pose(g) for g in gs if g.get('mesh')=='finger_frame_1']
    little=translation(99*side,-29,0)@rotation([0,0,1],np.radians(-14*side))
    return dict(zip(('Indice','Medio','Anular','Pulgar','Menique'),[*frames,little]))


def wrist_pose(side):
    # u across palm, v towards fingers, w towards back of hand. Z=845 is the wrist pin.
    frame=translation(236*side,0,845)
    frame[:3,:3]=[[0,0,side],[-side,0,0],[0,-1,0]]
    return frame@translation(-side*24.5,-WRIST_V,11.5)

