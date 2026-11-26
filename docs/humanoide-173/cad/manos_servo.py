"""Adaptación mecánica de cinco dígitos. CAD AmazingHand, Pollen Robotics, CC BY 4.0.
Ver referencias/AmazingHand/ATRIBUCION.md. Longitudes en mm.
"""
from functools import lru_cache
from math import atan2, degrees, hypot
import numpy as np
from scipy.spatial.transform import Rotation
from build123d import (Axis, Box, Color, Compound, Cylinder, Keep, Location, Plane, Pos,
                      RectangleRounded, RegularPolygon, Rot, Sphere, extrude, import_step, loft)
from cinematica_manos import REF, WRIST_V, finger, module_positions, wrist_pose
from instalacion_control import arm_module, hand_port, hand_loom

DARK='#34454A'
STEEL='#8A9D9F'
WHITE='#E5E4DA'
TEAL='#328781'
BRONZE='#C38650'
BODY='#49595D'
OPEN=(-32,32)
THUMB_REST=(35,-35)
WRIST_REST_DEG=20
PALMAR_FASTENER_Y=-93.2


def loc(matrix):
    q=Rotation.from_matrix(matrix[:3,:3]).as_rotvec()
    angle=float(np.linalg.norm(q))
    return Location(tuple(matrix[:3,3]),tuple(q/angle) if angle>1e-12 else (0,0,1),degrees(angle))


def painted(s,name,color):
    s.label=name
    s.color=Color(color)
    return s


def block(x,y,z,l,w,h):
    return Pos(x,y,z)*Box(l,w,h)


def cy(x,y,z,r,h):
    return Pos(x,y,z)*Rot(90,0,0)*Cylinder(r,h)


def single(s):
    solids=list(s.solids())
    if len(solids)!=1:
        raise ValueError(f'Se esperó una pieza continua y se obtuvieron {len(solids)}')
    return solids[0]


def strip(a,b,width,z0,z1):
    x,y=a
    dx,dy=b[0]-x,b[1]-y
    p=Pos(x+dx/2,y+dy/2,(z0+z1)/2)*Rot(0,0,degrees(atan2(dy,dx)))*Box(hypot(dx,dy),width,z1-z0)
    ends=[Pos(*pt,(z0+z1)/2)*Cylinder(width/2,z1-z0) for pt in [a,b]]
    return p.fuse(*ends)


@lru_cache()
def library():
    params={
        'finger_frame_1':('Finger_Frame-1','Marco proximal',STEEL),
        'finger_frame_2':('Finger_Frame-2','Marco de servos',STEEL),
        'proximal':('Proximal','Falange proximal',DARK),
        'distal':('Distal','Falange distal',DARK),
        'proximal_shell':('Proximal_Shell','Almohadilla proximal TPU',WHITE),
        'distal_shell':('Distal_Shell','Almohadilla distal TPU',WHITE),
        'link':('Link','Biela de acoplamiento',TEAL),
        'gimbal':('Gimbal','Cardan',BRONZE)}
    parts={key:painted(single(import_step(REF/(file+'.step'))),label,color)
           for key,(file,label,color) in params.items()}
    # STEP catalog shaft centre (x=0,z=-5.7); seat y=24.1. Match the two physical
    # shaft datums of the reference, with no reflection of the purchased servo.
    motor=import_step(REF.parent/'feetech_scs0009.step')
    parts['scs0009']=painted(Pos(0,9.575,5.7)*Rot(0,0,180)*Compound(children=list(motor.solids())),
                             'Feetech SCS0009',BODY)
    horn=strip((0,0),(-5,0),4.4,-1.4,1.5).fuse(Pos(0,0,-.25)*Cylinder(2.9,4.5))
    horn=horn.cut(Pos(-5,0,0)*Cylinder(1,8),Cylinder(1.98,8))
    parts['custom_servo_horn']=painted(horn,'Palanca de servo',BRONZE)
    parts['rotule_ball']=painted(Sphere(2.5)-Cylinder(1.05,7),'Bola de rotula',STEEL)
    socket=Cylinder(3.75,4).fuse(block(0,-7.2,0,4,14.4,4))
    socket=socket.cut(Sphere(2.55),cy(0,-11.5,0,1.06,9))
    parts['rotule_lever']=painted(socket,'Terminal de bieleta',TEAL)
    parts['m2_rod_l18']=painted(cy(0,-16.5,0,1,18),'Varilla M2 por 18',STEEL)
    parts['bushing_0608_04']=painted(Pos(0,0,-2)*(Cylinder(4,4)-Cylinder(3,6)),'Casquillo 6 por 8 por 4',BRONZE)
    for length in (10,16):
        parts['pin'+str(length)]=painted(Cylinder(1,length),f'Pasador 2 por {length}',STEEL)
    return parts


def part_key(name):
    if name.startswith('parallel_pin_2_x_16'): return 'pin16'
    if name.startswith('parallel_pin_2_x_10'): return 'pin10'
    return name


def segment_poses(frames):
    model=finger()
    for key,m in model.fixed:
        yield key,m
    for body,raw,tf in model.geometry:
        key=part_key(raw)
        if key in library():
            yield key,frames[body]@tf


@lru_cache()
def digit(angles=OPEN):
    frames,_,_=finger().solve(angles)
    parts=[]
    for key,tf in segment_poses(frames):
        item=loc(tf)*library()[key]
        item.label=library()[key].label
        parts.append(item)
    return Compound(label='Dedo con dos servos',children=parts)


def original_part(side,name):
    code='R' if side==1 else 'L'
    shift=0 if side==1 else 100
    return Pos(shift,0,0)*single(import_step(REF/f'{code}-{name}.step'))


def spine_holes(side,radius):
    return [Pos(side*x,y,25)*Cylinder(radius,30) for x,y in ((76,-33),(76,-67))]


def thumb_heel(side):
    # The two thumb brackets bolt directly to the heel of the palm. Transfer
    # their original seats and hole pattern from the old wrist to the palm.
    old=original_part(side,'Wrist_Interface')
    flange=old.intersect(block(side*24.5,-87,-10,125,4,95))
    thumb=loc(module_positions(side)['Pulgar'])
    rib=old.intersect(thumb*block(0,-23,17.5,38,55,10))
    # The rear servo bracket is offset 0.60 mm in the reference mechanism.
    # A raised land closes that gap while keeping its central fastener clear.
    land=Pos(0,-44.5,12.25)*(Cylinder(3.0,.7)-Cylinder(1.18,2))
    return single(single(flange).fuse(*rib.solids(),thumb*land))


THUMB_FIXINGS=((-6,-7,16.5),(6,-7,16.5),(0,-44.5,15.5))


@lru_cache()
def thumb_screws(side):
    frame=loc(module_positions(side)['Pulgar'])
    # M2 x 8 assembly envelopes. Threads are represented by the shank core.
    head=(Pos(0,0,.76)*Cylinder(1.9,1.52)).cut(
        Pos(0,0,.60)*extrude(RegularPolygon(.865,6),amount=1.05))
    screw=head.fuse(Pos(0,0,-4)*Cylinder(.9,8))
    screws=[painted(frame*Pos(x,y,z)*screw,'M2 x 8 - fijacion del pulgar',BRONZE)
            for x,y,z in THUMB_FIXINGS]
    return Compound(label='Tres tornillos de anclaje del pulgar',children=screws)


@lru_cache()
def palm(side):
    base=original_part(side,'Hand_Plate')
    # Transfer the manufacturer's real index mounting pockets to an extra station.
    # Two dorsal webs join the transplanted seat to the existing ring-finger sector.
    donor=base.intersect(block(0,-23.5,2,28.4,58,40))
    holder=loc(module_positions(side)['Menique'])*single(donor)
    links=[strip((side*a,b),(side*c,d),w,14.3,18.5)
           for a,b,c,d,w in [(59,-29,100,-39,12),(49,-65,89,-71,13),(89,-71,101,-39,9)]]
    plate=single(base.fuse(holder,*links,thumb_heel(side)).cut(*spine_holes(side,1.2)))
    return painted(plate,'Palma continua con asiento de pulgar y menique',STEEL)


def cuff_center(side):
    return np.array([side*24.5,WRIST_V,-11.5])


@lru_cache()
def yoke(side):
    u,v,w=cuff_center(side)
    stations=[(v-10,24.5,-11.5,26,47,7),(v-3,24.5,-11.5,35,55,8),
              (v+7,24.5,-11.5,42,56,9),(-99,24.5,-10.8,55,49,8),
              (-88.65,24.5,-9.5,66,42,6)]
    sections=[Pos(side*x,y,z)*(Plane.XZ*RectangleRounded(width,depth,radius))
              for y,x,z,width,depth,radius in stations]
    body=loft(sections)
    # The hand carries the thumb. The wrist bolts to its four-hole rear flange.
    tools=[Pos(u,v,w)*Cylinder(24.85,44.6),block(u,v-40,w,66,80,44.6),
           Pos(u,v,w)*Cylinder(4.18,75),cy(u,-104,w,7.2,44)]
    for x in (-6,55):
        for z in (-7.5,10.5):
            tools.extend([cy(side*x,-119,z,2.7,54),cy(side*x,-89,z,1.45,8)])
    for px in (14,36):
        tools.extend([Pos(side*px,PALMAR_FASTENER_Y,-41.05)*Cylinder(4.05,18),
                      Pos(side*px,PALMAR_FASTENER_Y,-28.2)*Cylinder(1.22,8)])
    frame=loc(module_positions(side)['Pulgar'])
    tools.extend(frame*Pos(x,y,z+1.1)*Cylinder(2.15,2.4) for x,y,z in THUMB_FIXINGS)
    tools.append(palm(side))
    # Remove the free island inside the original screw-head pocket.
    saddle=max(body.cut(*tools).solids(),key=lambda solid:solid.volume)
    return painted(saddle,'Base carpal integrada con apoyo doble',TEAL)


def thumb_rim():
    # A flared elliptical-rectangular mouth seats the thumb within the palm
    # skin. The palmar sector is scalloped to clear the opposing linkage.
    def profiles(stations):
        return [Pos(0,v,z)*(Plane.XZ*RectangleRounded(w,d,r)) for v,z,w,d,r in stations]
    outside=loft(profiles([(-16,1,43,48,11),(-8,1,40,45,10),(2,1.5,36,40,9)]))
    bore=loft(profiles([(-17,1,38.5,43.5,9),(-8,1,35.5,40.5,8),(3,1.5,32,36,7)]))
    return outside.cut(bore,block(0,3,-19,24,17,14))


@lru_cache()
def cover(side,dorsal):
    kind='Top_Shell' if dorsal else 'Palm_Shell'
    master=original_part(side,kind)
    x=45*side
    far,near=master.split(Plane(origin=(x,0,0),z_dir=(side,0,0)),Keep.BOTH)
    faces=[f for f in near.faces() if f.bounding_box().size.X<.001 and abs(f.center().X-x)<.01]
    middle=[extrude(f,amount=34,dir=(side,0,0)) for f in faces]
    skin=single(near.fuse(Pos(34*side,0,0)*far,*middle))
    stations=module_positions(side)
    # Cut individual knuckle openings in the extended uninterrupted skin.
    # A rounded profile leaves webs between adjacent fingers.
    openings=[]
    for tf in stations.values():
        cutter=Pos(0,40,-2)*(Plane.XZ*RectangleRounded(28.8,36,6))
        openings.append(loc(tf)*extrude(cutter,amount=46,dir=(0,-1,0)))
    skin=skin.cut(*openings)
    if not dorsal:
        skin=single(skin.fuse(loc(stations['Pulgar'])*thumb_rim()))
        little=loc(stations['Menique'])
        # Give the outside socket a continuous service pocket through its arc.
        # The extra material goes on the outside of the palmar skin.
        profiles=[Pos(12*side,-32,z)*RectangleRounded(w,h,r)
                  for z,w,h,r in [(-20,10,19,4),(-19,15,26,6),(-16.7,19,31,7),(-9,22,35,7)]]
        blister=loft(profiles)
        mouth=Pos(12*side,-29,-16.5)*RectangleRounded(20,40,4)
        pocket=extrude(mouth,amount=15)
        skin=skin.fuse(little*blister).cut(little*pocket)
        for px in (14,36):
            axis=Axis((side*px,PALMAR_FASTENER_Y,-65),(0,0,1))
            low=min(p.Z for p,_ in skin.find_intersection_points(axis))+1
            top=-32.25
            boss=Pos(side*px,PALMAR_FASTENER_Y,(low+top)/2)*Cylinder(3.5,top-low)
            hole=Pos(side*px,PALMAR_FASTENER_Y,-44)*Cylinder(1.6,28)
            skin=skin.fuse(boss).cut(hole)
    # Rear trim follows the saddle. This keeps the shell removable.
    trimmed=skin.cut(yoke(side),palm(side))
    skin=max(trimmed.solids(),key=lambda solid:solid.volume)
    if dorsal:
        for px,py in ((76,-33),(76,-67)):
            axis=Axis((side*px,py,-65),(0,0,1))
            hits=skin.find_intersection_points(axis)
            if hits:
                z=max(p.Z for p,_ in hits)-1.1
                h=z-18.65
                if h>0:
                    boss=Pos(side*px,py,18.65+h/2)*Cylinder(3.4,h)
                    skin=skin.fuse(boss)
        skin=skin.cut(cover(side,False),*spine_holes(side,1.45))
    label='Tapa dorsal de cinco dedos' if dorsal else 'Cubierta palmar con salida de pulgar'
    return painted(single(skin),label,WHITE if dorsal else '#B4C5BD')


def tenon(side,extra=0):
    u,v,w=cuff_center(side)
    return block(u,v-30,w,15+2*extra,32+2*extra,10+2*extra)


def crossbores(side,radius,length):
    u,v,w=cuff_center(side)
    return [Pos(u,v-d,w)*Cylinder(radius,length) for d in (31,41)]


@lru_cache()
def wrist_parts(side):
    u,v,w=cuff_center(side)
    drum=(Pos(u,v,w)*Cylinder(24,43)).fuse(tenon(side)).cut(
        Pos(u,v,w)*Cylinder(4.18,47),*crossbores(side,1.65,14))
    pin=Pos(u,v,w)*Cylinder(4,54)
    return [painted(drum,'Nucleo de muneca con espiga',DARK),painted(pin,'Eje transversal de 8 mm',BRONZE)]


def hand(side,angles=None):
    sections=[palm(side),yoke(side),cover(side,False),cover(side,True),thumb_screws(side)]
    for name,tf in module_positions(side).items():
        q=OPEN if angles is None else angles.get(name,OPEN)
        mod=loc(tf)*digit(q)
        mod.label=name+' - dos SCS0009'
        sections.append(mod)
    return Compound(label='Mano derecha' if side==1 else 'Mano izquierda',children=sections)


def bench_hand(side):
    return Compound(label='Mano y muneca para prueba de banco',
                    children=[*wrist_parts(side),hand(side)])


def equipped_forearm(group,side):
    wrist=loc(wrist_pose(side))
    key=wrist*tenon(side,.25)
    holes=[wrist*s for s in crossbores(side,1.75,145)]
    panels=[]
    for p in group.children:
        shape=(group.location*p).cut(key,*holes)
        panels.extend(shape.solids())
    return arm_module(panels,side)


def arm_with_hand(arm,side):
    from arnes_brazos import upper_with_pilots
    from equipo_brazos import elbow_mount, upper_panels
    wrist=loc(wrist_pose(side))
    new=[]
    for group in arm.children:
        label=group.label.lower()
        if label.startswith('mano') or label.startswith('mu') and not label.startswith('mus'):
            continue
        if label.startswith('antebrazo'):
            group=equipped_forearm(group,side)
        elif label=='brazo':
            group=upper_panels(side,upper_with_pilots(group,side))
        elif label=='codo':
            group=elbow_mount(group,side)
        new.append(group)
    new.extend(wrist*p for p in wrist_parts(side))
    # Forward tilt about the actual transverse pin keeps the opposed thumb clear
    # of the thigh when the arms hang at the sides.
    seat=Pos(236*side,0,845)*Rot(WRIST_REST_DEG,0,0)*Pos(-236*side,0,-845)
    assembled=hand(side,{'Pulgar':THUMB_REST})
    assembled.children=tuple(hand_port(p,side) if 'dorsal' in p.label.lower() else p
                             for p in assembled.children)
    new.append(seat*wrist*assembled)
    new.append(hand_loom(side,seat*wrist))
    return Compound(label=arm.label,children=new)

