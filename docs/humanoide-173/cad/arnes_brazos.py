"""Arnés de los antebrazos. Coordenadas de montaje en mm, +Y hacia el pecho.
CAN1 identifica la funda con H/L trenzados, referencia y 5 V auxiliar.
Las dos fundas se separan dentro del registro antes de J1 y VIN/GND.
"""
from functools import lru_cache
from math import acos, tan
from build123d import (Box, Circle, Compound, Edge, Plane, Pos, Vector, Wire,
                       sweep, fillet, Axis)
from instalacion_control import (name, solid, box, duct, bore_y, tray_frame, fore_frame,
                                regulator_frame, cable_axes)

CAN, POWER, METAL = '#19B4CA', '#F48137', '#637B82'
SIGNAL_R, POWER_R = 1.55, 1.95
TORSO_Y = -45.
LEVEL = {False:1308., True:1326.}
CLAMP_Y = -58.
CLAMP_LEVELS = (1280.,1185.)

def trunk_bores():
    return [duct((s*84,TORSO_Y,z),(s*169,TORSO_Y,z),4.)
            for s in (-1,1) for z in LEVEL.values()]

def rounded_path(points, target=16.):
    """Round spatial polygon corners with tangent circular arcs, never overshoot an anchor."""
    pts=[Vector(p) for p in points]
    edges=[]; start=pts[0]
    for a,b,c in zip(pts,pts[1:],pts[2:]):
        incoming,outgoing=(b-a).normalized(),(c-b).normalized()
        angle=acos(max(-1.,min(1.,incoming.dot(outgoing))))
        if angle<1e-5: continue
        trim=min(target*tan(angle/2),.46*(b-a).length,.46*(c-b).length)
        near,far=b-incoming*trim,b+outgoing*trim
        edges.append(Edge.make_line(start,near))
        edges.append(Edge.make_tangent_arc(near,incoming,far))
        start=far
    edges.append(Edge.make_line(start,pts[-1]))
    return Wire(edges)

def lead(label,points,r,color,radius=16.):
    path=rounded_path(points,radius)
    tube=sweep(Plane(origin=path.position_at(0),z_dir=path.tangent_at(0))*Circle(r),path=path)
    return name(solid(tube),label,color)

def pside(side,x,y,z):
    return (side*x,y,z)

def upper_x(side,power):
    return side*(242 if power else 230)

def board_plug(frame,u,v,n,ref):
    # Local centroids referenced to pin 1 of the native JST header.
    center=u+(n-1)*1.25
    p=frame*box(center,v-.45,1.6+8.4,(n-1)*2.5+3.6,4.1,2.8)
    tip=frame*Pos(center,v-.45,11.4)
    return name(p,ref+' - XHP-'+str(n)+' cara de salida','#E5E9D8'),tuple(tip.position)

def service_half(side,power,fixed):
    # Exposed, mated half-envelopes, not contact or mould tooling geometry.
    width,height,length=(10.5,6.,10.) if power else (13.,7.5,11.)
    cx=177+(-1 if fixed else 1)*length/2
    base=Box(length,width,height)
    shell=Pos(side*cx,TORSO_Y,LEVEL[power])*fillet(base.edges().filter_by(Axis.X),1.)
    label=('XT30U-F fuente' if fixed else 'XT30U-M brazo') if power else ('SMR-04V-B torso' if fixed else 'SMP-04V-BC brazo')
    return name(shell,label+' - envolvente acoplada',('#B88826' if fixed else '#F0C653') if power else ('#28464E' if fixed else '#476875'))

def guide_drills(side,z):
    return [bore_y(side*236+off,z,-65,-33,1.25) for off in (-12,12)]

def saddle(side,z,upper):
    # The actual posterior skin defines the seating face of each screw-on guide.
    x=side*236
    block=box(x,-48,z,32,32,8).cut(upper)
    holes=[bore_y(x+off,z,-68,-30,1.65) for off in (-12,12)]
    for power,r in [(False,SIGNAL_R),(True,POWER_R)]:
        cx=upper_x(side,power)
        holes.append(duct((cx,CLAMP_Y,z-8),(cx,CLAMP_Y,z+8),r+.45))
        # Rear slot lets a harness be installed before retaining it with a tie.
        holes.append(box(cx,-63,z,2*r+.4,10,12))
    return name(solid(block.cut(*holes)),'Peine dorsal M3 '+str(int(z)),METAL)

def upper_with_pilots(upper,side):
    holes=[h for z in CLAMP_LEVELS for h in guide_drills(side,z)]
    children=[]
    for child in upper.children:
        world=upper.location*child
        cut=world.cut(*holes)
        if cut is not None:
            for part in cut.solids():
                children.append(name(part,child.label,child.color))
    return Compound(label=upper.label,children=children)

def external_points(side,power):
    z=LEVEL[power]; lane=242 if power else 230
    pts=[(187 if power else 188,TORSO_Y,z),(204 if power else 201,TORSO_Y,z),
         (211 if power else 201,-69,z+16),(209 if power else 195,-105 if power else -91,1409),
         (252 if power else 238,-105 if power else -91,1409),
         (lane,-69,1350),(lane,CLAMP_Y,1310),(lane,CLAMP_Y,1162),
         (257 if power else 243,-69,1134),(287 if power else 274,-69,1102),
         (322,-65,1085) if power else (307,-62,1091),
         (323,-33,1058) if power else (311,-51.1,1080)]
    start,end,_=cable_axes(side)[1]
    dy=2.10 if power else -2.10
    a=Vector(start)+Vector(0,dy,0); b=Vector(end)+Vector(0,dy,0)
    return [pside(side,*p) for p in pts]+[tuple(b+(b-a).normalized()*(10 if power else 29)),tuple(b),tuple(a)]

def line_to_boards(side,power,entry,can_tip):
    x=side*236
    if power:
        # 2.54 mm from the PCB edge; first two 5 mm-pitch pads in the Pololu top view.
        frame=regulator_frame(side)
        tips=[(frame*Pos(u,2.54,1.57)).position for u in (7.62,12.62)]
        mid=(tips[0]+tips[1])*.5
        branch=(mid.X,5.,mid.Z)
        return [lead('12 V - llegada al regulador',[entry,(x+side*5,-4,1003),branch],POWER_R,POWER,5),
                *[lead(label,[branch,(t.X,12,t.Z),tuple(t)],.86,paint,3)
                  for t,label,paint in zip(tips,['VIN - positivo','GND - retorno'],[POWER,'#5E5151'])]]
    return [lead('CAN1 y 5 V - llegada a B.J1',
                 [entry,(x+side*16,-10.1,994),(x+side*18,-12,970),
                  (x+side*16,-16.5,949),(can_tip[0],-16.9,931),
                  can_tip],SIGNAL_R,CAN,5)]

def branch(side):
    short='D' if side==1 else 'I'
    plug,tip=board_plug(fore_frame(side),46-32.5,17-5,4,'B.J1 '+short)
    parts=[]
    for power,r,col in [(False,SIGNAL_R,CAN),(True,POWER_R,POWER)]:
        label='12 V y retorno' if power else 'CAN1 REF y 5 V'
        points=external_points(side,power)
        parts.extend([service_half(side,power,True),service_half(side,power,False),
                      lead(label+' - hombro y codo '+short,points,r,col,22),
                      *line_to_boards(side,power,points[-1],tip)])
    return Compound(label='Ramal '+short+' - hombro codo antebrazo',children=[*parts,plug])

def distribution():
    # The block designates the protected 12 V and auxiliary 5 V handoff.
    # It hangs from the fixed shelf; the tray is unplugged at A.J3 and B.J1.
    body=box(0,-39,1334,24,10,9)
    lug=box(0,-33,1337.5,18,11,3.5)
    parts=[name(solid(body.fuse(lug)),'Distribucion 12 V y 5 V - interfaz de mazo',METAL)]
    for side in (-1,1):
        def p(x,z):return pside(side,x,TORSO_Y,z)
        # Both fixed transverse runs pass ABOVE the pulled-out computer and PCB frames.
        parts.append(lead('12 V protegido desde distribucion '+str(side),
                    [pside(side,12,-39,1334),pside(side,79,-39,1334),
                     p(85,1326),p(167,1326)],POWER_R,POWER,4))
        parts.append(lead('Troncal CAN1 a hombro '+str(side),
                    [pside(side,0,-50,1334),pside(side,80,-50,1334),
                     pside(side,83,-49,1316),p(85,1308),p(166,1308)],SIGNAL_R,CAN,4))
        is_a=side==1
        plug,tip=board_plug(tray_frame(side,1272 if is_a else 1295),
                    44-28 if is_a else 46-32.5,18-26 if is_a else 17-5,3 if is_a else 4,
                    'A.J3 CAN1' if is_a else 'B.J1 cervical')
        stub=[tip,(side*83,tip[1],tip[2]),(side*83,-13,tip[2]+4),
              (side*84,-32,1340),(side*82,-44,1340),pside(side,72,-50,1334)]
        parts.extend([plug,lead('Derivacion corta '+('A.J3' if is_a else 'B.J1 cuello'),stub,1.35,CAN,5)])
    parts.append(lead('Inyeccion 5 V auxiliar al mazo logico',[(0,-44,1334),(0,-50,1334)],1.08,'#9679B0'))
    return Compound(label='Distribucion dorsal y derivaciones CAN1',children=parts)

def harness(robot):
    arms=[]
    for side,word in [(-1,'izquierdo'),(1,'derecho')]:
        arm=next(a for a in robot.children if 'brazo' in a.label.lower() and word in a.label.lower())
        upper=next(a for a in arm.children if a.label.lower()=='brazo')
        arm_harness=branch(side)
        arm_harness.children=(*arm_harness.children,*(saddle(side,z,upper) for z in CLAMP_LEVELS))
        arms.append(arm_harness)
    return Compound(label='Cableado bandeja a antebrazos - CAN1 y 12 V',children=[distribution(),*arms])

