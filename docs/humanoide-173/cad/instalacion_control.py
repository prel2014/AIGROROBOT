"""Interfaces mecánicas de la electrónica. mm; +Y anterior; Z desde la planta."""
from functools import lru_cache
from pathlib import Path
import sys

from build123d import (Box, Cylinder, Compound, Plane, Pos, Rot, Vector, Color,
                       RectangleRounded, Circle, Edge, Wire, sweep, extrude, import_step)

CAD = Path(__file__).resolve().parent
sys.path.insert(0, str(CAD.parent / 'electronica' / 'fuentes'))
from circuitos import motor, servo, holes, pads, package
sys.path.pop(0)
REF = CAD.parent / 'referencias'
GREEN, DARK, METAL, TEAL, SHELL = '#356851', '#344449', '#A9B7B4', '#267D79', '#AABAB9'
WRIST_X, PCB_Z, REG_Z = 236., 940., 995.
BOARD_BACK, REG_BACK, LID_SEAT = -3.5, 23., -19.
PCB_FLOOR, REG_FLOOR = 3., 29.
COVER_FIX = [(x,z) for x in (-16,16) for z in (901,1022)]
# Four mounting holes, distinct from the four terminal-block holes of identical diameter.
REG_FIX = [(x,y) for x in (2.159,41.021) for y in (4.191,29.591)]
REG_W, REG_H = 43.18, 31.75

def name(part, text, color):
    part.label = text
    part.color = Color(color)
    return part

def solid(part):
    shapes = list(part.solids())
    if len(shapes) != 1 or not shapes[0].is_valid:
        raise ValueError(f'Sólidos inesperados: {len(shapes)}')
    return shapes[0]

def box(x,y,z,w,d,h):
    return Pos(x,y,z)*Box(w,d,h)

def bore_y(x,z,y0,y1,r):
    return Pos(x,(y0+y1)/2,z)*Rot(90,0,0)*Cylinder(r,abs(y1-y0))

def duct(a,b,r):
    a,b = Vector(a), Vector(b)
    return Plane(origin=(a+b)*.5,z_dir=b-a).location * Cylinder(r,(b-a).length)

def rounded_xz(x,z,w,h,r,y0,y1):
    profile=Pos(x,y0,z)*(Plane.XZ * RectangleRounded(w,h,r))
    return extrude(profile, amount=y1-y0, dir=(0,1,0))

@lru_cache()
def connector(count):
    return import_step(REF/'JST-XH'/f'XH{count}.step')

@lru_cache()
def pcb(which):
    b = motor() if which=='A' else servo()
    w,h=b.size
    base=box(0,0,.8,w,h,1.6)
    drills=[Pos(u-w/2,h/2-v,.8)*Cylinder(1.35,3) for u,v in holes(b)]
    drills += [Pos(u-w/2,h/2-v,.8)*Cylinder(d/2,3)
               for _,_,_,u,v,_,_,d in pads(b) if d]
    items=[name(solid(base.cut(*drills)),f'PCB {which} - FR4 1.6 mm',GREEN)]
    for p in b.parts:
        u,v=p.at
        if p.package.startswith('XH'):
            item=Pos(u-w/2,h/2-v,1.6)*connector(int(p.package[2:]))
            name(item,p.ref+' - '+p.value,'#E1DDC9')
        else:
            _,(x1,y1,x2,y2)=package(p.package)
            height=1.6 if p.package.startswith(('LQFP','SOIC')) else 1
            item=box(u-w/2+(x1+x2)/2,h/2-v-(y1+y2)/2,1.6+height/2,x2-x1,y2-y1,height)
            name(item,p.ref+' - '+p.value,DARK if p.ref.startswith(('U','D')) else '#BEA47D')
        items.append(item)
    return Compound(label='Placa '+which+' equipada',children=items)

def tray_frame(side,z):
    # PCB length along Z; keyed JST headers face outwards on the extracted tray.
    return Plane(origin=(66*side,-8,z),x_dir=(0,0,1),z_dir=(side,0,0)).location

def tray_with_boards(tray):
    metalwork=[]
    boards=[]
    for key,side,z,b in [('A',1,1272,motor()),('B',-1,1295,servo())]:
        frame=tray_frame(side,z)
        w,h=b.size
        places=[(u-w/2,h/2-v) for u,v in holes(b)]
        rails=[]
        for across in sorted({y for _,y in places}):
            cy=-8-side*across
            top=z+w/2
            rails.append(box(side*60.5,cy,(top+1217)/2,4,6,top-1217))
        rails.append(box(side*60.5,-8,z+w/2-3,4,h,5))
        for u,v in places:
            rails.append(frame*Pos(u,v,-2.25)*Cylinder(2.85,4.5))
        bracket=rails[0].fuse(*rails[1:])
        pilots=[frame*Pos(u,v,-3.8)*Cylinder(1.025,8.0) for u,v in places]
        metalwork.append(bracket.cut(*pilots))
        p=frame*pcb(key)
        p.label='Interfaz CAN del Xavier - A' if key=='A' else 'Nodo cervical STM32 - B'
        boards.append(p)
    tray=name(solid(tray.fuse(*metalwork)),'Bandeja Xavier con dos bastidores verticales',TEAL)
    return tray,boards

def fore_frame(side):
    return Plane(origin=(side*WRIST_X,BOARD_BACK,PCB_Z),x_dir=(0,0,-1),z_dir=(0,-1,0)).location

def regulator_frame(side):
    return Plane(origin=(side*WRIST_X,REG_BACK,REG_Z),x_dir=(0,0,-1),z_dir=(0,-1,0)).location * Pos(-REG_W/2,-REG_H/2,0)

@lru_cache()
def commercial_regulator():
    return import_step(REF/'Pololu'/'d24v150fx-step-down-voltage-regulator.step')

def cable_axes(side):
    x=side*WRIST_X
    return [((x+side*20,-12,927),(x+side*36,-9,865),5.2),
            ((x+side*12,-8,1006),(x+side*49,-26,1048),4.5)]

def cover_plate(side):
    x=side*WRIST_X
    # A domed service cover follows the forearm outline. Its skirt and mounting
    # seats stay inside the existing register so both boards withdraw straight back.
    def dome(w,h,r,arc):
        outline=rounded_xz(x,961,w,h,r,-52,LID_SEAT+.2)
        return solid(outline.intersect(Pos(x,4,961)*Cylinder(arc,144)))
    outside=dome(52.3,129.3,5.65,45)
    inside=dome(47.1,124.1,3.1,42.3)
    lid=outside-inside
    tools=[]
    for dx,z in COVER_FIX:
        pillar=solid(bore_y(x+dx,z,-46,LID_SEAT,4.2).intersect(outside))
        lid=lid.fuse(pillar)
        tools.extend([bore_y(x+dx,z,-47,-18,1.65),bore_y(x+dx,z,-47,-21.1,3.15)])
    for z in (984,990,996,1002,1008):
        tools.append(rounded_xz(x,z,23,2.8,1.3,-47,-18))
    lid=lid.cut(*tools,box(x,LID_SEAT+1,961,65,2,145))
    return name(solid(lid),'Tapa posterior abombada y ventilada - '+('D' if side>0 else 'I'), '#E3E5DC')

def arm_module(panels,side):
    x=side*WRIST_X
    blank=solid(panels[0].fuse(*panels[1:],tol=1e-5))
    # Two stepped floors supporting the boards, visible from the same rear register.
    cuts=[rounded_xz(x,961,53,130,6,-85,LID_SEAT),
          rounded_xz(x,PCB_Z,42,70,3,-80,PCB_FLOOR),
          rounded_xz(x,REG_Z,40,47.5,3,-80,REG_FLOOR)]
    cuts.extend(duct(a,b,r) for a,b,r in cable_axes(side))
    # Lateral wiring gutter bridges the two chambers even when both PCBs are fitted.
    cuts.append(rounded_xz(x+side*19,968,10,92,3,-25,2.8))
    for dx,z in COVER_FIX:
        cuts.append(bore_y(x+dx,z,-21,-9.5,1.25))
    body=blank.cut(*cuts)
    bosses=[]
    pilots=[]
    for frame,points,height,r,pilot in [
        (fore_frame(side),[(u-32.5,17-v) for u,v in holes(servo())],PCB_FLOOR-BOARD_BACK,2.65,1.025),
        (regulator_frame(side),REG_FIX,REG_FLOOR-REG_BACK,2.0,.8)]:
        for u,v in points:
            bosses.append(frame*Pos(u,v,-(height+.4)/2)*Cylinder(r,height+.4))
            pilots.append(frame*Pos(u,v,-4)*Cylinder(pilot,8.3))
    body=solid(body.fuse(*bosses).cut(*pilots))
    body=name(body,'Carcasa portante con dos alojamientos - '+('D' if side>0 else 'I'),SHELL)
    control=fore_frame(side)*pcb('B')
    control.label='Controlador de mano STM32 - B'
    reg=name(regulator_frame(side)*commercial_regulator(),'Pololu D24V150F6 - reg18b',GREEN)
    return Compound(label='Antebrazo electronico '+('derecho' if side>0 else 'izquierdo'),
                    children=[body,control,reg,cover_plate(side)])

def hand_port(cover,side):
    # Palm-local +W is the back of the hand; this patch opens into the rear wiring chamber.
    x,y=34*side,-78
    ring=Pos(x,y,30.5)*Cylinder(8,6)
    hole=Pos(x,y,27)*Cylinder(5.2,24)
    mouth=Pos(x,y,33.25)*Cylinder(6.2,1.5)
    return name(solid(cover.fuse(ring).cut(hole,mouth)),'Tapa dorsal con entrada de mazo','#E5E4DA')

def hand_loom_path(side,hand_pose):
    # One neutral-axis bundle, not a model of four individual conductors.
    a,b,_=cable_axes(side)[0]
    a,b=Vector(a),Vector(b)
    outward=Vector(side,0,0)
    local=Plane(hand_pose)
    crown=local.from_local_coords(Vector(34*side,-78,40))
    inside=local.from_local_coords(Vector(34*side,-78,22))
    curve=Edge.make_bezier(b,b+(b-a).normalized()*27,crown+outward*35,crown)
    return Wire([Edge.make_line(a,b),curve,Edge.make_line(crown,inside)])

def hand_loom(side,hand_pose):
    path=hand_loom_path(side,hand_pose)
    a,b,_=cable_axes(side)[0]
    a,b=Vector(a),Vector(b)
    bundle=sweep(Plane(origin=a,z_dir=b-a)*Circle(3.25),path=path)
    return name(solid(bundle),'Mazo flexible 6 V y TTL - referencia en reposo','#C5823D')
