"""Verificación de las interfaces instaladas en los STEP. Milímetros."""
from pathlib import Path
import argparse
from build123d import Box, Cylinder, Location, Pos, Rot, import_step

from instalacion_control import (CAD, WRIST_X, PCB_Z, REG_Z, BOARD_BACK, REG_BACK,
                                REG_FIX, COVER_FIX, LID_SEAT, REG_W, REG_H,
                                fore_frame, regulator_frame, tray_frame, duct, cable_axes,
                                box, bore_y, hand_loom)
from circuitos import servo, motor, holes

EPS_VOLUME=.03

def check(ok,msg):
    if not ok:
        raise ValueError(msg)

def overlap(a,b):
    aa,bb=a.bounding_box(optimal=False),b.bounding_box(optimal=False)
    if any(amax < bmin or bmax < amin for amin,amax,bmin,bmax in zip(aa.min,aa.max,bb.min,bb.max)):
        return 0.
    common=a.intersect(b)
    return sum(s.volume for s in common.solids()) if common else 0.

def free(a,b,msg):
    v=overlap(a,b)
    check(v<EPS_VOLUME,f'{msg}: solape {v:.4f} mm3')

def bare(node):
    return type(node)(node.wrapped)

def mounted(frame,pts,support,inner,outer):
    for u,v in pts:
        ring=frame*Pos(u,v,-.12)*(Cylinder(outer,.22)-Cylinder(inner,.5))
        check(overlap(ring,support)/ring.volume>.98,f'Apoyo incompleto en {u}, {v}')
        bit=frame*Pos(u,v,-2.5)*Cylinder(inner*.55,4.8)
        free(bit,support,'Piloto de montaje obstruido')

def check_arm(module,side,exploded=False):
    shell,board,reg,lid=module.children
    shell,board,reg,lid=map(bare,(shell,board,reg,lid))
    if exploded:lid=Pos(0,68,0)*lid
    for part in (shell,lid):
        check(len(part.solids())==1 and part.is_valid,'Carcasa o tapa discontinua')
    for a,b,label in [(shell,board,'PCB/carcasa'),(shell,reg,'Pololu/carcasa'),(shell,lid,'Tapa/carcasa'),
                      (board,lid,'PCB/tapa'),(reg,lid,'Pololu/tapa'),(board,reg,'Placa/regulador')]:
        free(a,b,label)
    x=side*WRIST_X
    mounted(fore_frame(side),[(u-32.5,17-v) for u,v in holes(servo())],shell,1.4,2.48)
    mounted(regulator_frame(side),REG_FIX,shell,1.12,1.81)
    for dx,z in COVER_FIX:
        ring=bore_y(x+dx,z,LID_SEAT+.02,LID_SEAT+.22,2.7)-bore_y(x+dx,z,-21,-18,1.7)
        check(overlap(ring,shell)/ring.volume>.98,'Falta apoyo del tornillo de tapa')
        free(bore_y(x+dx,z,-25,-10,1.18),shell,'Tornillo ciego')
    for a,b,radius in cable_axes(side):
        free(duct(a,b,radius-.12),shell,'Conducto obstruido')
    # Continuously swept substrate/upper-component rectangles, normal to the boards.
    # Solder tails/back-side features initially project toward their mounting floor.
    for z,width,length,y,other,part in [(PCB_Z,34,65,BOARD_BACK,reg,board),
                                         (REG_Z,REG_H,REG_W,REG_BACK,board,reg)]:
        volume=box(x,(y-90)/2,z,width,y+90,length)
        free(volume,shell,'Paso de extracción del módulo')
        free(volume,other,'Cruce con segundo módulo')
        for shift in (0,1,2,3.5):
            free(Pos(0,-shift,0)*part,shell,'Salida de pines y cara posterior')
    print('Antebrazo',side,': cuatro apoyos por placa, cuatro apoyos de tapa, dos conductos, extracción posterior.',flush=True)

def check_tray(computing):
    tray=bare(computing.children[0])
    check(tray.is_valid and len(tray.solids())==1,'Bandeja o montantes discontinuos')
    for i,(side,z,b) in enumerate([(1,1272,motor()),(-1,1295,servo())]):
        board=bare(computing.children[4+i])
        frame=tray_frame(side,z)
        w,h=b.size
        mounted(frame,[(u-w/2,h/2-v) for u,v in holes(b)],tray,1.4,2.55)
        for p in computing.children:
            if p is computing.children[4+i]:continue
            free(board,bare(p),'Placa de bandeja contra '+p.label)
    print('Bandeja: ocho apoyos coaxiales, dos PCB y montantes unidos.',flush=True)

def swept_tray(computing,robot):
    fixed=[*robot.children[0].children,*computing.children[-2:]]
    # Bounds of the added rails/boards, extending 150 mm toward the back.
    for side,z,b in [(1,1272,motor()),(-1,1295,servo())]:
        w,h=b.size
        sweep=box(side*68,-83,(1217+z+w/2)/2,21,h+152,z+w/2-1217)
        for p in fixed:free(sweep,bare(p),'Extracción de bandeja contra '+p.label)
    print('Recorrido dorsal de las nuevas piezas: 150 mm libres.',flush=True)

def full():
    robot=import_step(CAD/'humanoide_xavier.step')
    comp=robot.children[1]
    check_tray(comp)
    swept_tray(comp,robot)
    for side in (-1,1):
        suffix='izquierdo' if side<0 else 'derecho'
        arm=next(p for p in robot.children if 'brazo_'+suffix in p.label.lower())
        module=arm.children[0]
        check_arm(module,side)
        loom=bare(arm.children[-1])
        check('mazo' in arm.children[-1].label.lower(),'Falta el mazo')
        for part in arm.children[:-1]:
            free(loom,bare(part),'Mazo contra '+part.label)
        print('Mazo continuo sin choque en reposo:',suffix,flush=True)
        hand=arm.children[-2]
        for delta in (-10,10):
            pose=Pos(side*WRIST_X,0,845)*Rot(delta,0,0)*Pos(-side*WRIST_X,0,-845)*hand.location
            loop=hand_loom(side,pose)
            for part in arm.children[:-2]:
                free(loop,bare(part),'Mazo en flexión contra '+part.label)
            moved=type(hand)(hand.wrapped.Located(pose.wrapped))
            free(loop,moved,'Mazo en flexión contra la mano')
        print('Mazo flexible reconstruido a 10, 20 y 30 grados:',suffix,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--subconjuntos',action='store_true')
    if parser.parse_args().subconjuntos:
        check_arm(import_step(CAD/'antebrazo_servicio.step'),1,True)
        check_tray(import_step(CAD/'torso_xavier_abierto.step').children[1])
    else:
        full()
