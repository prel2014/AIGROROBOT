"""Montaje del BMS del cartucho, en mm. +Y hacia el pecho, acceso desde -Y.

La placa viene del STEP generado por KiCad en el origen del contorno. Las carcasas
Mini-Fit se representan por sus volúmenes de acoplamiento, según los planos
Molex 55570302-SD y 55590300-SD. Las posiciones de los terminales son nominales.
"""
import re
from pathlib import Path
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from build123d import Compound, Cylinder, Pos, import_step
from instalacion_control import box, name, solid
from arnes_brazos import lead

FLOOR = 1148.5
PCB_X, PCB_Y, PCB_W, PCB_D = -12., -2., 72., 34.
PCB_BOTTOM = FLOOR + 4
FIX = [(PCB_X+x, PCB_Y+y) for x in (-31,31) for y in (-12,12)]
HATCH_FIX = [(x,y) for x in (-51,51) for y in (-33,21)]
PORT_X, PORT_Z = 36.,1157.
DARK, GREEN, IVORY = '#384447','#276858','#E3E3C8'
RED, GREY, GOLD = '#BE523F','#505D65','#D9A651'


def post(x,y,bottom,top,r):
    return Pos(x,y,(bottom+top)/2)*Cylinder(r,top-bottom)


def hatch_cuts():
    return [box(0,-6,1169.6,112,67,2.8),box(0,-6,1168,94,52,10),
            *[post(x,y,1164,1172,.85) for x,y in HATCH_FIX]]


def hatch():
    cover=box(0,-6,1169.1,111.4,66.4,1.8)
    return name(solid(cover.cut(*[post(x,y,1167,1171,1.15) for x,y in HATCH_FIX])),
                'Trampilla superior desmontable - cuatro M2', '#718C8C')


def service_deck():
    shelf=box(0,-7,FLOOR-1,120,76,2)
    standoffs=[post(x,y,FLOOR-.1,PCB_BOTTOM,3.1) for x,y in FIX]
    # Cuna abierta con dos pasos para la brida retenedora del cuerpo 5557.
    saddle=box(PORT_X,-38.7,(FLOOR-.1+PORT_Z-3.7)/2,19,10,PORT_Z-3.7-FLOOR+.1)
    cheeks=[box(PORT_X+s*8.2,-38.7,PORT_Z-2.5,2,10,3.5) for s in (-1,1)]
    rails=[box(44+s*8.5,8,FLOOR+1.1,1.5,11,2.4) for s in (-1,1)]
    shelf=shelf.fuse(*standoffs,saddle,*cheeks,*rails)
    tools=[post(x,y,FLOOR-2.5,PCB_BOTTOM+.5,1.04) for x,y in FIX]
    tools.extend([box(-29,-36.4,FLOOR-1,25,7.5,5),
                  box(-50,-36.4,FLOOR-1,8.5,7.5,5),
                  *[box(PORT_X+s*10.7,-38.7,FLOOR-1,2.2,4.5,6) for s in (-1,1)],
                  *[box(44,8+s*6,FLOOR-1,4.5,2.2,6) for s in (-1,1)]])
    return solid(shelf.cut(*tools))


def protector():
    source=Path(__file__).resolve().parents[1]/'electronica/bateria_4s9p/protector_cartucho/bms_4s_equipado.step'
    pcb=import_step(source)
    # El lector de build123d recupera el nombre de encapsulado. KiCad guarda la
    # referencia de cada instancia en NEXT_ASSEMBLY_USAGE_OCCURRENCE del producto raíz.
    occurrences=re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('\d+','([^']*)','',\s*(#\d+),",source.read_text())
    root=next(parent for ref,parent in occurrences if ref=='J1')
    labels=[ref for ref,parent in occurrences if parent==root]
    assert len(labels)==len(pcb.children)==42
    pieces=[]
    origin=Pos(PCB_X-PCB_W/2, PCB_Y+PCB_D/2, PCB_BOTTOM)
    for ref,body in zip(labels,pcb.children):
        # Conservar designadores distintos incluso cuando KiCad instancia el
        # mismo modelo (J2/J3, resistencias, condensadores).
        item=origin*Compound.cast(BRepBuilderAPI_Copy(body.wrapped).Shape())
        if re.fullmatch(r'[A-Z]+\d+',ref):
            color={'J':IVORY,'C':'#B8976D','U':DARK,'Q':DARK,'D':'#555354','R':'#687374'}
            name(item,ref+' - '+body.label,color[ref[0]])
        else:
            assert abs(body.bounding_box().size.X-PCB_W)<.01
            name(item,'PCB BMS 4S - FR4 72 x 34',GREEN)
        pieces.append(item)
    return Compound(label='Protector y balanceador embarcado 4S',children=pieces)

def bms_plugs():
    # Sólo el tramo visible de las carcasas acopladas XHP. No duplicar en CAD
    # el material interno de los cabezales incluidos en el STEP de KiCad.
    z=PCB_BOTTOM+4.65
    return [name(box(-30,-20.1,z,13.6,4,4.1),'J1 XHP-5 - salida de los cinco cables',IVORY),
            name(box(10.75,-20.1,z,6.1,4,4.1),'J2 XHP-2 - salida de potencia',IVORY),
            name(box(-49.1,2.75,z,4,6.1,4.1),'J3 XHP-2 - salida de sonda',IVORY)]


def fuse_parts():
    # Portafusible aislado, envolvente propuesta; las dos hojas entran en alojamientos.
    x,y=44,8
    holder=box(x,y,FLOOR+3,15,8,6)
    blades=[box(x+s*3.8,y,FLOOR+4.7,2.8,.8,4.6) for s in (-1,1)]
    holder=holder.cut(*[box(x+s*3.8,y,FLOOR+4.5,3.2,1.2,6) for s in (-1,1)])
    fuse=box(x,y,FLOOR+10.5,10.9,3.8,7)
    return [name(solid(holder),'Alojamiento aislado F1 - interfaz propuesta',DARK),
            name(solid(fuse.fuse(*blades)),'F1 MINI 3 A - envolvente','#96668F')]


def pack_port():
    x,z=PORT_X,PORT_Z
    rear=box(x,-39.05,z,13.8,12.1,7.4)
    nose=box(x,-49.35,z,12.6,8.5,5.4)
    cavities=[box(x+offset,-50.1,z,2.7,9,2.7) for offset in (-4.2,0,4.2)]
    return name(solid(rear.fuse(nose).cut(*cavities)),'J1 cartucho - Molex 39-01-4031',IVORY)


def robot_plug():
    x,z=PORT_X,PORT_Z
    shell=box(x,-56.8,z,15.2,23.4,8.4)
    mouth=box(x,-49.2,z,13.3,9.8,6)
    ears=box(x,-57,z,21,2,9)
    return name(solid(shell.fuse(ears).cut(mouth)),'Molex 39-01-4032 - mazo fijo del robot','#9CA9AC')


def robot_lead_points():
    return [(PORT_X,-68.5,PORT_Z),(PORT_X,-89,PORT_Z),(PORT_X,-97,1171),
            (PORT_X,-97,1190),(PORT_X,-68,1190),(82,-49,1190),
            (82,-24,1190),(82,-24,1225)]


def feedthrough():
    return lead('Conducto del ramal auxiliar',robot_lead_points(),4.6,GREY,8)


def robot_loom():
    wire=lead('Ramal auxiliar dos polos hasta la bahia del torso',robot_lead_points(),2.5,GREY,8)
    return Compound(label='Conexion auxiliar del lado del robot',children=[robot_plug(),wire])


def thermal():
    items=[]
    for label,x,z,w,h in [('RT1 NTC del protector',-14,1116,5,8),
                          ('TS1 corte termico NC',-30,1116,10,13),
                          ('RT2 NTC de carga',18,1136,5,8)]:
        items.append(name(box(x,-35.25,z,w,.5,h),'Interfaz termica aislante '+label,'#E2B36A'))
        items.append(name(box(x,-37,z,w,3,h),label,'#B98842'))
    return items


def pack_wires():
    z=PCB_BOTTOM+4.65
    items=[]
    for i in range(5):
        x=-35+i*2.5
        # Extremos en el plano de colectores dorsal. Secciones de rutas, no pletinas liberadas.
        items.append(lead('Toma B'+str(i)+' al J1 del protector',
            [(x,-22.1,z),(x,-26,z),(x,-36.4,1153.6),(x,-36.4,1142.5)],
            .82 if i in (0,4) else .48,('#454D57','#62A7B4','#88AD64','#D8AC4F',RED)[i],4))
    items+= [
        lead('J2 positivo a F1',[(9.5,-22.1,z),(9.5,-25.5,z),(9.5,-25.5,1164),(55,-25.5,1164),(55,0,1154),(55,8,1154),(51.5,8,1154)],.85,RED,2),
        lead('F1 a pin 1',[(36.5,8,1154),(32,8,1154),(31,-9,1153),(31.8,-25,1157),(31.8,-33,1157)],.85,RED,4),
        lead('J2 retorno a pin 2',[(12,-22.1,z),(12,-26,z),(12,-28,1160),(36,-28,1160),(36,-30,1157),(36,-33,1157)],.85,GREY,2),
        lead('J3 a TS1 y RT1 - funda de dos hilos',[(-51.1,2.75,z),(-54.5,2.75,z),(-54.5,-36,z),(-51,-38,1152),(-51,-38,1126),(-51,-41,1121),(-30,-41,1116),(-30,-38.5,1116)],.9,GOLD,2),
        lead('Cola termica del BMS a RT1',[(-30,-37,1109.5),(-30,-37,1105),(-14,-37,1105),(-14,-37,1112)],.7,GOLD,2),
        lead('RT2 al puerto - funda sonda y retorno',[(18,-38.5,1136),(18,-41.5,1136),(42,-45.8,1142),(47,-46.8,1150),(47,-29,1157),(40.2,-29,1157),(40.2,-33,1157)],.72,'#D59C44',3)]
    return items
