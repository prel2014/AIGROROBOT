"""Interior de servicio del humanoide de 1730 mm. Milímetros, +Y hacia el pecho.
Los volúmenes electrónicos son reservas de integración. Origen Z en la planta.
La cara de acoplamiento de cada cajón es su brida dorsal.
"""
from pathlib import Path
import unicodedata

from build123d import Axis, Box, Color, Compound, Cylinder, Location, Pos, Rot, fillet, import_step
from instalacion_control import tray_with_boards
from arnes_brazos import trunk_bores
import cartucho_auxiliar as aux

CAD = Path(__file__).resolve().parent
CELL_DIAMETER, CELL_LENGTH, CELL_PITCH = 18., 65., 20.
GRID = 6
JETSON_FACE, JETSON_HEIGHT = 105., 65.
CELL_FIRST_Z = 1036.
# Cámara inferior y superior: ancho, frente interior, piso, techo y asiento dorsal.
BANK_BAY = (128, 38, 1020, 1172, -47)
CPU_BAY = (176, 43, 1212, 1365, -58)
TEAL, COPPER, SHELL, DARK = '#208482', '#CB7542', '#607475', '#29373B'
CELL, METAL, BOARD = '#378495', '#B3C1BD', '#486C56'


def named(part, label, paint):
    part.label = label
    part.color = Color(paint) if isinstance(paint, str) else paint
    return part


def block(x,y,z,dx,dy,dz):
    return Pos(x,y,z)*Box(dx,dy,dz)


def along_y(x,y,z,r,length):
    return Pos(x,y,z)*Rot(90,0,0)*Cylinder(r,length)


def flat(x,y,z,w,d,h,r=3):
    obj = Box(w,d,h)
    return Pos(x,y,z)*fillet(obj.edges().filter_by(Axis.Y),r)


def normal(label):
    try:
        label=label.encode('latin1').decode('utf8')
    except UnicodeError:
        pass
    return unicodedata.normalize('NFKD', label.replace('_',' ')).encode('ascii','ignore').decode().lower()


def bore_pattern(width,zlow,zhigh,y,length,radius):
    return [along_y(sign*width/2,y,z,radius,length)
            for sign in (-1,1) for z in (zlow,zhigh)]


def tunnel(width,front,low,high):
    return block(0,(front-240)/2,(low+high)/2,width,front+240,high-low)


def seats(bay):
    width, front, bottom, top, dorsal = bay
    # The larger, shallow counter-recess makes a flat seating plane in the curved back.
    return [tunnel(width,front,bottom,top),
            tunnel(width+22,dorsal,bottom-7,top+7),
            *bore_pattern(width+10,bottom+12,top-12,dorsal+4,8,1.65)]


def overlap(a,b):
    aa,bb=a.bounding_box(),b.bounding_box()
    return all(getattr(aa.min,k)<getattr(bb.max,k) and getattr(aa.max,k)>getattr(bb.min,k)
               for k in 'XYZ')


def modified_torso(source, with_servo=False):
    tools = seats(BANK_BAY)+seats(CPU_BAY)+trunk_bores()
    tools.extend(along_y(x,47,1339.5,1.65,8) for x in (-62,62))
    # Passage for a grommet and loom from the bank's ceiling to the electronics floor.
    tools.append(Pos(0,8,1192)*Cylinder(10,48))
    tools.append(aux.feedthrough())
    if with_servo:
        # Same displayed neck section as the assembly film; lateral branch leaves its four-point head datum.
        tools.extend([block(0,27,1508,95,92,50),block(0,0,1494.7,95,105,5.2)])
    result=[]
    for item in source.children:
        shape=source.location*item
        nearby=[tool for tool in tools if overlap(shape,tool)]
        if nearby:
            shape=shape.cut(*nearby)
        for n,solid in enumerate(shape.solids() if shape is not None else []):
            if solid.volume>1e-4:
                name=normal(item.label)+(f' {n+1}' if n else '')
                result.append(named(solid,name,item.color))
    return Compound(label='Carroceria con dos registros dorsales',children=result)


def flange(bay,thick,paint,label,aperture=None):
    width,_,bottom,top,dorsal=bay
    obj=flat(0,dorsal-thick/2,(bottom+top)/2,width+20,thick,top-bottom+12)
    if aperture:
        obj=obj-flat(0,dorsal-thick/2,(bottom+top)/2,aperture[0],thick+3,aperture[1])
    obj=obj.cut(*bore_pattern(width+10,bottom+12,top-12,dorsal-thick/2,thick+2,2.25))
    return named(obj,label,paint)


def cell_centers():
    return [((c-(GRID-1)/2)*CELL_PITCH,CELL_FIRST_Z+r*CELL_PITCH)
            for r in range(GRID) for c in range(GRID)]


def bank_case():
    # Sliding shell, open to the rear. The flange seats on the abdomen.
    shell=flat(0,-6.5,1096,124,81,148)-flat(0,-10,1096,119,84,143)
    ring=flange(BANK_BAY,3,SHELL,'',aperture=(117,140))
    # A real through port in the roof aligns with the trunk passage.
    port=Pos(0,8,1169)*Cylinder(8,10)
    pads=[aux.post(x,y,1164,1169,3.1) for x,y in aux.HATCH_FIX]
    return named(shell.fuse(ring,aux.service_deck(),*pads).cut(port,*aux.hatch_cuts()),
                 'Casete extraible con brida, piso y cuatro apoyos del protector',SHELL)


def separator(y):
    plate=flat(0,y,1086,118.5,2.5,120.5)
    holes=[along_y(x,y,z,9.3,4) for x,z in cell_centers()]
    return named(plate.cut(*holes),'Separador aislante de 36 alveolos',DARK)


def bank():
    items=[bank_case(),separator(-26),separator(23)]
    # Cuatro grupos físicos 3 x 3. BT1–9, BT10–18, BT19–27 y BT28–36.
    for index,(x,z) in enumerate(cell_centers()):
        row,col=divmod(index,GRID)
        i=1+9*(2*(row//3)+col//3)+3*(row%3)+col%3
        cell=along_y(x,-2.5,z,CELL_DIAMETER/2,CELL_LENGTH)
        items.append(named(cell,f'Celda 18650 Naylamp BT{i:02d} grupo {(i-1)//9+1}',CELL))
    items.extend([aux.protector(),*aux.bms_plugs(),*aux.fuse_parts(),aux.pack_port(),*aux.thermal(),
                  *aux.pack_wires(),aux.hatch()])
    return Compound(label='Un banco extraible - 36 celdas y proteccion',children=items)


def battery_lid():
    lid=flange(BANK_BAY,3,DARK,'Tapa protectora del banco')
    lid=Pos(0,-3,0)*lid
    # Two low ribs let the operator grasp the cover.
    grip=flat(0,-54,1096,60,3,11,2)
    port=block(aux.PORT_X,-52,aux.PORT_Z,23,12,13)
    return named(lid.fuse(grip).cut(port),'Tapa protectora del banco con asidero y puerto J1',DARK)


def retaining_bands():
    loops=[]
    for x in (-41,41):
        outer=block(x,-2.5,1268.5,6,68,112.6)
        inner=block(x,-2.5,1268.225,10,65.5,110.05)
        loops.append(named(outer-inner,'Cincha lateral de retencion',COPPER))
    return loops


def cradle():
    # Base, front stop and short lateral guides form one slide-in tray.
    tray=block(0,-7.5,1215.5,153,101,5)
    front=block(0,40,1226,150,5,21)
    lips=[block(s*54.25,-2,1225.5,2.5,70,16) for s in (-1,1)]
    ring=flange(CPU_BAY,3,TEAL,'',aperture=(158,145))
    # Two narrow ribs join the inset tray to its mounting flange.
    feet=[block(s*70,-56,1214,10,6,4) for s in (-1,1)]
    cutouts=[block(x,-13,1216,9,48,12) for x in (-36,-12,12,36)]
    return named(tray.fuse(front,*lips,ring,*feet).cut(*cutouts,*retaining_bands()),'Bandeja universal de servicio Xavier',TEAL)


def jetson():
    # Vertical installation, fan towards -Y. All features except the overall
    # dimensions are schematic. OEM mounting holes are intentionally pending.
    z=1218+JETSON_FACE/2
    front=30.
    back=front-JETSON_HEIGHT
    base=flat(0,front-3,z,JETSON_FACE,6,JETSON_FACE)
    core=flat(0,(front-6+back+6)/2,z,101,JETSON_HEIGHT-12,101)
    grille=flat(0,back+3,z,JETSON_FACE,6,JETSON_FACE)
    slots=[flat(x,back+3,z,3.1,9,65-abs(x)*.6,1.45) for x in range(-32,33,8)]
    grille=grille.cut(*slots)
    return Compound(label='Jetson AGX Xavier Developer Kit P2972 - envolvente',children=[
        named(base,'Base de referencia 105 x 105',METAL),named(core,'Volumen termico indicativo',DARK),
        named(grille,'Frente ventilado indicativo',DARK)])


def electronics():
    deck=block(0,-1,1341,150,63,3.5)
    # Cantilever with a mounting back seated in the forward wall.
    stops=[block(x,35,1339.5,16,16,6.5) for x in (-62,62)]
    deck=deck.fuse(*stops).cut(*(along_y(x,1,1339.5,2.25,100) for x in (-62,62)))
    tray,boards=tray_with_boards(cradle())
    return Compound(label='Computo y regulacion',children=[tray,jetson(),*retaining_bands(),*boards,
        named(deck,'Estante del convertidor',SHELL),
        named(block(0,-2,1351.25,74,36,17),'Reserva convertidor DC-DC - sin modelo',COPPER)])


def cpu_lid():
    lid=flange(CPU_BAY,3,SHELL,'Registro ventilado')
    lid=Pos(0,-3,0)*lid
    slots=[flat(x,-62.5,z,6,10,36,2.7) for x in range(-60,61,12) for z in (1250,1302)]
    slots.extend(flat(x,-62.5,1344,7,10,13,3) for x in range(-60,61,12))
    return named(lid.cut(*slots),'Tapa dorsal con entradas y salidas de aire',SHELL)


def original():
    return import_step(CAD/'humanoide_refinado.step')


def interior(source, covers):
    items=[modified_torso(source),electronics(),bank(),aux.robot_loom()]
    if covers:
        items.extend([cpu_lid(),battery_lid()])
    return Compound(label='Torso desmontable con Jetson AGX Xavier y banco 18650',children=items)


def find_torso(robot):
    return next(n for n in robot.children if normal(n.label).startswith('tronco'))

