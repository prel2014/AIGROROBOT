"""Interfaces de tracción. +u es el exterior de la pierna, Y frontal, Z vertical."""
from functools import lru_cache
from pathlib import Path
from build123d import Box, Cylinder, Plane, Pos, import_step
from arnes_brazos import lead
from tren_inferior import AL, DARK, FRAME, HIP, KNEE, STEEL, WHITE, group, part

REF=Path(__file__).resolve().parent.parent/'referencias'
CAN, POWER, BOOT='#18A6B8','#E48537','#465258'
LANES=(('CAN0 H/L',-6,1.65,CAN),('48 V y retorno',6,2.6,POWER))
GUIDE_HEIGHTS=(604,748)


def loom(g,label,points,r,color,bend=10):
    return lead(label,[(g.x(u),y,z) for u,y,z in points],r,color,bend)


def hip_points():
    # The round sleeve enters on the X pivot. Rotation thus preserves its fixed endpoint.
    return [(-52.5,0,HIP),(-2,0,HIP),(19,0,HIP),(31,0,850),(31,0,806)]


def hip_bore(g):
    return loom(g,'Canal en tambor',hip_points(),6.1,DARK)


def outlet(g):
    return [g.pin(27,70,553,r+1.4,y*g.side) for _,y,r,_ in LANES]


def thigh_harness(g):
    # Terminal móvil de rodilla. El antiguo tubo pasaba por el nuevo rotor de flexión.
    parts=[part(g.block(31,0,801,10,18,10),'Salida de funda y separación de ramales',DARK)]
    sleeves=[]
    for label,offset,r,color in LANES:
        y=offset*g.side
        # Ends at the back of the OEM mating plug. The stator and both conduits stay
        # on the thigh side of the knee, so no wire spans the moving tibial clevis.
        tip_y=g.side*(-3.9 if offset<0 else 1.5)
        points=[(31,y,796),(31,y,581),(31,y,553),(55,y,553),(98,y,553),
                (115,y,547),(115,tip_y,535),(115,tip_y,519)]
        parts.append(loom(g,label+' - muslo a conector de rodilla',points,r,color,8))
        outer=loom(g,label,points,r+1.15,BOOT,8)
        void=loom(g,label,points,r+.35,BOOT,8)
        window=g.block(82,0,547,80,28,48)
        tube=(outer-void).intersect(window).solids()
        if len(tube)!=1: raise ValueError('Funda de protección discontinua')
        sleeves.append(part(tube[0],'Funda de protección '+label,BOOT))
    for z in GUIDE_HEIGHTS:
        guide=g.block(28.8,0,z,9.6,20,6)-g.pin(23,35,z,1.7)
        for _,y,r,_ in LANES:
            guide=guide-Pos(g.x(31),y*g.side,z)*Cylinder(r+.45,8)
        parts.append(part(guide,'Guía doble sobre larguero lateral',FRAME))
        parts.append(part(g.pin(21.7,33.6,z,1.5).fuse(g.pin(33.6,35.5,z,2.7)),
                          'Fijación M3 de guía',STEEL))
    return group('Arnés CAN0 y 48 V',parts),group('Fundas exteriores hasta tapa',sleeves)


def pelvis_route():
    # Coordenada transversal relativa al centro de la cadera. Los extremos son
    # la unión de los dos ramales del conector y el prensaestopas del muslo.
    return [(-73,-77.5,HIP),(-73,-76,858),(-73,-82,833),
            (-66,-98,802),(-27,-107,785),(11,-88,780),(31,-67,790),
            (31,-43,792),(31,-22,802),(31,-15,813),(31,-8,816),
            (31,0,814),(31,0,806)]


def pelvis_parts(g):
    parts=[]
    # External service connector envelopes stop at the upstream harness interface.
    for label,y,r,color in LANES:
        u=-73+y
        parts.append(loom(g,label+' - entrada pélvica',[(u,-83,HIP),(u,-80,HIP),(-73,-77.5,HIP)],r,color,4))
        parts.append(part(g.block(u,-87,HIP,9,8,8),label+' - conector de servicio',color))
        parts.append(part(g.block(u,-94,HIP,9,6,8),label+' - mitad del troncal',DARK))
    parts.append(loom(g,'Lazo posterior de servicio al muslo',pelvis_route(),4.8,BOOT,12))
    return group('Paso pélvico '+('derecho' if g.side>0 else 'izquierdo'),parts)


def pelvis_with_ports(torso,legs):
    cuts=[]
    bosses=[]
    fixtures=[]
    for g in legs:
        cuts.append(loom(g,'Conducto pélvico',pelvis_route()[:4],6.1,DARK,12))
        cuts.append(g.block(-73,-75,HIP,24,28,13))
        # Anclar arriba y abajo del conector deja libre el barrido del motor posterior.
        plate=g.block(-73,-85,HIP,32,4,34)-g.block(-73,-85,HIP,23,6,10)
        fasteners=[]
        u_fijacion=-82  # 9 mm hacia el centro, fuera del descenso de la funda.
        for z in (HIP-12.5,HIP+12.5):
            bosses.append(g.cross(u_fijacion,-83,-49,z,4.2))
            cuts.append(g.cross(u_fijacion,-84,-68,z,1.25))
            plate-=g.cross(u_fijacion,-89,-82,z,1.7)
            screw=g.cross(u_fijacion,-87,-75,z,1.5).fuse(g.cross(u_fijacion,-90,-87,z,2.7))
            fasteners.append(part(screw,'Tornillo M3 del soporte pélvico',STEEL))
        fixtures.append(group('Conectores - soporte posterior',[part(plate,'Placa de conectores',AL),*fasteners]))
    updated=[]
    for p in torso.children:
        if 'lv' in p.label.lower() and p.bounding_box().min.Z<HIP+15:
            forma=p.fuse(*bosses) if 'escudo' in p.label.lower() else p
            p=part(forma-cuts,p.label,p.color)
        updated.append(p)
    torso.children=tuple(updated)
    return group('Conectores fijos de pelvis',fixtures+[pelvis_parts(g) for g in legs])


@lru_cache()
def driver():
    return import_step(REF/'AK54-4810-2D-A2-integracion.step')


def equipped_case(g,case):
    # Board X points up to put the genuine P2 right-angle connector under the top opening.
    frame=Plane(origin=(g.x(113),0,KNEE),x_dir=(0,0,1),z_dir=(g.side,0,0)).location
    source=driver()
    board=frame*source
    board.label='Controlador integrado CubeMars AK54-4810-2D-A2'
    case=case-g.pin(106,120,KNEE,31)-g.block(114,0,514,16,26,29)
    # Three PCB seats follow the manufacturer's through-holes (diameter 3.04 mm).
    board_face=next(p for p in source.children if p.label=='Board').bounding_box().min.Z
    underside=[p.bounding_box() for p in source.children if p.label!='Board' and p.bounding_box().min.Z<board_face]
    retainers=[]
    for x,y in ((-15.56,-15.56),(-15.56,15.56),(15.56,15.56)):
        boss=Pos(x,y,(board_face-7.1)/2)*Cylinder(2.8,board_face+7.1)
        for b in underside:
            if b.min.X-3.1<x<b.max.X+3.1 and b.min.Y-3.1<y<b.max.Y+3.1:
                boss=boss-Pos(*b.center())*Box(b.size.X+.6,b.size.Y+.6,b.size.Z+.6)
        pilot=frame*Pos(x,y,-5.1)*Cylinder(1.25,8)
        case=case.fuse(frame*boss)-pilot
        screw=(Pos(x,y,-2.7)*Cylinder(1.23,5.4)).fuse(Pos(x,y,.8)*Cylinder(2.1,1.6))
        retainers.append(part(frame*screw,'Tornillo de PCB M2.5',STEEL))
    # Body of the removable plug; the OEM connector terminates at board X=27.0.
    mating_z=KNEE+next(p for p in source.children if p.label=='P2').bounding_box().max.X
    plug=g.block(115,-2.25*g.side,(519+mating_z)/2,7.5,15,519-mating_z)
    parts=[part(case,'Cuerpo con asiento de la tarjeta',DARK),board]
    # Ventilated cap carried on the rear stationary bolt circle.
    cap=g.ring(72,124,KNEE,45.5,42.7).fuse(g.pin(124,127,KNEE,45.5))
    bolts=[]
    # Alternate holes of the rear eight-hole pattern leave the P2 plug removable.
    for y,h in g.circle(34,4,55.05):
        cap=cap.fuse(g.pin(118.5,124,KNEE+h,3.8,y))-g.pin(117,128,KNEE+h,1.7,y)
        case=case-g.pin(115.5,119,KNEE+h,1.5,y)
        bolts.append(part(g.pin(116,127,KNEE+h,1.5,y).fuse(g.pin(127,129,KNEE+h,2.8,y)),
                          'Tornillo M3 de tapa',STEEL))
    for h in (-16,-8,0,8,16):
        cap=cap-g.block(125.5,0,KNEE+h,7,30,3.2)
    cap=cap-g.block(115,0,525,13,19,18)
    parts[0]=part(case,'Cuerpo con asiento de la tarjeta',DARK)
    parts.extend([group('Tapa exterior ventilada',[part(cap,'Cubierta lateral del motor',WHITE),*bolts]),
                  part(plug,'Envolvente de clavija XT30 (2+2)', '#D4AC4A'),group('Fijación PCB',retainers)])
    return group('Estator equipado',parts)
