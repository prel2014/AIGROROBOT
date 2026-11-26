"""Carenado del húmero y mazos de tracción. Coordenadas del robot en mm."""
from math import hypot
from build123d import Box, Circle, Compound, Plane, Pos, RectangleRounded, loft
from arnes_brazos import lead
from instalacion_control import box, bore_y, duct, name, solid
from tren_inferior import Leg, AL, DARK, STEEL, WHITE, group

MOUNTS = (1202, 1268)
LANES = (('CAN0 ida y vuelta', 260, 2., '#22A98E'),
         ('48 V y retorno', 270, 2.6, '#D67039'))
LEFT_LANES=((260.,-62.),(271.,-49.))
LEFT_MOUTH=(8.5,-7.5)


def elbow_motor(side):
    # Same actuator in all four joints; translate mounting face and pivot together.
    motor = Pos(side*113, 0, 620) * Leg(side).motor()
    motor.label = 'AK60-39 codo ' + ('D' if side > 0 else 'I')
    return motor


def elbow_mount(hinge,side):
    # Local outboard coordinate is the unsigned distance to the robot centre.
    def axial(a,b,y,z,r):
        return duct((side*a,y,z),(side*b,y,z),r)
    world=[hinge.location*p for p in hinge.children[:3]]
    outer=max(range(2),key=lambda i:side*world[i].bounding_box().center().X)
    bridge=box(side*283.75,0,1145,22.5,34,15)
    rim=axial(290.5,295,0,1100,38.5)-axial(289,297,0,1100,29.7)
    tab=box(side*292.75,0,1137.5,4.5,34,30)
    bracket=solid(rim.fuse(tab,bridge))-axial(289,297,0,1100,29.7)
    bolts=[]
    for y in (-10,10):
        bracket=bracket-axial(272,296,y,1145,2.25)
        world[outer]=world[outer]-axial(263,273,y,1145,1.65)
        bolts.append(name(solid(axial(264,295,y,1145,1.63).fuse(axial(295,297.8,y,1145,3.35))),
                          'Anclaje de soporte a orejeta M4',STEEL))
    for y,h in Leg(side).circle(34,4,10.05):
        bracket=bracket-axial(290,296,y,1100+h,1.65)
        bolts.append(name(solid(axial(290.5,300,y,1100+h,1.45).fuse(
                    axial(288.4,290.5,y,1100+h,2.5))),'Fijación de estator M3',STEEL))
    drive=axial(199.5,273.15,0,1100,11).fuse(axial(273.15,276.5,0,1100,20))
    coupler_bolts=[]
    for y,h in Leg(side).circle(15,3,30):
        drive=drive-axial(272,277,y,1100+h,2.1)
        # Countersunk envelope leaves clearance from the fixed fork face at 272.5.
        drive=drive-axial(273.1,274.7,y,1100+h,3.35)
        coupler_bolts.append(name(solid(axial(274.7,280,y,1100+h,1.9).fuse(
                    axial(273.3,274.7,y,1100+h,3.2))),'Fijación brida de salida M4',STEEL))
    bore=bore_y(side*236,1100,-31,31,2.53)
    drive=drive-bore
    world[2]=world[2]-bore
    shaft=group('Eje y brida solidarios al antebrazo',[
        name(solid(drive),'Brida y eje mecanizados',AL),
        name(bore_y(side*236,1100,-27,27,2.5),'Pasador de arrastre de 5 mm',STEEL),*coupler_bolts])
    return group(hinge.label,[*[name(solid(p),o.label,o.color) for p,o in zip(world,hinge.children)],
                              shaft,group('Soporte de estator',[name(solid(bracket),'Soporte anular',AL),*bolts])])


def upper_panels(side, original):
    x = side*236
    # Constant sections at the screw heads provide planar seats on both sides.
    profiles = [(1168, 99, -74, 47), (1196, 106, -76, 55),
                (1208, 106, -76, 55), (1262, 108, -76, 57),
                (1274, 108, -76, 57), (1292, 104, -74, 54)]
    def envelope(t):
        return loft([Plane(origin=(x, (a+b)/2, z)) *
                     RectangleRounded(w-2*t, b-a-2*t, 22-t)
                     for z,w,a,b in profiles], ruled=True)
    skin = envelope(0) - envelope(2.8)
    base = Compound(children=[original.location*c for c in original.children])
    hardware, shells, holes = [], [], []
    for sign, label in ((1,'Panel anterior'), (-1,'Panel dorsal')):
        panel = solid(skin.intersect(box(x, sign*50.35, 1240, 140, 100, 180)))
        if sign<0:
            for i,(_,_,r,_) in enumerate(LANES):
                panel=panel-lead('Paso del ramal',traction_points(side,bool(i))[6:],
                                 r+1.9,DARK,12)
        screws = []
        for z in MOUNTS:
            face = (55 if z==1202 else 57) if sign==1 else -76
            seat = face-sign*2.8
            for dx in (-14,14):
                cx=x+dx
                # Custom spacers match the actual underlying curved skin.
                def bore(a,b,r):
                    return bore_y(cx,z,min(a,b),max(a,b),r)
                post = bore(sign*17,seat,4.4).cut(base)
                tap = bore(sign*17,seat+sign,1.25)
                hardware.append(name(solid(post-tap),'Separador con asiento curvo M3',AL))
                holes.append(bore(sign*17,seat,1.25))
                panel=panel-bore(seat-sign,face+sign,1.7)
                # Through fastener into the parent arm, passing through the spacer.
                shank=bore(sign*28,face,1.22)
                screws.append(name(solid(shank.fuse(bore(face,face+sign*2,2.8))),
                                   'Tornillo de panel M3',STEEL))
        shells.append(group(label+' con sus fijaciones',
                            [name(solid(panel),label+' de húmero',WHITE),*screws]))
    revised=[]
    for piece in original.children:
        cut=(original.location*piece).cut(*holes)
        revised.extend(name(p,piece.label,piece.color) for p in cut.solids())
    return group(original.label,[*revised,group('Separadores de carenado',hardware),*shells])


def traction_points(side, power):
    x = LANES[int(power)][1]
    z = 1406 if power else 1394
    rear = -137 if power else -122
    y = -55 if power else -60
    end_z=1139
    # No cable spans the rotating elbow flange. The CAN sheath carries two pairs
    # up to the splice at the motor plug, so the elbow is an intermediate bus node.
    tip_y = side*(1.5 if power else -3.9)
    if side<0:
        x,y=LEFT_LANES[int(power)]
        tip_y=LEFT_MOUTH[int(power)]
        end_z=1154
    # The motor is turned over on the opposite arm. Route the lead reaching the
    # farthest P2 outlet above its neighbour before they descend side by side.
    drop = 1177 if power==(side>0) else 1168
    crest=drop+8 if side<0 and not power else drop
    crown=1438 if power else 1420
    points = [(177,-44,z),(177,-65,z),(168 if power else 181,rear,crown),(x,rear,crown+2),
              (x,-78,1350),(x,y,1320),(x,y,1300),(x,y,1190),
              (x,y,drop),(305,y,drop),(340,y,crest),(340,tip_y,crest-5),
              (340,tip_y,end_z)]
    return [(side*a,b,c) for a,b,c in points]


def shoulder_bracket(side):
    x=side*177
    # Carrier bears on the fixed 36 mm cross shaft, clear of the rotating shoulder.
    seat=box(x,-17,1400,20,30,30)-duct((side*163,0,1400),(side*190,0,1400),18)
    parts=[]
    for i,(label,_,_,color) in enumerate(LANES):
        z=1394+12*i
        seat=seat-box(x,-24,z,11.6,32,9.5)
        parts.extend([name(box(x,-26,z,11.2,12,9.1),label+' - base de servicio',DARK),
                      name(box(x,-38,z,11.2,12,9.1),label+' - clavija de brazo',color)])
    for dx in (-8.3,8.3):
        hole=bore_y(x+dx,1400,-35,-7,1.25)
        seat=seat-hole
        parts.append(name(solid(bore_y(x+dx,1400,-32,-11,1.22).fuse(
                    bore_y(x+dx,1400,-34,-32,2.35))),'Fijación de soporte M2.5',STEEL))
    return group('Conectores desmontables en hombro',[name(solid(seat),'Soporte sobre travesaño',AL),*parts])


def shaft_pilots(torso):
    holes=[bore_y(side*177+dx,1400,-35,-8,1.25) for side in (-1,1) for dx in (-8.3,8.3)]
    torso.children=tuple(name(solid(p.cut(*holes)),p.label,p.color) if 'travesa' in p.label.lower()
                         else p for p in torso.children)


def left_guide(z):
    # Local X joins the two cable centres; local Y points into the fixed jaw.
    (a,b),(c,d)=LEFT_LANES
    span=hypot(c-a,d-b)
    frame=Plane(origin=(-(a+c)/2,(b+d)/2,z),x_dir=(-(c-a),d-b,0),z_dir=(0,0,-1)).location
    def jaw(y,depth):
        return frame*Pos(0,y,0)*Box(span+15.8,depth,7)
    def cross(u,v0,v1,r):
        return frame*bore_y(u,0,v0,v1,r)
    base=jaw(3,6)
    cap=jaw(-2.1,4.2)
    foot=box(-253.75,-65.5,z,19.5,3,7)
    # Clip the mounting foot at the split face so the keeper slides straight off.
    foot=foot-(frame*Pos(0,-21,0)*Box(80,42,15))
    base=solid(base.fuse(foot))-bore_y(-248,z,-70,-62,1.65)
    base=base-box(-242.15,-54,z,20,20,10)-box(-272,-29.3,z,20,24,10)
    for (x,y),(_,_,r,_) in zip(LEFT_LANES,LANES):
        # Flared entrances follow the first bend below the lower clamp.
        seat=loft([Plane(origin=(-x,y,z+h))*Circle(r+clearance)
                   for h,clearance in ((-4.5,1.25),(-2,.65),(2,.65),(4.5,1.25))],ruled=True)
        base=base-seat
        cap=cap-seat
    screws=[]
    for u in (-(span/2+5.25),span/2+5.25):
        base=base-cross(u,-.2,5.4,1.025)
        cap=cap-cross(u,-5,.2,1.4)
        screws.append(name(solid(cross(u,-4.2,4.8,1.).fuse(cross(u,-6,-4.2,2.15))),
                           'Tornillo M2.5 de mordaza',STEEL))
    return group('Guía izquierda de dos canales con tapa desmontable',
                 [name(solid(base),'Base con dos cunas',AL),
                  name(solid(cap),'Mordaza desmontable',DARK),*screws])


def left_plug_boot():
    # The cable-side boot sits on the OEM plug. The jacket ends 2 mm inside
    # its entry. Only insulated conductor bundles continue down to the P2 exits.
    x=-340.
    def section(z,w,h,y):
        return Plane(origin=(x,y,z))*RectangleRounded(w,h,1.3)
    body=loft([section(1136.8,8.9,16.3,2.25),section(1139.0,8.9,16.3,2.25),
               section(1140.6,8.9,16.1,.25),section(1145.8,8.9,16.1,.25),
               section(1153,10.4,26,.5),section(1156,10.4,26,.5)],ruled=True)
    blank=body
    body=body-box(x,2.25,1135.5,7.8,15.3,7)
    cores=[]
    for i,(_,_,r,color) in enumerate(LANES):
        y=LEFT_MOUTH[i]
        pin=(3.9,-1.5)[i]
        def passage(rad):
            return loft([Plane(origin=(x,cy,z))*Circle(rad) for z,cy in
                        ((1139,pin),(1144,pin),(1152,y),(1154,y))],ruled=True)
        # Contain each conductor bore, with at least 0.6 mm radial wall on
        # the outside, including where the strain relief flares above the cap.
        if any(s.volume>1.e-5 for s in (passage(r+.37)-blank).solids()):
            raise ValueError('Pared exterior insuficiente en la transición de P2')
        body=body-passage(r-.23)-duct((x,y,1153.99),(x,y,1157),r+.20)
        cores.append(name(solid(passage(r-.4)),'Haz aislado dentro del manguito '+LANES[i][0],color))
    # A moulded elastomer boot terminates both jackets and pulls out with P2.
    return group('Manguito bifurcado de P2 izquierdo',[name(solid(body),'Manguito de dos entradas',DARK),*cores])


def traction_branch(side):
    cables=[lead(label+' - hombro a P2',traction_points(side,bool(i)),r,color,12)
            for i,(label,_,r,color) in enumerate(LANES)]
    guides=[]
    for i,(label,_,r,_) in enumerate(LANES):
        points=traction_points(side,bool(i))[6:]
        tube=lead(label,points,r+1.25,DARK,12)-lead(label,points,r+.35,DARK,12)
        sleeve=solid(tube.intersect(box(side*303,-58,1172,51,34,34)))
        guides.append(name(sleeve,'Funda de salida al codo - '+label,DARK))
    for z in (1185,1280):
        if side<0:
            guides.append(left_guide(z))
            guides.append(name(solid(bore_y(-248,z,-67,-37,1.22).fuse(
                              bore_y(-248,z,-69,-67,2.5))),'Tornillo común M3',STEEL))
            continue
        tab=box(side*254.5,-65.5,z,21,3,7)
        comb=box(side*264,-56,z,24,16,7)
        guide=tab.fuse(comb)
        for i,(_,lane,r,_) in enumerate(LANES):
            y=-55 if i else -60
            guide=guide-duct((side*lane,y,z-6),(side*lane,y,z+6),r+.6)
            guide=guide-box(side*lane,y+7,z,2*r+.6,14,9)
        guide=guide-bore_y(side*248,z,-70,-62,1.65)
        guides.append(name(solid(guide),'Guía de tracción en peine dorsal',AL))
        guides.append(name(solid(bore_y(side*248,z,-67,-37,1.22).fuse(
                          bore_y(side*248,z,-69,-67,2.5))),'Tornillo común M3',STEEL))
    if side<0:
        guides.append(left_plug_boot())
    return group('Ramal de tracción '+('derecho' if side>0 else 'izquierdo'),
                 [shoulder_bracket(side),*cables,group('Retención en brazo',guides)])
