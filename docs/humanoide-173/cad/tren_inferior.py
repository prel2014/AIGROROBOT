"""Tren inferior, mm. u crece hacia fuera de cada pierna; Y apunta al frente."""
from math import cos, radians, sin

from build123d import (Axis, Box, Color, Compound, Cylinder, Plane, Pos, RectangleRounded,
                       Rot, loft)


AL, STEEL, FRAME, DRIVE, WHITE, DARK = ('#849695', '#BDC6C5', '#276F70',
                                         '#CD7946', '#E3E5DC', '#323F44')
KNEE, ANKLE, HIP = 480, 85, 875


def part(shape, label, color):
    shape.label, shape.color = label, Color(color)
    return shape


def group(label, children):
    return Compound(label=label, children=children)


def common_solid(a,b):
    solids=a.intersect(b).solids()
    if len(solids)!=1:
        raise ValueError(f'Intersección de carenado discontinua: {len(solids)} sólidos')
    return solids[0]


class Leg:
    def __init__(self, side):
        self.side = side

    def x(self, u):
        return self.side * (112 + u)

    def block(self, u, y, z, w, d, h):
        return Pos(self.x(u), y, z) * Box(w, d, h)

    def pin(self, a, b, z, r, y=0):
        return Pos(self.x((a+b)/2), y, z) * Rot(0,90,0) * Cylinder(r, b-a)

    def ring(self, a, b, z, outside, inside):
        return self.pin(a,b,z,outside) - self.pin(a-1,b+1,z,inside)

    def cross(self, u, a, b, z, r):
        return Pos(self.x(u),(a+b)/2,z) * Rot(90,0,0) * Cylinder(r,b-a)

    def circle(self, radius, count, phase):
        # Both actuators are identical; reversing the motor axis also reverses Y.
        return [(self.side*radius*cos(radians(phase+i*360/count)),
                 radius*sin(radians(phase+i*360/count))) for i in range(count)]

    def bolt(self, a, b, z, y, radius, head_radius, head_length):
        return self.pin(a,b,z,radius,y).fuse(self.pin(a-head_length,a,z,head_radius,y))

    def motor(self):
        from equipo_rodilla import equipped_case
        bolt_pattern = self.circle(34,8,10.05)
        case = self.pin(70,118.5,KNEE,39.5) - [
            self.pin(69,76,KNEE+h,1.5,y) for y,h in bolt_pattern]
        neck = self.pin(62,70,KNEE,27.3)
        rotor = self.pin(51.5,62,KNEE,28.5)
        for rad,phase,depth in [(15,30,4),(14.25,-30,3.5)]:
            rotor = rotor - [self.pin(51,51.5+depth,KNEE+h,2,y)
                             for y,h in self.circle(rad,3,phase)]
        return group('AK60-39 rodilla '+('D' if self.side>0 else 'I'),[
            equipped_case(self,case),part(neck,'Cuello',AL),
            part(rotor,'Brida de salida AK60-39',DRIVE)])

    def rail_end(self,z,direction):
        # Machined tenons plug the tubes; both walls and tenon share a cross screw.
        bridge = self.block(0,0,z-direction*4,48,32,8)
        for u in (-17,17):
            tenon = self.block(u,0,z+direction*11,8.8,26.8,22)
            bridge = bridge.fuse(tenon-self.cross(u,-19,19,z+direction*13,2.65))
        return bridge

    def rails(self,low,high,name):
        pieces=[]
        for u in (-17,17):
            tube=self.block(u,0,(low+high)/2,14,32,high-low)
            tube=tube-self.block(u,0,(low+high)/2,9,27,high-low+2)
            for z in (low+13,high-13):
                tube=tube-self.cross(u,-19,19,z,2.65)
                bolt=self.cross(u,-19,17,z,2.5).fuse(self.cross(u,16,20,z,4.2))
                pieces.append(part(bolt,'Perno pasante M5',STEEL))
                pieces.append(part(self.cross(u,-20,-16,z,4.4)-self.cross(u,-21,-15,z,2.5),
                                   'Tuerca M5',DARK))
            pieces.append(part(tube,'Larguero tubular 14 x 32 x 2.5',FRAME))
        # Four removable quadrants per link. Constant sections at the fasteners
        # give each screw a flat seat on a removable front or rear shell.
        thigh=low>KNEE
        base,top=(535,823) if thigh else (122,440)
        sections=([(base,96,78,0),(577,102,85,2),(630,111,98,3),
                   (724,116,107,3),(779,108,92,2),(top,98,78,0)] if thigh else
                  [(base,80,62,0),(169,83,69,-1),(226,94,83,-4),
                   (322,110,104,-8),(387,104,92,-4),(top,84,73,0)])
        mounts=sections[1:-1]
        profiles=[sections[0],*[(z+dz,w,d,y) for z,w,d,y in mounts for dz in (-5,5)],sections[-1]]
        wall=2.8
        def skin(delta):
            return loft([Plane(origin=(self.x(0),y,z))*RectangleRounded(w-2*delta,d-2*delta,w*.28-delta)
                         for z,w,d,y in profiles],ruled=True)
        inside=skin(wall)
        hollow=skin(0)-inside
        if thigh:
            # The two looms exit through the outboard wall above the fixed motor.
            from equipo_rodilla import GUIDE_HEIGHTS,outlet
            hollow=hollow-outlet(self)
            rail=pieces[9]
            pieces[9]=part(rail-[self.pin(20,25,z,1.5) for z in GUIDE_HEIGHTS],rail.label,FRAME)
        middle=(base+top)/2
        covers=[]
        for front in (1,-1):
            for lower,upper in ((base,middle-.45),(middle+.45,top)):
                clip=self.block(0,-4+front*60.45,(lower+upper)/2,160,120,upper-lower)
                panel=common_solid(hollow,clip)
                for z,w,d,y in mounts:
                    if not lower<z<upper: continue
                    face=y+front*d/2
                    seat=face-front*wall
                    support=common_solid(self.block(0,(front*16+seat)/2,z,43,abs(seat-front*16),8),inside)
                    for index,u in ((4,-17),(9,17)):
                        a,b=sorted((front*13,face+front*3))
                        drill=self.cross(u,a,b,z,1.7)
                        panel=panel-drill
                        support=support-drill
                        rail=pieces[index]
                        pieces[index]=part(rail-drill,rail.label,FRAME)
                        a,b=sorted((front*13.5,face))
                        c,e=sorted((face,face+front*2))
                        screw=self.cross(u,a,b,z,1.5).fuse(self.cross(u,c,e,z,3))
                        pieces.append(part(screw,'Tornillo de carcasa M3',STEEL))
                    pieces.append(part(support,'Separador de carcasa',AL))
                label=('Frontal' if front==1 else 'Dorsal')+(' inferior' if lower==base else ' superior')
                covers.append(part(panel,label,WHITE if front==1 else '#A5B6B6'))
        pieces.append(group('Carcasa exterior desmontable',covers))
        return group(name,pieces)

    def fork(self,z,knee):
        bore,outer,width,inner,ear_r=(10,21,12,24,30) if knee else (8.5,17.5,10,20,25)
        end=z+(65 if knee else 59)
        bracket=self.rail_end(end,1)
        fixed, moving = [],[]
        for s in (-1,1):
            def seg(a,b):
                return tuple(sorted((s*a,s*b)))
            a,b=seg(inner-2,inner+width)
            ear=self.pin(a,b,z,ear_r).fuse(self.block((a+b)/2,0,(z+end-4)/2,
                                 b-a,ear_r*1.46,end-4-z))
            a1,b1=seg(inner-2,inner)
            a2,b2=seg(inner,inner+width)
            ear=ear-self.pin(a-1,b+1,z,outer-2)-self.pin(a2,b2,z,outer)
            # Upper bridge spans the full fork, including the outboard ears.
            bracket=bracket.fuse(self.block(0,0,end-6,2*(inner+width),32,4),ear)
            label='6004-2RS' if knee else '6003-2RS'
            fixed.append(part(self.ring(a2,b2,z,outer,outer-2.2),label+' aro exterior',STEEL))
            moving.append(part(self.ring(a2,b2,z,bore+2.4,bore),label+' aro interior',STEEL))
            for face in (a2+.35,b2-.85):
                fixed.append(part(self.ring(face,face+.5,z,outer-2.2,bore+2.6),label+' sello',DARK))
            # Removable outer-ring keeper with three M3 fasteners; turned inner shoulder.
            ca,cb=seg(inner+width,inner+width+2)
            cap=self.ring(ca,cb,z,ear_r-1,outer-1.6)
            for y,h in self.circle(outer+3.4,3,90):
                cap=cap-self.pin(ca-1,cb+1,z+h,1.7,y)
                ha,hb=seg(inner+width-5,inner+width)
                bracket=bracket-self.pin(ha,hb,z+h,1.5,y)
                sa,sb=seg(inner+width-4.5,inner+width+2)
                da,db=seg(inner+width+2,inner+width+4)
                fixed.append(part(self.pin(sa,sb,z+h,1.5,y).fuse(self.pin(da,db,z+h,2.7,y)),
                                  'Tornillo tapa M3',STEEL))
            fixed.append(part(cap,'Retenedor del aro exterior',FRAME))
        return bracket,fixed,moving

    def axle(self,z,knee):
        radius,inner,width,tab=(10,24,12,13) if knee else (8.5,20,10,12)
        edge=inner+width
        shaft=self.pin(-edge,edge,z,radius)
        key_width,key_h=(6,6) if knee else (5,5)
        key=self.block(0,0,z+radius,tab*2,key_width,key_h)
        shaft=shaft-key
        tongue_r=24 if knee else 19
        lower=419 if knee else 25
        tongue=self.pin(-tab,tab,z,tongue_r).fuse(self.block(0,0,(lower+z)/2,tab*2,
                                                            2*tongue_r-4,z-lower))
        tongue=tongue-self.pin(-tab-1,tab+1,z,radius)-key
        if knee:
            tongue=tongue.fuse(self.rail_end(419,-1))
        moving=[part(tongue,'Oreja enchavetada de la tibia' if knee else 'Pedestal enchavetado del pie',FRAME),
                part(key,'Chaveta paralela',DRIVE)]
        for s in (-1,1):
            a,b=sorted((s*tab,s*inner))
            moving.append(part(self.ring(a,b,z,radius+2.2,radius),'Separador del aro interior',AL))
        # Retain axle with axial screws and washers, inboard on knee, both ends on ankle.
        for s in ((-1,) if knee else (-1,1)):
            a,b=sorted((s*(edge-8),s*edge))
            shaft=shaft-self.pin(a,b,z,2.5)
            a,b=sorted((s*edge,s*(edge+2)))
            moving.append(part(self.ring(a,b,z,radius+2.2,2.65),'Arandela de retención',STEEL))
            a,b=sorted((s*(edge-7),s*(edge+2)))
            c,d=sorted((s*(edge+2),s*(edge+5)))
            moving.append(part(self.pin(a,b,z,2.5).fuse(self.pin(c,d,z,4.3)),'Tornillo axial M5',DARK))
        if knee:
            flange=self.pin(46,51.5,z,22.5)
            shaft=shaft.fuse(self.pin(edge,edge+3,z,radius+2.2),self.pin(edge+3,46,z,radius),flange)
            for y,h in self.circle(15,3,30):
                shaft=shaft-self.pin(45,52,z+h,2.25,y)
                moving.append(part(self.bolt(46,55,z+h,y,2,3.5,3.8),'Tornillo de salida M4',STEEL))
            for y,h in self.circle(14.25,3,-30):
                shaft=shaft-self.pin(46,52,z+h,2,y)
                moving.append(part(self.pin(47,54.5,z+h,2,y),'Pasador de centrado Ø4',STEEL))
        moving.append(part(shaft,'Semieje con brida integral Ø20' if knee else 'Eje transversal Ø17',DRIVE))
        return moving

    def assemble(self,old):
        from equipo_rodilla import hip_bore,thigh_harness
        foot,_,_,_,_,hip=old.children
        # Convert original local component placements into robot coordinates exactly once.
        def world(g,p):
            return old.location*g.location*p
        hip_parts=[world(hip,p) for p in hip.children]
        drum=hip_parts[0]
        hip_parts[0]=part(drum-hip_bore(self),drum.label,drum.color)
        saddle=self.rail_end(810,-1).fuse(self.block(0,0,828,48,32,24))
        saddle=saddle-hip_parts[0]
        # Saddle bolts into the drum from below.
        for u in (-6,6):
            for y in (-9,9):
                def vertical(z,h,r):
                    return Pos(self.x(u),y,z)*Cylinder(r,h)
                saddle=saddle-vertical(822,32,2.7)
                drum=hip_parts[0]
                hip_parts[0]=part(drum-vertical(831,12,2.5),drum.label,drum.color)
                hip_parts.append(part(vertical(823,26,2.5).fuse(vertical(808,4,4.5)),
                                      'Fijación del asiento M5',STEEL))
        hip_parts.append(part(saddle,'Asiento de los largueros del muslo',AL))
        bracket,bearings,krot=self.fork(KNEE,True)
        mount=self.ring(63,70,KNEE,42,29.5)
        mount=mount.fuse(self.block(66.5,0,504,7,40,48)-self.pin(62,71,KNEE,29.5),
                         self.block(49.5,0,527,41,30,8))
        bracket=bracket.fuse(mount)
        for y,h in self.circle(34,8,10.05):
            bracket=bracket-self.pin(62,71,KNEE+h,1.7,y)
            bearings.append(part(self.bolt(63,75,KNEE+h,y,1.5,2.7,2.8),'Fijación de estator M3',STEEL))
        knee=group('Rodilla',[part(bracket,'Horquilla femoral y bancada del estator',AL),
                 group('Apoyos fijos de la rodilla',bearings),
                 group('Acople móvil de salida a tibia',krot+self.axle(KNEE,True))])
        lower,ab,arot=self.fork(ANKLE,False)
        ankle=group('Tobillo',[part(lower,'Horquilla tibial de tobillo',AL),*ab])
        plate=self.block(0,0,21.5,72,81,7)
        anchor=self.block(0,0,15.5,72,81,5)
        stem_clear=self.block(0,0,60,25,36,76)
        pocket=self.pin(-44,44,ANKLE,40)
        new_foot=[]
        for i,p in enumerate(foot.children):
            solid=world(foot,p)
            trimmed=solid-anchor if i==0 else solid-plate-stem_clear-pocket
            new_foot.append(part(trimmed,p.label,p.color))
        for u in (-27,27):
            for y in (-30,30):
                def vert(z,h,r):
                    return Pos(self.x(u),y,z)*Cylinder(r,h)
                plate=plate-vert(22,10,2.7)
                anchor=anchor-vert(16,7,2.5)
                # Counterbore the instep above the screw heads for top-side access.
                for i in (1,2):
                    s=new_foot[i]
                    new_foot[i]=part(s-vert(51,52,5),s.label,s.color)
                new_foot.append(part(vert(19.5,11,2.5).fuse(vert(27,4,4.3)),'Fijación plantar M5',STEEL))
        new_foot.extend([part(plate,'Placa base del pedestal',AL),part(anchor,'Contraplaca plantar',FRAME)])
        new_foot.extend(arot+self.axle(ANKLE,False))
        thigh=self.rails(545,810,'Muslo')
        thigh.children=(*thigh.children,*thigh_harness(self))
        return group(old.label,[group('Pie',new_foot),ankle,self.rails(144,419,'Pantorrilla'),
                                knee,thigh,group('Cadera',hip_parts)])
