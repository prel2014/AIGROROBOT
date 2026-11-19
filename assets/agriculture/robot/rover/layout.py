"""Geometría nominal en mm, X al frente, Y a la izquierda."""
from dataclasses import dataclass
from math import cos, dist, hypot, radians, sin

DECK = (1480, 890)
BASKET_FRACTION = .60
POST_X = -360
SHOULDER = (POST_X + 75, 319, 1080)
BASKET_FRONT = DECK[0]/2
BASKET_BACK = BASKET_FRONT - DECK[0]*BASKET_FRACTION
BASKET_FLOOR = 458
RAMP_END = 330
EYE = (-680, 220, 1720)
EYE_YAW = 22.0
EYE_PITCH_DOWN = 43.0
LIDAR = (-620, 0, 1900)
ROVER_TRAVEL_DEG = {
    "Left": (-44., -23., 83., -75., -86., 0.),
    "Right": (-4., -32., 110., -68., -86., 0.),
}
LEFT_OPERATING_LIMITS_DEG = {
    "shoulder_pan_joint": (-45., 60.),
    "shoulder_lift_joint": (-60., 0.),
    "elbow_joint": (-10., 85.),
}


def eye_axis(side):
    yaw, pitch = radians(side*EYE_YAW), radians(EYE_PITCH_DOWN)
    return cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch)


def eye_housing_angles(name):
    if name.startswith(("Visor_RGBD_", "Ojo_RGBD_")):
        return EYE_YAW if name.endswith("_izq") else -EYE_YAW, EYE_PITCH_DOWN
    return None


@dataclass(frozen=True)
class Tube:
    name: str
    start: tuple
    end: tuple
    width: float
    wall: float

    @property
    def center(self):
        return tuple((a+b)/2 for a,b in zip(self.start, self.end))

    @property
    def length(self):
        return dist(self.start, self.end)

    @property
    def axes(self):
        z = tuple((b-a)/self.length for a,b in zip(self.start,self.end))
        horizontal = hypot(z[0], z[1])
        x = (-z[1]/horizontal, z[0]/horizontal, 0.) if horizontal else (1., 0., 0.)
        y = (z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0])
        return x, y, z

    @property
    def mass_kg(self):
        return self.length*(self.width**2-(self.width-2*self.wall)**2)*7850e-9


def rear_frame():
    tubes = [
        Tube("Parachoques", (-735,-425,480), (-735,425,480), 80, 4),
        Tube("Travesano_columnas", (-360,-345,905), (-360,345,905), 65, 4),
        Tube("Travesano_mastil", (-620,-345,530), (-620,345,530), 65, 4),
    ]
    for sign, side in [(1,"izq"), (-1,"der")]:
        y = sign*345
        tubes.extend([
            Tube("Larguero_"+side, (-735,y,480), (-232,y,480), 75, 4),
            Tube("Riostra_hombro_"+side, (-690,y,515), (-395,y,910), 55, 4),
            Tube("Riostra_mastil_"+side, (-620,y,545), (-620,sign*15,1150), 45, 3),
        ])
    return tubes


@dataclass(frozen=True)
class Plate:
    """Prisma convexo con perfil en XZ (antihorario) extruido en Y; cotas en mm."""
    name: str
    outline: tuple
    y: float
    width: float
    color: str
    density: float = 0.0  # Masa adicional en kg/m³; las piezas de base ya van en la tara.

    def vertices(self):
        return [(x, self.y + side*self.width/2, z)
                for side in (-1, 1) for x, z in self.outline]

    def added_mass_and_center(self):
        edges = list(zip(self.outline, self.outline[1:] + self.outline[:1]))
        cross = [x*v - u*z for (x,z), (u,v) in edges]
        area2 = sum(cross)
        if area2 <= 0:
            raise ValueError(f"Perfil invertido o degenerado: {self.name}")
        cx = sum((a[0]+b[0])*c for (a,b), c in zip(edges,cross))/(3*area2)
        cz = sum((a[1]+b[1])*c for (a,b), c in zip(edges,cross))/(3*area2)
        return self.density * self.width * area2/2 * 1e-9, (cx, self.y, cz)


def plates():
    parts = []
    back, front, floor = BASKET_BACK, BASKET_FRONT, BASKET_FLOOR
    inner = DECK[1]/2 - 16
    for sign, side in ((1, "izq"), (-1, "der")):
        # La pata conserva la unión al bastidor. La corona adelanta la brida de cada UR10e.
        foot, head = POST_X, SHOULDER[0]
        parts.append(Plate("Columna_"+side, ((foot-105,440),(foot+105,440),
                           (head+105,1020),(head-105,1020)), sign*SHOULDER[1],170,"#468675"))
        # Borde proximal bajo para depositar por el lado que trabaja junto a la planta.
        sections = ((back, 100, 622, 622), (100, 340, 622, 710), (340, front, 710, 710))
        for idx,(a,b,za,zb) in enumerate(sections):
            parts.append(Plate(f"Lateral_{side}_{idx}", ((a,floor),(b,floor),(b,zb),(a,za)),
                               sign*(inner+8),16,"#D99842"))
        a, b = back+16, RAMP_END
        parts.append(Plate("Receptor_"+side, ((a,591),(b,462),(b,474),(a,603)),
                           sign*(inner+1)/2, inner-1, "#44695B", 650))
        # Tres caras convexas por guardabarros conservan el paso circular de la rueda en PhysX.
        for axle,x in (("frontal",550), ("trasero",-550)):
            lower = ((-218,375),(-126,477),(126,477),(218,375))
            for idx,((u,z),(v,h)) in enumerate(zip(lower,lower[1:])):
                parts.append(Plate(f"Guardabarros_{axle}_{side}_{idx}",
                                   ((x+u,z),(x+v,h),(x+v,h+5),(x+u,z+5)),
                                   sign*510,130,"#42776B",2700))
            parts.append(Plate(f"Pie_guardabarros_{axle}_{side}",
                               ((x-27,440),(x+27,440),(x+27,477),(x-27,477)),
                               sign*437,10,"#344D49",2700))
            parts.append(Plate(f"Puente_guardabarros_{axle}_{side}",
                               ((x-27,477),(x+27,477),(x+27,482),(x-27,482)),
                               sign*438.5,13,"#344D49",2700))
    a, b = RAMP_END, front-16
    parts.append(Plate("Lecho_canasta", ((a,floor),(b,floor),(b,474),(a,474)),
                       0,2*inner,"#44695B",650))
    return parts

def shapes():
    rows = []

    def part(name, shape, xyz, color):
        rows.append((name, shape, xyz, color))

    def box(name, size, xyz, color):
        part(name, ("box", size), xyz, color)

    def cylinder_x(r, h):
        return "X", (r, h)

    def cylinder_y(r, h):
        return "Y", (r, h)

    def cylinder_z(r, h):
        return "Z", (r, h)

    frame, shell, rubber = "#284346", "#468675", "#30363B"
    metal, basket, sensor = "#B2C3BE", "#E3AC53", "#225777"
    bridge_x = EYE[0]-60

    box("Chasis", (1500, 860, 155), (0, 0, 327.5), frame)
    box("Cubierta", (*DECK, 35), (0, 0, 422.5), shell)
    for x, axle in [(550, "frontal"), (-550, "trasero")]:
        part("Eje_" + axle, cylinder_y(45, 1020), (x, 0, 230), metal)
        for y, side in [(510, "izq"), (-510, "der")]:
            suffix = axle + "_" + side
            part("Neumatico_" + suffix, cylinder_y(230, 130),
                 (x, y, 230), rubber)
            part("Llanta_" + suffix, cylinder_y(137, 134),
                 (x, y, 230), metal)
            part("Cubo_" + suffix, cylinder_y(62, 140),
                 (x, y, 230), frame)
    for sign, side in [(1, "izq"), (-1, "der")]:
        x, y, z = SHOULDER
        y *= sign
        box("Placa_" + side, (240, 196, 48), (x, y, 1044), frame)
        part("Apoyo_UR10e_" + side, cylinder_z(95, 12), (x, y, z-6), metal)
        ex, ey, ez = EYE
        axis = eye_axis(sign)
        def behind_lens(distance):
            return tuple(v-distance*a for v,a in zip((ex, sign*ey, ez), axis))
        box("Anclaje_visor_"+side, (36,28,62), (bridge_x+5,sign*(ey-46),ez+28), frame)
        box("Visor_RGBD_" + side, (90, 121, 66), behind_lens(65), frame)
        part("Ojo_RGBD_" + side, cylinder_x(23, 18), behind_lens(11), sensor)
    lx, ly, lz = LIDAR
    # El voladizo central retrasa los ojos para despejar la vista entre los hombros.
    box("Soporte_travesano", (lx-bridge_x, 48, 40), ((lx+bridge_x)/2, 0, EYE[2]), frame)
    box("Travesano_camaras", (48, 2*EYE[1], 40), (bridge_x, 0, EYE[2]), frame)
    box("Mastil", (62, 62, 1376), (lx, ly, 1128), frame)
    part("Carcasa_lidar", cylinder_z(60, 60), (lx, ly, lz-54), sensor)
    part("Tapa_lidar", cylinder_z(62, 9), (lx, ly, lz+32), metal)
    box("IMU_interna", (58, 44, 25), (0, 0, 360), sensor)
    length, width = DECK[0]*BASKET_FRACTION, DECK[1]
    front = BASKET_FRONT
    center = front - length/2
    box("Fondo_canasta", (length, width, 18), (center, 0, 449), basket)
    for x, end, top in [(BASKET_BACK+8, "trasero", 622), (front-8, "frontal", 710)]:
        height = top - BASKET_FLOOR
        box("Testero_" + end, (16, width-32, height), (x, 0, BASKET_FLOOR+height/2), basket)
    return rows
