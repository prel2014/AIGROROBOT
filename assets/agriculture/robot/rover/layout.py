"""Geometría nominal en mm, X al frente, Y a la izquierda."""
from dataclasses import dataclass
from math import dist, hypot

DECK = (1480, 890)
BASKET_FRACTION = .60
SHOULDER = (-360, 345, 1080)
EYE = (-560, 409, 1760)
LIDAR = (-620, 0, 1900)
LEFT_OPERATING_LIMITS_DEG = {
    "shoulder_pan_joint": (-45., 60.),
    "shoulder_lift_joint": (-60., 0.),
    "elbow_joint": (-10., 85.),
}


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
        box("Columna_" + side, (210, 170, 580), (x, y, 730), shell)
        box("Placa_" + side, (240, 196, 48), (x, y, 1044), frame)
        part("Apoyo_UR10e_" + side, cylinder_z(95, 12), (x, y, z-6), metal)
        ex, ey, ez = EYE
        ey *= sign
        box("Visor_RGBD_" + side, (90, 121, 66), (ex-60, ey, ez), frame)
        part("Ojo_RGBD_" + side, cylinder_x(23, 18), (ex-8, ey, ez), sensor)
    lx, ly, lz = LIDAR
    # El travesaño se monta directamente sobre el mástil, detrás de los hombros.
    box("Travesano_camaras", (48, 2*EYE[1], 40), (lx, 0, EYE[2]), frame)
    box("Mastil", (62, 62, 1376), (lx, ly, 1128), frame)
    part("Carcasa_lidar", cylinder_z(60, 60), (lx, ly, lz-54), sensor)
    part("Tapa_lidar", cylinder_z(62, 9), (lx, ly, lz+32), metal)
    box("IMU_interna", (58, 44, 25), (0, 0, 360), sensor)
    length, width = DECK[0]*BASKET_FRACTION, DECK[1]
    front = DECK[0]/2
    center = front - length/2
    box("Fondo_canasta", (length, width, 18), (center, 0, 449), basket)
    for sign, side in [(1, "izq"), (-1, "der")]:
        box("Lateral_" + side, (length, 16, 252), (center, sign*(width/2-8), 584), basket)
    for x, end in [(front-length+8, "trasero"), (front-8, "frontal")]:
        box("Testero_" + end, (16, width-32, 252), (x, 0, 584), basket)
    return rows
