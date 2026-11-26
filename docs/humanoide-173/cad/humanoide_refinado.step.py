"""Estudio de carrocería en postura neutra. Milímetros, +Y frontal, +Z arriba.

Los grupos representan volúmenes exteriores y ejes de referencia. Las secciones
de control son editables. Las carcasas todavía requieren vaciado y fijaciones.
"""

from build123d import (
    Box, Color, Compound, Cylinder, Edge, Face, Keep, Plane, Pos, Rot,
    Shell, SlotCenterToCenter, Solid, Wire, extrude,
)


from integracion_h9c import HEAD_HEIGHT, NECK_PATTERN, head as vision_head


HEIGHT = 1730
HIP_X, SHOULDER_X = 112, 236
ANKLE_Z, KNEE_Z, HIP_Z = 85, 480, 875
WRIST_Z, ELBOW_Z, SHOULDER_Z = 845, 1100, 1400
HUB_CLEARANCE = 2

PORCELAIN = "#E5E2D9"
TITANIUM = "#899DA1"
GRAPHITE = "#293944"
RUBBER = "#17252D"
COPPER = "#D88245"


def named(shape, label, paint):
    shape.label = label
    shape.color = paint if isinstance(paint, Color) else Color(paint)
    return shape


def skin(*stations, fullness=0.73):
    """Four tangent Bézier panels between end profiles.

    Each (z, half width, half depth, Y offset) is a control section.
    Interior sections control curvature, endpoints are exact mating sections.
    """
    quadrants = []
    for z, a, b, y in stations:
        u, v = a * fullness, b * fullness
        ring = [
            [(a, y, z), (a, y+v, z), (u, y+b, z), (0, y+b, z)],
            [(0, y+b, z), (-u, y+b, z), (-a, y+v, z), (-a, y, z)],
            [(-a, y, z), (-a, y-v, z), (-u, y-b, z), (0, y-b, z)],
            [(0, y-b, z), (u, y-b, z), (a, y-v, z), (a, y, z)],
        ]
        quadrants.append(ring)
    panels = [Face.make_bezier_surface(list(rows)) for rows in zip(*quadrants)]
    for end in (quadrants[0], quadrants[-1]):
        panels.append(Face(Wire([Edge.make_bezier(*row) for row in end])))
    return Solid(Shell(panels))


def split_at(shape, origin, normal=(0, 0, 1)):
    """Return the positive and negative half of a styling or assembly datum."""
    return shape.split(Plane(origin=origin, z_dir=normal), Keep.BOTH)


def x_drum(x, z, radius, width):
    return Pos(x, 0, z) * Rot(0, 90, 0) * Cylinder(radius, width)


def pivot(x, z, r, width, side, label):
    """Layered joint cover with a recessed central plug, facing outwards."""
    outer_x = x + side * width / 2
    carrier = x_drum(x, z, r, width)
    ring = x_drum(outer_x + side * 1.5, z, r * .84, 3)
    rim = x_drum(outer_x + side * 4.5, z, r * .74, 3)
    bore_radius = r * .40
    rim = rim - x_drum(outer_x + side * 4.5, z, bore_radius, 5)
    plug = x_drum(outer_x + side * 4, z, bore_radius - 0.8, 2)
    return Compound(label=label, children=[
        named(carrier, "tambor", GRAPHITE),
        named(ring, "anillo de contraste", COPPER),
        named(rim, "tapa anular", TITANIUM),
        named(plug, "tapón central rebajado", RUBBER),
    ])


def hinge(x, z, radius, width, side, label):
    """Two fork ears, a central tongue and a through pin, on a common X axis."""
    ear_width = round(width * .15)
    tongue_width = width - 2 * (ear_width + .8)
    pin_radius = round(radius * .32)
    bore = x_drum(x, z, pin_radius + .35, width + 2)

    def strap(center_x, span, extension, end_radius):
        outline = Plane.YZ * SlotCenterToCenter(abs(extension), 2*end_radius, rotation=90)
        blank = Pos(center_x - span/2, 0, z + extension/2) * extrude(outline, amount=span)
        return blank - bore

    ears = [strap(x + s*(width-ear_width)/2, ear_width, radius*1.45, radius*.92)
            for s in (-1, 1)]
    # Open the two pockets all the way to the outside of the fairing. A blind
    # pocket here would leave a thin, disconnected skin over each fork ear.
    pocket_extension = 24
    outer_pockets = [
        strap(x + s*(width-ear_width+pocket_extension)/2, ear_width+pocket_extension,
              radius*1.45, radius*.92) for s in (-1, 1)
    ]
    tongue = strap(x, tongue_width, -radius*1.48, radius*.79)
    # During a quarter-turn the ear tips sweep over the sides of the distal
    # link. Their centres remain above the pin plane in distal coordinates.
    # Relieve each outer cheek below that plane by the end radius plus clearance.
    cheek_start = width/2 - ear_width - .8
    cheek_reliefs = [
        Pos(x + s*(cheek_start+40), 0, z + 50 - radius*.92 - HUB_CLEARANCE)
        * Box(80, 4*radius, 100) for s in (-1, 1)
    ]
    pin = x_drum(x, z, pin_radius, width)
    face_x = x + side*width/2
    bezel = x_drum(face_x+side*1.5, z, radius*.67, 3)
    collar = x_drum(face_x+side*3.7, z, radius*.52, 1.4)
    cap = x_drum(face_x+side*4.8, z, radius*.31, .8)
    assembly = Compound(label=label, children=[
        named(ears[0], "orejeta A", TITANIUM), named(ears[1], "orejeta B", TITANIUM),
        named(tongue, "lengüeta distal", GRAPHITE), named(pin, "pasador transversal", GRAPHITE),
        named(bezel, "bisel exterior", RUBBER), named(collar, "anillo del eje", COPPER),
        named(cap, "cabeza del pasador", TITANIUM),
    ])
    return assembly, outer_pockets, [tongue, *cheek_reliefs]


def head():
    return Pos(0, 0, HEIGHT-HEAD_HEIGHT) * vision_head()


def torso():
    chest = skin((1162, 98, 69, -1), (1228, 151, 101, 7),
                 (1398, 210, 119, 4), (1460, 121, 72, -2))
    # A separate back panel has real recessed ventilation slots.
    rear, front = split_at(chest, (0, -62, 1320), (0, -1, .03))
    vents = [Pos(0, -101, 1297 + 15*i) * Box(74 - 5*abs(i-2), 58, 5)
             for i in range(5)]
    # Back the narrow openings with graphite inserts at their pocket floors.
    vent_floor = Pos(0, -74, 1325) * Box(95, 4, 86)
    grilles = [named(vent & vent_floor, f"fondo de ranura {i+1}", GRAPHITE)
               for i, vent in enumerate(vents)]
    rear = rear - vents
    parts = [named(rear, "panel dorsal con cinco ranuras", TITANIUM)]
    parts.extend(grilles)
    # A recessed, off-centre status window just below the right clavicle.
    slot = Plane.XZ * SlotCenterToCenter(35, 10)
    recess = Pos(63, 130, 1391) * extrude(slot, amount=45)
    front = front - recess
    window = Pos(63, 88, 1391) * extrude(slot, amount=3)
    strip = Plane.XZ * SlotCenterToCenter(19, 3.8)
    parts.extend([named(window, "ventana de estado", RUBBER),
                  named(Pos(63, 89, 1391) * extrude(strip, amount=1), "testigo frontal", COPPER)])
    # A split breastplate and swept band form a shallow V around the sternum.
    right, left = split_at(front, (0, 0, 0), (1, 0, 0))
    for sign, half, suffix in ((-1, left, "I"), (1, right, "D")):
        high, low = split_at(half, (0, 0, 1303), (-sign*.24, -.06, 1))
        band, ribs = split_at(low, (0, 0, 1281), (-sign*.24, -.06, 1))
        wing, center = split_at(high, (sign*3, 0, 0), (sign, 0, 0))
        parts.extend([named(wing, "pectoral " + suffix, PORCELAIN),
                      named(center, "junta del esternón " + suffix, GRAPHITE),
                      named(band, "franja pectoral " + suffix, GRAPHITE),
                      named(ribs, "costado inferior " + suffix, TITANIUM)])
    neck = skin((1460, 42, 36, -2), (1478, 32, 32, -4),
                (1511, 29, 32, -2), (HEIGHT-HEAD_HEIGHT, 35, 34, 2))
    neck_top, neck_base = split_at(neck, (0, 0, 1480))
    # Pilot bores for four tapped M4 seats in the cervical prototype.
    seat_z = HEIGHT - HEAD_HEIGHT
    holes = [Pos(x, y, seat_z - 6) * Cylinder(1.65, 14) for x, y in NECK_PATTERN]
    neck_top = neck_top - holes
    parts.extend([named(neck_base, "base cervical", TITANIUM),
                  named(neck_top, "columna cervical", GRAPHITE)])
    abdomen = skin((964, 79, 56, -4), (1000, 72, 56, -5),
                   (1117, 105, 79, 0), (1162, 98, 69, -1))
    for index, (z, angle) in enumerate(((1010, -.09), (1066, -.12), (1121, -.15))):
        abdomen, plate = split_at(abdomen, (0, 0, z), (0, angle, 1))
        gasket, plate = split_at(plate, (0, 0, z-9), (0, angle, 1))
        parts.extend([named(plate, f"placa abdominal {index+1}", GRAPHITE),
                      named(gasket, f"junta abdominal {index+1}", RUBBER)])
    parts.append(named(abdomen, "costilla subtorácica", GRAPHITE))
    pelvis = skin((HIP_Z-28, 96, 60, 0), (HIP_Z-10, 155, 82, 3),
                  (940, 167, 98, -1), (964, 99, 68, -4))
    sockets = [x_drum(s*HIP_X, HIP_Z, 49 + HUB_CLEARANCE, 118) for s in (-1, 1)]
    belt, lower = split_at(pelvis - sockets, (0, 0, 938), (0, -.12, 1))
    parts.extend([named(belt, "cintura pélvica", GRAPHITE), named(lower, "escudo pélvico", PORCELAIN)])
    # Axles bridge the shoulder gap. The shared bore keeps the neutral assembly interference-free.
    shaft = x_drum(0, SHOULDER_Z, 18, 2*(SHOULDER_X-47))
    for i, part in enumerate(parts):
        if part.bounding_box().min.Z < SHOULDER_Z + 18 and part.bounding_box().max.Z > SHOULDER_Z - 18:
            parts[i] = named(part - shaft, part.label, part.color)
    parts.append(named(shaft, "travesaño de hombros", GRAPHITE))
    return Compound(label="Tronco y pelvis", children=parts)


def fairing(x, lower, upper, controls, split_height, tilt, label, lower_color, upper_color,
            fittings=(), rear_offset=None, bend=None):
    """Shape a limb between two circular keep-outs around the transverse axes."""
    low_z, low_r = lower
    high_z, high_r = upper
    shape = Pos(x, 0, low_z) * skin(*controls)
    keepouts = [x_drum(x, z, radius + HUB_CLEARANCE, 220) for z, radius in (lower, upper)]
    shape = shape - keepouts
    if bend is not None:
        # The inner faces of the two covers open into a V at the bending joint.
        # Positive Z points into a proximal link and negative Z into a distal one.
        bend_z, flex_y = bend
        toward_link = 1 if bend_z == low_z else -1
        root_clearance = 8
        shape, _ = split_at(shape, (x, 0, bend_z + toward_link*root_clearance),
                            (0, -1.12*flex_y, toward_link))
    if fittings:
        shape = shape - list(fittings)
    top, rest = split_at(shape, (x, 0, low_z + split_height), (0, tilt, 1))
    seam, bottom = split_at(rest, (x, 0, low_z + split_height-5), (0, tilt, 1))
    panels = [named(bottom, "sección distal", lower_color), named(seam, "junta", RUBBER)]
    if rear_offset is not None:
        rear, top = split_at(top, (x, rear_offset, (low_z + high_z)/2), (0, -1, -.085))
        panels.append(named(rear, "panel posterior", TITANIUM))
    panels.append(named(top, "sección proximal", upper_color))
    return Compound(label=label, children=panels)


def foot(x):
    sole = skin((0, 54, 121, 40), (2, 62, 129, 42),
                (12, 65, 130, 42), (18, 64, 128, 42))
    instep = skin((18, 64, 128, 42), (26, 63, 128, 41),
                  (43, 61, 111, 28), (ANKLE_Z-9, 45, 50, -9))
    instep = instep - x_drum(0, ANKLE_Z, 39 + HUB_CLEARANCE, 160)
    toe, body = split_at(instep, (0, 74, 24), (0, 1, .50))
    return Pos(x, 0, 0) * Compound(label="Pie", children=[
        named(sole, "suela continua", RUBBER), named(toe, "puntera", TITANIUM),
        named(body, "empeine y talón", PORCELAIN),
    ])


def hand(x, side):
    # The palm faces the thigh. Fingers spread along Y and curl towards the palm.
    palm = skin((-85, 15, 35, 0), (-74, 24, 43, 0),
                (-44, 23, 41, -1), (-14, 16, 26, -1))
    palm = palm - x_drum(0, 0, 24 + HUB_CLEARANCE, 80)
    parts = [named(palm, "dorso y palma", TITANIUM)]
    for name, y, length in (("índice", 26, 63), ("medio", 8, 72),
                            ("anular", -10, 66), ("meñique", -27, 51)):
        finger = skin((0, 4, 5, 0), (4, 8.8, 7.7, 0),
                      (length-6, 9.5, 8.3, 0), (length, 7, 7.8, 0))
        # Rotate about the root of the finger, keeping the four knuckles on the palm.
        finger = Pos(0, y, -85) * Rot(0, side*12, 0) * Pos(0, 0, -length) * finger
        finger = finger - palm
        proximal, tip = split_at(finger, (0, y, -85-length*.49))
        parts.extend([named(proximal, name + " proximal", GRAPHITE),
                      named(tip, name + " distal", PORCELAIN)])
    thumb = skin((0, 5, 6, 0), (3, 9, 9, 0), (53, 11, 11, 0), (58, 8, 9, 0))
    thumb = Pos(-side*7, 32, -45) * Rot(25, side*24, 0) * Pos(0, 0, -58) * thumb
    parts.append(named(thumb - palm, "pulgar opuesto", GRAPHITE))
    return Pos(x, 0, WRIST_Z) * Compound(label="Mano", children=parts)


def limbs(sign, side):
    x = sign * HIP_X
    ankle, knee, hip = (ANKLE_Z, 39), (KNEE_Z, 47), (HIP_Z, 49)
    knee_joint, thigh_pockets, shin_pockets = hinge(x, *knee, 103, sign, "Rodilla")
    shin_span, thigh_span = KNEE_Z-ANKLE_Z, HIP_Z-KNEE_Z
    shin = fairing(x, ankle, knee, [
        (15, 30, 33, -2), (100, 35, 46, -8), (shin_span-95, 68, 77, -16),
        (shin_span-23, 46, 45, 0)], 85, -.38, "Pantorrilla", GRAPHITE, PORCELAIN,
        fittings=shin_pockets, rear_offset=-36, bend=(KNEE_Z, -1))
    thigh = fairing(x, knee, hip, [
        (22, 44, 45, 0), (90, 60, 69, 12), (thigh_span-90, 77, 86, -4),
        (thigh_span-28, 50, 53, 0)], 89, .43, "Muslo", TITANIUM, PORCELAIN,
        fittings=thigh_pockets, rear_offset=-47, bend=(KNEE_Z, -1))
    leg = Compound(label="Pierna " + side, children=[
        foot(x), pivot(x, *ankle, 79, sign, "Tobillo"), shin,
        knee_joint, thigh, pivot(x, *hip, 105, sign, "Cadera"),
    ])
    x = sign * SHOULDER_X
    wrist, elbow, shoulder = (WRIST_Z, 24), (ELBOW_Z, 35), (SHOULDER_Z, 55)
    elbow_joint, upper_arm_pockets, forearm_pockets = hinge(x, *elbow, 73, sign, "Codo")
    fore_span, arm_span = ELBOW_Z-WRIST_Z, SHOULDER_Z-ELBOW_Z
    forearm = fairing(x, wrist, elbow, [
        (11, 20, 24, 0), (55, 25, 35, 3), (fore_span-56, 57, 57, 2),
        (fore_span-14, 34, 35, 0)], 56, -.42, "Antebrazo", GRAPHITE, PORCELAIN,
        fittings=forearm_pockets, rear_offset=-31, bend=(ELBOW_Z, 1))
    upper = fairing(x, elbow, shoulder, [
        (15, 31, 35, 0), (60, 39, 49, -4), (arm_span-70, 52, 58, 4),
        (arm_span-28, 39, 40, 0)], 63, .38, "Brazo", TITANIUM, PORCELAIN,
        fittings=upper_arm_pockets, bend=(ELBOW_Z, 1))
    cap = Pos(x, 0, SHOULDER_Z) * skin((13, 48, 51, 0), (28, 55, 62, 1),
                                       (62, 50, 49, -1), (71, 24, 25, -3))
    cap = cap - x_drum(x, SHOULDER_Z, shoulder[1]+HUB_CLEARANCE, 140)
    arm = Compound(label="Brazo " + ("derecho" if sign > 0 else "izquierdo"), children=[
        hand(x, sign), pivot(x, *wrist, 43, sign, "Muñeca"), forearm,
        elbow_joint, upper,
        pivot(x, *shoulder, 94, sign, "Hombro"), named(cap, "casquete deltoideo", PORCELAIN),
    ])
    return leg, arm


def gen_step():
    return Compound(label="Humanoide 1730 mm - estudio de carrocería", children=[
        torso(), head(), *limbs(-1, "izquierda"), *limbs(1, "derecha"),
    ])
