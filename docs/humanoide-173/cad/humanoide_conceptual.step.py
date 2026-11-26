from build123d import Box, Color, Compound, Cylinder, Pos, Rot, Sphere

# Envelope proportions and joint-center locations. Millimeters, +Z up, +Y front.
HEIGHT = 1730
HIP_X = 112
SHOULDER_X = 236
ANKLE_Z = 85
KNEE_Z = 480
HIP_Z = 875
SHOULDER_Z = 1400
ELBOW_Z = 1100
WRIST_Z = 845
BODY = Color("#567287")
JOINT = Color("#1F374C")
MODULE = Color("#4599A9")


def gen_step():
    shapes = []

    def add(shape, label, color=BODY):
        shape.label = label
        shape.color = color
        shapes.append(shape)

    def segment(x, bottom, top, radius, label):
        add(Pos(x, 0, (bottom + top) / 2) * Cylinder(radius, top - bottom),
            label, MODULE)

    def joint(x, height, radius, name):
        shape = Pos(x, 0, height) * Rot(0, 90, 0) * Cylinder(radius, 2 * radius)
        add(shape, name, JOINT)

    add(Pos(0, 0, 907) * Box(305, 160, 108), "envolvente de pelvis")
    add(Pos(0, 0, 998) * Cylinder(66, 74), "envolvente de unión lumbar", JOINT)
    add(Pos(0, 0, 1113) * Box(240, 148, 156), "envolvente del abdomen")
    add(Pos(0, 0, 1324) * Box(364, 194, 266), "envolvente del tórax")
    add(Pos(0, 0, 1493.5) * Cylinder(36, 73), "envolvente de cuello", JOINT)
    add(Pos(0, 0, HEIGHT - 101) * Sphere(101), "envolvente de cabeza")

    for sign, side in ((-1, "izquierdo"), (1, "derecho")):
        leg_x, arm_x = sign * HIP_X, sign * SHOULDER_X
        add(Pos(leg_x, 42, 22) * Box(120, 246, 44), f"pie {side}")
        joint(leg_x, ANKLE_Z, 44, f"centro de tobillo {side}")
        segment(leg_x, ANKLE_Z + 41, KNEE_Z - 37, 40, f"pantorrilla {side}")
        joint(leg_x, KNEE_Z, 45, f"centro de rodilla {side}")
        segment(leg_x, KNEE_Z + 40, HIP_Z - 38, 51, f"muslo {side}")
        joint(leg_x, HIP_Z, 49, f"centro de cadera {side}")
        joint(arm_x, SHOULDER_Z, 62, f"centro de hombro {side}")
        segment(arm_x, ELBOW_Z + 33, SHOULDER_Z - 49, 42, f"brazo {side}")
        joint(arm_x, ELBOW_Z, 37, f"centro de codo {side}")
        segment(arm_x, WRIST_Z + 25, ELBOW_Z - 30, 33, f"antebrazo {side}")
        joint(arm_x, WRIST_Z, 29, f"centro de muñeca {side}")
        add(Pos(arm_x, 0, 777) * Box(65, 42, 100), f"envolvente de mano {side}")

    return Compound(label="Humanoide 1730 mm - envolventes conceptuales", children=shapes)
