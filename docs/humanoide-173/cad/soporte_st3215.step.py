from build123d import (
    Align, BuildPart, BuildSketch, Color, Cone, Cylinder, Locations, Mode,
    Rectangle, RectangleRounded, extrude,
)

# Coordinates in millimeters. The rear idler and output shaft share X=Y=0.
PLATE_SIZE = (75.0, 44.0, 4.2)
PLATE_CENTER_X = 12.5
SERVO_HOLES = [(x, y) for x in (8.30, 32.75) for y in (-10.25, 10.25)]
BENCH_HOLES = [(x, y) for x in (-20.0, 45.0) for y in (-16.0, 16.0)]
SERVO_CLEARANCE_DIAMETER = 2.4
HEAD_DIAMETER = 4.2
REAR_IDLER_CLEARANCE = 20.4
CONNECTOR_OPENING_CENTER_X = 23.45
CONNECTOR_OPENING = (13.7, 19.6)
BOTTOM_ON_PLANE = (Align.CENTER, Align.CENTER, Align.MIN)


def gen_step():
    width, depth, thickness = PLATE_SIZE
    with BuildPart() as plate:
        with BuildSketch():
            with Locations((PLATE_CENTER_X, 0)):
                RectangleRounded(width, depth, 5.0)
        extrude(amount=thickness)

        Cylinder(REAR_IDLER_CLEARANCE / 2, thickness, align=BOTTOM_ON_PLANE,
                 mode=Mode.SUBTRACT)
        with BuildSketch():
            with Locations((CONNECTOR_OPENING_CENTER_X, 0)):
                Rectangle(*CONNECTOR_OPENING)
        extrude(amount=thickness, mode=Mode.SUBTRACT)
        with Locations(*SERVO_HOLES):
            Cylinder(SERVO_CLEARANCE_DIAMETER / 2, thickness,
                     align=BOTTOM_ON_PLANE, mode=Mode.SUBTRACT)
            Cone(HEAD_DIAMETER / 2, SERVO_CLEARANCE_DIAMETER / 2,
                 (HEAD_DIAMETER - SERVO_CLEARANCE_DIAMETER) / 2,
                 align=BOTTOM_ON_PLANE, mode=Mode.SUBTRACT)
        with Locations(*BENCH_HOLES):
            Cylinder(1.7, thickness, align=BOTTOM_ON_PLANE, mode=Mode.SUBTRACT)

    plate.part.label = "Soporte de banco en PLA para ST3215"
    plate.part.color = Color("#2A8199")
    return plate.part
