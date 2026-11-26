from math import cos, radians, sin

from build123d import (
    Align, BuildPart, BuildSketch, Color, Cylinder, Locations, Mode,
    RectangleRounded, extrude,
)

# Coordinates in millimeters. The opening is centered on the motor output.
PLATE_SIZE = (104.0, 104.0, 8.5)
ROTOR_CLEARANCE_DIAMETER = 59.0
MOTOR_BOLT_CIRCLE = 68.0
# Clocking from the axes of the eight holes in the official STEP, front face.
FIRST_MOTOR_HOLE_DEG = 10.05
MOTOR_HOLES = [
    (MOTOR_BOLT_CIRCLE / 2 * cos(radians(FIRST_MOTOR_HOLE_DEG + i * 45)),
     MOTOR_BOLT_CIRCLE / 2 * sin(radians(FIRST_MOTOR_HOLE_DEG + i * 45)))
    for i in range(8)
]
BENCH_HOLES = [(x, y) for x in (-44.0, 44.0) for y in (-44.0, 44.0)]
BOTTOM_ON_PLANE = (Align.CENTER, Align.CENTER, Align.MIN)


def gen_step():
    width, depth, thickness = PLATE_SIZE
    with BuildPart() as plate:
        with BuildSketch():
            RectangleRounded(width, depth, 8.0)
        extrude(amount=thickness)
        Cylinder(ROTOR_CLEARANCE_DIAMETER / 2, thickness,
                 align=BOTTOM_ON_PLANE, mode=Mode.SUBTRACT)
        with Locations(*MOTOR_HOLES, *BENCH_HOLES):
            Cylinder(1.7, thickness, align=BOTTOM_ON_PLANE, mode=Mode.SUBTRACT)

    plate.part.label = "Plantilla de interfaz frontal AK60-39 V3.0 en PLA"
    plate.part.color = Color("#2A8199")
    return plate.part
