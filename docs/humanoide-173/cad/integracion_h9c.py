"""Cabeza H9c. Milímetros, +Y hacia delante. Z=0 en el asiento cervical.

Envolvente comercial basada en el plano EZVIZ CS-H9c-R100-8H33WKFL.
Las formas internas, ejes ópticos y dimensiones del soporte original son
aproximaciones. El patrón de la plantilla física se incorpora en OEM_HOLES.
"""
from math import cos, radians, sin
from build123d import (
    Axis, Box, Color, Compound, Cylinder, Plane, Polygon, Pos, RectangleRounded,
    Rot, SlotCenterToCenter, Sphere, extrude, fillet, loft,
)

HEAD_HEIGHT = 198
CAMERA_WIDTH, CAMERA_DEPTH, CAMERA_HEIGHT = 116, 142.4, 145
CAMERA_Y, CAMERA_Z = 14, 29
FRONT = 52
SEAT_Y = CAMERA_Y + FRONT - CAMERA_DEPTH
TOP = CAMERA_Z + CAMERA_HEIGHT
PIVOT_Z = 38
SEAM_Z = 95
NECK_PATTERN = [(x, y) for x in (-17, 17) for y in (-14, 14)]
# (x, z relative to TOP), measured on the rear wall-mount template.
OEM_HOLES = ()
OEM_HOLE_D = 4.5
CABLE_WIDTH, CABLE_HEIGHT = 30, 24
WHITE, METAL, DARK = "#E5E2D9", "#879CA0", "#23343E"
BLACK, GLASS, LAMP = "#182327", "#244B59", "#BFCCC9"


def colored(shape, label, color):
    shape.label, shape.color = label, Color(color)
    return shape


def rounded_block(x, y, z, width, depth, height, radius):
    """Rounded XY profile extruded upward from z."""
    return Pos(x, y, z) * extrude(RectangleRounded(width, depth, radius), amount=height)


def carrier():
    base = rounded_block(0, -27, 0, 74, 114, 8, 22)
    # Two rear ribs leave the centre open for the factory pigtail. Their
    # forward edges retreat behind the optical and pan envelope as they rise.
    outline = [(-8, 6), (-75, 6), (SEAT_Y-9, TOP+12), (SEAT_Y, TOP+12),
               (SEAT_Y, TOP-54), (-62, 49), (-46, 26)]
    rib_profile = Plane.YZ * Polygon(*outline, align=None)
    ribs = [Pos(x, 0, 0) * extrude(rib_profile, amount=9, dir=(1, 0, 0))
            for x in (-33, 24)]
    # Flat vertical blank for the EZVIZ wall bracket. Transfer the original
    # template once available. The lower U-slot remains accessible from below.
    pad = Pos(0, SEAT_Y-5, TOP-21) * Box(92, 10, 66)
    pad = fillet(pad.edges().filter_by(Axis.Y), radius=6)
    cable = Pos(0, SEAT_Y-5, TOP-54+CABLE_HEIGHT/2-1) * Box(CABLE_WIDTH, 18, CABLE_HEIGHT+2)
    pad = pad - cable
    for x, z_offset in OEM_HOLES:
        pad = pad - Pos(x, SEAT_Y-5, TOP+z_offset) * Rot(90, 0, 0) * Cylinder(OEM_HOLE_D/2, 16)
    # Shorter than the front of the lenses. The crown contacts the back ribs
    # only and leaves the upper surface of the purchased camera exposed below.
    crown = loft([
        Pos(0, -18, TOP+8) * RectangleRounded(134, 142, 44),
        Pos(0, -18, HEAD_HEIGHT-8) * RectangleRounded(139, 143, 46),
        Pos(0, -21, HEAD_HEIGHT) * RectangleRounded(110, 124, 40),
    ], ruled=True)
    recess = loft([
        Pos(0, -12, TOP+7) * RectangleRounded(124, 123, 42),
        Pos(0, -17, HEAD_HEIGHT-4) * RectangleRounded(103, 107, 37),
    ], ruled=True)
    crown = crown - recess
    frame = base.fuse(*ribs, pad, crown)
    for x, y in NECK_PATTERN:
        frame = frame - Pos(x, y, 4) * Cylinder(2.25, 12)
    return colored(frame, "Bastidor cervical H9c - plantilla pendiente", METAL)


def disk_y(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(-90, 0, 0) * Cylinder(radius, length)


def optics(upper, barrel=1):
    """Flat optical face in its own XZ frame, including separate unglazed emitters."""
    ring = 9 if upper else 10
    parts = [colored(disk_y(0, barrel/2, 0, ring, barrel), "Aro de lente", BLACK),
             colored(disk_y(0, barrel+.5, 0, ring-2.4, 1), "Lente", GLASS)]
    lamps = [(-19, 0), (19, 0)] if upper else [(-13, 12), (13, 12), (-13, -12), (13, -12)]
    for i, (x, z) in enumerate(lamps, 1):
        parts.append(colored(disk_y(x, .35, z, 3.2, .7), f"Iluminador {i}", LAMP))
    return parts


def camera(pan=0, tilt=0):
    # Local Z=0 at the camera's lowest point. Folded antennas define the 116 mm width.
    upper = Pos(0, 0, (CAMERA_HEIGHT+SEAM_Z)/2) * Cylinder(50, CAMERA_HEIGHT-SEAM_Z)
    window = extrude(Plane.XZ * SlotCenterToCenter(41, 20), amount=20, dir=(0, 1, 0))
    window_pose = Pos(0, 42, 120) * Rot(-15, 0, 0)
    upper = upper - window_pose * window
    # Make the window follow the front curvature. Optical seats are flat and
    # recessed, with separate lenses oriented 15 degrees down from horizontal.
    insert = window_pose * window & (Pos(0, 0, 120) * Cylinder(49.8, 44))
    seat = 43.8
    optical_pose = Pos(0, seat, 120) * Rot(-15, 0, 0)
    cutter = [disk_y(x, 6, z, r, 12) for x,z,r in [(0,0,9),(-19,0,3.2),(19,0,3.2)]]
    insert = insert - [optical_pose * cut for cut in cutter]
    # The overall depth datum is the nose of the upper lens, 52 mm ahead of
    # the pan axis. The seated barrel extends from the recess to that datum.
    barrel_length = (FRONT - seat - 6.6*sin(radians(15))) / cos(radians(15)) - 1
    fixed = [colored(upper, "Cuerpo superior", WHITE), colored(insert, "Ventana panoramica", BLACK),
             *[optical_pose * p for p in optics(True, barrel_length)]]
    for sign in (-1, 1):
        # Pivots and blades are fixed to the upper body, independent of pan.
        pivot = Pos(sign*50, 0, 119) * Rot(0, 90, 0) * Cylinder(6, 6)
        pivot = pivot - Pos(0, 0, 120) * Cylinder(50, 44)
        blade = Pos(sign*(CAMERA_WIDTH/2-2.5), 0, 86) * Box(5, 11, 76)
        blade = fillet(blade.edges().filter_by(Axis.X), radius=4)
        fixed.extend([colored(pivot, "Pivote de antena", METAL), colored(blade, "Antena plegada", WHITE)])
    # Rear mounting bracket envelope. Its vertical seat is the documented depth datum.
    rear_y = FRONT - CAMERA_DEPTH
    original_pad = Pos(0, rear_y+4, CAMERA_HEIGHT-24) * Box(66, 8, 48)
    original_pad = fillet(original_pad.edges().filter_by(Axis.Y), radius=4)
    arm = rounded_block(0, (rear_y+8-38)/2, CAMERA_HEIGHT-14, 48, -38-(rear_y+8), 14, 5)
    arm = arm - Pos(0, 0, CAMERA_HEIGHT-7) * Cylinder(50, 18)
    fixed.append(colored(original_pad.fuse(arm), "Soporte original aproximado", WHITE))

    # Rounded pan shell and exposed tilting insert. The sphere/cheeks are
    # separated by real side clearances even at nonzero pan and tilt.
    sphere = Pos(0, 0, 50) * Sphere(49.5)
    straight = Pos(0, 0, 72) * Cylinder(49.5, 44)
    yoke = sphere.fuse(straight) & (Pos(0, 0, 47) * Box(106, 106, 94))
    yoke = yoke - Pos(0, 0, 39) * Box(55, 112, 79)
    # The tilt cartridge is contained in a sphere, so its envelope is invariant
    # under tilt. Only the front optical disks need separate clearance checks.
    drum = (Pos(0, 0, PIVOT_Z) * Sphere(PIVOT_Z)) & (Pos(0,0,PIVOT_Z) * Box(53, 90, 90))
    drum = drum - Pos(0, 52, PIVOT_Z) * Box(66, 42, 91)
    tilt_pose = Pos(0, 0, PIVOT_Z) * Rot(tilt, 0, 0) * Pos(0, 0, -PIVOT_Z)
    lower_optics = [Pos(0, 31, PIVOT_Z) * part for part in optics(False)]
    rotor = Compound(label="Modulo panoramico movil", children=[
        colored(yoke, "Horquilla giratoria EZVIZ", WHITE),
        tilt_pose * Compound(label="Teleobjetivo basculante", children=[
            colored(drum, "Tambor optico", BLACK), *lower_optics]),
    ])
    collar = colored(Pos(0, 0, 94.5) * Cylinder(46, 1), "Junta de giro", DARK)
    return Pos(0, CAMERA_Y, CAMERA_Z) * Compound(label="EZVIZ H9c R100 - envolvente", children=[
        *fixed, collar, Rot(0, 0, pan) * rotor])


def head():
    return Compound(label="Cabeza de vision EZVIZ H9c", children=[carrier(), camera()])
