"""Ensamble del chasis en milímetros."""
from pathlib import Path
from runpy import run_path

from build123d import Axis, Box, Color, Compound, Cylinder, Face, Plane, Pos, Rot, Shell, Solid, Wire


def wheel_solid(component):
    if component.profile:
        wire = Wire.make_polygon([(r, y, 0) for y, r in component.profile], close=True)
        return Solid.revolve(Face(wire), 360, Axis.Y)
    faces = [Face(Wire.make_polygon([component.vertices[k] for k in face], close=True))
             for face in component.faces]
    return Solid(Shell(faces))


def gen_step():
    layout = run_path(str(Path(__file__).with_name("layout.py")))
    parts = []
    head = {"HeadPan": [], "HeadTilt": []}
    for name, (kind, size), xyz, color in layout["shapes"]():
        if kind == "box":
            solid = Box(*size)
        else:
            angles = {"X": (0, 90, 0), "Y": (90, 0, 0), "Z": (0, 0, 0)}
            solid = Rot(*angles[kind]) * Cylinder(*size)
        angles = layout["eye_housing_angles"](name)
        if angles:
            yaw, pitch = angles
            solid = Rot(0, 0, yaw) * Rot(0, pitch, 0) * solid
        item = Pos(*xyz) * solid
        item.label = name
        item.color = Color(color)
        link = layout["head_link"](name)
        (head[link] if link else parts).append(item)
    for plate in layout["plates"]():
        lower_side = plate.vertices()[:len(plate.outline)]
        face = Face(Wire.make_polygon(lower_side, close=True))
        item = Solid.extrude(face, (0, plate.width, 0))
        item.label, item.color = plate.name, Color(plate.color)
        parts.append(item)
    for tube in layout["rear_frame"]():
        inner = tube.width-2*tube.wall
        hollow = Box(tube.width,tube.width,tube.length) - Box(inner,inner,tube.length+2)
        x, _, z = tube.axes
        item = Plane(origin=tube.center,x_dir=x,z_dir=z).location * hollow
        item.label, item.color = tube.name, Color("#53736B")
        parts.append(item)
    for side in (1, -1):
        components = []
        for spec in layout["wheel_meshes"](side):
            item = wheel_solid(spec)
            item.label, item.color = spec.name, Color(spec.color)
            components.append(item)
        wheel = Compound(children=components)
        for name, center in layout["wheel_centers"]():
            if center[1]*side > 0:
                copy = Pos(*center) * wheel
                copy.label = "Rueda_" + name
                parts.append(copy)
    for name, components in head.items():
        parts.append(Compound(label=name, children=components))
    return Compound(label="Chasis_cosechador", children=parts)
