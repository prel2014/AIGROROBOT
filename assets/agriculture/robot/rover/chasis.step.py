"""Ensamble del chasis en milímetros."""
from pathlib import Path
from runpy import run_path

from build123d import Box, Color, Compound, Cylinder, Plane, Pos, Rot


def gen_step():
    layout = run_path(str(Path(__file__).with_name("layout.py")))
    parts = []
    for name, (kind, size), xyz, color in layout["shapes"]():
        if kind == "box":
            solid = Box(*size)
        else:
            angles = {"X": (0, 90, 0), "Y": (90, 0, 0), "Z": (0, 0, 0)}
            solid = Rot(*angles[kind]) * Cylinder(*size)
        item = Pos(*xyz) * solid
        item.label = name
        item.color = Color(color)
        parts.append(item)
    for tube in layout["rear_frame"]():
        inner = tube.width-2*tube.wall
        hollow = Box(tube.width,tube.width,tube.length) - Box(inner,inner,tube.length+2)
        x, _, z = tube.axes
        item = Plane(origin=tube.center,x_dir=x,z_dir=z).location * hollow
        item.label, item.color = tube.name, Color("#53736B")
        parts.append(item)
    return Compound(label="Chasis_cosechador", children=parts)
