"""Vista del subconjunto montado, con sus dos salidas de muslo."""
from pathlib import Path
from build123d import Compound, import_step
from montaje_cadera import PIEZAS_TORSO


def gen_step():
    robot = import_step(Path(__file__).resolve().parent / "humanoide_xavier.step")
    torso = robot.children[0]
    bancada = next(p for p in robot.children if p.label.replace("_", " ").startswith("Caderas CubeMars"))
    fijo, gi, ai, gd, ad = bancada.children
    paneles = [p for p in torso.children if p.label.lower().replace("_", " ").removesuffix(" adaptada") in PIEZAS_TORSO]
    salidas = [p.children[-1] for p in robot.children if p.label.lower().startswith("pierna")]
    if len(paneles) != 5 or len(salidas) != 2:
        raise ValueError("Revisar las interfaces del ensamble completo")
    fijo.children = (*fijo.children, *paneles)
    for salida in salidas:
        salida.children = tuple(p for p in salida.children if "M5" not in p.label)
    pelvis = Compound(label="Xavier - cadera instalada en la pelvis adaptada",
                      children=[fijo, gi, ai, salidas[0], gd, ad, salidas[1]])
    for p in pelvis.descendants:
        p.label = p.label.replace("_", " ")
    return {"shape":pelvis,"params":"cadera_xavier_instalada.params.js"}
