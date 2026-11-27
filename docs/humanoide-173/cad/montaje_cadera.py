"""Variante de la pelvis de Xavier, centros articulares en coordenadas del robot (mm)."""
import runpy
from itertools import product
from math import cos, radians, sin
from pathlib import Path

from build123d import Box, Compound, Polygon, Pos, extrude
from scipy.spatial import ConvexHull

from cadera_xavier import HIP_X, HIP_Z, cilindro, pieza, placa
from flexion_xavier import hueco_medial, instalar


CAD = Path(__file__).resolve().parent
CANDIDATO = runpy.run_path(str(CAD / "cadera_ak60_encaje.step.py"))


def barrido_giro(ancho, posterior, anterior, abajo, arriba):
    # Envolvente convexa del giro del rectángulo en planta. Los márgenes de 2 mm
    # o más superan la flecha entre muestras de 2 grados a este radio (<0,03 mm).
    puntos = []
    for angulo in range(-20, 21, 2):
        a = radians(angulo)
        for x, y in product((-ancho/2, ancho/2), (posterior, anterior)):
            puntos.append((x*cos(a)-y*sin(a), x*sin(a)+y*cos(a)))
    indices = ConvexHull(puntos).vertices
    perfil = Polygon(*[puntos[i] for i in indices], align=None)
    return Pos(0,0,abajo) * extrude(perfil, amount=arriba-abajo)


def vaciados():
    cuna = barrido_giro(120, -55, 55, -62, 53)
    motor_lateral = barrido_giro(84, -94, -16, -44, 55)
    cuello = cilindro("Z", 26, 50, 91)
    motor_vertical = Pos(0,0,123.5)*Box(84,82,85)
    hueco = cuna.fuse(motor_lateral, cuello, motor_vertical)
    huecos = [Pos(lado*HIP_X,0,HIP_Z)*hueco for lado in (-1,1)]
    huecos.extend(Pos(lado*HIP_X,0,HIP_Z)*hueco_medial(lado) for lado in (-1,1))
    huecos.append(Pos(0,0,954)*Box(310,80,10))
    huecos.append(Pos(0,0,961)*Box(56,60,8))
    return huecos


def modulo_motorizado(lado):
    fijo,giro,apertura,flexion = CANDIDATO["modulo_de_encaje"](lado, compacto=True)
    instalar(lado,apertura,flexion,CANDIDATO)
    return [fijo,giro,apertura,flexion]


PIEZAS_TORSO = {"cintura pelvica", "escudo pelvico", "placa abdominal 1",
                "junta abdominal 1", "placa abdominal 2"}


def mecanismo():
    return CANDIDATO["ensamblar"](modulo_motorizado)


def adaptar_torso(torso):
    huecos = vaciados()
    adaptadas = {}
    for original in torso.children:
        nombre = original.label.lower().replace("_", " ")
        if nombre not in PIEZAS_TORSO:
            continue
        forma = original
        if nombre == "cintura pelvica":
            forma = forma.fuse(placa(62,60,14,942,5))
        forma = forma.cut(*huecos)
        if nombre == "cintura pelvica":
            for x,y in product((-21,-7,7,21),(-20,20)):
                forma -= Pos(x,y,0)*cilindro("Z",2.5,934,950)
        adaptadas[nombre] = pieza(forma,nombre+" adaptada",original.color)
    if set(adaptadas) != PIEZAS_TORSO:
        raise ValueError("El conjunto de la pelvis de origen ha cambiado")
    torso.children = tuple(adaptadas.get(p.label.lower().replace("_", " "), p) for p in torso.children)


def integrar(partes):
    torso = partes[0]
    adaptar_torso(torso)
    fijo, gi, ai, fi, gd, ad, fd = mecanismo().children
    piernas = {p.label.lower().replace("_", " "):p for p in partes if p.label.lower().startswith("pierna")}
    for nombre, salida in (("pierna izquierda",fi),("pierna derecha",fd)):
        pierna = piernas[nombre]
        anterior = pierna.children[-1]
        herrajes = list(anterior.children[-5:])
        if not (anterior.label == "Cadera" and len(herrajes) == 5 and
                all("M5" in p.label for p in herrajes[:4]) and "largueros" in herrajes[-1].label):
            raise ValueError("Revisar los cuatro tornillos y el asiento del muslo")
        salida.children = (*salida.children, *herrajes)
        pierna.children = (*pierna.children[:-1], salida)
    partes.append(Compound(label="Caderas CubeMars - bancada, giro y apertura",
                           children=[fijo, gi, ai, gd, ad]))
