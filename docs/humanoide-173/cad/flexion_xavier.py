"""Accionamiento medial de la flexión. Todas las longitudes en mm."""
from itertools import product
from math import cos, radians, sin

import numpy as np
from build123d import Box, Face, Pos, Rot, Shell, Solid, Wire
from scipy.spatial import HalfspaceIntersection

from cadera_xavier import APERTURA, FLEXION, casquillo, cilindro, pieza, placa


CARA_SALIDA = 40.
ASIENTO = CARA_SALIDA + 48.5


def instalar(lado, apertura, flexion, ak):
    # +Z del montaje es la dirección medial. El eje físico permanece en el pivote X.
    medial = Rot(0, -lado*90, 0)
    fase = 13.
    orientar = medial * Pos(0,0,CARA_SALIDA + ak["Z_ROTOR"]) * Rot(0,0,fase) * Rot(180,0,0)
    estator, rotor = ak["motor"]()
    estator = pieza(orientar*estator, "AK60-39 V3 flexion - carcasa", "#AA653F")
    rotor = pieza(orientar*rotor, "AK60-39 V3 flexion - rotor", "#C1A078")
    circular, agujeros = ak["circular"], ak["agujeros"]

    soporte = (medial.inverse()*apertura[0]) - cilindro("Z",22.5,26,44)
    soporte -= Pos(0,0,45)*Box(84,84,11)
    pared = placa(82,82,6,ASIENTO+3,7) - cilindro("Z",29.5,ASIENTO-1,ASIENTO+7)
    pared = agujeros(pared,circular(34,fase+10.05,8),1.7,ASIENTO,ASIENTO+6)
    # Dos nervios pasan por delante del motor. El estator acompaña la abducción.
    talon = Pos(0,32.5,34)*Box(67,9,8)
    nervios = [Pos(signo*30.5,33.5,63)*Box(7,9,62) for signo in (-1,1)]
    soporte = soporte.fuse(talon, *nervios, pared)
    apertura[:] = [p for p in apertura if not (
        p.label == "Casquillo de flexion" and p.center().X*lado < 0)]
    apertura[0] = pieza(medial*soporte, "Bastidor con soporte medial de flexion", APERTURA)
    apertura.append(estator)

    # La brida y su cuello forman una sola pieza con el asiento móvil del muslo.
    disco = casquillo("Z",20.3,6.8,33,CARA_SALIDA)
    tornillos = circular(15,fase-57.9,3)
    disco = agujeros(disco,tornillos,2.2,33,CARA_SALIDA)
    disco = agujeros(disco,circular(14.25,fase+2.1,3),2,33,CARA_SALIDA)
    for x,y in tornillos:
        disco -= Pos(x,y,0)*cilindro("Z",3.6,32,37.2)
    cuello = casquillo("Z",10,6.8,25.5,34)
    salida = flexion[0].fuse(medial*cuello,medial*disco)
    flexion[:] = [p for p in flexion if not (
        p.label in ("Munon desmontable de flexion","Pasador transversal de flexion")
        and p.center().X*lado < 0)]
    flexion[0] = pieza(salida,"Brida de flexion y asiento del muslo",FLEXION)
    for i,p in enumerate(flexion):
        if p.label == "Munon desmontable de flexion":
            # Recortar el plato de mando antiguo en el lado que queda como apoyo.
            corte = Pos(lado*52,0,0)*Box(20.8,32,32)
            flexion[i] = pieza(p-corte,"Munon exterior de apoyo", "#A7B1B3")
    flexion.append(rotor)


def hueco_medial(lado):
    """Poliedro circunscrito al volumen barrido. El margen cubre el muestreo angular."""
    puntos = []
    for d,r in ((23,23),(40,40),(90,40),(96,30),(107,30)):
        for a in range(0,360,10):
            puntos.append((-lado*d,r*cos(radians(a)),r*sin(radians(a))))
    for d,y,z in product((30,94.5),(-41,41),(-41,41)):
        puntos.append((-lado*d,y,z))
    nube = np.asarray(puntos)
    poses = []
    for giro in range(-20,21,4):
        g = radians(giro)
        rz = np.array([[cos(g),-sin(g),0],[sin(g),cos(g),0],[0,0,1]])
        for apertura in (-8,-4,0,4,8,12,16,18):
            a = radians(-lado*apertura)
            ry = np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])
            poses.append(nube @ (rz @ ry).T)
    puntos = np.concatenate(poses)
    normales = np.array([v for v in product((-1,0,1),repeat=3) if any(v)],dtype=float)
    normales /= np.linalg.norm(normales,axis=1)[:,None]
    limites = (puntos@normales.T).max(axis=0)+2
    vertices = HalfspaceIntersection(np.column_stack((normales,-limites)),puntos.mean(axis=0)).intersections
    caras = []
    for normal,limite in zip(normales,limites):
        vs = vertices[np.abs(vertices@normal-limite)<1e-5]
        if len(vs)<3:
            continue
        centro = vs.mean(axis=0)
        u = vs[0]-centro
        v = np.cross(normal,u)
        orden = np.argsort(np.arctan2((vs-centro)@v,(vs-centro)@u))
        caras.append(Face(Wire.make_polygon([tuple(p) for p in vs[orden]],close=True)))
    return Solid(Shell(caras))
