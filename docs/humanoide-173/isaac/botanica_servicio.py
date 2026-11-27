"""Macetas torneadas y plantas con tallos, pecíolos y hojas de nervadura continua."""
import math

import numpy as np
from pxr import Sdf, UsdGeom


def superficie(t, p, nombre, filas, uv, acabado, costura=False):
    """Cuadrícula de vértices compartidos con normales suaves y UV explícitas."""
    puntos = np.asarray(filas, dtype=np.float32)
    m, n, _ = puntos.shape
    vertices = puntos.reshape(-1, 3)
    indice = np.arange(m * n).reshape(m, n)
    caras = np.stack((indice[:-1, :-1], indice[:-1, 1:], indice[1:, 1:], indice[1:, :-1]),
                     axis=-1).reshape(-1, 4)
    a, b, c, d = vertices[caras].transpose(1, 0, 2)
    normales = np.zeros_like(vertices)
    normal_cara = np.cross(c - a, d - b)
    for columna in caras.T:
        np.add.at(normales, columna, normal_cara)
    if costura:
        bordes = normales[indice[:, 0]] + normales[indice[:, -1]]
        normales[indice[:, 0]] = bordes
        normales[indice[:, -1]] = bordes
    normales /= np.maximum(np.linalg.norm(normales, axis=1, keepdims=True), 1e-10)
    mesh = UsdGeom.Mesh.Define(t.stage, t.ruta(p, nombre))
    mesh.CreatePointsAttr(vertices)
    mesh.CreateNormalsAttr(normales)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateFaceVertexCountsAttr([4] * len(caras))
    mesh.CreateFaceVertexIndicesAttr(caras.ravel())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(np.stack([vertices.min(0), vertices.max(0)]))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            "vertex").Set(np.asarray(uv, dtype=np.float32).reshape(-1, 2))
    t.acabado(mesh.GetPrim(), acabado)
    return mesh


def torno(t, p, nombre, perfil, acabado):
    filas, uv = [], []
    for radio, z in perfil:
        filas.append([(radio * math.cos(i * math.tau / 64), radio * math.sin(i * math.tau / 64), z)
                      for i in range(65)])
        uv.append([(i / 64, z / .22) for i in range(65)])
    return superficie(t, p, nombre, filas, uv, acabado, True)


def hoja(t, p, inicio, angulo, largo, ancho, subida, arqueo, acabado, ondulada):
    filas, uv = [], []
    co, si = math.cos(angulo), math.sin(angulo)
    for i in range(13):
        u = i / 12
        contorno = math.sin(math.pi * u)**.78
        ancho_local = ancho * (.024 + contorno) * (1 + .09 * math.sin(9 * math.pi * u) * ondulada)
        z = inicio[2] + subida * u + arqueo * math.sin(math.pi * u)
        cx = inicio[0] + u * largo * co
        cy = inicio[1] + u * largo * si
        fila = []
        for j in range(7):
            v = (j - 3) / 3
            lado = ancho_local * v
            pliegue = ancho_local * (.20 * (1 - abs(v)) - .20 * v * v + .16 * v * math.sin(5 * u))
            fila.append((cx + si * lado, cy - co * lado, z + pliegue))
        filas.append(fila)
        uv.append([(j / 6, u) for j in range(7)])
    superficie(t, p, "Limbo", filas, uv, acabado)


def planta_en_maceta(t, p, especie, marcada=False, solido_follaje=True):
    acabado = "barro_ocre" if marcada else "barro"
    torno(t, p, "Tiesto",
          [(.001, .001), (.095, .001), (.10, .006), (.102, .017), (.128, .176), (.139, .179), (.144, .186),
           (.144, .210), (.141, .218), (.132, .221), (.126, .216), (.125, .201), (.117, .163), (.087, .024),
           (.001, .024)], acabado)
    proxy = t.disco(p, "ContactoTiesto", (0, 0, .108), .141, .216, "oxido", True)
    UsdGeom.Imageable(proxy).MakeInvisible()
    # Alturas y granulometría reproducibles; la tierra queda bajo el labio del tiesto.
    radios = (.001, .031, .061, .094, .122)
    tierra, uv = [], []
    for r in radios:
        anillo = []
        for i in range(33):
            a = math.tau * i / 32
            x, y = r * math.cos(a), r * math.sin(a)
            anillo.append((x, y, .198 - .028 * r + .002 * math.sin(30 * x + 15 * y)))
        tierra.append(anillo)
        uv.append([(x / .22 + .5, y / .22 + .5) for x, y, _ in anillo])
    # Invertir el recorrido radial para que las normales de la tierra apunten hacia arriba.
    superficie(t, p, "Sustrato", tierra[::-1], uv[::-1], "tierra_maceta", True)
    for j in range(28):
        a = j * 2.39996 + t.rng.uniform(-.1, .1)
        r = .116 * math.sqrt((j + .5) / 28)
        geo = UsdGeom.Sphere.Define(t.stage, t.ruta(p, "Perlita" if j % 4 == 0 else "Corteza"))
        geo.CreateRadiusAttr(1.)
        geo.AddTranslateOp().Set((r * math.cos(a), r * math.sin(a), .197))
        geo.AddRotateZOp().Set(t.rng.uniform(0, 180))
        s = t.rng.uniform(.003, .008)
        geo.AddScaleOp().Set((1.4 * s, .82 * s, .57 * s))
        t.acabado(geo.GetPrim(), "lino" if j % 4 == 0 else "tierra")
    if especie == "albahaca":
        for k in range(4):
            a = 2.4 * k + t.rng.uniform(-.24, .24)
            alto = t.rng.uniform(.24, .34)
            dx, dy = .076 * math.cos(a), .076 * math.sin(a)
            t.varilla(p, "Tallo", (.12 * dx, .12 * dy, .193), (dx, dy, .197 + alto), .0038, "brote")
            for j in range(4):
                u = .29 + .22 * j
                base = (dx * (.12 + .88 * u), dy * (.12 + .88 * u), .197 + alto * u)
                for lado in (0, 1):
                    b = a + j * math.pi / 2 + lado * math.pi
                    peciolo = (base[0] + .018 * math.cos(b), base[1] + .018 * math.sin(b), base[2] + .009)
                    t.varilla(p, "Peciolo", base, peciolo, .0018, "brote")
                    hoja(t, p, peciolo, b, .108 - .014 * j, .041 - .004 * j, .014 + .004 * j, .029,
                         "hoja_tierna" if j == 3 else "hoja_cultivo", True)
    else:
        for j in range(21):
            u = j / 21
            a = 2.39996 * j + t.rng.uniform(-.13, .13)
            hoja(t, p, (.014 * math.cos(a), .014 * math.sin(a), .198 + .05 * u), a, .165 * (1 - .73 * u),
                 .038 * (1 - .38 * u), .09 + .085 * u, .045, "hoja_tierna" if u > .68 else "hoja_cultivo",
                 False)
    if solido_follaje:
        proxy = UsdGeom.Sphere.Define(t.stage, t.ruta(p, "ContactoFollaje"))
        proxy.CreateRadiusAttr(.162)
        proxy.AddTranslateOp().Set((0, 0, .365))
        t.colision(proxy.GetPrim(), "obstaculo")
        UsdGeom.Imageable(proxy).MakeInvisible()


def planta_decorativa(t, p, pos, escala):
    grupo = t.grupo(p, "Planta", pos, t.rng.uniform(-180, 180))
    factor = escala * 1.72
    UsdGeom.Xformable(t.stage.GetPrimAtPath(grupo)).AddScaleOp().Set((factor, ) * 3)
    planta_en_maceta(t, grupo, "roseta", solido_follaje=False)
