"""Geometría arquitectónica en metros y materiales portátiles para las escenas Xavier."""
from itertools import product
import math
import random

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

# Color lineal, rugosidad y metalicidad. Las superficies texturizadas usan albedo sRGB.
ACABADOS = {
    "roble": ((.48, .25, .10), .48, 0),
    "terrazo": ((.65, .60, .49), .46, 0),
    "asfalto": ((.12, .13, .13), .92, 0),
    "ladrillo": ((.53, .21, .10), .85, 0),
    "hormigon": ((.48, .47, .43), .83, 0),
    "tejido": ((.35, .40, .29), .91, 0),
    "marfil": ((.79, .75, .65), .76, 0),
    "cal": ((.90, .87, .78), .70, 0),
    "arena": ((.64, .43, .27), .83, 0),
    "petroleo": ((.034, .13, .14), .58, 0),
    "oxido": ((.48, .16, .075), .72, 0),
    "grafito": ((.055, .064, .065), .45, .38),
    "aluminio": ((.49, .53, .53), .30, .78),
    "vidrio": ((.15, .29, .32), .13, .26),
    "ceramica": ((.85, .84, .76), .24, 0),
    "laton": ((.58, .36, .12), .34, .63),
    "ocre": ((.84, .53, .12), .67, 0),
    "tinta": ((.10, .19, .27), .67, 0),
    "oliva": ((.27, .33, .15), .84, 0),
    "hoja": ((.115, .26, .095), .75, 0),
    "brote": ((.27, .43, .14), .70, 0),
    "tierra": ((.11, .075, .043), .98, 0),
    "lino": ((.70, .58, .43), .93, 0),
    "coral": ((.64, .28, .20), .84, 0),
}
# Anchura y altura reales del motivo repetido.
REPETICION = {
    "roble": (1.65, 1.05),
    "terrazo": (.60, .60),
    "asfalto": (.80, .80),
    "ladrillo": (.89, .69),
    "hormigon": (1.70, 1.70),
    "tejido": (.15, .15)
}


def material(stage, raiz, nombre, archivo=None):
    mat = UsdShade.Material.Define(stage, raiz + "/" + nombre)
    shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Superficie")
    shader.CreateIdAttr("UsdPreviewSurface")
    color, rugosidad, metal = ACABADOS.get(nombre, ((1., 1., 1.), .65, 0.))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rugosidad)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set(color)
    if archivo:
        lector = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Coordenadas")
        lector.CreateIdAttr("UsdPrimvarReader_float2")
        lector.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Albedo")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./texturas/" + archivo))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        for eje in ("wrapS", "wrapT"):
            tex.CreateInput(eje, Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(lector.ConnectableAPI(), "result")
        diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


class Taller:

    def __init__(self, stage):
        self.stage = stage
        self.raiz = "/World/Escenario"
        UsdGeom.Xform.Define(stage, self.raiz)
        self.cuentas = {}
        self.materiales = {}
        self.rng = random.Random(47)

    def ruta(self, parent, nombre):
        base = parent + "/" + nombre
        n = self.cuentas.get(base, 0)
        self.cuentas[base] = n + 1
        return base if not n else f"{base}_{n:02}"

    def grupo(self, parent, nombre, pos=(0, 0, 0), giro=0):
        grupo = UsdGeom.Xform.Define(self.stage, self.ruta(parent, nombre))
        grupo.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if giro:
            grupo.AddRotateZOp().Set(float(giro))
        return str(grupo.GetPath())

    def acabado(self, prim, nombre):
        if nombre not in self.materiales:
            archivo = (nombre + ".jpg" if nombre in REPETICION or nombre.startswith("vitrina_") else nombre +
                       ".png" if nombre.startswith("rotulo_") else None)
            self.materiales[nombre] = material(self.stage, self.raiz + "/Materiales", nombre, archivo)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.materiales[nombre])

    @staticmethod
    def colision(prim, tipo, aproximacion=None):
        if tipo:
            UsdPhysics.CollisionAPI.Apply(prim)
            if aproximacion:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(aproximacion)
            prim.CreateAttribute("entorno:tipoContacto", Sdf.ValueTypeNames.Token).Set(tipo)

    def malla(self, parent, nombre, caras, acabado, pos=(0, 0, 0), colision=None, uv=None, aprox="none"):
        """Polígonos en coordenadas locales. El tamaño de las texturas se fija en metros."""
        mesh = UsdGeom.Mesh.Define(self.stage, self.ruta(parent, nombre))
        puntos = np.concatenate(caras).astype(float)
        mesh.CreatePointsAttr(puntos.astype("float32"))
        mesh.CreateFaceVertexCountsAttr([len(f) for f in caras])
        mesh.CreateFaceVertexIndicesAttr(list(range(len(puntos))))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*puntos.min(0)), Gf.Vec3f(*puntos.max(0))])
        mesh.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if acabado in REPETICION and uv is None:
            uv = []
            escala = REPETICION[acabado]
            for cara in caras:
                f = np.asarray(cara)
                normal = np.cross(f[1] - f[0], f[2] - f[0])
                plano = ((1, 2), (0, 2), (0, 1))[int(np.abs(normal).argmax())]
                uv.extend(f[:, plano] / escala)
        if uv is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.faceVarying).Set(np.asarray(
                                                        uv, np.float32))
        self.acabado(mesh.GetPrim(), acabado)
        self.colision(mesh.GetPrim(), colision, aprox)
        return mesh

    def caja(self, parent, nombre, pos, medidas, acabado, solido=False, bisel=0):
        if not bisel and acabado not in REPETICION:
            cubo = UsdGeom.Cube.Define(self.stage, self.ruta(parent, nombre))
            cubo.CreateSizeAttr(1.)
            cubo.AddTranslateOp().Set(Gf.Vec3d(*pos))
            cubo.AddScaleOp().Set(Gf.Vec3d(*medidas))
            self.acabado(cubo.GetPrim(), acabado)
            self.colision(cubo.GetPrim(), "obstaculo" if solido else None)
            return cubo
        mitad = np.asarray(medidas, dtype=float) / 2
        b = min(bisel, min(mitad) * .75)
        # Seis caras, doce chaflanes de arista y ocho de esquina para atrapar los brillos.
        caras = []
        for normal in product((-1, 0, 1), repeat=3):
            activos = np.flatnonzero(normal)
            if not len(activos) or (len(activos) > 1 and not b):
                continue
            vertices = []
            libres = [k for k in range(3) if k not in activos]
            for signos in product((-1, 1), repeat=len(libres)):
                signo = np.array(normal)
                signo[libres] = signos
                for principal in activos:
                    v = signo * (mitad - b)
                    v[principal] += signo[principal] * b
                    vertices.append(v)
            f = np.asarray(vertices)
            n = np.asarray(normal, dtype=float)
            n /= np.linalg.norm(n)
            tangente = np.eye(3)[np.argmin(abs(n))]
            u = np.cross(n, tangente)
            v = np.cross(n, u)
            d = f - f.mean(0)
            caras.append(f[np.argsort(np.arctan2(d @ v, d @ u))])
        return self.malla(parent,
                          nombre,
                          caras,
                          acabado,
                          pos,
                          "obstaculo" if solido else None,
                          aprox="boundingCube")

    def disco(self, parent, nombre, pos, radio, altura, acabado, solido=False, eje="Z"):
        geo = UsdGeom.Cylinder.Define(self.stage, self.ruta(parent, nombre))
        geo.CreateAxisAttr(eje)
        geo.CreateRadiusAttr(radio)
        geo.CreateHeightAttr(altura)
        geo.AddTranslateOp().Set(Gf.Vec3d(*pos))
        self.acabado(geo.GetPrim(), acabado)
        self.colision(geo.GetPrim(), "obstaculo" if solido else None)
        return geo

    def varilla(self, parent, nombre, a, b, radio, acabado):
        a, b = Gf.Vec3d(*a), Gf.Vec3d(*b)
        geo = self.disco(parent, nombre, (a + b) / 2, radio, (b - a).GetLength(), acabado)
        geo.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), b - a).GetQuat())

    def rotulo(self, parent, nombre, pos, ancho, alto, mapa):
        self.panel(parent, nombre, pos, ancho, alto, "rotulo_" + mapa)

    def panel(self, parent, nombre, pos, ancho, alto, acabado):
        # Frente hacia -Y. La cara tiene UV completas, sin repetir la imagen.
        a, h = ancho / 2, alto / 2
        cara = [(-a, 0, -h), (a, 0, -h), (a, 0, h), (-a, 0, h)]
        self.malla(parent, nombre, [cara], acabado, pos, uv=[(0, 0), (1, 0), (1, 1), (0, 1)])

    def suelo(self, parent, nombre, xy, dimensiones, acabado, z=.001):
        a, b = np.asarray(dimensiones) / 2
        self.malla(parent, nombre, [[(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0)]], acabado, (*xy, z))


def cojin(t, p, nombre, pos, medidas, acabado, inclinacion=0):
    # Superelipsoide achatado. La sección conserva caras anchas y cantos blandos.
    def potencia(x):
        return math.copysign(abs(x)**.48, x)

    a, b, c = np.array(medidas) * .5
    anillos = []
    for lat in np.linspace(-math.pi / 2, math.pi / 2, 13)[1:-1]:
        r = potencia(math.cos(lat))
        anillos.append([(a * r * potencia(math.cos(lon)), b * r * potencia(math.sin(lon)),
                         c * potencia(math.sin(lat))) for lon in np.arange(24) * math.tau / 24])
    caras = []
    for i in range(24):
        j = (i + 1) % 24
        caras.extend([[(0, 0, -c), anillos[0][j], anillos[0][i]], [anillos[-1][i], anillos[-1][j],
                                                                   (0, 0, c)]])
        for inferior, superior in zip(anillos, anillos[1:]):
            caras.append([inferior[i], inferior[j], superior[j], superior[i]])
    forma = t.malla(p, nombre, caras, acabado, pos)
    if inclinacion:
        forma.AddRotateXOp().Set(float(inclinacion))


def silla(t, p, pos, giro=0):
    g = t.grupo(p, "Silla", pos, giro)
    t.caja(g, "Asiento", (0, 0, .46), (.44, .47, .055), "roble", True, .025)
    t.caja(g, "Respaldo", (0, .20, .69), (.44, .046, .33), "roble", True, .020)
    for x, y in product((-.16, .16), (-.16, .16)):
        t.varilla(g, "Pata", (x * 1.18, y * 1.20, .02), (x, y, .45), .017, "grafito")


def mesa(t, p, pos, ancho=1.6, fondo=.85, alto=.76):
    g = t.grupo(p, "Mesa", pos)
    t.caja(g, "Tablero", (0, 0, alto - .025), (ancho, fondo, .05), "roble", True, .018)
    for x, y in product((-.5, .5), repeat=2):
        t.caja(g, "Pata", (x * (ancho - .16), y * (fondo - .15), alto / 2), (.055, .055, alto), "grafito",
               True)
    return g


def estante(t, p, pos, giro=0):
    g = t.grupo(p, "Estanteria", pos, giro)
    t.caja(g, "Fondo", (0, .18, .94), (1.42, .045, 1.88), "roble", True)
    for x in (-.69, .69):
        t.caja(g, "Costado", (x, 0, .94), (.045, .39, 1.88), "roble", True)
    for z in (.10, .53, .96, 1.39, 1.83):
        t.caja(g, "Balda", (0, 0, z), (1.42, .40, .038), "roble")
        if z > 1.8: continue
        x = -.60
        for _ in range(t.rng.randint(9, 14)):
            ancho = t.rng.uniform(.04, .09)
            alto = t.rng.uniform(.21, .34)
            x += ancho / 2
            t.caja(g, "Libro", (x, -.015, z + .02 + alto / 2), (ancho, .22, alto),
                   t.rng.choice(("tinta", "oxido", "marfil", "oliva", "petroleo", "ocre")))
            t.caja(g, "Cinta", (x, -.127, z + .07), (ancho * .67, .004, .009), "lino")
            x += ancho / 2 + .012


def planta(t, p, pos, escala=1.):
    g = t.grupo(p, "Planta", pos)
    s = escala
    t.disco(g, "Tiesto", (0, 0, .16 * s), .21 * s, .32 * s, "oxido", True)
    t.disco(g, "Borde", (0, 0, .31 * s), .225 * s, .057 * s, "oxido")
    t.disco(g, "Sustrato", (0, 0, .341 * s), .198 * s, .012 * s, "tierra")
    for i in range(14):
        ang = i * 2.4 + t.rng.uniform(-.15, .15)
        largo = t.rng.uniform(.30, .56) * s
        alto = t.rng.uniform(.35, .87) * s
        d = np.array((math.cos(ang), math.sin(ang), 0))
        w = np.array((-d[1], d[0], 0))
        base = d * .018 * s + [0, 0, .32 * s]
        tallo = base + d * .13 * largo + [0, 0, alto * .42]
        t.varilla(g, "Peciolo", base, tallo, .006 * s, "oliva")
        franjas = []
        anterior = None
        for paso in range(11):
            f = paso / 10.
            nervio = tallo + d * largo * (.15 * f + .85 * f * f) + [0, 0, alto * (1.37 * f - 1.05 * f * f)]
            ancho = s * .079 * math.sin(math.pi * f)**.8
            borde = nervio - [0, 0, .22 * ancho]
            fila = (borde - w * ancho, nervio, borde + w * ancho)
            if anterior is not None:
                for k in range(2):
                    if paso == 1:
                        franjas.append([anterior[1], fila[k + 1], fila[k]])
                    elif paso == 10:
                        franjas.append([anterior[k], anterior[k + 1], fila[1]])
                    else:
                        franjas.append([anterior[k], anterior[k + 1], fila[k + 1], fila[k]])
            anterior = fila
        t.malla(g, "Hoja", franjas, "brote" if i % 3 else "hoja")


def banco(t, p, pos, giro=0):
    g = t.grupo(p, "Banco", pos, giro)
    for x in (-.72, .72):
        t.caja(g, "Soporte", (x, 0, .21), (.13, .52, .42), "hormigon", True, .018)
    for i in range(5):
        t.caja(g, "Liston", (0, (i - 2) * .098, .45), (1.92, .085, .06), "roble", True, .008)
    for x in (-.73, .73):
        t.varilla(g, "Montante", (x, .20, .38), (x, .26, .86), .024, "grafito")
    for z in (.69, .82):
        t.caja(g, "Respaldo", (0, .25, z), (1.92, .045, .10), "roble", True, .01)
