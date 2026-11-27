"""Ocho entornos de servicio en metros. Ejecutar con el Python de Isaac Sim."""
import argparse
import math
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from acabados_servicio import TallerServicio, generar_texturas, preparar_materiales
from botanica_servicio import hoja, planta_en_maceta, planta_decorativa, superficie, torno
from escenarios_marcha import plafon
from geometria_escenarios import banco, material, mesa, silla

DESTINO = Path(__file__).resolve().parent / "escenarios"
FPS = 24


def cartel(t, p, clave, texto, pos, ancho, fondo="#234954"):
    imagen = Image.new("RGB", (768, 128), fondo)
    dibujo = ImageDraw.Draw(imagen)
    try:
        fuente = ImageFont.truetype("arialbd.ttf", 70)
    except OSError:
        fuente = ImageFont.load_default(size=60)
    dibujo.rectangle((8, 9, 18, 119), fill="#f6c772")
    dibujo.text((391, 64), texto, font=fuente, anchor="mm", fill="#fff8df")
    imagen.save(DESTINO / "texturas" / f"rotulo_{clave}.png")
    alto = ancho / 6
    t.caja(p, "Placa", pos, (ancho, .036, alto), "grafito")
    t.rotulo(p, "Rotulo", (pos[0], pos[1] - .02, pos[2]), ancho, alto, clave)


def cuerpo(t, ruta, kg, animacion=()):
    prim = t.stage.GetPrimAtPath(ruta)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(kg)
    if animacion:
        rb.CreateKinematicEnabledAttr(True)
        posicion = prim.GetAttribute("xformOp:translate")
        for segundos, punto in animacion:
            posicion.Set(Gf.Vec3d(*punto), segundos * FPS)


def tablero(t, p, clave, titulo, filas, pos, ancho, fondo="#174749", ruta=False):
    """Panel de lectura frontal con tipografía, jerarquía y un esquema opcional de paradas."""
    alto = ancho * .72
    img = Image.new("RGB", (900, 648), "#f2eee2")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 900, 126), fill=fondo)

    def fuente(tamano, negrita=False):
        try:
            return ImageFont.truetype("arialbd.ttf" if negrita else "arial.ttf", tamano)
        except OSError:
            return ImageFont.load_default(size=tamano)

    d.text((40, 63), titulo, fill="white", font=fuente(51, True), anchor="lm")
    paso = 455 / len(filas)
    if ruta:
        d.line((54, 168 + paso / 2, 54, 623 - paso / 2), fill="#16847c", width=12)
    for i, (izq, der) in enumerate(filas):
        y = int(168 + paso * (i + .5))
        if ruta:
            d.ellipse((40, y - 14, 68, y + 14), fill="white", outline="#16847c", width=6)
        d.text((90 if ruta else 40, y), izq, fill="#243332", font=fuente(40, True), anchor="lm")
        d.text((855, y), der, fill="#365450", font=fuente(34), anchor="rm")
        d.line((90, y + paso * .42, 855, y + paso * .42), fill="#c5c4af", width=2)
    img.save(DESTINO / "texturas" / f"rotulo_{clave}.png")
    x, y, z = pos
    t.caja(p, "Bastidor", pos, (ancho + .06, .07, alto + .06), "aluminio", True)
    t.rotulo(p, "PanelInformativo", (x, y - .038, z), ancho, alto, clave)


def caja_movil(t, nombre, pos, medidas, acabado, kg):
    grupo = t.grupo(t.raiz, nombre, pos)
    t.caja(grupo, "Caja", (0, 0, 0), medidas, acabado, True)
    cuerpo(t, grupo, kg)
    return grupo


def esfera(t, p, nombre, pos, radio, acabado, solida=False):
    geo = UsdGeom.Sphere.Define(t.stage, t.ruta(p, nombre))
    geo.CreateRadiusAttr(radio)
    geo.AddTranslateOp().Set(pos)
    t.acabado(geo.GetPrim(), acabado)
    t.colision(geo.GetPrim(), "obstaculo" if solida else None)
    return geo


def silueta(alto, angulo, tipo):
    z = math.sin(alto)
    r = math.cos(alto)
    if tipo == "tomate":
        r *= 1 + .060 * math.cos(7 * angulo + .35 * z) * (.42 + abs(z))
        z *= 1 - .22 * abs(z)**7
    elif tipo == "limon":
        z *= 1 + .18 * abs(z)**10
    x, y = r * math.cos(angulo), r * math.sin(angulo)
    if tipo == "berenjena":
        y = y * (1 + .40 * x) + .17 * x * x
        z = z * (1 + .40 * x) + .13 * x * x
    elif tipo == "escamas":
        y *= 1 + .26 * x
        z = z * (1 + .33 * x) + .10 * x * x
    elif tipo == "corteza":
        z = max(-.75, z) + .025 * math.sin(3 * angulo) * r * r
        if z > .3:
            d = min(((x - .44 * n - .27 * y) / .11)**2 + (y / .60)**2 for n in (-1, 0, 1))
            z -= .09 * math.exp(-1.6 * d)
    return Gf.Vec3d(x, y, z)


def ovalo(t, p, nombre, pos, radios, acabado, giro=0):
    if acabado in ("tomate", "naranja", "limon", "berenjena", "corteza", "tejido", "escamas"):
        modelo = "/Formas/" + acabado
        if not t.stage.GetPrimAtPath(modelo):
            t.stage.CreateClassPrim("/Formas")
            m = UsdGeom.Mesh.Define(t.stage, modelo)
            puntos, normales, uv, indices = [], [], [], []
            for j in range(33):
                lat = math.pi * (j / 32 - .5)
                for i in range(65):
                    a = math.tau * i / 64
                    puntos.append(silueta(lat, a, acabado))
                    du = silueta(lat, a + .001, acabado) - silueta(lat, a - .001, acabado)
                    dv = silueta(lat + .001, a, acabado) - silueta(lat - .001, a, acabado)
                    normales.append(
                        Gf.Cross(du, dv).GetNormalized() if 0 < j < 32 else (0, 0, 1 if j else -1))
                    uv.append((i / 64, j / 32))
            for j, k in product(range(32), range(64)):
                n = j * 65 + k
                indices.extend((n, n + 1, n + 66, n + 65))
            m.CreatePointsAttr(puntos)
            m.CreateNormalsAttr(normales)
            m.SetNormalsInterpolation("vertex")
            m.CreateFaceVertexCountsAttr([4] * 2048)
            m.CreateFaceVertexIndicesAttr(indices)
            m.CreateSubdivisionSchemeAttr("none")
            UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv)
        geo = UsdGeom.Mesh.Define(t.stage, t.ruta(p, nombre))
        geo.GetPrim().GetReferences().AddInternalReference(modelo)
        geo.AddTranslateOp().Set(pos)
        t.acabado(geo.GetPrim(), acabado)
    else:
        geo = esfera(t, p, nombre, pos, 1., acabado)
    if giro:
        geo.AddRotateZOp().Set(float(giro))
    geo.AddScaleOp().Set(radios)
    return geo


def materiales_puesto(t):
    for nombre, color in {
            "tomate": (.65, .032, .012),
            "naranja": (.93, .28, .012),
            "limon": (.40, .57, .048),
            "berenjena": (.16, .042, .13),
            "corteza": (.53, .24, .058),
            "rosa": (.68, .075, .20)
    }.items():
        # Cada material pertenece a esta escena y usa el mismo shader que el mobiliario.
        mat = material(t.stage, t.raiz + "/Materiales", nombre)
        shader = UsdShade.Shader.Get(t.stage, mat.GetPath().AppendChild("Superficie"))
        shader.GetInput("diffuseColor").Set(color)
        shader.GetInput("roughness").Set(.52)
        t.materiales[nombre] = mat


def naranja_abierta(t, g, pos):
    mitad = t.grupo(g, "MediaNaranja", pos)
    UsdGeom.Xformable(t.stage.GetPrimAtPath(mitad)).AddRotateXOp().Set(32.)
    radio = .09
    perfil = [(max(.001, radio * math.cos(a * math.pi / 24)), -radio * math.sin(a * math.pi / 24))
              for a in range(12, -1, -1)]
    torno(t, mitad, "Cascara", perfil, "naranja")
    borde = [(radio * math.cos(i * math.tau / 64), radio * math.sin(i * math.tau / 64), .0003)
             for i in range(64)]
    t.malla(mitad, "Gajos", [borde], "pulpa_naranja",
            uv=[(.5 + x / (2 * radio), .5 + y / (2 * radio)) for x, y, _ in borde])


def productos(t, g, tipo):
    if tipo == "Flores":
        for k, color in enumerate(("rosa", "ocre", "cal", "coral")):
            x = -.93 + k * .60
            cubo = t.grupo(g, "CuboFlorista", (x, 0, .80))
            torno(t, cubo, "Chapa", [(.001, 0), (.103, 0), (.112, .015), (.152, .37), (.155, .38),
                                     (.147, .382), (.144, .365), (.105, .021), (.001, .021)], "aluminio")
            for j in range(9):
                a = j * 2.40
                dx, dy = .19 * math.cos(a), .19 * math.sin(a)
                z = 1.39 + .14 * (j % 3)
                t.varilla(g, "Tallo", (x + dx * .3, dy * .3, 1.05), (x + dx, dy, z), .0038, "hoja")
                hoja(t, g, (x + dx * .73, dy * .73, z - .16), a, .085, .018, .046, .022, "hoja_cultivo",
                     False)
                corola = t.grupo(g, "Corola", (x + dx, dy, z), j * 33)
                UsdGeom.Xformable(t.stage.GetPrimAtPath(corola)).AddRotateXOp().Set(20 + j % 4 * 12)
                for nivel in range(2):
                    for n in range(12):
                        b = (n + nivel * .5) * math.tau / 12
                        hoja(t, corola, (.009 * math.cos(b), .009 * math.sin(b), .002 * nivel), b,
                             .065 - .021 * nivel, .013 - .002 * nivel, -.014 + .037 * nivel, .014, color,
                             False)
                ovalo(t, corola, "DiscoFloral", (0, 0, .012), (.018, .018, .010), "ocre")
                for n in range(17):
                    r = .015 * math.sqrt(n / 17)
                    esfera(t, corola, "Flosculo",
                           (r * math.cos(n * 2.40), r * math.sin(n * 2.40), .020 - .003 * n / 17), .0023,
                           "lino")
        return
    if tipo == "Granos":
        for x, acabado in zip((-.86, 0, .86), ("lino", "oxido", "ocre")):
            saco = t.grupo(g, "SacoAbierto", (x, 0, .80))
            torno(t, saco, "Arpillera", [(.001, 0), (.23, 0), (.28, .065), (.306, .21), (.29, .344),
                                         (.286, .383), (.302, .392), (.303, .406), (.29, .415), (.27, .407),
                                         (.26, .37), (.23, .23)], "tejido")
            t.disco(saco, "FondoCarga", (0, 0, .354), .267, .044, "tierra")
            for capa, cantidad in enumerate((390, 330)):
                for j in range(cantidad):
                    a = j * 2.39996 + .7 * capa + t.rng.uniform(-.012, .012)
                    r = math.sqrt((j + .5) / cantidad)
                    xx = .271 * r * math.cos(a)
                    yy = .271 * r * math.sin(a)
                    z = .377 + .018 * capa + .066 * (1 - r * r) + t.rng.uniform(-.003, .003)
                    giro = t.rng.uniform(0, 180)
                    s = t.rng.uniform(.90, 1.08)
                    ovalo(t, saco, "Legumbre", (xx, yy, z), (.016 * s, .011 * s, .008 * s), acabado, giro)
                    if capa and j % 3 == 0:
                        ovalo(t, saco, "Hilio", (xx, yy, z + .0075 * s), (.0048, .0014, .0014), "arena",
                              giro)
        return
    # Reservar la mitad izquierda del mostrador de frutas para la canasta.
    centros = (.63, ) if tipo == "Frutas" else (-.68, .68)
    for caja, x in enumerate(centros):
        t.caja(g, "FondoJaba", (x, 0, .828), (1.19, .86, .055), "roble")
        for lado in (-1, 1):
            t.caja(g, "TesteroJaba", (x + lado * .576, 0, .917), (.034, .86, .18), "roble")
            for alto in (.87, .943):
                t.caja(g, "ListonJaba", (x, lado * .414, alto), (1.17, .032, .044), "roble")
        if tipo == "Pescados":
            t.caja(g, "Hielo", (x, 0, .89), (1.10, .78, .07), "cal", bisel=.02)
            for j in range(5):
                y = -.30 + j * .15
                ovalo(t, g, "Pescado", (x, y, .975), (.31, .083, .049), "escamas")
                t.malla(g, "ColaAhorquillada", [[(x - .26, y, .998), (x - .43, y - .071, .966),
                                                 (x - .374, y, .992), (x - .43, y + .071, 1.014)]],
                        "grafito")
                t.malla(g, "AletaPectoral", [[(x + .11, y - .018, 1.018), (x + .063, y - .070, 1.002),
                                              (x - .047, y - .077, .982), (x + .025, y - .039, 1.013)]],
                        "tinta")
                for n in range(7):
                    t.varilla(g, "RadioCaudal", (x - .26, y, .982),
                              (x - .38 - .048 * abs(n - 3) / 3, y + (n - 3) * .020, .99 + (n - 3) * .005),
                              .0016, "tinta")
                ovalo(t, g, "OjoSuperior", (x + .21, y - .014, 1.016), (.019, .014, .009), "laton")
                ovalo(t, g, "IrisSuperior", (x + .21, y - .016, 1.024), (.011, .009, .004), "grafito")
                arco = [(x + .16, y - .052, .995), (x + .137, y - .035, 1.021), (x + .129, y, 1.028),
                        (x + .139, y + .036, 1.024), (x + .156, y + .064, 1.004)]
                for a, b in zip(arco, arco[1:]):
                    t.varilla(g, "Operculo", a, b, .0021, "tinta")
            for j in range(20):
                xx = x + t.rng.uniform(-.51, .51)
                yy = t.rng.uniform(-.34, .34)
                t.caja(g, "HieloPicado", (xx, yy, .941), (.040, .053, .033), "vidrio_satinado", bisel=.009)
            continue
        for j, k in product(range(4), range(3)):
            xx = x + (j - 1.5) * .262 + t.rng.uniform(-.02, .02)
            yy = (k - 1) * .238 + t.rng.uniform(-.018, .018)
            if tipo == "Pan":
                pieza = t.grupo(g, "Panecillo", (xx, yy, .928), j * 9 + k * 6)
                ovalo(t, pieza, "Corteza", (0, 0, 0), (.122, .085, .081), "corteza")
                for n in range(14):
                    a = n * 2.39996
                    lat = .73 + .49 * ((n * 7) % 13) / 12
                    pos = silueta(lat, a, "corteza")
                    ovalo(t, pieza, "Ajonjoli", (pos[0] * .122, pos[1] * .085, pos[2] * .081 + .0006),
                          (.0037, .00165, .00115), "lino", n * 31)
            elif tipo == "Verduras" and not caja:
                ovalo(t, g, "Berenjena", (xx, yy, .945), (.116, .075, .085), "berenjena", j * 12)
                esfera(t, g, "Caliz", (xx - .083, yy, .979), .029, "hoja")
                t.varilla(g, "Pezon", (xx - .088, yy, .976), (xx - .132, yy + .014, 1.005), .007, "oliva")
            else:
                color = ("naranja" if j < 2 else "limon") if tipo == "Frutas" else "tomate"
                if tipo == "Frutas" and j == 1 and k == 0:
                    naranja_abierta(t, g, (xx, yy, .948))
                    continue
                tam = t.rng.uniform(.92, 1.05)
                ovalo(t, g, "Cosecha", (xx, yy, .948), (.104 * tam, .093 * tam, .092 * tam), color,
                      t.rng.uniform(0, 360))
                z = .948 + .092 * tam * (.78 if color == "tomate" else 1.)
                t.varilla(g, "Pedunculo", (xx, yy, z - .004), (xx + .007, yy, z + .018), .004, "oliva")
                if color == "tomate":
                    for n in range(5):
                        hoja(t, g, (xx, yy, z + .005), n * math.tau / 5 + .4 * j, .041, .005, -.003, .007,
                             "hoja_cultivo", False)


def muro_acristalado(t, p, ancho, alto, huecos, antepecho, dintel, acabado):
    """Muro frontal en el plano XZ. Cada vano contiene solo el marco y el vidrio."""
    t.caja(p, "Antepecho", (0, 0, antepecho / 2), (ancho, .16, antepecho), acabado, True)
    t.caja(p, "DintelVanos", (0, 0, (dintel + alto) / 2), (ancho, .16, alto - dintel), acabado, True)
    inicio = -ancho / 2
    h = dintel - antepecho
    z = (dintel + antepecho) / 2
    for x, luz in huecos:
        borde = x - luz / 2
        if borde > inicio:
            t.caja(p, "Machon", ((inicio + borde) / 2, 0, z), (borde - inicio, .16, h), acabado, True)
        t.caja(p, "CristalVentana", (x, 0, z), (luz, .012, h), "vidrio", True)
        for lado in (-1, 1):
            t.caja(p, "Perfil", (x + lado * (luz / 2 - .02), 0, z), (.04, .09, h), "aluminio")
            t.caja(p, "Travesano", (x, 0, z + lado * (h / 2 - .02)), (luz, .09, .04), "aluminio")
        t.caja(p, "Parteluz", (x, 0, z), (.038, .075, h), "aluminio")
        t.caja(p, "Vierteaguas", (x, -.035, antepecho), (luz + .12, .24, .04), "hormigon")
        inicio = x + luz / 2
    fin = ancho / 2
    if fin > inicio:
        t.caja(p, "Machon", ((inicio + fin) / 2, 0, z), (fin - inicio, .16, h), acabado, True)


def frente_comercial(t, p, x, tinte, negocio):
    g = t.grupo(p, "Comercio", (x, 10.2, 0))
    muro_acristalado(t, g, 5.95, 3.38, [(-.77, 3.54), (2.04, 1.16)], .42, 2.52, "revoque_" + tinte)
    t.caja(g, "MuroInterior", (0, 2.95, 1.67), (5.95, .16, 3.34), "revoque_marfil", True)
    for lado in (-1, 1):
        t.caja(g, "Medianera", (lado * 2.94, 1.5, 1.68), (.12, 3.0, 3.36), "ladrillo", True)
    t.caja(g, "Cubierta", (0, 1.4, 3.42), (6.08, 3.3, .12), "hormigon", True)
    for i in range(12):
        t.caja(g, "LonaMarquesina", ((i - 5.5) * .47, -.38, 2.64), (.47, .9, .065),
               tinte if i % 2 else "marfil")
    t.rotulo(g, "Letrero", (0, -.095, 2.98), 2.55, .60, negocio)
    mesa(t, g, (-.75, .84, 0), 3.15, .66, .78)
    for i in range(9):
        x_obj = -2.02 + i * .32
        if negocio == "pan":
            ovalo(t, g, "Hogaza", (x_obj, .81, .87), (.143, .11, .09), "corteza", i * 11)
        elif negocio == "papel":
            for j in range(3):
                t.caja(g, "Cuaderno", (x_obj, .81, .798 + j * .032), (.26, .31, .031),
                       ("tinta", "oxido", "marfil")[j])
        else:
            t.disco(g, "Conserva", (x_obj, .81, .87), .078, .18, "metal_torneado")
            t.disco(g, "Faja", (x_obj, .81, .87), .079, .13, "oliva" if i % 2 else "ocre")
    plafon(t, g, (0, 1.3, 3.30), 1.35, 14000.)


def mercado(t):
    p = t.raiz
    materiales_puesto(t)
    t.suelo(p, "LosetaDeMercado", (0, 0), (16.7, 10.5), "terrazo", .002)
    t.caja(p, "FondoGaleria", (0, 4.15, 1.65), (17, .18, 3.30), "ladrillo", True)
    cartel(t, p, "galeria", "MERCADO DE ABASTOS", (0, 4.03, 2.83), 3.4)
    # Nave abierta al frente. Los apoyos quedan fuera del corredor transversal.
    for y in (-4.5, -.15, 4.2):
        for lado in (-1, 1):
            t.caja(p, "PilarNave", (lado * 8.2, y, 2.05), (.16, .16, 4.1), "petroleo", True)
            t.varilla(p, "CordonSuperior", (lado * 8.2, y, 4.08), (0, y, 5.30), .065, "petroleo")
            t.varilla(p, "Tirante", (lado * 8.2, y, 4.06), (0, y, 4.06), .047, "petroleo")
            for i in range(4):
                x = lado * i * 2.05
                t.varilla(p, "DiagonalCercha", (x, y, 5.3 - i * .305), (x + lado * 2.05, y, 4.06), .034,
                          "petroleo")
        plafon(t, p, (0, y, 4.01), 1.75, 24000.)
    for lado in (-1, 1):
        ala = t.grupo(p, "CubiertaDeCalamina", (lado * 4.16, -.15, 4.69))
        UsdGeom.Xformable(t.stage.GetPrimAtPath(ala)).AddRotateYOp().Set(lado * 8.47)
        t.caja(ala, "Plancha", (0, 0, 0), (8.45, 9.24, .045), "marfil")
        for j in range(19):
            t.caja(ala, "Nervadura", ((j - 9) * .45, 0, -.042), (.035, 9.22, .07), "aluminio")
    for x in (-7.3, -5.1, 5.2, 7.3):
        for i in range(2):
            z = .15 + i * .30
            t.caja(p, "CajonApilado", (x, 3.57, z), (.77, .63, .28), "roble", True)
            for dz in (-.07, .07):
                t.caja(p, "RanuraCajon", (x, 3.25, z + dz), (.65, .012, .03), "grafito")
    puestos = [(-2.6, 2.4, 0, "Frutas", "oxido"), (2.6, 2.4, 0, "Verduras", "hoja"),
               (-6.1, 2.4, 0, "Pan", "ocre"), (6.1, 2.4, 0, "Pescados", "tinta"),
               (-6.5, -1.5, 65, "Granos", "oliva"), (6.5, -1.5, -65, "Flores", "coral")]
    for x, y, giro, tipo, color in puestos:
        g = t.grupo(p, "Puesto" + tipo, (x, y, 0), giro)
        mesa(t, g, (0, 0, 0), 2.8, 1.1, .8)
        for lado in (-1, 1):
            t.caja(g, "Pilar", (lado * 1.32, .42, 1.15), (.08, .08, 2.3), "roble", True)
        for i in range(9):
            t.caja(g, "Lona", ((i - 4) * .34, .04, 2.35), (.34, 1.45, .08), color if i % 2 == 0 else "lino")
            t.caja(g, "FaldonToldo", ((i - 4) * .34, -.67, 2.24), (.34, .035, .17),
                   color if i % 2 == 0 else "lino")
        cartel(t, g, "puesto_" + tipo.lower(), tipo.upper(), (0, -.715, 2.22), 1.55)
        plafon(t, g, (0, -.16, 2.276), 1.36, 19000.)
        productos(t, g, tipo)
        t.caja(g, "FrentePuesto", (0, -.50, .40), (2.72, .07, .76), "roble", True)
        for j in range(3):
            t.caja(g, "Liston", (0, -.54, .14 + j * .24), (2.73, .025, .035), "arena")
    g = t.grupo(p, "Canasta", (-3.05, 2.22, .8))
    t.caja(g, "Base", (0, 0, .023), (.40, .30, .046), "ocre", True)
    for lado in (-1, 1):
        t.caja(g, "Testero", (lado * .185, 0, .10), (.03, .30, .20), "ocre", True)
        t.caja(g, "Costado", (0, lado * .14, .10), (.40, .02, .20), "ocre", True)
        t.caja(g, "SoporteAsa", (lado * .185, 0, .25), (.025, .03, .15), "laton", True)
    t.caja(g, "Asa", (0, 0, .331), (.39, .034, .028), "laton", True)
    cuerpo(t, g, .45)
    mesa(t, p, (3.3, -3.3, 0), 1.6, .84, .80)
    t.caja(p, "Balanza", (3.83, -3.36, .85), (.38, .36, .10), "petroleo", True, bisel=.02)
    t.caja(p, "PlatoBalanza", (3.83, -3.36, .919), (.36, .34, .035), "aluminio")
    t.caja(p, "VisorBalanza", (3.83, -3.545, .854), (.20, .015, .060), "grafito", bisel=.006)
    cartel(t, p, "lcd_balanza", "0.000 kg", (3.83, -3.555, .852), .16, "#334637")
    t.caja(p, "PieCartel", (3.3, -2.93, 1.02), (.04, .04, .44), "aluminio")
    cartel(t, p, "caja_mercado", "CAJA", (3.3, -2.94, 1.24), .85)
    g = t.grupo(p, "Carretilla", (-4, -.6, 0))
    t.caja(g, "Plataforma", (0, 0, .29), (.92, .57, .12), "petroleo", True)
    t.caja(g, "Jaba", (.06, 0, .52), (.62, .50, .34), "ocre", True)
    t.caja(g, "Manillar", (-.43, 0, .67), (.05, .49, .67), "aluminio", True)
    for x, y in product((-.32, .32), (-.24, .24)):
        t.disco(g, "Rueda", (x, y, .14), .14, .07, "grafito", True, "Y")
    cuerpo(t, g, 15.)
    carro = t.stage.GetPrimAtPath(g)
    UsdPhysics.RigidBodyAPI(carro).CreateKinematicEnabledAttr(True)
    posicion = carro.GetAttribute("xformOp:translate")
    posicion.Clear()
    variantes = carro.GetVariantSets().AddVariantSet("momentoCruce")
    for nombre, inicio in (("temprano", 2), ("tardio", 12)):
        variantes.AddVariant(nombre)
        variantes.SetVariantSelection(nombre)
        with variantes.GetVariantEditContext():
            posicion.Set(Gf.Vec3d(-4, -.6, 0))
            for s, x in ((0, -4), (inicio, -4), (inicio + 8, 4)):
                posicion.Set(Gf.Vec3d(x, -.6, 0), s * FPS)
    variantes.SetVariantSelection("temprano")


def butaca(t, p, pos, giro):
    g = t.grupo(p, "Butaca", pos, giro)
    t.caja(g, "Pedestal", (0, .012, .215), (.14, .23, .43), "grafito", True, bisel=.019)
    t.caja(g, "Cuba", (0, 0, .435), (.455, .455, .087), "petroleo", True, bisel=.028)
    t.caja(g, "Carcasa", (0, .202, .747), (.448, .084, .635), "petroleo", True, bisel=.036)
    # Una banda continua curvada bajo las piernas y en el respaldo, con más apoyo en los laterales.
    perfil = [(-.225, .439), (-.217, .474), (-.190, .495), (-.140, .489), (-.04, .465), (.090, .459),
              (.135, .485), (.142, .545), (.128, .62), (.117, .72), (.123, .82), (.142, .96), (.154, 1.027),
              (.166, 1.044), (.192, 1.034)]
    filas, uv = [], []
    for i, (y, z) in enumerate(perfil):
        fila = []
        for j in range(13):
            u = (j - 6) / 6
            hombro = max(0, (z - .88) * .32)
            fila.append(
                (u * (.207 - hombro), y - .028 * (u**4) * min(1, max(0, (z - .49) * 9)), z + .014 * u**4))
        filas.append(fila)
        uv.append([(j / 6, i / 3) for j in range(13)])
    superficie(t, g, "Moqueta", filas, uv, "asiento_bus")
    for lado in (-1, 1):
        t.varilla(g, "SoporteAsa", (lado * .124, .195, 1.045), (lado * .11, .195, 1.137), .016, "ocre")
    t.varilla(g, "Asa", (-.11, .195, 1.137), (.11, .195, 1.137), .016, "ocre")
    for x in (-.15, .15):
        t.disco(g, "Remache", (x, .249, .81), .010, .008, "metal_torneado", eje="Y")


def salon_bus(t, g):
    for x in (-3.86, -3.04, -2.22, -1.4, 1.4, 2.22, 3.04, 3.86):
        giro = 90 if x < 0 else -90
        butaca(t, g, (x, .74, .40), giro)
        if abs(x) > 1.5:
            butaca(t, g, (x, -.76, .40), giro)
    for lado in (-1, 1):
        y = lado * .34
        t.disco(g, "RielSuperior", (0, y, 2.26), .023, 8.55, "ocre", eje="X")
        for x in (-3.38, -1.72, 1.72, 3.38):
            t.disco(g, "PosteSalon", (x, y, 1.34), .024, 1.88, "ocre", True)
            t.disco(g, "Casquillo", (x, y, .447), .043, .092, "metal_torneado")
            t.caja(g, "CajaParada", (x, y - lado * .018, 1.38), (.075, .068, .134), "grafito", bisel=.013)
            t.disco(g, "BotonParada", (x, y - lado * .057, 1.39), .023, .018, "coral", eje="Y")
        for x in (-3.85, -2.63, -1.13, 1.12, 2.66, 3.90):
            t.caja(g, "Correa", (x, y, 2.20), (.043, .023, .138), "grafito", bisel=.004)
            vertices = [(x, y, 2.145), (x - .092, y, 2.012), (x - .072, y, 1.992), (x + .072, y, 1.992),
                        (x + .092, y, 2.012), (x, y, 2.145)]
            for a, b in zip(vertices, vertices[1:]):
                t.varilla(g, "Asidero", a, b, .013, "ocre")
        for x in (-3.20, -1.09, 1.09, 3.2):
            t.caja(g, "PerfilLuminaria", (x, lado * .66, 2.462), (1.48, .095, .036), "aluminio")
            t.caja(g, "Difusor", (x, lado * .66, 2.44), (1.43, .077, .013), "ceramica")
            luz = UsdLux.RectLight.Define(t.stage, t.ruta(g, "LuzSalon"))
            luz.AddTranslateOp().Set((x, lado * .66, 2.431))
            luz.CreateWidthAttr(1.43)
            luz.CreateHeightAttr(.077)
            luz.CreateIntensityAttr().Set(44000.)
            luz.CreateColorAttr().Set((.94, .97, 1.))
            for j in range(6):
                t.caja(g, "RejillaAire", (x - .105 + j * .042, lado * .94, 2.462), (.018, .135, .014),
                       "grafito")
        t.caja(g, "JuntaVentana", (0, lado * 1.082, 2.34), (9.14, .043, .055), "grafito")
    frente = t.grupo(g, "Cabecera", (4.52, 0, 0), -90)
    cartel(t, frente, "bus_interior", "EXPRESO  01", (0, 0, 2.16), 1.43, "#182c29")
    t.caja(frente, "Consola", (0, -.05, .996), (1.12, .34, .22), "grafito", True, bisel=.022)
    cartel(t, g, "bus_prioridad", "ASIENTO PREFERENTE", (-2.51, 1.064, 1.33), .77)


def estacion(t):
    p = t.raiz
    t.caja(p, "Anden", (0, -1.70, .20), (12, 6, .4), "terrazo", True)
    t.suelo(p, "LineaDeEspera", (0, .76), (11.9, .13), "ocre", .403)
    t.suelo(p, "CarrilExclusivo", (0, 3.38), (60, 4), "asfalto", .002)
    for x in range(-12, 13, 3):
        t.suelo(p, "MarcaVial", (x, 4.74), (1.2, .095), "marfil", .004)
    cartel(t, p, "estacion", "METROPOLITANO", (-3.8, .90, 2.23), 2.6)
    t.caja(p, "Poste", (-3.8, .93, 1.12), (.09, .09, 2.24), "grafito", True)
    for x in (-3.65, 3.65):
        banco(t, p, (x, -2.24, .4), -125 if x < 0 else 125)
    for x, clave, texto in ((-5.48, "salida_anden", "←  SALIDA"), (2.25, "embarque", "EMBARQUE  01")):
        t.caja(p, "SoporteSenal", (x, .55, 1.50), (.072, .072, 2.20), "aluminio", True)
        cartel(t, p, clave, texto, (x, .49, 2.42), 1.68, "#206351" if x < 0 else "#214767")
    for x in (-5.5, -2.83):
        t.caja(p, "SoporteMarquesina", (x, -3.42, 1.80), (.09, .13, 2.80), "grafito", True)
    t.caja(p, "AleroEspera", (-4.16, -2.67, 3.2), (3.2, 2.02, .09), "aluminio", True)
    t.caja(p, "RespaldoParadero", (-4.16, -3.44, 1.63), (2.65, .06, 2.12), "vidrio", True)
    t.caja(p, "PieDirectorio", (5.32, -1.6, 1.05), (.12, .12, 1.30), "aluminio", True)
    tablero(t, p, "plano_anden", "GUÍA DEL ANDÉN", [("Acceso", "01"), ("Zona de espera", "02"),
                                                    ("Embarque", "03"), ("Salida", "04")],
            (5.32, -1.64, 1.94), 1.04, ruta=True)
    for i in range(39):
        x = (i - 19) * .30
        t.suelo(p, "BaldosaPodotactil", (x, 1.02), (.294, .30), "ocre", .404)
        for j, k in product(range(3), repeat=2):
            t.disco(p, "Relieve", (x + (j - 1) * .09, 1.02 + (k - 1) * .09, .408), .020, .008, "arena")
    for x, tinte, negocio in ((-7.1, "ocre", "papel"), (-.7, "marfil", "pan"), (5.7, "petroleo", "bodega")):
        frente_comercial(t, p, x, tinte, negocio)
    g = t.grupo(p, "BusDetenido", (0, 2.52, 0))
    t.caja(g, "PisoBus", (0, 0, .22), (9.4, 2.32, .36), "grafito", True)
    t.suelo(g, "GomaDeBus", (0, 0), (9.34, 2.24), "piso_goma", .401)
    for x, ancho in ((-2.69, 4.02), (2.69, 4.02)):
        t.caja(g, "Carroceria", (x, -1.12, .93), (ancho, .08, 1.06), "laca_bus", True, bisel=.012)
        t.caja(g, "Ventanilla", (x, -1.12, 1.89), (ancho, .010, .86), "vidrio", True)
    t.caja(g, "Fondo", (0, 1.12, .93), (9.4, .08, 1.06), "laca_bus", True)
    t.caja(g, "LunaLateral", (0, 1.12, 1.89), (9.4, .010, .86), "vidrio", True)
    for x in (-4.65, 4.65):
        t.caja(g, "Testero", (x, 0, 1.37), (.10, 2.32, 1.94), "vidrio", True)
    t.caja(g, "Techo", (0, 0, 2.55), (9.5, 2.42, .13), "marfil", True)
    for lado in (-1, 1):
        for x in (-2.82, 2.82):
            t.disco(g, "Neumatico", (x, lado * 1.12, .43), .418, .20, "grafito", True, "Y")
            t.disco(g, "Llanta", (x, lado * 1.225, .43), .23, .02, "aluminio", False, "Y")
        t.caja(g, "Parachoques", (lado * 4.7, 0, .52), (.15, 2.28, .24), "grafito", True)
        for y in (-.83, .83):
            t.caja(g, "Faro", (lado * 4.73, y, .84), (.08, .27, .12), "ocre")
    for x in (-3.5, -2.25, -1.1, 1.1, 2.25, 3.5):
        t.caja(g, "MarcoVentana", (x, -1.16, 1.90), (.055, .04, .88), "grafito")
        t.caja(g, "MarcoTrasero", (x, 1.12, 1.90), (.050, .04, .88), "grafito")
    cartel(t, g, "ruta_bus", "EXPRESO  01", (2.36, -1.18, 2.34), 1.50)
    salon_bus(t, g)
    for x in (-2.82, 2.82):
        for i in range(8):
            a = i * math.tau / 8
            t.disco(g, "TuercaLlanta", (x + .128 * math.cos(a), -1.248, .43 + .128 * math.sin(a)), .022,
                    .027, "metal_torneado", eje="Y")


def escalinata(t):
    p = t.raiz
    # Las terrazas flanquean la escalera y continúan a la cota superior del barrio.
    for lado in (-1, 1):
        for j in range(3):
            y, h = -2.5 + j * 1.55, .27 + .29 * j
            t.caja(p, "Terraza", (lado * 4.02, y, h / 2), (3.85, 1.55, h), "ladrillo", True)
            t.caja(p, "TierraJardin", (lado * 4.02, y, h + .035), (3.72, 1.40, .07), "tierra", True)
            planta_decorativa(t, p, (lado * (3.09 + j % 2 * .8), y, h + .07), .83 + j * .12)
    t.caja(p, "CalleAlta", (0, 5.6, .45), (20, 7.4, .90), "hormigon", True)
    for x, tinte in ((-4.1, "oxido"), (2.5, "ocre")):
        casa = t.grupo(p, "ViviendaDeLadera", (x, 6.55, .9))
        muro_acristalado(t, casa, 6.3, 3.30, [(-1.5, 1.72)], 1.17, 2.25, "revoque_" + tinte)
        t.caja(casa, "ParedInterior", (0, 2.80, 1.65), (6.3, .16, 3.30), "revoque_cal", True)
        for lado in (-1, 1):
            t.caja(casa, "Costado", (lado * 3.08, 1.43, 1.65), (.14, 2.86, 3.30), "revoque_" + tinte, True)
        mesa(t, casa, (-1.50, .88, 0), 1.2, .65, .76)
        planta_decorativa(t, casa, (-1.50, .88, .76), .52)
        t.caja(casa, "Cornisa", (0, 1.35, 3.32), (6.5, 3.15, .14), "hormigon")
        t.caja(casa, "Puerta", (1.16, -.038, 1.06), (1.12, .10, 2.12), "roble", True)
        t.disco(casa, "Manija", (1.51, -.108, 1.07), .043, .045, "laton", eje="Y")
    for x in (-2.38, 2.38):
        t.disco(p, "Farola", (x, 1.56, 2.63), .048, 3.34, "grafito", True)
        esfera(t, p, "Globo", (x, 1.56, 4.36), .19, "lino")
    for x, n, huella, alto in ((-1.1, 6, .34, .15), (1.1, 9, .36, .10)):
        for i in range(n):
            z = (i + 1) * alto
            y = -(n - i - .5) * huella
            t.caja(p, "Peldano", (x, y, z / 2), (1.8, huella, z), "hormigon", True)
            t.suelo(p, "Nariz", (x, y - huella / 2 + .03), (1.77, .045), "ocre", z + .002)
        borde = x * 1.75
        for i in (0, n - 1):
            z = (i + 1) * alto
            y = -(n - i - .5) * huella
            t.disco(p, "Poste", (borde, y, z + .45), .035, .90, "grafito", True)
        barra = t.disco(p, "Baranda", (borde, -n * huella / 2, (.9 + alto) / 2 + .9), .04,
                        ((n - 1) * huella) * (1 + (alto / huella)**2)**.5, "laton", True)
        barra.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, huella, alto)).GetQuat())
    t.caja(p, "Descanso", (0, .9, .45), (4.0, 1.8, .90), "terrazo", True)
    t.caja(p, "MuroDeContencion", (0, 1.84, .94), (4.30, .14, 1.88), "ladrillo", True)
    cartel(t, p, "mirador", "MIRADOR", (0, 1.75, 1.55), 1.6)


def invernadero(t):
    p = t.raiz
    for x in (-3.4, 3.4):
        t.caja(p, "PanelTranslucido", (x, 0, 1.36), (.010, 9, 2.72), "vidrio", True)
        for y in (-4.4, -2.2, 0, 2.2, 4.4):
            t.caja(p, "Montante", (x, y, 1.4), (.09, .09, 2.8), "aluminio", True)
            t.varilla(p, "Cercha", (x, y, 2.8), (0, y, 3.65), .045, "aluminio")
        ala = t.grupo(p, "AlaTecho", (x / 2, 0, 3.225))
        UsdGeom.Xformable(t.stage.GetPrimAtPath(ala)).AddRotateYOp().Set(14.04 if x > 0 else -14.04)
        t.caja(ala, "Cubierta", (0, 0, 0), (3.51, 9, .010), "vidrio_satinado")
    for x in (-1.55, 1.55):
        mesa(t, p, (x, .9, 0), .85, 5.2, .72)
        exterior = x + math.copysign(.31, x)
        t.disco(p, "TuboDeRiego", (exterior, .90, .753), .019, 5.1, "grafito", eje="Y")
        for y in (-1.2, -.5, .2, .9, 1.6, 2.3, 3.0):
            marcada = x < 0 and y == .9
            g = t.grupo(p, "MacetaMarcada" if marcada else "Cultivo", (x, y, .72))
            especie = "albahaca" if marcada or y in (-1.2, .2, 2.3) else "roseta"
            planta_en_maceta(t, g, especie, marcada)
            if marcada:
                t.varilla(g, "Estaca", (.075, -.072, .188), (.075, -.072, .265), .004, "roble")
                cartel(t, g, "lote_albahaca", "ALBAHACA   04", (.075, -.073, .269), .092, "#365a36")
            # La boquilla queda al borde exterior para permitir retirar el tiesto.
            t.varilla(p, "Gotero", (exterior, y, .755), (x + math.copysign(.16, x), y, .92), .009,
                      "grafito")
            if marcada:
                cuerpo(t, g, .70)
    mesa(t, p, (1.55, -3.4, 0), 1.4, .82, .72)
    t.caja(p, "PieCartel", (1.55, -3.05, .97), (.04, .04, .5), "aluminio")
    cartel(t, p, "semillero", "SEMILLERO", (1.55, -3.06, 1.21), 1.2)
    t.disco(p, "DepositoRiego", (-2.58, 3.62, .66), .49, 1.32, "petroleo", True)
    for z in (.16, .44, 1.09, 1.30):
        t.disco(p, "AroDeposito", (-2.58, 3.62, z), .502, .038, "grafito")
    t.disco(p, "TapaDeposito", (-2.58, 3.62, 1.35), .25, .06, "grafito")
    for x in (-1.55, 1.55):
        exterior = x + math.copysign(.31, x)
        t.varilla(p, "Colector", (exterior, 3.44, .755), (exterior, 3.79, .755), .025, "tinta")
    t.varilla(p, "Colector", (-2.58, 3.79, .755), (1.86, 3.79, .755), .025, "tinta")
    for lado in (-1, 1):
        t.caja(p, "MureteVivero", (lado * 2.43, 4.43, .45), (1.84, .13, .90), "ladrillo", True)
        planta_decorativa(t, p, (lado * 2.55, 5.25, 0), 1.65)
    # Bancal de propagación al fondo y líneas de suspensión sobre cada cultivo.
    mesa(t, p, (0, 4.01, 0), 2.40, .70, .76)
    for x in (-.74, 0, .74):
        t.caja(p, "Almaciguera", (x, 4.01, .81), (.64, .46, .095), "grafito", True)
        for j, k in product(range(5), range(3)):
            xx, yy = x + (j - 2) * .116, 4.01 + (k - 1) * .125
            t.disco(p, "Alveolo", (xx, yy, .862), .043, .018, "tierra")
            for lado in (-1, 1):
                ovalo(t, p, "Cotiledon", (xx + lado * .020, yy, .889), (.027, .013, .017), "brote",
                      15 * lado)
    for x in (-1.55, 1.55):
        t.varilla(p, "CableTutor", (x, -3.95, 2.63), (x, 4.0, 2.63), .009, "grafito")
        for y in (-2.2, 0, 2.2):
            t.varilla(p, "Suspension", (x, y, 2.63), (x, y, 3.25), .008, "aluminio")
            plafon(t, p, (x, y, 2.60), .75, 16000.)


def interior(t, ancho, fondo, alto, acabado, ventanas=()):
    """Sala abierta por el frente, con cubierta y luz instalada en el cielo raso."""
    p = t.raiz
    t.suelo(p, "Pavimento", (0, 0), (ancho, fondo), "terrazo", .001)
    t.caja(p, "ParedFondo", (0, fondo / 2, alto / 2), (ancho, .15, alto), acabado, True)
    if ventanas:
        lateral = t.grupo(p, "VentanasComedor", (-ancho / 2, 0, 0), 90)
        muro_acristalado(t, lateral, fondo, alto, ventanas, 1.45, 2.59, acabado)
    else:
        t.caja(p, "ParedLateral", (-ancho / 2, 0, alto / 2), (.15, fondo, alto), acabado, True)
    t.caja(p, "ZocaloFondo", (0, fondo / 2 - .09, .12), (ancho, .05, .24), "grafito")
    t.caja(p, "ZocaloLateral", (-ancho / 2 + .09, 0, .12), (.05, fondo, .24), "grafito")
    t.caja(p, "ParedDerecha", (ancho / 2, 0, alto / 2), (.15, fondo, alto), acabado, True)
    t.caja(p, "CieloRaso", (0, 0, alto + .06), (ancho, fondo, .12), "marfil", True)
    for x, y in product((-.23 * ancho, .23 * ancho), (-.23 * fondo, .23 * fondo)):
        plafon(t, p, (x, y, alto - .03), min(1.55, ancho * .25), 44000.)


def almacen(t):
    p = t.raiz
    interior(t, 9.6, 11.0, 3.65, "hormigon")
    for y in range(-4, 5, 2):
        t.caja(p, "DefensaRack", (-4.67, y, .95), (.06, 1.8, .12), "ocre")
    for x in (-1.12, 1.12):
        t.suelo(p, "LineaPasillo", (x, 0), (.055, 8.4), "ocre", .003)
    cartel(t, p, "recepcion", "ENCOMIENDAS", (0, 5.40, 2.6), 3.2)
    t.caja(p, "PersianaMuelle", (0, 5.37, 1.10), (2.9, .09, 2.20), "aluminio", True)
    for j in range(17):
        t.caja(p, "LamaPersiana", (0, 5.315, .11 + j * .125), (2.86, .025, .022), "grafito")
    for x in (-3.35, 3.35):
        for j in range(5):
            t.caja(p, "TablaPalet", (x + (j - 2) * .22, 4.22, .16), (.19, 1.10, .05), "roble", True)
        for dx in (-.41, 0, .41):
            t.caja(p, "TacoPalet", (x + dx, 4.22, .07), (.13, 1.03, .14), "roble", True)
        for j, k in product((-1, 1), repeat=2):
            t.caja(p, "CajaEnPalet", (x + j * .24, 4.22 + k * .24, .41), (.46, .44, .45), "carton", True)
    for x, fila in ((-2.1, "A"), (2.1, "B")):
        for y in (-1.8, 1.8):
            g = t.grupo(p, "Rack" + fila, (x, y, 0))
            codigo = fila + ("1" if y < 0 else "2")
            cartel(t, g, "rack_" + codigo, codigo, (0, -1.33, 1.57), .74)
            for dx, dy in product((-.55, .55), (-1.27, 1.27)):
                t.caja(g, "Puntal", (dx, dy, 1.16), (.065, .065, 2.32), "tinta", True)
            for z in (.18, .84, 1.53, 2.25):
                t.caja(g, "Balda", (0, 0, z), (1.14, 2.62, .06), "ocre", True)
                if z < 2:
                    for dy in (-.82, .82):
                        altura = t.rng.uniform(.29, .49)
                        t.caja(g, "Bulto", (0, dy, z + .03 + altura / 2), (.68, .56, altura), "carton",
                               True)
                        t.caja(g, "Precinto", (0, dy, z + .031 + altura), (.074, .56, .005), "lino")
                        t.caja(g, "EtiquetaEnvio", (0, dy - .283, z + .18), (.20, .006, .10), "cal")
                        for j in range(9):
                            t.caja(g, "BarraCodigo", (-.078 + j * .015, dy - .288, z + .184),
                                   (.005 if j % 3 else .009, .003, .064), "grafito")
    mesa(t, p, (0, -3.95, 0), 1.5, .85, .76)
    g = caja_movil(t, "Encomienda", (0, -3.95, .925), (.44, .34, .33), "carton", 1.4)
    t.caja(g, "Cinta", (0, 0, .165), (.07, .34, .003), "ocre")
    t.caja(g, "Etiqueta", (.12, .04, .166), (.12, .12, .003), "cal")
    t.suelo(p, "HuecoDeEntrega", (1.91, 1.8), (.65, .80), "hoja", .872)
    cartel(t, p, "despacho", "DESPACHO  B2", (2.1, .46, 2.24), 1.12)


def boton(t, p, nombre, pos):
    # El muelle devuelve el pulsador a cero. Su desplazamiento +Y se puede medir.
    g = t.grupo(p, nombre, pos)
    t.disco(g, "Pulsador", (0, 0, 0), .041, .028, "ocre", True, "Y")
    cuerpo(t, g, .035)
    junta = UsdPhysics.PrismaticJoint.Define(t.stage, g + "/Guia")
    junta.CreateBody1Rel().SetTargets([g])
    junta.CreateAxisAttr("Y")
    junta.CreateLocalPos0Attr(pos)
    junta.CreateLowerLimitAttr(0.)
    junta.CreateUpperLimitAttr(.012)
    drive = UsdPhysics.DriveAPI.Apply(junta.GetPrim(), "linear")
    drive.CreateStiffnessAttr(240.)
    drive.CreateDampingAttr(3.)
    drive.CreateTargetPositionAttr(0.)


def ascensor(t):
    p = t.raiz
    interior(t, 7.3, 8.0, 3.0, "revoque_arena")
    for x in (-1.94, 1.94):
        t.caja(p, "Frente", (x, 0, 1.38), (2.44, .16, 2.76), "revoque_marfil", True)
    t.caja(p, "Dintel", (0, 0, 2.56), (1.44, .16, .40), "aluminio", True)
    for lado in (-1, 1):
        t.caja(p, "LateralCabina", (lado * 1.38, 1.45, 1.27), (.09, 2.76, 2.54), "aluminio", True)
        a, b = (lado * .355, .15, 1.16), (lado * 1.09, .15, 1.16)
        g = t.grupo(p, "Hoja", a)
        t.caja(g, "Panel", (0, 0, 0), (.708, .065, 2.32), "aluminio", True)
        cuerpo(t, g, 35., ((0, a), (1, a), (3, b), (16, b), (18, a)))
    t.caja(p, "FondoCabina", (0, 2.86, 1.3), (2.84, .13, 2.60), "petroleo", True)
    t.suelo(p, "PavimentoCabina", (0, 1.48), (2.66, 2.62), "roble", .004)
    for x in (-.76, .76):
        t.caja(p, "JambaInox", (x, -.098, 1.19), (.082, .10, 2.38), "aluminio", True)
    t.caja(p, "MarcoSuperior", (0, -.098, 2.34), (1.59, .10, .065), "aluminio")
    t.caja(p, "MarcoEspejo", (-.28, 2.783, 1.67), (1.72, .024, .96), "metal_torneado", bisel=.008)
    t.caja(p, "EspejoCabina", (-.28, 2.762, 1.67), (1.67, .012, .91), "espejo", bisel=.003)
    t.disco(p, "PasamanosCabina", (0, 2.66, .92), .028, 2.27, "aluminio", True, "X")
    for x in (-.91, .91):
        t.disco(p, "AnclajeBarra", (x, 2.73, .92), .022, .15, "aluminio", eje="Y")
    lateral = t.grupo(p, "PasamanosLateral", (-1.30, 1.55, 0), 90)
    t.disco(lateral, "Barra", (0, -.08, .92), .028, 1.99, "metal_torneado", True, "X")
    for x in (-.77, .77):
        t.disco(lateral, "Anclaje", (x, -.026, .92), .023, .11, "aluminio", eje="Y")
    cartel(t, lateral, "placa_cabina", "CABINA  02", (0, -.012, 1.62), .80)
    for y in (.035, .081, .127, .173, .219):
        t.caja(p, "RanuraGuia", (0, y, .007), (1.42, .012, .004), "aluminio")
    tablero(t, p, "directorio_edificio", "DIRECTORIO", [("02", "Administración"), ("01", "Recepción"),
                                                        ("PB", "Vestíbulo")], (-1.93, -.124, 1.50), 1.13)
    planta_decorativa(t, p, (-2.75, -.69, .002), 1.10)
    cartel(t, p, "indicador_cabina", "01   ▲", (1.04, -.114, 1.75), .47, "#182e2c")
    t.suelo(p, "Umbral", (0, .12), (1.45, .28), "grafito", .003)
    for x, y, nombre in ((1.04, -.11, "Llamada"), (.81, 2.755, "PisoDos")):
        t.caja(p, "Botonera", (x, y, 1.02), (.18, .035, .32), "grafito", True)
        boton(t, p, nombre, (x, y - .048, 1.02))
    cartel(t, p, "ascensor", "ASCENSOR  02", (0, -.09, 2.58), 1.65)


def vajilla(t, p, nombre, pos, perfil, acabado):
    """Perfil cerrado de revolución; el borde y la pared interior se ven a través del cristal."""
    n = 48
    anillos = [[(r * math.cos(i * math.tau / n), r * math.sin(i * math.tau / n), z) for i in range(n)]
               for r, z in perfil]
    caras = [anillos[0][::-1], anillos[-1]]
    for a, b in zip(anillos, anillos[1:]):
        caras.extend([a[i - 1], a[i], b[i], b[i - 1]] for i in range(n))
    return t.malla(p, nombre, caras, acabado, pos)


def comedor(t):
    p = t.raiz
    interior(t, 9.4, 10.4, 3.45, "revoque_marfil", ventanas=[(-2.58, 2.01), (.1, 2.01), (2.75, 2.01)])
    t.caja(p, "Salpicadero", (0, 5.08, 1.35), (8.95, .028, .71), "azulejo")
    # La línea de servicio ocupa la pared del fondo, fuera del corredor entre mesas.
    t.caja(p, "IslaDeServicio", (-2.43, 4.38, .48), (3.28, .88, .96), "petroleo", True)
    t.caja(p, "EncimeraInox", (-2.43, 4.38, .99), (3.36, .96, .075), "inox", True, bisel=.009)
    for x, comida in ((-3.47, "ocre"), (-2.58, "hoja"), (-1.69, "oxido")):
        t.caja(p, "CubetaGastronomica", (x, 4.38, 1.04), (.72, .52, .08), "aluminio", bisel=.027)
        t.caja(p, "Racion", (x, 4.38, 1.085), (.61, .43, .035), comida, bisel=.015)
    for x in (-3.90, -.96):
        t.disco(p, "MontanteVitrina", (x, 4.03, 1.28), .024, .55, "aluminio")
    t.caja(p, "ProtectorVitrina", (-2.43, 4.03, 1.37), (2.98, .020, .38), "vidrio")
    tablero(t, p, "menu_universitario", "MENÚ DEL DÍA", [("Entrada", "Sopa de verduras"),
                                                         ("Fondo", "Arroz con pollo"),
                                                         ("Bebida", "Chicha morada")], (.52, 5.075, 2.22),
            1.65)
    for x, y in product((-2.20, 2.20), (-1.1, 1.8)):
        mesa(t, p, (x, y, 0), 1.65, .92, .76)
        for dx in (-.48, .48):
            silla(t, p, (x + dx, y + .75, 0))
        vajilla(t, p, "Plato", (x + .48, y + .20, .76), [(.067, .001), (.078, .008), (.142, .025),
                                                         (.144, .031), (.133, .030), (.087, .013)],
                "ceramica")
        t.caja(p, "Servilleta", (x + .19, y + .18, .766), (.11, .18, .007), "lino")
        t.varilla(p, "MangoCuchara", (x + .19, y + .12, .774), (x + .19, y + .239, .774), .008, "aluminio")
        ovalo(t, p, "Cuchara", (x + .19, y + .254, .777), (.017, .027, .006), "aluminio")
        taza = t.grupo(p, "Taza", (x + .68, y + .24, .76))
        vajilla(t, taza, "Cuerpo", (0, 0, 0), [(.034, 0), (.043, .012), (.050, .109), (.048, .112),
                                               (.044, .110), (.038, .018)], "ceramica")
        puntos = [(.042 + .041 * math.sin(a * math.pi / 8), 0, .064 + .033 * math.cos(a * math.pi / 8))
                  for a in range(9)]
        for a, b in zip(puntos, puntos[1:]):
            t.varilla(taza, "Asa", a, b, .007, "ceramica")
    mesa(t, p, (0, -3.8, 0), 2.1, .84, .80)
    g = caja_movil(t, "Bandeja", (0, -3.8, .813), (.46, .34, .026), "petroleo", .36)
    for lado in (-1, 1):
        t.caja(g, "Borde", (lado * .222, 0, .022), (.016, .34, .028), "petroleo", True)
        t.caja(g, "Borde", (0, lado * .162, .022), (.46, .016, .028), "petroleo", True)
    vaso = t.grupo(p, "Vaso", (.08, -3.79, .904))
    vajilla(t, vaso, "Cristal", (0, 0, -.078), [(.033, 0), (.038, .008), (.048, .152), (.047, .156),
                                                (.044, .155), (.034, .016)], "vidrio")
    colision = t.disco(vaso, "VolumenVaso", (0, 0, 0), .048, .156, "grafito", True)
    UsdGeom.Imageable(colision).MakeInvisible()
    cuerpo(t, vaso, .22)
    t.caja(p, "MostradorRetorno", (1.3, 4.66, .53), (2.8, .79, 1.06), "petroleo", True)
    for i in range(6):
        t.disco(p, "PilaDePlatos", (1.80, 4.6, 1.08 + .022 * i), .16, .025, "ceramica")
    cartel(t, p, "comedor", "COMEDOR", (-1.65, 5.10, 2.44), 2.6)


def taller(t):
    p = t.raiz
    interior(t, 5.2, 4.6, 2.8, "revoque_marfil")
    t.caja(p, "ArmarioRepuestos", (-1.93, 1.77, .84), (.77, .66, 1.68), "tinta", True)
    for z in (.37, .77, 1.17):
        t.caja(p, "Cajon", (-1.93, 1.42, z), (.70, .05, .35), "grafito")
        t.caja(p, "Tirador", (-1.93, 1.38, z + .08), (.30, .04, .035), "aluminio")
    mesa(t, p, (0, 1, 0), 2.5, .96, .84)
    t.caja(p, "PanelHerramientas", (0, 1.53, 1.47), (2.52, .075, 1.24), "petroleo", True)
    for i in range(40):
        for j in range(13):
            t.disco(p, "Perforacion", (-1.17 + i * .060, 1.49, 1.03 + j * .078), .009, .006, "grafito",
                    eje="Y")
    for i, x in enumerate((-.98, -.76, -.54, -.32)):
        largo = .27 + i * .038
        t.disco(p, "Gancho", (x, 1.452, 1.77), .013, .092, "aluminio", eje="Y")
        t.caja(p, "MangoLlave", (x, 1.429, 1.74 - largo / 2), (.033, .021, largo), "aluminio", bisel=.006)
        for lado in (-1, 1):
            t.varilla(p, "BocaLlave", (x, 1.428, 1.73), (x + lado * .043, 1.428, 1.77), .014, "aluminio")
            t.varilla(p, "Mordiente", (x + lado * .043, 1.428, 1.77), (x + lado * .035, 1.428, 1.806), .013,
                      "aluminio")
    for i in range(4):
        x = .18 + i * .225
        t.caja(p, "CajaTornilleria", (x, 1.375, 1.16), (.204, .23, .13), "tinta", bisel=.012)
        t.caja(p, "EtiquetaCajetin", (x, 1.255, 1.16), (.105, .006, .043), "lino")
    t.caja(p, "PieTornilloBanco", (-.91, .87, .883), (.32, .29, .086), "tinta", True, bisel=.015)
    for y in (.78, .89):
        t.caja(p, "Mordaza", (-.91, y, .99), (.235, .077, .16), "tinta", True, bisel=.009)
        t.caja(p, "BocaAcero", (-.91, y, 1.066), (.244, .069, .031), "aluminio", True)
    t.disco(p, "Husillo", (-.91, .724, .969), .026, .21, "aluminio", eje="Y")
    t.disco(p, "Maneral", (-.91, .613, .969), .010, .19, "aluminio", eje="X")
    t.disco(p, "BaseLampara", (.99, 1.15, .868), .115, .056, "grafito", True)
    t.varilla(p, "BrazoLampara", (.99, 1.15, .88), (1.04, 1.20, 1.30), .020, "grafito")
    t.varilla(p, "BrazoLampara", (1.04, 1.20, 1.30), (.79, 1.07, 1.49), .017, "grafito")
    plafon(t, p, (.77, 1.06, 1.48), .25, 1200.)
    cartel(t, p, "ajuste", "BANCO DE AJUSTE", (.60, 1.47, 1.80), 1.10)
    g = t.grupo(p, "PiezaPerforada", (-.28, .85, .84))
    # Cuatro sólidos dejan un paso cuadrado real de 32 mm por toda la pieza.
    for lado in (-1, 1):
        t.caja(g, "Larguero", (lado * .053, 0, .034), (.074, .14, .068), "tinta", True)
        t.caja(g, "Puente", (0, lado * .043, .034), (.032, .054, .068), "tinta", True)
    cuerpo(t, g, .32)
    pin = t.grupo(p, "Pasador", (.29, .85, .84))
    t.disco(pin, "Vastago", (0, 0, .054), .012, .108, "laton", True)
    t.disco(pin, "Cabeza", (0, 0, .115), .023, .014, "ocre", True)
    cuerpo(t, pin, .06)


# Los puntos indican dónde se apoyan los pies; la meta de un depósito queda frente al mueble.
ESCENAS = {
    "mercado": ("Mercado de abastos", mercado, [(0, -3.6, 0), (0, -1.3, 0), (-3.05, 1.25, 0), (0, -1.5, 0),
                                                (3.3, -2.4, 0)], (1.3, -12.9, 3.35)),
    "estacion": ("Estación del Metropolitano", estacion, [(1, -3, .4), (1, .4, .4), (0, .6, .4),
                                                          (0, 1.8, .4), (1, 2.3, .4)], (.45, -10.0, 3.45)),
    "escalinata": ("Escalinata en ladera", escalinata, [(-1.1, -3.5, 0), (-1.1, -1.19, .45), (-1.1, .8, .9),
                                                        (1.1, .8, .9), (1.1, -3.8, 0)], (5, -7, 4.8)),
    "invernadero": ("Invernadero", invernadero, [(0, -4, 0), (0, .9, 0), (-.94, .9, 0), (0, -2.5, 0),
                                                 (.56, -3.4, 0)], (.45, -6.5, 2.4)),
    "almacen": ("Almacén de encomiendas", almacen, [(0, -2.95, 0), (0, -1.4, 0), (0, 1.8, 0),
                                                    (1.12, 1.8, 0)], (1.2, -5.85, 2.12)),
    "ascensor": ("Ascensor de edificio", ascensor, [(1, -2.5, 0), (1.04, -.60, 0), (1.2, -1., 0),
                                                    (0, -.62, 0), (0, 1, 0),
                                                    (.81, 2.27, 0)], (1.05, -4.65, 1.78)),
    "comedor": ("Comedor universitario", comedor, [(0, -2.91, 0), (0, -1.1, 0), (0, .6, 0),
                                                   (1.7, .97, 0)], (2.92, -5.06, 1.93)),
    "taller": ("Taller de mantenimiento", taller, [(0, -1.35, 0), (0, .12, 0)], (1.75, -2.25, 2.35)),
}

# Posiciones en metros en el sistema local del objeto. La vista del andén usa el sistema del mundo.
PRIMEROS_PLANOS = {
    "mercado":
    [("DetalleCitricos", "Cítricos y pulpa", "PuestoFrutas", (.52, -.83, 1.47), (.56, -.09, .94), 42),
     ("DetalleTomates", "Tomates y cálices", "PuestoVerduras", (.32, -.73, 1.37), (.66, -.13, .96), 43),
     ("DetallePan", "Corteza y ajonjolí", "PuestoPan", (-.32, -.68, 1.36), (-.70, -.07, .98), 47),
     ("DetallePescado", "Escamas, aletas y hielo", "PuestoPescados", (.35, -.77, 1.51), (.67, -.05, .99),
      38),
     ("DetalleGranos", "Tejido del saco y legumbres", "PuestoGranos", (.22, -.77, 1.67), (0, -.03, 1.22),
      40), ("DetalleFlores", "Pétalos y follaje", "PuestoFlores", (-.54, -.70, 1.78), (-.91, 0, 1.50), 40)],
    "estacion": [("DesdeAnden", "Desde el andén · a través de la ventanilla", "", (-1.57, .70, 2.02),
                  (-2.70, 3.20, 1.48), 23),
                 ("DentroDelBus", "Desde el pasillo · hacia la cabecera", "BusDetenido", (-.48, -.55, 1.66),
                  (3.72, .29, 1.46), 20)],
    "invernadero": [("DetalleAlbahaca", "Albahaca · tallos, hojas y pecíolos", "MacetaMarcada",
                     (.53, -.86, .51), (0, 0, .267), 31),
                    ("DetalleMaceta", "Arcilla torneada, borde y sustrato", "Cultivo_01", (.23, -.38, .39),
                     (.035, -.065, .192), 40)],
}


def camara(t, nombre, ojo, centro, focal=23):
    cam = UsdGeom.Camera.Define(t.stage, t.raiz + "/" + nombre)
    vista = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*ojo), Gf.Vec3d(*centro), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(vista.GetInverse())
    cam.CreateHorizontalApertureAttr(32.)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr((.02, 100.))


def construir(nombre):
    titulo, diseno, puntos, ojo = ESCENAS[nombre]
    # Construir en memoria evita truncar la escena existente si falla su diseño.
    contexto = Ar.ResolverContext(Ar.DefaultResolverContext([str(DESTINO)]))
    stage = Usd.Stage.CreateInMemory("servicio.usda", contexto)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetFramesPerSecond(FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(30 * FPS)
    UsdPhysics.Scene.Define(stage, "/World/Fisica").CreateGravityMagnitudeAttr(9.81)
    cielo = UsdLux.DomeLight.Define(stage, "/World/Luz")
    cielo.CreateIntensityAttr(540.)
    cielo.CreateTextureFileAttr(Sdf.AssetPath("./texturas/cielo_costero.jpg"))
    cielo.CreateTextureFormatAttr("latlong")
    sol = UsdLux.DistantLight.Define(stage, "/World/Sol")
    sol.CreateIntensityAttr(1800.)
    sol.AddRotateXYZOp().Set((35, -40, 0))
    generar_texturas(DESTINO / "texturas" / "servicio_pbr")
    t = TallerServicio(stage)
    t.caja(t.raiz, "Suelo", (0, 0, -.08), (120, 120, .16), "hormigon", True)
    diseno(t)
    preparar_materiales(t)
    ruta = t.grupo(t.raiz, "Recorrido")
    for i, punto in enumerate(puntos):
        hito = "Inicio" if i == 0 else "Meta" if i == len(puntos) - 1 else f"Paso_{i:02}"
        t.grupo(ruta, hito, punto)
    centro = (0, 1, .96) if nombre == "taller" else (0, 0, .85)
    camara(t, "Vista", ojo, centro)
    if nombre == "mercado":
        camara(t, "VistaDetalle", (.80, -.62, 1.76), (3.0, 2.43, 1.18))
    elif nombre == "estacion":
        camara(t, "VistaDetalle", (3.72, -4.48, 1.98), (4.10, -.55, 1.40))
    elif nombre == "ascensor":
        camara(t, "VistaMaterial", (.68, .54, 1.58), (-.30, 2.76, 1.48))
    elif nombre == "invernadero":
        camara(t, "VistaMaterial", (-4.20, -1.77, 1.51), (-1.50, .35, 1.09))
    cache = UsdGeom.XformCache()
    for identificador, _, ancla, pos, objetivo, focal in PRIMEROS_PLANOS.get(nombre, []):
        prim = stage.GetPrimAtPath(t.raiz + "/" + ancla if ancla else t.raiz)
        matriz = cache.GetLocalToWorldTransform(prim)
        camara(t, identificador, matriz.Transform(Gf.Vec3d(*pos)), matriz.Transform(Gf.Vec3d(*objetivo)),
               focal)
    stage.GetRootLayer().customLayerData = {
        "titulo": titulo,
        "cameraSettings": {
            "Perspective": {
                "position": Gf.Vec3d(*ojo),
                "target": Gf.Vec3d(*centro)
            }
        }
    }
    archivo = DESTINO / f"{nombre}.usda"
    stage.GetRootLayer().Export(str(archivo))
    print(archivo, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("escenarios", nargs="*", help="Nombres a regenerar. Por defecto genera los ocho.")
    args = parser.parse_args()
    for nombre in args.escenarios:
        if nombre not in ESCENAS:
            parser.error(f"Escenario desconocido {nombre}. Opciones: {', '.join(ESCENAS)}")
    for nombre in args.escenarios or ESCENAS:
        construir(nombre)
