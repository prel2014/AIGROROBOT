"""Construye la vivienda, la calle y el campus con la pista articulada común."""
from itertools import product
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from geometria_escenarios import Taller, banco, cojin, estante, mesa, planta, silla

HERE = Path(__file__).resolve().parent
DESTINO = HERE / "escenarios"


def rotulos():
    letreros = {
        "pan": ("PAN  DE  BARRIO", "PANADERÍA   /   CAFÉ", "#904a30"),
        "botica": ("BOTICA", "SALUD   /   CUIDADO", "#226964"),
        "papel": ("PAPELERÍA", "LIBROS   ·   COPIAS", "#344f65"),
        "bodega": ("LA  ESQUINA", "ABARROTES   /   MERCADO", "#816433"),
        "biblioteca": ("BIBLIOTECA", "CAMPUS   /   SALA DE LECTURA", "#274e57"),
        "aulas": ("AULAS", "PABELLÓN   /   01", "#8b4c35"),
        "paradero": ("PARADERO", "TRANSPORTE URBANO", "#285865"),
    }
    try:
        fuente = ImageFont.truetype("arialbd.ttf", 72)
    except OSError:
        fuente = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    for clave, (titulo, pie, fondo) in letreros.items():
        img = Image.new("RGB", (768, 192), fondo)
        d = ImageDraw.Draw(img)
        d.rectangle((24, 21, 32, 168), fill="#e6bd75")
        d.text((58, 77), titulo, anchor="lm", font=fuente, fill="#faf2df")
        d.text((61, 143), pie, anchor="lm", font=fuente.font_variant(size=26), fill="#f0d8a7")
        img.save(DESTINO / "texturas" / f"rotulo_{clave}.png")


def ventana(t, p, x, y, z, ancho, alto, luz_natural=False):
    if luz_natural:
        for lado in (-1, 1):
            t.caja(p, "Marco", (x + lado * ancho / 2, y, z), (.055, .10, alto), "grafito")
            t.caja(p, "Travesano", (x, y, z + lado * alto / 2), (ancho, .10, .05), "grafito")
    else:
        t.caja(p, "Marco", (x, y, z), (ancho + .10, .10, alto + .10), "grafito")
        t.caja(p, "Cristal", (x, y - .057, z), (ancho, .026, alto), "vidrio")
    for k in (-.25, .25):
        t.caja(p, "Parteluz", (x + ancho * k, y - .08, z), (.034, .04, alto), "aluminio")
    t.caja(p, "Alfeizar", (x, y - .05, z - alto / 2 - .035), (ancho + .18, .23, .065), "hormigon")


def plafon(t, p, pos, ancho=1.2, intensidad=5200.):
    t.caja(p, "PerfilLED", pos, (ancho, .38, .038), "ceramica")
    luz = UsdLux.RectLight.Define(t.stage, t.ruta(p, "LuzInterior"))
    luz.AddTranslateOp().Set((pos[0], pos[1], pos[2] - .025))
    luz.CreateWidthAttr(ancho)
    luz.CreateHeightAttr(.36)
    luz.CreateIntensityAttr().Set(intensidad)
    luz.CreateColorAttr().Set((1., .89, .75))


def muro_casa(t, p):
    # Dos puertas de 1.8 m, alineadas con los recorridos de la pista.
    for centro, ancho in ((-6.45, 4.7), (0, 4.6), (6.45, 4.7)):
        t.caja(p, "Antepecho", (centro, 0, .49), (ancho, .20, .98), "cal", True)
        t.caja(p, "Zocalo", (centro, -.115, .09), (ancho, .04, .18), "roble")
        ventana(t, p, centro, 0, 1.82, ancho - .32, 1.55, luz_natural=True)
        t.caja(p, "Dintel", (centro, 0, 2.96), (ancho, .20, .70), "cal", True)
        for lado in (-1, 1):
            t.caja(p, "Jamba", (centro + lado * (ancho / 2 - .08), 0, 1.83), (.16, .22, 1.70), "cal", True)
        cristal = t.caja(p, "CierreVentana", (centro, .04, 1.83), (ancho - .32, .04, 1.65), "vidrio", True)
        UsdGeom.Imageable(cristal.GetPrim()).MakeInvisible()
    for x in (-3.2, 3.2):
        t.caja(p, "VigaPuerta", (x, 0, 2.90), (1.8, .26, .82), "roble", True)


def vivienda(t):
    p = t.raiz
    t.suelo(p, "Jardin", (0, 0), (46, 46), "hormigon")
    t.suelo(p, "Parquet", (0, 0), (17.65, 17.65), "roble", .004)
    for xy, giro in (((0, 8.8), 0), ((8.8, 0), -90), ((0, -8.8), 180), ((-8.8, 0), 90)):
        muro_casa(t, t.grupo(p, "Cerramiento", (*xy, 0), giro))
    for xy, giro in (((0, 13.8), 0), ((13.8, 0), -90), ((0, -13.8), 180), ((-13.8, 0), 90)):
        jardin = t.grupo(p, "PatioExterior", (*xy, 0), giro)
        t.caja(jardin, "Medianera", (0, 0, .87), (27.6, .22, 1.74), "arena", True)
        t.caja(jardin, "Albardilla", (0, 0, 1.77), (27.6, .30, .07), "marfil")
        for x in (-7.2, 0, 7.2):
            planta(t, jardin, (x, -1.6, 0), 2.15)
        palmera(t, jardin, (5.1, -1.0, 0), 3.45)
    for lado in (-1, 1):
        t.caja(p, "TechoGaleria", (0, lado * 6.17, 3.33), (17.72, 5.38, .16), "cal", True)
        t.caja(p, "TechoLateral", (lado * 6.17, 0, 3.33), (5.38, 6.96, .16), "cal", True)
    for x in (-3.44, 0, 3.44):
        t.caja(p, "VigaLucernario", (x, 0, 3.30), (.14, 7.0, .22), "roble")
    for x, y in product((-6.1, 6.1), (-6., 6.)):
        plafon(t, p, (x, y, 3.20), 1.48, 12500.)
    for x, y in ((0, -5.4), (0, 5.4), (-5.4, 0), (5.4, 0)):
        plafon(t, p, (x, y, 3.20), 1.40, 9500.)
    t.caja(p, "TabiqueDormitorio", (7.14, -4.83, 1.6), (3.15, .14, 3.2), "marfil", True)
    t.caja(p, "TabiqueComedor", (-7.30, -4.83, 1.6), (2.92, .14, 3.2), "arena", True)
    salon = t.grupo(p, "Sala", (-5.95, 5.80, 0))
    t.suelo(salon, "Alfombra", (0, -.42), (3.65, 2.50), "tejido", .009)
    t.caja(salon, "SofaBase", (0, .72, .32), (2.65, .94, .46), "tejido", True, .10)
    t.caja(salon, "Respaldo", (0, 1.11, .72), (2.62, .23, .74), "tejido", True, .10)
    for x in (-1.23, 1.23):
        t.caja(salon, "Brazo", (x, .70, .55), (.24, .94, .55), "tejido", True, .085)
    for x in (-.72, 0, .72):
        cojin(t, salon, "CojinAsiento", (x, .60, .565), (.72, .73, .17), "tejido")
    for x, acabado in ((-.72, "ocre"), (.77, "coral")):
        cojin(t, salon, "Almohadon", (x, .89, .82), (.49, .20, .47), acabado, -12)
    mesa(t, salon, (0, -.66, 0), 1.28, .62, .39)
    t.caja(salon, "Revista", (-.27, -.65, .41), (.33, .22, .035), "coral")
    planta(t, salon, (-1.89, .58, 0), 1.35)
    t.disco(salon, "PieLampara", (1.76, .74, .045), .21, .08, "laton")
    t.disco(salon, "BarraLampara", (1.76, .74, .78), .022, 1.50, "laton")
    t.disco(salon, "Pantalla", (1.76, .74, 1.49), .27, .35, "lino")
    cocina = t.grupo(p, "Cocina", (5.95, 7.95, 0))
    t.suelo(cocina, "Baldosa", (0, -1.30), (4.72, 4.08), "terrazo", .009)
    t.caja(cocina, "MuebleBajo", (0, 0, .44), (4.35, .71, .88), "petroleo", True)
    t.caja(cocina, "Encimera", (0, -.02, .92), (4.42, .79, .10), "terrazo", True, .022)
    for x in (-1.65, -.83, 0, .83, 1.65):
        t.caja(cocina, "Frente", (x, -.366, .47), (.78, .029, .75), "petroleo", False, .016)
        t.caja(cocina, "Tirador", (x, -.393, .73), (.22, .035, .025), "laton")
        t.caja(cocina, "Alacena", (x, .06, 1.95), (.80, .48, .73), "marfil", True, .015)
    t.caja(cocina, "Fregadero", (-.8, -.03, .98), (.73, .47, .026), "aluminio", False, .018)
    t.caja(cocina, "Cubeta", (-.8, -.04, .996), (.60, .34, .015), "grafito", False, .014)
    for x, y in ((.65, -.15), (1.01, .12), (1.44, -.15)):
        t.disco(cocina, "Quemador", (x, y, .985), .13, .02, "grafito")
    t.varilla(cocina, "Grifo", (-.81, .21, .97), (-.81, .21, 1.23), .018, "aluminio")
    t.varilla(cocina, "Cano", (-.81, .21, 1.23), (-.81, .05, 1.23), .017, "aluminio")
    cuarto = t.grupo(p, "Dormitorio", (5.98, -6.12, 0), 180)
    t.caja(cuarto, "Cabecero", (0, 1.12, .63), (2.21, .16, 1.25), "roble", True, .045)
    t.caja(cuarto, "Base", (0, 0, .24), (1.90, 2.20, .43), "roble", True, .03)
    t.caja(cuarto, "Colchon", (0, 0, .50), (1.86, 2.13, .26), "lino", True, .09)
    t.caja(cuarto, "Cubrecama", (0, -.35, .635), (1.89, 1.48, .12), "tejido", False, .055)
    for x in (-.46, .46):
        cojin(t, cuarto, "Almohada", (x, .75, .70), (.75, .44, .20), "cal", 5)
    for x in (-1.39, 1.39):
        t.caja(cuarto, "Mesita", (x, .78, .27), (.52, .55, .54), "roble", True, .025)
    mesa(t, p, (-5.9, -5.98, 0), 1.85, .93)
    for x, y in product((-6.43, -5.37), (-6.77, -5.19)):
        silla(t, p, (x, y, 0), 180 if y < -6 else 0)
    estante(t, p, (-6.1, -7.87, 0), 180)
    planta(t, p, (.95, .75, 0), 1.62)
    return (-3.34, 3.14, 1.57), (-5.96, 6.42, .93)


def palmera(t, p, pos, altura=4.8):
    g = t.grupo(p, "Palmera", pos)
    t.disco(g, "Tronco", (0, 0, altura / 2), .13, altura, "arena", True)
    for i in range(1, int(altura / .19)):
        t.disco(g, "Cicatriz", (0, 0, i * .19), .137, .027, "lino")
    caras = [[], []]
    for i in range(16):
        a = i * 2.40
        d = np.array((math.cos(a), math.sin(a), 0))
        lateral = np.array((-d[1], d[0], 0))
        longitud = t.rng.uniform(1.34, 2.16)

        def nervio(s):
            return d * longitud * s + [0, 0, altura + .70 * math.sin(s * math.pi) - .48 * s]

        for j in range(1, 15):
            s = j / 16
            base = nervio(s)
            t0 = nervio(s + .055)
            caras[i % 2].append(
                [base - .016 * lateral, t0 - .014 * lateral, t0 + .014 * lateral, base + .016 * lateral])
            for signo in (-1, 1):
                largo = .09 + .36 * math.sin(math.pi * s)
                punta = base + signo * lateral * largo + d * (.10 + .13 * s) - [0, 0, largo * .56]
                # Foliolos arqueados con una nervadura central y una punta colgante.
                ancho = .032 * math.sin(math.pi * s) + .008
                filas = []
                for u in np.linspace(0., 1., 8):
                    centro = (1. - u) * base + u * punta
                    centro[2] += .14 * math.sin(math.pi * u)
                    mitad = ancho * (.35 + .65 * math.sin(math.pi * u)) * (1. - u)
                    filas.append((centro - d * mitad, centro + [0, 0, mitad * .35], centro + d * mitad))
                for inferior, superior in zip(filas[:-2], filas[1:-1]):
                    for k in (0, 1):
                        caras[i % 2].append([inferior[k], inferior[k+1], superior[k+1], superior[k]])
                for k in (0, 1):
                    caras[i % 2].append([filas[-2][k], filas[-2][k+1], punta])
    for hojas, acabado in zip(caras, ("hoja", "brote")):
        t.malla(g, "Corona", hojas, acabado)


def local(t, p, pos, giro, acabado, letrero):
    g = t.grupo(p, "Comercio", pos, giro)
    t.caja(g, "Edificio", (0, 1.9, 2.90), (5.95, 3.8, 5.80), acabado, True)
    t.caja(g, "Cornisa", (0, 0, 5.73), (6.08, .32, .18), "cal")
    t.caja(g, "Basamento", (0, -.09, .29), (5.94, .20, .58), "hormigon")
    for x in (-1.48, 1.48):
        ventana(t, g, x, -.12, 4.30, 2.13, 1.32)
    ventana(t, g, -.88, -.13, 1.41, 3.38, 1.63)
    t.panel(g, "Escaparate", (-.88, -.205, 1.41), 3.37, 1.62, "vitrina_" + letrero)
    t.caja(g, "MarcoPuerta", (1.72, -.13, 1.19), (1.29, .20, 2.38), "grafito")
    t.caja(g, "Puerta", (1.72, -.245, 1.22), (1.11, .03, 2.25), "vidrio")
    t.caja(g, "Manija", (2.07, -.28, .98), (.037, .05, .31), "laton")
    t.caja(g, "Fascia", (0, -.18, 2.96), (5.70, .21, .69), "marfil")
    t.rotulo(g, "Letrero", (0, -.295, 2.97), 2.57, .61, letrero)
    for i in range(12):
        x = (i - 5.5) * .436
        t.malla(g, "Toldo", [[(x - .218, -.15, 2.59), (x + .218, -.15, 2.59), (x + .218, -.84, 2.40),
                              (x - .218, -.84, 2.40)],
                             [(x - .218, -.84, 2.40), (x + .218, -.84, 2.40), (x + .218, -.84, 2.27),
                              (x - .218, -.84, 2.27)]], acabado if i % 2 else "marfil")


def automovil(t, p, pos, giro, color):
    from scipy.interpolate import PchipInterpolator
    g = t.grupo(p, "AutomovilEstacionado", pos, giro)
    proxy = t.caja(g, "ColisionCarroceria", (0, 0, .56), (3.89, 1.62, .59), color, True)
    UsdGeom.Imageable(proxy).MakeInvisible()
    # Secciones transversales de la chapa. Los arcos recortan el faldón sobre cada neumático.
    estaciones = [-1.95, -1.75, -1.25, -.76, .70, 1.30, 1.77, 1.95]
    manga = PchipInterpolator(estaciones, [.57, .76, .82, .83, .83, .82, .77, .64])
    altura = PchipInterpolator(estaciones, [.64, .73, .81, .88, .88, .84, .81, .72])
    anillos = []
    for x in np.linspace(estaciones[0], estaciones[-1], 81):
        dx = min(abs(x + 1.24), abs(x - 1.23))
        base = .29 if dx >= .36 else .33 + math.sqrt(.36**2 - dx**2)
        alto, ancho = float(altura(x)), float(manga(x))
        perfil = [(ancho * .91, base), (ancho, base + .32 * (alto - base)),
                  (ancho, alto - .05), (ancho * .91, alto), (ancho * .67, alto + .025)]
        anillos.append([(x, y, z) for y, z in perfil] + [(x, -y, z) for y, z in reversed(perfil)])
    caras = [anillos[0], anillos[-1][::-1]]
    for a, b in zip(anillos, anillos[1:]):
        for k in range(len(a)):
            j = (k + 1) % len(a)
            caras.append([a[k], a[j], b[j], b[k]])
    t.malla(g, "Chapa", caras, color)
    # Silueta longitudinal, capó, parabrisas, techo y luneta.
    perfil = [(-1.21, .81), (-.63, 1.38), (.58, 1.40), (1.24, .82)]
    lados = [[(x, y, z) for x, z in perfil] for y in (-.70, .70)]
    for lado in lados:
        t.malla(g, "CristalLateral", [lado], "vidrio")
    for i, acabado in enumerate(("vidrio", color, "vidrio")):
        t.malla(g, "Cabina", [[lados[0][i], lados[0][i + 1], lados[1][i + 1], lados[1][i]]], acabado)
    t.caja(g, "VolumenCabina", (0, 0, 1.06), (1.30, 1.30, .55), color, True)
    for x, y in product((-1.24, 1.23), (-.79, .79)):
        t.disco(g, "Neumatico", (x, y, .33), .32, .20, "grafito", False, "Y")
        t.disco(g, "Llanta", (x, y * 1.135, .33), .19, .018, "aluminio", False, "Y")
        lado = math.copysign(1., y)
        cara = lado * .909
        t.disco(g, "DiscoFreno", (x, cara, .33), .155, .009, "grafito", eje="Y")
        for ang in np.arange(8) * math.tau / 8:
            dx, dz = math.cos(ang), math.sin(ang)
            t.varilla(g, "RadioLlanta", (x + dx * .042, cara + lado * .007, .33 + dz * .042),
                      (x + dx * .168, cara + lado * .007, .33 + dz * .168), .016, "aluminio")
        t.disco(g, "Buje", (x, cara + lado * .015, .33), .055, .022, "aluminio", eje="Y")
    for lado in (-1, 1):
        t.varilla(g, "PilarB", (0, lado * .705, .85), (0, lado * .705, 1.39), .032, "grafito")
        for x in (-.39, .65):
            t.caja(g, "Tirador", (x, lado * .83, .785), (.13, .026, .026), "aluminio", bisel=.01)
        t.caja(g, "Espejo", (-.75, lado * .84, 1.02), (.21, .17, .10), color, bisel=.034)
    for y in (-.57, .57):
        t.caja(g, "Faro", (-1.934, y, .65), (.04, .36, .14), "ceramica", False, .015)
        t.caja(g, "Piloto", (1.934, y, .65), (.04, .34, .13), "oxido")


def calle(t):
    p = t.raiz
    t.suelo(p, "Calzada", (0, 0), (56, 52), "asfalto")
    for y in (-7.3, 7.3):
        t.suelo(p, "Vereda", (0, y), (52, 4.5), "hormigon", .006)
        # En los pasos la vereda queda enrasada con la calzada.
        for x, ancho in ((-15, 20.2), (0, 3.5), (15, 20.2)):
            t.caja(p, "Sardinel", (x, math.copysign(5.1, y), .064), (ancho, .20, .128), "hormigon", True,
                   .019)
        for x in (-11, -7.4, 0, 7.4, 11):
            t.suelo(p, "Junta", (x, y), (.014, 4.4), "grafito", .008)
    for x in (-3.2, 3.2):
        for y in np.arange(-4.62, 4.7, .80):
            t.suelo(p, "PasoCebra", (x, y), (1.70, .32), "marfil", .006)
    for x in (-21, -16, -11, -7, 0, 7, 11, 16, 21):
        t.suelo(p, "LineaCentral", (x, 0), (2.15, .095), "ocre", .006)
    for x, acabado, nombre in ((-9, "ocre", "papel"), (-3, "oxido", "pan"), (3, "petroleo", "botica"),
                               (9, "arena", "bodega")):
        local(t, p, (x, 9.15, 0), 0, acabado, nombre)
        local(t, p, (-x, -9.65, 0), 180, acabado, nombre)
    for lado in (-1, 1):
        for x, acabado in ((15, "marfil"), (21, "arena")):
            for y in (-9.3, 9.3):
                g = t.grupo(p, "CuadraContigua", (lado * x, y, 0), 0 if y > 0 else 180)
                t.caja(g, "Vivienda", (0, 1.50, 2.45), (5.85, 3, 4.90), acabado, True)
                for vx in (-1.5, 1.5):
                    ventana(t, g, vx, -.07, 3.32, 1.57, 1.12)
                    ventana(t, g, vx, -.07, 1.37, 1.45, 1.16)
    parada = t.grupo(p, "ParadaBus", (-6.65, 6.97, 0))
    banco(t, parada, (0, 0, 0))
    for x in (-1.38, 1.38):
        t.caja(parada, "Pilar", (x, .25, 1.20), (.08, .10, 2.40), "grafito", True)
    t.caja(parada, "Cubierta", (0, 0, 2.44), (3.08, 1.38, .10), "petroleo", True, .025)
    t.rotulo(parada, "Senal", (0, -.71, 2.44), 1.22, .30, "paradero")
    for x, y in ((-7, -6.22), (8, 6.33), (-12, 6.3), (12, -6.30)):
        palmera(t, p, (x, y, 0), 4.2)
    for x, y in ((-10, 5.82), (7.5, -5.80)):
        t.disco(p, "Farola", (x, y, 2.12), .075, 4.24, "grafito", True)
        sentido = -np.sign(y)
        t.varilla(p, "Brazo", (x, y, 4.17), (x, y + .76 * sentido, 4.34), .038, "grafito")
        t.caja(p, "Luminaria", (x, y + .75 * sentido, 4.34), (.28, .55, .10), "aluminio", False, .03)
    automovil(t, p, (8.5, 4.50, 0), 180, "tinta")
    automovil(t, p, (-8.9, -4.58, 0), 0, "coral")
    return (-7.50, -1.68, 2.15), (.32, 8.12, 2.0)


def pabellon(t, p, pos, giro, letrero):
    g = t.grupo(p, "Pabellon", pos, giro)
    t.caja(g, "MuroTrasero", (0, 3.17, 3.28), (16, .25, 6.56), "ladrillo", True)
    for x in (-7.89, 7.89):
        t.caja(g, "Testero", (x, 1.6, 3.28), (.23, 3.42, 6.56), "ladrillo", True)
    t.caja(g, "Forjado", (0, 1.35, 3.12), (16.15, 4.78, .28), "hormigon", True)
    t.caja(g, "Azotea", (0, 1.36, 6.45), (16.38, 4.67, .24), "marfil", True)
    t.caja(g, "AntepechoPisoAlto", (0, 0, 3.63), (15.97, .25, .86), "ladrillo", True)
    for x in (-6, -2, 2, 6):
        ventana(t, g, x, -.03, 5.03, 3.78, 1.74)
    for i in range(39):
        x = (i - 19) * .403
        t.caja(g, "Quiebrasol", (x, -.39, 5.10), (.066, .46, 1.98), "roble")
    for x in (-7.56, -.02, 7.55):
        t.caja(g, "Columna", (x, -.83, 1.52), (.30, .30, 3.04), "hormigon", True)
    t.caja(g, "Friso", (0, -.98, 3.14), (16.24, .18, .44), "marfil")
    for x in (-4., 4.):
        plafon(t, g, (x, 1.13, 2.94))
    t.rotulo(g, "Directorio", (2.42, -1.077, 3.14), 1.72, .40, letrero)
    for x in (-6.3, -4.1, 4.1, 6.3):
        if letrero == "biblioteca":
            estante(t, g, (x, 2.63, 0))
        else:
            mesa(t, g, (x, 1.40, 0), 1.12, .61)
            silla(t, g, (x, 2.04, 0))
    if letrero == "biblioteca":
        for x in (-2.6, 2.6):
            mesa(t, g, (x, .64, 0), 1.58, .83)
            silla(t, g, (x, 1.35, 0))
    else:
        t.caja(g, "MarcoPizarra", (0, 3.01, 1.63), (3.18, .08, 1.42), "aluminio")
        t.caja(g, "Pizarra", (0, 2.95, 1.63), (3.08, .05, 1.32), "petroleo")


def campus(t):
    p = t.raiz
    t.suelo(p, "Patio", (0, 0), (55, 52), "hormigon")
    for x in (-3.2, 3.2):
        t.suelo(p, "Circulacion", (x, 0), (1.96, 16.7), "terrazo", .003)
    for y in (-3.2, 3.2):
        t.suelo(p, "Galeria", (0, y), (20.4, 1.96), "terrazo", .006)
    for x in np.arange(-10, 11, 1.25):
        for y in (-6.05, 0, 6.05):
            t.suelo(p, "Junta", (x, y), (.015, 3.46), "grafito", .002)
    pabellon(t, p, (0, 8.75, 0), 0, "biblioteca")
    pabellon(t, p, (0, -8.75, 0), 180, "aulas")
    for x, y in ((-6.58, 5.69), (6.58, 5.69), (-6.58, -5.69), (6.58, -5.69)):
        t.caja(p, "Jardinera", (x, y, .21), (2.02, 1.60, .42), "hormigon", True, .045)
        t.caja(p, "Sustrato", (x, y, .427), (1.85, 1.43, .026), "tierra")
        palmera(t, p, (x, y, .435), 4.56)
    for x in (-6.9, 6.9):
        banco(t, p, (x, .37, 0), 90 if x < 0 else -90)
    for x in (-9.80, 9.80):
        for y in (-1.57, 1.57):
            t.caja(p, "PostePergola", (x, y, 1.48), (.16, .18, 2.96), "grafito", True)
        for y in np.arange(-1.78, 1.80, .28):
            t.caja(p, "LamaPergola", (x, y, 3.0), (2.32, .12, .17), "roble")
    for lado in (-1, 1):
        g = t.grupo(p, "PabellonDeFondo", (lado * 18.7, 0, 0), -lado * 90)
        t.caja(g, "Volumen", (0, 2.1, 3.74), (15.60, 4.20, 7.48), "ladrillo", True)
        for z in (1.57, 4.18, 6.52):
            for x in (-5., 0., 5.):
                ventana(t, g, x, -.08, z, 4.53, 1.35)
            t.caja(g, "Alero", (0, -.19, z + .79), (16.0, .68, .13), "marfil")
    return (-8.52, -2.26, 2.48), (1.25, 8.78, 2.35)


ESCENAS = {"vivienda": vivienda, "calle": calle, "campus": campus}


def construir(nombre, diseno):
    stage = Usd.Stage.CreateNew(str(DESTINO / (nombre + ".usda")))
    stage.GetRootLayer().subLayerPaths = ["../pista_marcha_16.usda"]
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.SetTimeCodesPerSecond(20.)
    stage.SetStartTimeCode(0.)
    stage.SetEndTimeCode(20. * 86400)
    piso = stage.GetPrimAtPath("/World/Suelo")
    piso.GetAttribute("xformOp:scale").Set((56., 56., .07))
    UsdGeom.Imageable(piso).CreateVisibilityAttr().Set("invisible")
    cielo = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Luz"))
    cielo.CreateTextureFileAttr(Sdf.AssetPath("./texturas/cielo_costero.jpg"))
    cielo.CreateTextureFormatAttr("latlong")
    cielo.CreateIntensityAttr().Set(520.)
    sol = UsdLux.DistantLight(stage.GetPrimAtPath("/World/LuzPrincipal"))
    sol.CreateIntensityAttr().Set(1900.)
    sol.CreateColorAttr().Set((1., .92, .79))
    sol.CreateAngleAttr().Set(1.6)
    taller = Taller(stage)
    ojo, centro = diseno(taller)
    cam = UsdGeom.Camera.Define(stage, taller.raiz + "/Vista")
    matriz = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*ojo), Gf.Vec3d(*centro), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(matriz.GetInverse())
    cam.CreateFocalLengthAttr(19.)
    cam.CreateHorizontalApertureAttr(32.)
    cam.CreateClippingRangeAttr((.05, 150.))
    stage.GetRootLayer().customLayerData = {
        "cameraSettings": {
            "Perspective": {
                "position": Gf.Vec3d(*ojo),
                "target": Gf.Vec3d(*centro)
            }
        }
    }
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    rotulos()
    for nombre, diseno in ESCENAS.items():
        construir(nombre, diseno)
        print(DESTINO / (nombre + ".usda"), flush=True)
