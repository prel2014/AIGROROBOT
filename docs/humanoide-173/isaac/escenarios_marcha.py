"""Construye los escenarios de Xavier con la pista articulada común."""
from itertools import product
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from entornos_xavier import acopio, arbol, casa, residencial, vehiculo
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
        # Las lamas y el vano quedan detrás del cristal, con profundidad visible al sesgo.
        for k in range(6):
            t.caja(p, "LamaInterior", (x, y - .058, z + alto * (.43 - .09*k)),
                   (ancho * .97, .018, alto * .048), "lino" if k % 3 else "arena")
        t.caja(p, "Cristal", (x, y - .083, z), (ancho, .013, alto), "vidrio")
    for k in (-.25, .25):
        t.caja(p, "Parteluz", (x + ancho * k, y - .103, z), (.034, .04, alto), "aluminio")
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
        t.caja(jardin, "Medianera", (0, 0, .87), (27.6, .22, 1.74), "revoque", True)
        t.suelo(jardin, "Cesped", (0, -1.80), (25.8, 3.3), "cesped", .003)
        for x in (-8.2, -.4, 7.0):
            arbol(t, jardin, (x, 2.2, 0), 1.8)
        t.caja(jardin, "Albardilla", (0, 0, 1.77), (27.6, .30, .07), "marfil")
        for x in (-7.2, 0, 7.2):
            planta(t, jardin, (x, -1.6, 0), 2.15)
        arbol(t, jardin, (4.1, -1.5, 0), 1.66)
    for lado in (-1, 1):
        t.caja(p, "TechoGaleria", (0, lado * 6.17, 3.33), (17.72, 5.38, .16), "cal", True)
        t.caja(p, "TechoLateral", (lado * 6.17, 0, 3.33), (5.38, 6.96, .16), "cal", True)
    for x in (-3.44, 0, 3.44):
        t.caja(p, "VigaLucernario", (x, 0, 3.30), (.14, 7.0, .22), "roble")
    for x, y in product((-6.1, 6.1), (-6., 6.)):
        plafon(t, p, (x, y, 3.20), 1.48, 12500.)
    for x, y in ((0, -5.4), (0, 5.4), (-5.4, 0), (5.4, 0)):
        plafon(t, p, (x, y, 3.20), 1.40, 9500.)
    rebote = UsdLux.RectLight.Define(t.stage, p + "/LuzDeLucernario")
    rebote.AddTranslateOp().Set((-1.7, 1.8, 3.42))
    rebote.CreateWidthAttr(3.3)
    rebote.CreateHeightAttr(3.3)
    rebote.CreateIntensityAttr(5300.)
    rebote.CreateColorAttr((.95, .97, 1.))
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
    return (-.55, -.55, 1.59), (-5.7, 5.65, 1.04)


def local(t, p, pos, giro, acabado, letrero):
    g = t.grupo(p, "Comercio", pos, giro)
    t.caja(g, "Edificio", (0, 1.9, 2.90), (5.95, 3.8, 5.80), "revoque", True)
    t.caja(g, "Cornisa", (0, 0, 5.73), (6.08, .32, .18), "cal")
    t.caja(g, "Basamento", (0, -.09, .29), (5.94, .20, .58), "hormigon")
    for x in (-1.48, 1.48):
        ventana(t, g, x, -.12, 4.30, 2.13, 1.32)
    if letrero in ("pan", "bodega"):
        t.caja(g, "Balcon", (0, -.38, 3.54), (5.20, .83, .14), "hormigon", True)
        t.varilla(g, "Pasamanos", (-2.5, -.74, 4.43), (2.5, -.74, 4.43), .028, "grafito")
        for k in range(27):
            x = -2.47 + k * .19
            t.varilla(g, "Barrote", (x, -.74, 3.59), (x, -.74, 4.42), .012, "grafito")
    t.disco(g, "Bajante", (2.80, -.16, 2.72), .048, 5.44, "aluminio")
    ventana(t, g, -.88, -.13, 1.41, 3.38, 1.63)
    t.panel(g, "Escaparate", (-.88, -.241, 1.41), 3.37, 1.62, "vitrina_" + letrero)
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


def edificio_de_esquina(t, pos, giro, pisos):
    g = t.grupo(t.raiz, "EdificioDeEsquina", pos, giro)
    altura = 3.*pisos
    t.caja(g, "Cuerpo", (0,3.9,altura/2), (11.8,7.8,altura), "revoque", True)
    laterales = [t.grupo(g,"FachadaLateral",(lado*5.90,3.9,0),lado*90) for lado in (-1,1)]
    for i in range(pisos):
        z = 1.52 + 3.*i
        for x in (-4.0,-1.35,1.35,4.0):
            ventana(t,g,x,-.08,z,1.90,1.38)
        for pared in laterales:
            for x in (-2.0,.65):
                ventana(t,pared,x,-.07,z,1.46,1.38)
        t.caja(g,"Forjado",(0,-.17,z+1.35),(12.0,.49,.14),"hormigon")
        if i:
            t.caja(g,"VueloDeBalcon",(-2.7,-.55,z-.88),(5.0,1.1,.16),"hormigon")
            for x in np.linspace(-5.1,-.3,11):
                t.varilla(g,"Montante",(x,-1.04,z-.81),(x,-1.04,z+.13),.023,"grafito")
            t.varilla(g,"Pasamanos",(-5.1,-1.04,z+.13),(-.3,-1.04,z+.13),.035,"grafito")
    t.caja(g,"CuartoDeAzotea",(2.5,4.8,altura+.88),(3.2,3.5,1.76),"ladrillo")
    t.disco(g,"Tanque",(-1.7,5.0,altura+.70),.74,1.4,"grafito")


def cuadras_del_fondo(t):
    p = t.raiz
    # Cruces perpendiculares y remates en T, visibles desde los dos sentidos.
    for signo in (-1,1):
        for distancia in (29.,74.):
            t.suelo(p,"CalleTransversal",(signo*distancia,0),(8.,76.),"asfalto",.002)
            for y in np.arange(-4.2,4.3,.85):
                t.suelo(p,"CrucePeatonal",(signo*(distancia-5.1),y),(1.45,.27),"marfil",.006)
        for lado in (-1,1):
            for x, longitud in ((44.,21.8),(62.1,14.),(82.,7.0)):
                t.suelo(p,"VeredaDeCuadra",(signo*x,lado*7.2),(longitud,4.3),"hormigon",.009)
                t.caja(p,"Bordillo",(signo*x,lado*5.10,.065),(longitud,.16,.13),"hormigon")
            frente = 0 if lado > 0 else 180
            edificio_de_esquina(t,(signo*39.2,lado*9.5,0),frente,3 if signo==lado else 2)
            casa(t,p,"house_1",(signo*55.6,lado*16.1,0),frente)
            local(t,p,(signo*66.0,lado*9.5,0),frente,"arena","bodega")
            for x in (35.,48.5,63.,83.):
                arbol(t,p,(signo*x,lado*7.1,0),1.45 + (x%3)*.12)
            casa(t,p,"house_2",(signo*87.,lado*14.7,0),frente)
        edificio_de_esquina(t,(signo*90.,-1.8,0),-signo*90,4)
        vehiculo(t,p,(signo*36.,3.85,.003),180,"prius_hybrid")
        vehiculo(t,p,(signo*61.,-3.65,.003),0,"pickup")
        for x in range(32,85,6):
            if abs(x-74)>4:
                t.suelo(p,"MarcaDeEje",(signo*x,0),(2.2,.095),"ocre",.005)
    # Cableado con flecha entre apoyos. El tendido atraviesa las cuadras en perspectiva.
    grupo = t.grupo(p,"RedDeDistribucion")
    apoyos = (-84.,-62.,-40.,-18.,5.,27.,49.,71.,92.)
    for x in apoyos:
        t.disco(grupo,"Poste",(x,7.9,3.75),.11,7.5,"hormigon")
        t.caja(grupo,"Cruceta",(x,7.9,7.27),(.13,1.45,.10),"grafito")
        for y in (7.37,7.9,8.43):
            t.disco(grupo,"Aislador",(x,y,7.43),.055,.25,"ceramica")
    for xa,xb in zip(apoyos,apoyos[1:]):
        for y in (7.37,7.9,8.43):
            pts = [(xa+(xb-xa)*u,y,7.53-2.6*u*(1-u)) for u in np.linspace(0,1,7)]
            for u,v in zip(pts,pts[1:]):
                t.varilla(grupo,"Conductor",u,v,.014,"grafito")


def calle(t):
    p = t.raiz
    t.suelo(p, "TerrenoUrbano", (0, 0), (238, 235), "suelo_seco", -.025)
    t.suelo(p, "Calzada", (0, 0), (181, 10.3), "asfalto")
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
        for y in (-15.9, 15.9):
            casa(t, p, "house_2", (lado * 19.1, y, 0), 0 if y > 0 else 180)
    parada = t.grupo(p, "ParadaBus", (-6.65, 6.97, 0))
    banco(t, parada, (0, 0, 0))
    for x in (-1.38, 1.38):
        t.caja(parada, "Pilar", (x, .25, 1.20), (.08, .10, 2.40), "grafito", True)
    t.caja(parada, "Cubierta", (0, 0, 2.44), (3.08, 1.38, .10), "petroleo", True, .025)
    t.rotulo(parada, "Senal", (0, -.71, 2.44), 1.22, .30, "paradero")
    for x, y in ((-7, -6.22), (8, 6.33), (-12, 6.3), (12, -6.30)):
        arbol(t, p, (x, y, 0), 1.52)
    for x, y in ((-10, 5.82), (7.5, -5.80)):
        t.disco(p, "Farola", (x, y, 2.12), .075, 4.24, "grafito", True)
        sentido = -np.sign(y)
        t.varilla(p, "Brazo", (x, y, 4.17), (x, y + .76 * sentido, 4.34), .038, "grafito")
        t.caja(p, "Luminaria", (x, y + .75 * sentido, 4.34), (.28, .55, .10), "aluminio", False, .03)
    vehiculo(t, p, (7.2, 3.94, .003), 180, "prius_hybrid")
    vehiculo(t, p, (-8.9, -3.78, .003), 0, "pickup")
    cuadras_del_fondo(t)
    return (-2.6, -1.7, 1.58), (4.3, 6.5, 1.60)


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
        arbol(t, p, (x, y, .435), 1.58)
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
    return (-6.1, -2.3, 1.61), (-.80, 7.45, 1.62)


ESCENAS = {"vivienda": vivienda, "calle": calle, "campus": campus,
           "residencial": residencial, "acopio": acopio}


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
    lado_suelo = 238. if nombre == "calle" else 56.
    piso.GetAttribute("xformOp:scale").Set((lado_suelo, lado_suelo, .07))
    UsdGeom.Imageable(piso).CreateVisibilityAttr().Set("invisible")
    cielo = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Luz"))
    panorama = "kloofendal_48d_partly_cloudy_4k.hdr" if nombre == "calle" else "kloofendal_48d_partly_cloudy_puresky_1k.hdr"
    cielo.CreateTextureFileAttr(Sdf.AssetPath("./texturas/" + panorama))
    cielo.CreateTextureFormatAttr("latlong")
    cielo.CreateIntensityAttr().Set(470.)
    sol = UsdLux.DistantLight(stage.GetPrimAtPath("/World/LuzPrincipal"))
    sol.CreateIntensityAttr().Set(880.)
    sol.CreateColorAttr().Set((1., .92, .79))
    sol.CreateAngleAttr().Set(3.2)
    taller = Taller(stage)
    if nombre == "calle":
        stage.OverridePrim("/XavierMarcha/cabeza/Camara").GetAttribute("clippingRange").Set((.02, 220.))
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
