"""Entornos residenciales y de poscosecha de Xavier, en metros y con eje Z vertical."""
from pxr import UsdGeom

from geometria_escenarios import mesa


def modelo(t, parent, nombre, archivo, pos, giro=0, escala=1.):
    grupo = t.grupo(parent, nombre, pos, giro)
    if escala != 1.:
        UsdGeom.Xformable(t.stage.GetPrimAtPath(grupo)).AddScaleOp().Set((escala,) * 3)
    visual = UsdGeom.Xform.Define(t.stage, grupo + "/Visual")
    visual.GetPrim().GetReferences().AddReference("./objetos/" + archivo)
    visual.GetPrim().SetInstanceable(True)
    return grupo


def vehiculo(t, parent, pos, giro, tipo):
    grupo = modelo(t, parent, "Vehiculo", tipo + "/visual.usdc", pos, giro)
    largo, ancho, alto = (4.62, 1.86, 1.54) if tipo == "prius_hybrid" else (5.66, 2.31, 1.87)
    for nombre, centro, medidas in (
        ("Bastidor", (0, 0, .53), (largo * .97, ancho, .74)),
        ("Cabina", (0, 0, (alto + .84) / 2), (largo * .52, ancho * .86, alto - .84)),
    ):
        proxy = t.caja(grupo, nombre, centro, medidas, "grafito", True)
        UsdGeom.Imageable(proxy).MakeInvisible()


def casa(t, parent, tipo, pos, giro):
    grupo = modelo(t, parent, "Vivienda", tipo + "/visual.usdc", pos, giro)
    # La cubierta, el porche y los retranqueos conservan su geometría de colisión estática.
    visual = t.stage.GetPrimAtPath(grupo + "/Visual")
    visual.SetInstanceable(False)
    for pieza in visual.GetChildren():
        if pieza.IsA(UsdGeom.Mesh):
            t.colision(pieza, "obstaculo", "none")


def arbol(t, parent, pos, escala=1.):
    grupo = modelo(t, parent, "Arbol", "tree_small_02/visual.usdc", pos, t.rng.uniform(0, 360), escala * .60)
    tronco = t.disco(grupo, "ColisionTronco", (0, 0, .58), .085, 1.16, "tierra", True)
    UsdGeom.Imageable(tronco).MakeInvisible()


def residencial(t):
    p = t.raiz
    t.suelo(p, "Terreno", (0, 0), (92, 76), "cesped")
    t.suelo(p, "Calzada", (0, 0), (90, 8.7), "asfalto", .004)
    for lado in (-1, 1):
        y = lado * 5.5
        t.suelo(p, "Acera", (0, y), (88, 2.3), "hormigon", .008)
        for centro in (-27., -17., -7., 7., 17., 27.):
            t.caja(p, "Bordillo", (centro, lado * 4.38, .073), (7.8, .17, .146),
                   "hormigon", True, .014)
        for x in range(-35, 36, 2):
            t.suelo(p, "Junta", (x, y), (.014, 2.25), "grafito", .009)
        for i, x in enumerate((-30., -10., 9., 29.)):
            tipo = "house_2" if (i + lado) % 3 == 1 else "house_1"
            casa(t, p, tipo, (x, lado * (15.1 + i % 2), 0), 180 if lado < 0 else 0)
            t.suelo(p, "Entrada", (x + 2.6, lado * 8.9), (2.8, 4.7), "hormigon", .006)
            arbol(t, p, (x - 4., lado * 8.5, 0), 1.30 + i * .12)
    vehiculo(t, p, (7.3, 3.19, .006), 180, "prius_hybrid")
    vehiculo(t, p, (-13., -3.12, .006), 0, "pickup")
    for x in (-19., 1.2, 22.):
        t.disco(p, "PosteDeAlumbrado", (x, 6.22, 2.9), .078, 5.8, "grafito", True)
        t.varilla(p, "BrazoFarola", (x, 6.22, 5.66), (x, 5.17, 5.95), .043, "grafito")
        t.caja(p, "Farola", (x, 5.12, 5.90), (.29, .55, .09), "aluminio", bisel=.035)
    return (-2.5, -.75, 1.56), (3.8, 6.6, 1.65)


def jaba(t, parent, pos, giro, llena):
    g = t.grupo(parent, "CajaFrutera", pos, giro)
    t.caja(g, "Fondo", (0, 0, .025), (.62, .42, .05), "roble", True)
    for x in (-.29, .29):
        for y in (-.19, .19):
            t.caja(g, "Esquinero", (x, y, .13), (.035, .035, .26), "roble")
    for z in (.09, .195):
        for y in (-.20, .20):
            t.caja(g, "Tabla", (0, y, z), (.62, .019, .079), "roble", bisel=.003)
        for x in (-.30, .30):
            t.caja(g, "Testero", (x, 0, z), (.02, .40, .079), "roble")
    if llena:
        for iz, nx, ny in ((0, 6, 4), (1, 5, 3), (2, 4, 2)):
            for ix in range(nx):
                for iy in range(ny):
                    modelo(t, g, "Naranja", "orange/visual.usda",
                           ((ix - (nx-1)/2)*.085, (iy - (ny-1)/2)*.084, .05 + .065*iz),
                           t.rng.uniform(0, 360), t.rng.uniform(.96, 1.08))


def acopio(t):
    p = t.raiz
    t.suelo(p, "TierraAgricola", (0, 0), (95, 90), "suelo_seco")
    t.suelo(p, "Losa", (0, 2.4), (15.8, 15.), "hormigon", .006)
    for x in (-6., -2., 2., 6.):
        t.suelo(p, "JuntaDeLosa", (x, 2.4), (.015, 15.), "grafito", .007)
    for x in (-7.75, 0., 7.75):
        for y in (5.9, 10.2):
            alto = 3.72 + .08 * (y - 5.6)
            t.caja(p, "PlacaBase", (x, y, .035), (.31, .31, .07), "grafito", True)
            t.caja(p, "Columna", (x, y, alto/2), (.13, .13, alto), "grafito", True)
        viga = t.grupo(p, "VigaDePortico", (x, 8.06, 3.88))
        UsdGeom.Xformable(t.stage.GetPrimAtPath(viga)).AddRotateXOp().Set(4.574)
        t.caja(viga, "Perfil", (0, 0, 0), (.16, 5.60, .17), "grafito")
        for y in (6., 10.10):
            t.varilla(p, "Cartela", (x, y, 3.0), (x, 7.7 if y < 7 else 8.4, 3.74), .047, "grafito")
    # Chapa plegada con caída hacia el canalón, sobre correas y pórticos.
    for y in (5.6, 7., 8.45, 10.5):
        t.caja(p, "Correa", (0, y, 3.77 + (y-5.6)*.08), (16.5, .09, .14), "grafito")
    caras = []
    for i in range(164):
        x = -8.2 + .10*i
        z0, z1 = (3.89, 3.85) if i % 2 else (3.85, 3.89)
        caras.append([(x, 5.25, z0), (x+.1, 5.25, z1), (x+.1, 10.9, z1+.452), (x, 10.9, z0+.452)])
    t.malla(p, "ChapaGalvanizada", caras, "aluminio")
    for x in (-5., -1.8, 2.3, 5.4):
        mesa(t, p, (x, 6.40, 0), 2.6, 1., .86)
        for dx in (-.70, 0, .70):
            jaba(t, p, (x+dx, 6.38, .864), t.rng.uniform(-3, 3), dx != .70)
    for x in (-23., -19., -15., -11., 11., 15., 19., 23.):
        for y in (-11., -6., -1., 4., 9., 14.):
            arbol(t, p, (x, y + t.rng.uniform(-.3, .3), 0), t.rng.uniform(1.22, 1.60))
    for y in (15., 20.):
        for x in (-8., -4., 0., 4., 8.):
            arbol(t, p, (x, y, 0), 1.4)
    vehiculo(t, p, (-7.5, 2.0, .007), -12, "pickup")
    return (5.0, -.85, 1.59), (-.40, 6.1, 1.42)
