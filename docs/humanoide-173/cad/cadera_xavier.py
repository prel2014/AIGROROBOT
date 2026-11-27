"""Módulo bilateral pasivo de ensayos. Milímetros, +Y al frente, +Z arriba."""
from itertools import product
from math import cos, radians, sin

from build123d import Axis, Box, Color, Compound, Cylinder, Pos, RevoluteJoint, Rot, fillet

HIP_X, HIP_Z = 112., 875.
# Convención anatómica bilateral. Positivo abre la pierna o gira la punta hacia fuera.
LIMITES = {"giro": (-20., 20.), "apertura": (-8., 18.), "flexion": (-15., 35.)}
FIJO, GIRO, APERTURA, FLEXION = "#62717A", "#CB763D", "#247E7D", "#E1E2D7"
BRONCE, ACERO = "#CBA467", "#9DAAB1"


def pieza(forma, nombre, color):
    forma.label, forma.color = nombre, Color(color)
    return forma


def placa(ancho, fondo, espesor, z, radio=3):
    bloque = Box(ancho, fondo, espesor)
    return Pos(0, 0, z) * fillet(bloque.edges().filter_by(Axis.Z), radio)


def cilindro(eje, radio, inicio, fin):
    medio = (inicio + fin) / 2
    lugar = {"X": Pos(medio, 0, 0) * Rot(0, 90, 0),
             "Y": Pos(0, medio, 0) * Rot(90, 0, 0), "Z": Pos(0, 0, medio)}[eje]
    return lugar * Cylinder(radio, fin - inicio)


def casquillo(eje, radio, agujero, a, b):
    return cilindro(eje, radio, a, b) - cilindro(eje, agujero, a-1, b+1)


def semicuna(lado):
    # Medio travesaño. En la zona central los dos se unen con una pletina independiente.
    ala = Pos(-lado*36, 0, 0) * placa(150, 76, 8, 78, 7)
    cuello = cilindro("Z", 24, 54, 76)
    ala = (ala + cuello) - cilindro("Z", 16, 53, 83)
    for x,y in product((-lado*91,-lado*105),(-20,20)):
        ala -= Pos(x,y,0) * cilindro("Z", 3.3, 73, 84)
    return pieza(ala, "Semitravesano y caja del pivote", FIJO)


def horquilla():
    tapa = placa(38, 98, 8, 48, 5)
    for direccion in (-1, 1):
        y = direccion*44
        oreja = Pos(0,y,24) * Box(30, 8, 48)
        oreja += cilindro("Y",15, y-4, y+4)
        oreja -= cilindro("Y",10, y-5,y+5)
        tapa += oreja
    # Espiga hueca integrada en la pieza móvil; la brida superior recibe el mando del eje Z.
    tapa += cilindro("Z", 11.85, 50, 87)
    tapa -= cilindro("Z", 5, 43, 88)
    for x in (-8,8):
        tapa -= Pos(x,0,0)*cilindro("Z",1.25,78,88)
    return pieza(tapa, "Horquilla de giro y eje hueco", GIRO)


def cardan():
    anillo = placa(82, 74, 20, 0, 9) - placa(62, 52, 24, 0, 6)
    anillo -= cilindro("X", 9, -42, 42)
    anillo -= cilindro("Y", 7.8, -38, 38)
    for y in (-32,32):
        anillo -= Pos(0,y,0)*cilindro("Z",1.6,-11,11)
    return pieza(anillo, "Bastidor de apertura lateral", APERTURA)


def salida():
    # La cara cilíndrica inferior copia el radio del tambor de Xavier. La silla del muslo
    # utiliza sus cuatro tornillos verticales a ±6 mm y ±9 mm del centro.
    asiento = cilindro("X", 49, -24, 24) & (Pos(0,0,-44) * Box(48,32,14))
    vastago = placa(25, 21, 39, -20.5)
    cabeza = cilindro("X", 13, -27, 27)
    salida = asiento + vastago + cabeza
    salida -= cilindro("X", 6.8, -28, 28)
    for x in (-20,20):
        salida -= Pos(x,0,0)*cilindro("Z",1.6,-14,14)
    for x,y in product((-6,6),(-9,9)):
        salida -= Pos(x,y,0) * cilindro("Z", 2.5, -50, -38)
    return pieza(salida, "Salida de flexion y asiento de muslo", FLEXION)


def brida(eje, signo, inicio, radio):
    # Plato de ensayo para arrastre externo. Patrón de interfaz, sin motor asignado.
    lugar = {"X": Rot(0,90*signo,0), "Y": Rot(-90*signo,0,0), "Z": Rot(0,0,0)}[eje]
    forma = casquillo("Z", radio, 4.5, inicio, inicio+4)
    for angulo in (0, 90, 180, 270):
        a = radians(angulo)
        forma -= Pos((radio-4)*cos(a), (radio-4)*sin(a),0)*cilindro("Z", 1.7, inicio-1, inicio+5)
    return pieza(lugar*forma, "Brida de mando "+eje, ACERO)


def modulo(lado):
    """Una lista por eslabón; todos los sólidos usan el centro articular como origen."""
    fijo = [semicuna(lado), pieza(casquillo("Z",15.97,12.,54,82), "Casquillo vertical", BRONCE)]
    plato = brida("Z",1,87,18)
    for x in (-8,8):
        plato -= Pos(x,0,0)*cilindro("Z",1.7,86,92)
    giro = [horquilla(), pieza(plato,"Plato desmontable de giro",ACERO)]
    apertura = [cardan()]
    flexion = [salida()]
    for signo in (-1,1):
        def intervalo(a,b):
            return sorted((signo*a,signo*b))
        giro.append(pieza(casquillo("Y",9.98,8.0,*intervalo(40,48)), "Casquillo lateral", BRONCE))
        munon = cilindro("Y",7.75,*intervalo(27,48.5))
        if signo == -1:
            munon += brida("Y",signo,48.5,15)
        munon -= Pos(0,32*signo,0)*cilindro("Z",1.6,-9,9)
        apertura.append(pieza(munon, "Munon desmontable de apertura", ACERO))
        apertura.append(pieza(Pos(0,32*signo,0)*cilindro("Z",1.5,-9.7,9.7),"Pasador transversal de apertura",ACERO))
        apertura.append(pieza(casquillo("X",8.98,7.,*intervalo(31,41)), "Casquillo de flexion", BRONCE))
        munon = cilindro("X",6.75,*intervalo(15,41.6))
        if signo == lado:
            munon += brida("X",signo,41.6,14)
        munon -= Pos(20*signo,0,0)*cilindro("Z",1.6,-8,8)
        flexion.append(pieza(munon, "Munon desmontable de flexion", ACERO))
        flexion.append(pieza(Pos(20*signo,0,0)*cilindro("Z",1.5,-12.5,12.5),"Pasador transversal de flexion",ACERO))
    fijo.extend(pieza(casquillo("Z",18,12.05,a,b), "Arandela axial", BRONCE)
                for a,b in [(52.15,53.95),(82.05,86.75)])
    return [fijo, giro, apertura, flexion]


def puente():
    pieza_central = placa(52, 56, 6, 85, 4)
    for x,y in product((-21,-7,7,21),(-20,20)):
        pieza_central -= Pos(x,y,0)*cilindro("Z",3.3,81,89)
    return pieza(pieza_central,"Union del travesano",FIJO)


def emplazar(lado, giro=0, apertura=0, flexion=0):
    base = Pos(lado*HIP_X,0,HIP_Z)
    z = base * Rot(0,0,-lado*giro)
    y = z * Rot(0,-lado*apertura,0)
    return [base,z,y,y*Rot(flexion,0,0)]


def ensamblar(generar_modulo=modulo):
    fijos = [Pos(0,0,HIP_Z)*puente()]
    moviles = []
    for lado, nombre in ((-1,"izquierda"),(1,"derecha")):
        grupos = generar_modulo(lado)
        base = emplazar(lado)[0]
        fijos.extend(base*p for p in grupos[0])
        for etiqueta, partes, eje in zip(("Giro","Apertura","Flexion"), grupos[1:],
                                         ((0,0,-lado),(0,-lado,0),(1,0,0))):
            nodo = Compound(label=etiqueta+" "+nombre, children=[base*p for p in partes])
            RevoluteJoint(etiqueta, nodo, Axis((lado*HIP_X,0,HIP_Z),eje), angular_range=LIMITES[etiqueta.lower()])
            moviles.append(nodo)
    return Compound(label="Xavier - caderas de tres ejes",
                    children=[Compound(label="Bancada pelvica",children=fijos),*moviles])
