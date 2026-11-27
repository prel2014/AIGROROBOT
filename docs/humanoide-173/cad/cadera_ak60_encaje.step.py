"""Banco bilateral de encaje en mm. Cuatro AK60-39 V3.0 de referencia."""
from functools import lru_cache
from itertools import product
from math import cos, radians, sin
from pathlib import Path

from build123d import Box, Compound, Pos, Rot, import_step
from cadera_xavier import ACERO, APERTURA, FIJO, GIRO, casquillo, cilindro, ensamblar, modulo, pieza, placa

REFERENCIA = Path(__file__).resolve().parents[1] / "referencias" / "AK60-39-V3.step"
Z_ROTOR, Z_ASIENTO_TRASERO = 38., -10.5
COLUMNAS = list(product((-32., 32.), (-31., 31.)))

@lru_cache(maxsize=1)
def motor():
    solidos = list(import_step(REFERENCIA).solids())
    candidatos = [s for s in solidos if any(
        abs(f.center().Z - Z_ROTOR) < 0.001 and 870 < f.area < 885 and
        len(f.inner_wires()) == 7 for f in s.faces())]
    if len(candidatos) != 1:
        raise ValueError("El STEP no coincide con la revisión del rotor acotada")
    rotor, = candidatos
    carcasa = Compound(children=[s for s in solidos if s is not rotor])
    return pieza(carcasa, "CubeMars AK60-39 V3.0 - conjunto fijo", "#454B51"), pieza(
        rotor, "CubeMars - brida de salida", ACERO)


def agujeros(forma, patron, radio, a, b):
    for x, y in patron:
        forma -= Pos(x,y,0) * cilindro("Z", radio, a-1, b+1)
    return forma


def circular(radio, fase, cantidad):
    angulos = [radians(fase + i*360/cantidad) for i in range(cantidad)]
    return [(radio*cos(a), radio*sin(a)) for a in angulos]


def conjunto(inicio, cara_rotor, base_a_columnas, patron_eje):
    """Coordenada axial creciente desde la articulación hacia la electrónica."""
    # Escalonar los patrones para separar los pasadores de los M3 laterales.
    fase = 13
    transformacion = Pos(0,0,cara_rotor + Z_ROTOR) * Rot(0,0,fase) * Rot(180,0,0)
    carcasa, rotor = motor()
    asiento = cara_rotor + Z_ROTOR - Z_ASIENTO_TRASERO
    retencion = placa(77,75,6,asiento+3,4) - cilindro("Z",29.5,asiento-1,asiento+7)
    retencion = agujeros(retencion,circular(34,fase+10.05,8),1.7,asiento,asiento+6)
    retencion = agujeros(retencion,COLUMNAS,2.2,asiento,asiento+6)
    fijos = [transformacion*carcasa, pieza(retencion,"Placa de sujecion de carcasa",FIJO)]
    for x,y in COLUMNAS:
        separador = casquillo("Z",4,2.2,base_a_columnas,asiento)
        fijos.append(pieza(Pos(x,y,0)*separador,"Separador tubular M4",ACERO))
    acople = casquillo("Z",20,4.5,inicio,cara_rotor)
    acople = agujeros(acople,patron_eje,1.7,inicio,cara_rotor)
    acople = agujeros(acople,circular(15,fase-57.9,3),2.2,inicio,cara_rotor)
    acople = agujeros(acople,circular(14.25,fase+2.1,3),2.0,inicio,cara_rotor)
    moviles = [transformacion*rotor,pieza(acople,"Acople al patron de salida CubeMars","#CB9753")]
    return fijos,moviles


def apertura_compacta(giro, apertura):
    # Y positivo hacia el pecho. La cara de salida queda a 24 mm detrás del pivote.
    cara = 24.
    atraso = Rot(90,0,0)
    fase = 13.
    carcasa, rotor = motor()
    montar_motor = atraso * Pos(0,0,cara+Z_ROTOR) * Rot(0,0,fase) * Rot(180,0,0)
    asiento = cara + Z_ROTOR - Z_ASIENTO_TRASERO
    # Conservar la oreja delantera y el eje vertical. Eliminar el apoyo posterior.
    horquilla = giro[0] - (Pos(0,-46,10)*Box(45,24,67))
    tapa = Pos(0,-43,48)*Box(38,72,8)
    pared = Pos(0,6,0)*placa(79,92,6,asiento+3,5)
    pared -= cilindro("Z",29.5,asiento-1,asiento+7)
    pared = agujeros(pared,circular(34,fase+10.05,8),1.7,asiento,asiento+6)
    giro[0] = pieza(horquilla.fuse(tapa,atraso*pared),"Horquilla compacta con soporte posterior",GIRO)
    # Tras retirar el plato vertical, el primer casquillo de giro es el trasero.
    del giro[1]
    giro.append(montar_motor*carcasa)

    anillo = apertura[0] - (Pos(0,-38,0)*Box(90,44,26))
    travesano = Pos(0,-20,0)*Box(82,8,20)
    disco = cilindro("Z",20.5,17.5,24)
    brida = (atraso.inverse()*travesano).fuse(disco)
    brida = agujeros(brida,circular(15,fase-57.9,3),2.2,16,24)
    brida = agujeros(brida,circular(14.25,fase+2.1,3),2,16,24)
    brida -= cilindro("Z",6,15,25)
    # Rebaje desde el lado accesible para las tres cabezas M4.
    for x,y in circular(15,fase-57.9,3):
        brida -= Pos(x,y,0)*cilindro("Z",3.6,15,21.7)
    apertura[0] = pieza(anillo.fuse(atraso*brida),"Bastidor con brida directa de apertura",APERTURA)
    del apertura[1:3]  # Se eliminan el muñón trasero y su pasador.
    apertura.append(montar_motor*rotor)


def modulo_de_encaje(lado, compacto=False):
    fijo,giro,apertura,flexion = modulo(lado)
    fijo[0] = pieza(agujeros(fijo[0],COLUMNAS,2.2,74,82),"Semitravesano taladrado",FIJO)
    giro.pop(1)  # El acople reemplaza el plato de ensayo vertical.
    estator,rotor = conjunto(87,96,82,[(-8,0),(8,0)])
    fijo.extend(estator)
    giro.extend(rotor)
    if compacto:
        apertura_compacta(giro,apertura)
        return [fijo,giro,apertura,flexion]
    atras = Rot(90,0,0)
    trasera = placa(78,76,8,44,5) - cilindro("Z",16,39,49)
    trasera = agujeros(trasera,COLUMNAS,2.2,40,48)
    giro[0] = pieza(giro[0]+atras*trasera,"Horquilla con apoyo de motor lateral",GIRO)
    estator,rotor = conjunto(52.5,60.5,48,circular(11,0,4))
    giro.extend(atras*p for p in estator)
    apertura.extend(atras*p for p in rotor)
    return [fijo,giro,apertura,flexion]


def gen_step():
    return {"shape": ensamblar(modulo_de_encaje), "params": "cadera_ak60_encaje.params.js"}
