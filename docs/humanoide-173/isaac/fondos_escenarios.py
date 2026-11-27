"""Fondo tridimensional y obras de riego de los escenarios de Xavier. Unidades en metros."""
import random

import numpy as np

from entornos_xavier import modelo


def estacion_riego(t):
    g = t.grupo(t.raiz, "CasetaDeRiego", (4.1, 26., 0), -10)
    t.caja(g,"Losa",(0,0,.12),(7.1,4.7,.24),"hormigon")
    t.caja(g,"Caseta",(-1.2,.4,1.65),(3.9,3.4,2.9),"ladrillo",True)
    t.caja(g,"Porton",(-1.7,-1.32,1.37),(1.42,.07,2.26),"petroleo")
    t.caja(g,"MarcoDeRejilla",(-.23,-1.34,2.03),(.75,.10,.65),"grafito")
    for z in np.linspace(1.8,2.26,7):
        t.caja(g,"Lama",(-.23,-1.40,z),(.68,.065,.035),"aluminio")
    # Techo de chapa con grecas y pendiente hacia la parte trasera.
    caras = []
    for i in range(84):
        x = -3.48+i*.055
        z0, z1 = (3.29,3.34) if i%2 else (3.34,3.29)
        caras.append([(x,-1.78,z0),(x+.055,-1.78,z1),(x+.055,2.6,z1-.26),(x,2.6,z0-.26)])
    t.malla(g,"Techo",caras,"aluminio")
    t.disco(g,"Deposito",(2.05,.15,1.49),1.02,2.48,"grafito",True)
    for z in (.42,.69,1.03,1.43,1.83,2.24,2.66):
        t.disco(g,"Aro",(2.05,.15,z),1.045,.046,"grafito")
    t.disco(g,"TapaRegistro",(2.05,.15,2.78),.43,.11,"grafito")
    t.varilla(g,"SalidaDeAgua",(2.04,-.80,.54),(2.04,-1.95,.54),.063,"petroleo")
    t.varilla(g,"Bajada",(2.04,-1.95,.54),(2.04,-1.95,.03),.064,"petroleo")


def callejones_huerto(t):
    """Cubierta vegetal irregular y goteo a los lados de las calles de servicio."""
    rng = random.Random(191)
    for x in (-14.,-11.,-8.,-5.,-2.,1.,4.,7.,10.,13.,16.):
        for inicio,fin in ((1.1,20.),(30.,55.)):
            caras = []
            hierbas = []
            borde = None
            for y in np.arange(inicio,fin,.42):
                mitad = rng.uniform(.44,.68)
                nuevo = [(x+mitad*u,y,.052*(1.-u*u)-.006) for u in (-1.,-.60,0.,.60,1.)]
                if borde:
                    for j in range(len(nuevo)-1):
                        caras.append([borde[j],borde[j+1],nuevo[j+1],nuevo[j]])
                borde = nuevo
                for _ in range(2):
                    centro = x+rng.uniform(-.4,.4)
                    for _ in range(6):
                        px,py = centro+rng.uniform(-.08,.08),y+rng.uniform(-.07,.07)
                        dx,dy = rng.uniform(-.05,.05),rng.uniform(-.05,.05)
                        hierbas.append([(px-.006,py,.042),(px+.006,py,.042),
                                        (px+dx,py+dy,rng.uniform(.091,.16))])
            t.malla(t.raiz,"Camellon",caras,"suelo_seco")
            t.malla(t.raiz,"HierbaEspontanea",hierbas,"oliva")
            t.varilla(t.raiz,"LineaDeGoteo",(x-.20,inicio,.056),(x-.20,fin,.056),.009,"grafito")
    t.suelo(t.raiz,"AccesoCaseta",(4.1,24.),(5.8,3.4),"hormigon",.004)
    estacion_riego(t)
    # Cortina rompevientos al fondo, a otra escala que los frutales de producción.
    for i in range(19):
        modelo(t,t.raiz,"Cortaviento","tree_small_02/visual.usdc",
               ((i-9)*4.1,59.+rng.uniform(-1.5,1.5),0),rng.randrange(360),rng.uniform(1.35,1.85))


def frutos_de_fondo(t, arbol, indice):
    # Puntos medidos en las ramas del modelo local, antes de su giro y escala.
    apoyos = [(1.08,.42,3.19),(.74,-1.08,2.79),(-.23,-2.08,3.),(1.15,.73,3.28),
              (.45,.80,1.95),(-.35,-.76,3.42),(-.16,-2.42,2.73),(1.26,.37,2.95)]
    for j,(x,y,z) in enumerate(apoyos):
        if (indice+j)%4:
            modelo(t,arbol,"FrutoEnRama","orange/visual.usda",(x,y,z-.095),j*43,1.25)
