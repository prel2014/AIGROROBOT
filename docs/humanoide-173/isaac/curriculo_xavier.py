"""Currículo y árbitro físico. Metros, segundos, kilogramos y radianes."""
from dataclasses import dataclass, field
from math import pi
import numpy as np

VERSION = "xavier-habilidades-v1"
DT = 1/20
LADOS = ("izq", "der")
CUERPO = tuple(f"{e}_{s}" for s in LADOS for e in
               ("giro_cadera", "apertura_cadera", "cadera", "rodilla", "tobillo", "hombro", "codo", "muneca"))
DEDOS = tuple(f"{d}_{s}_m{m}" for s in LADOS for d in
              ("pulgar", "indice", "medio", "anular", "menique") for m in (1, 2))
ACCIONES = CUERPO + DEDOS + ("cuello",)


@dataclass(frozen=True)
class Plan:
    entornos: tuple
    fases: tuple
    segundos: int
    objeto: str = ""


PLANES = {
    "caminar": Plan(("vivienda", "calle", "campus"), ("puerta", "rodeo", "parar"), 45),
    "levantarse": Plan(("vivienda", "campus"), ("incorporar", "paso", "parar"), 35),
    "transportar": Plan(("vivienda", "campus"), ("recoger", "portar", "depositar"), 55, "caja"),
    "imitar": Plan(("vivienda", "campus"), ("gesto1", "gesto2", "gesto3"), 30),
    "tijeras": Plan(("vivienda", "campus"), ("recoger", "depositar"), 30, "tijeras"),
    "correr": Plan(("campus", "calle"), ("acelerar", "carrera", "parar"), 40),
    "encestar": Plan(("campus",), ("recoger", "soltar", "canasta", "equilibrar"), 30, "pelota"),
    "pirueta": Plan(("campus",), ("girar", "aterrizar"), 25),
    "mision": Plan(("vivienda",), ("incorporar", "paso", "recoger", "portar", "depositar"), 80, "caja"),
}


@dataclass
class Medida:
    pelvis: np.ndarray
    vertical: float
    velocidad: float
    pies: np.ndarray             # fracciones del peso
    obstaculo: bool = False
    asiento: bool = False
    alcance: float = 0.
    objeto: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rapidez_objeto: float = 0.
    agarre: bool = False
    toca_mano: bool = False
    en_destino: bool = False
    apoyo_objeto: bool = False
    toca_hoja: bool = False
    delta_giro: float = 0.
    error_gesto: float = 1.
    avance_pies: float = 0.
    velocidad_horizontal: float = 0.
    error_aterrizaje: float = 0.
    pies_sin_contacto: bool = True
    velocidad_vertical_objeto: float = 0.
    contacto_cuerpo: bool = False


def diseno(tarea, nivel):
    """Posiciones locales compartidas por la escena, las marcas y el árbitro."""
    distancia = (1.2, 1.8, 2.4)[nivel]
    destino = np.array([.82, distancia+.95, .80])
    origen = np.array([0., .45, .82])
    if tarea == "mision":
        origen[1] += .65
        destino[1] += .65
    if tarea == "tijeras":
        origen[:] = [.235, .30, .79]
        destino[:] = [.23, .75, .77]
    if tarea == "encestar":
        origen[:] = [.23, .30, .93]
        destino[:] = [0., (1., 1.4, 1.8)[nivel], (1.30, 1.55, 1.80)[nivel]]
    final = np.array([0., distancia + 1., 0.])
    if tarea == "correr":
        final = np.array([(.0, .25, .60)[nivel], 3.8 + nivel, 0.])
    if tarea == "levantarse":
        final = np.array([0., .60, 0.])
    return origen, destino, final


class Episodio:
    def __init__(self, tarea, nivel):
        self.plan = PLANES[tarea]
        self.fases = self.plan.fases
        if tarea=="correr" and nivel>0:
            self.fases = ("acelerar","curva","carrera","parar")
        self.tarea, self.nivel = tarea, nivel
        self.origen, self.destino, self.final = diseno(tarea, nivel)
        self.indice = self.ticks = self.sostiene = 0
        self.vuelo = self.max_vuelo = self.giro = 0.
        self.anterior = None
        self.distancia_previa = None
        self.hecho = self.exito = self.apoyado = self.puerta_cruzada = False
        self.vuelos = self.ticks_fase = 0
        self.ultimo_apoyo = -1
        self.alternancias = 0
        self.xy_anterior = np.zeros(2)
        self.inicio_valido = False
        self.colocado = False
        self.suelo_inicial = tarea == "levantarse" and nivel == 2

    @property
    def fase(self):
        return self.fases[min(self.indice, len(self.fases)-1)]

    def meta(self):
        destinos = {"puerta": (.0, .72), "rodeo": (.65, 1.55), "paso": (0., .48),
                    "acelerar": (0., .75), "curva": (-.60*self.nivel, 1.65),
                    "carrera": (self.final[0], self.final[1]-.65),
                    "portar": (.73, self.destino[1]-.6), "recoger": self.origen[:2],
                    "depositar": self.destino[:2], "canasta": self.destino[:2]}
        return np.asarray(destinos.get(self.fase, self.final[:2]))

    def evaluar(self, m):
        if self.hecho:
            return 0.
        fase = self.fase
        self.ticks += 1
        self.ticks_fase += 1
        linea = .72 if self.tarea=="caminar" else self.destino[1]-.72
        centro = 0. if self.tarea=="caminar" else .73
        if self.xy_anterior[1] < linea <= m.pelvis[1] and (self.tarea=="caminar" or fase=="portar"):
            f = (linea-self.xy_anterior[1])/(m.pelvis[1]-self.xy_anterior[1])
            x = self.xy_anterior[0] + f*(m.pelvis[0]-self.xy_anterior[0])
            self.puerta_cruzada |= abs(x-centro) < .48
        self.xy_anterior = m.pelvis[:2].copy()
        distancia = np.linalg.norm(m.pelvis[:2]-self.meta())
        firme = m.vertical > .94 and m.velocidad < .17 and min(m.pies) > .035
        if fase == "incorporar":
            self.inicio_valido |= (m.vertical < .45 if self.suelo_inicial else m.asiento)
        caida = m.vertical < .60 or m.pelvis[2] < .65 or m.contacto_cuerpo
        acrobacia = fase == "girar" and self.nivel == 2 and max(m.pies) < .025 and not m.contacto_cuerpo
        if m.obstaculo or m.toca_hoja or (caida and not (fase == "incorporar" or acrobacia)):
            self.hecho = True
            return -6.
        aire = m.pies_sin_contacto and max(m.pies) < .025
        if max(m.pies)>.15 and min(m.pies)<.035:
            lado = int(np.argmax(m.pies))
            self.alternancias += int(self.ultimo_apoyo not in (-1,lado))
            self.ultimo_apoyo = lado
        if not aire:
            if self.vuelo >= .05:
                self.vuelos += 1
            self.apoyado |= max(m.pies)>.10
            if fase=="girar" and self.nivel>0:
                self.giro = 0.
        self.vuelo = self.vuelo + DT if aire and self.apoyado else 0.
        self.max_vuelo = max(self.max_vuelo, self.vuelo)
        if fase == "girar" and (self.nivel == 0 or self.vuelo>0.):
            self.giro += m.delta_giro
        sostener = .15
        if fase in ("puerta", "rodeo", "paso", "acelerar", "curva", "parar"):
            ok = distancia < .25 and m.vertical > .9
            if fase == "parar":
                ok &= firme
                sostener = 1.
            if fase == "paso":
                ok &= m.avance_pies > .20
            if fase == "puerta":
                ok &= self.puerta_cruzada
        elif fase == "incorporar":
            distancia = abs(1.17-m.pelvis[2])
            ok = self.inicio_valido and firme and not m.asiento and distancia < .12
            sostener = .6
        elif fase == "carrera":
            ok = distancia < .33 and m.velocidad_horizontal > .60 and self.vuelos >= 2 and self.alternancias>=1
            sostener = DT
        elif fase.startswith("gesto"):
            distancia = m.error_gesto
            ok = distancia < .13 and m.vertical > .9 and max(m.pies) > .15 and self.ticks_fase*DT>=.65
            if self.ticks_fase*DT > (5.,4.,3.)[self.nivel]:
                self.hecho = True
                return -3.
            sostener = .4
        elif fase == "recoger":
            elevacion = m.objeto[2]-self.origen[2]
            distancia = m.alcance + max(.09-elevacion, 0.)
            ok = m.agarre and elevacion > .075 and not m.apoyo_objeto
            sostener = .25
        elif fase in ("portar", "depositar"):
            if not m.toca_mano and not m.en_destino and m.objeto[2] < self.origen[2]-.18:
                self.hecho = True
                return -5.
            if fase == "portar":
                distancia = np.linalg.norm(m.objeto[:2]-self.meta())
                ok = distancia < .32 and self.puerta_cruzada and m.agarre and not m.apoyo_objeto
                if m.apoyo_objeto or not m.toca_mano:
                    self.hecho = True
                    return -4.
            else:
                distancia = np.linalg.norm(m.objeto-self.destino)
                self.colocado |= m.en_destino and m.toca_mano and m.rapidez_objeto < .25
                ok = self.colocado and distancia < .25 and m.en_destino and not m.toca_mano and m.rapidez_objeto < .10 and firme
                sostener = .65
        elif fase == "soltar":
            distancia = np.linalg.norm(m.objeto-self.destino)
            ok = (not m.toca_mano and not m.apoyo_objeto and m.velocidad_vertical_objeto > .30
                  and np.linalg.norm(m.objeto[:2]-self.destino[:2]) > .30)
            sostener = DT
        elif fase == "canasta":
            if m.toca_mano or m.apoyo_objeto:
                self.hecho = True
                return -4.
            distancia = np.linalg.norm(m.objeto-self.destino)
            ok = False
            if self.anterior is not None and self.anterior[2] > self.destino[2] >= m.objeto[2]:
                f = (self.anterior[2]-self.destino[2])/(self.anterior[2]-m.objeto[2])
                centro = self.anterior + f*(m.objeto-self.anterior)
                ok = not m.toca_mano and np.linalg.norm(centro[:2]-self.destino[:2]) < .225-.11
            sostener = DT
        elif fase == "girar":
            objetivo = (pi, pi/2, 2*pi)[self.nivel]
            distancia = abs(objetivo-abs(self.giro))
            ok = distancia < .30 and (self.nivel == 0 or self.max_vuelo >= .10)
            sostener = DT
        else:  # aterrizar o equilibrar después del lanzamiento
            distancia = np.linalg.norm(m.pelvis[:2])
            ok = firme and (fase == "equilibrar" or (distancia < .45 and m.error_aterrizaje < .32))
            sostener = 1.
        self.anterior = m.objeto.copy()
        progreso = 0. if self.distancia_previa is None else self.distancia_previa-distancia
        self.distancia_previa = distancia
        self.sostiene = self.sostiene+1 if ok else 0
        premio = 3.*np.clip(progreso, -.25, .25) + .012*np.exp(-3*distancia) - .015
        if self.sostiene*DT + 1e-8 >= sostener:
            self.indice += 1
            self.ticks_fase = self.vuelos = 0
            self.ultimo_apoyo, self.alternancias = -1, 0
            self.sostiene = 0
            self.distancia_previa = None
            self.max_vuelo = self.vuelo = 0.
            self.exito = self.indice == len(self.fases)
            premio += 8. if self.exito else 2.
        self.hecho = self.exito or self.ticks*DT >= self.plan.segundos
        return float(premio)
