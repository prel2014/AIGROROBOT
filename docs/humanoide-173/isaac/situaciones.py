"""Secuencias físicas por robot. Ángulos en radianes, posiciones en metros, tiempo en segundos."""
import numpy as np

TAREAS = ("flexionar_y_extender", "alternar_brazos", "flexion_bilateral", "caminar_parar_y_senalar")
CONTRATO = "xavier-situaciones-v1"
OBS_ADICIONALES = ("objetivo_izq_rad", "objetivo_der_rad", "codo_izq_rad", "codo_der_rad",
                   "orden_marcha", "residuo_previo_izq", "residuo_previo_der")
ACCIONES_ADICIONALES = ("residuo_codo_izq", "residuo_codo_der")


class Secuencias:
    def __init__(self, rng, cantidad, solo_frenado=False):
        self.tipo = rng.permutation(np.arange(cantidad) % len(TAREAS))
        self.solo_frenado = solo_frenado
        if solo_frenado:
            self.tipo.fill(3)
        self.lado = rng.integers(0, 2, cantidad)
        self.angulo = np.deg2rad(rng.uniform(28., 48., cantidad))
        self.recorrido = np.where(self.tipo == 3, rng.uniform(.38, .72, cantidad), 0.)
        self.fase = np.zeros(cantidad, dtype=int)
        self.sostenido = np.zeros(cantidad)
        self.punto_parada = np.zeros((cantidad, 2))

    def caminar(self):
        return (self.tipo == 3) & (self.fase == 0)

    def consignas(self):
        objetivo = np.zeros((len(self.tipo), 2))
        if self.solo_frenado:
            filas = np.flatnonzero(self.fase == 2)
            objetivo[filas, self.lado[filas]] = self.angulo[filas]
            return objetivo
        primera = (self.fase == 0) & (self.tipo != 3)
        segunda = (self.fase == 1) & ((self.tipo == 1) | (self.tipo == 3))
        filas = np.flatnonzero(primera | segunda)
        lado = np.where((self.tipo == 1) & segunda, 1 - self.lado, self.lado)
        objetivo[filas, lado[filas]] = self.angulo[filas]
        bilateral = primera & (self.tipo == 2)
        objetivo[bilateral] = self.angulo[bilateral, None]
        return objetivo

    def actualizar(self, dt, activos, cerca_meta, pasos, pies, vertical, rapidez, angulos, carga_relativa):
        objetivo = self.consignas()
        caminando = self.caminar()
        estable = (vertical > .96) & (rapidez < .16) & (carga_relativa > .55)
        estacionado = (self.tipo != 3) | (np.linalg.norm(pies - self.punto_parada, axis=1) < .22)
        error = np.max(abs(angulos - objetivo), axis=1)
        postura = estable & estacionado & (error <= np.deg2rad(8.))
        self.sostenido = np.where(activos & postura & ~caminando, self.sostenido + dt, 0.)
        mover_fase = activos & ((self.sostenido >= .35 - 1e-8) | (caminando & cerca_meta & (pasos >= 2)))
        al_parar = mover_fase & caminando
        self.punto_parada[al_parar] = pies[al_parar]
        self.fase += mover_fase
        self.sostenido[mover_fase] = 0.
        final = np.where((self.tipo == 1) | (self.tipo == 3), 3, 2)
        if self.solo_frenado:
            final.fill(4)
        acabo = mover_fase & (self.fase == final)
        premio = np.where(caminando, 0., .10 * np.exp(-5. * error) - .03 * ~estable)
        return acabo, premio + 1.5 * mover_fase
