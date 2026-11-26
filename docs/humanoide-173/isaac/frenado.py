"""Política de realimentación sagital. Unidades SI, en el marco de avance."""
import numpy as np

INICIAL = np.array([.32, .384, -.10, 2.8, .76, 0., 0., 0., .08, 0.])
NOMBRES = ('cadera', 'rodilla', 'sesgo_tobillo', 'kp_pos', 'kd_vel', 'kp_cadera',
           'kd_cadera', 'kp_inclinacion', 'rampa_s', 'centro_m')


def ventana_parada(restante, fase, fase_objetivo, ancho=.23):
    # Cada medio ciclo cambia el pie delantero. Hay una ventana homóloga por lado.
    return (restante <= .12) & ((fase-fase_objetivo) % np.pi < ancho)


def consignas(parametros, inicial, q, posicion, velocidad, cabeceo, tiempo):
    hip, knee, tobillo, kp, kv, hp, hv, tilt, rampa, centro = parametros.T
    error = posicion - centro
    orden = inicial.copy()
    orden[:, 0::3] = (hip + hp * error + hv * velocidad)[:, None]
    orden[:, 1::3] = knee[:, None]
    orden[:, 2::3] = q[:, 2::3] + (tobillo - kp*error - kv*velocidad + tilt*cabeceo)[:, None]
    mezcla = np.minimum(tiempo / rampa, 1.)[:, None]
    return inicial + mezcla * (orden - inicial)
