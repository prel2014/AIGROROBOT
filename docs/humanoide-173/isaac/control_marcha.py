"""Referencias articulares en radianes para el Xavier con los pies libres."""
import numpy as np

ARTICULACIONES = [f"{eje}_{lado}" for lado in ("izq","der") for eje in ("cadera","rodilla","tobillo")]
EJES_CADERA = [f"{eje}_cadera_{lado}" for lado in ("izq","der") for eje in ("giro","apertura")]
EJES_CONTROL = ARTICULACIONES + EJES_CADERA
ACCIONES_MARCHA = ARTICULACIONES + ["cadencia"] + EJES_CADERA


# Parámetros identificados con poblaciones de dieciséis Xavier en PhysX.
MARCHA = np.array([1.615, 1.561, 14.675, 12.205, .4417, 3.2384, .0682,
                   .5492, 1.041, 1.6143, .8405, 5.3419, .0244, .1654])
SESGO_FLEXION_GRADOS = 2.674
SESGO_TOBILLO_GRADOS = -3.826
APERTURA_MARCHA_GRADOS = 2.041
REALIMENTACION_LATERAL = .356
# Autoridad de los residuos medidos sobre los diez ejes, expresada en grados.
RESIDUO_SAGITAL = .15
RESIDUO_GIRO = .55
RESIDUO_APERTURA = .32
RESIDUO_CADENCIA = .008
GANANCIAS_CADERA_TRANSVERSAL = ([550., 480.], [16., 16.])
UMBRAL_CAPTURA_M = .006
ESPERA_EN_META_S = 1.
CAPTURA = np.array([.0168, .1884, .831, .2341, 1.1396, .216, -.1637, .0393])
GANANCIAS_MARCHA = ([745., 350., 248.], [13.8, 17.6, 7.9])
GANANCIAS_REPOSO = ([1000., 900., 600.], [10., 8., 4.])


def punto_captura(centro, velocidad, pies, frente):
    """Proyección longitudinal del centro de masa extrapolado, relativa a los tobillos."""
    altura = np.maximum(centro[:, 2] - pies[:, 2] + .029, .05)
    proyeccion = centro[:, :2] - pies[:, :2] + velocidad[:, :2] * np.sqrt(altura[:, None] / 9.81)
    return (proyeccion * frente).sum(axis=1) + .042


def postura_de_parada(inicial, captura, velocidad):
    """Conserva parte del escalonamiento de los pies al entrar en doble apoyo."""
    piernas = inicial.reshape(-1, 2, 3)
    postura = .4 * piernas + .6 * piernas.mean(axis=1, keepdims=True)
    postura[:, :, 0] += np.deg2rad(2.)
    postura[:, :, 1] += np.deg2rad(1.)
    error = np.clip(.002 - captura, -.12, .12)
    postura[:, :, 2] += (5. * error - .4 * velocidad - np.deg2rad(2.))[:, None]
    return postura.reshape(-1, 6)


def frenar_con_apoyo(inicial, q, separacion, velocidad, segundos):
    """Flexiona cadera y rodilla y regula el tobillo respecto al centro de los dos pies."""
    postura = inicial.copy()
    postura[:, 0::3] = .32
    postura[:, 1::3] = .384
    correccion = np.clip(1.4 * (separacion + .023) + .38 * velocidad, -.20, .20)
    postura[:, 2::3] = q[:, 2::3] - .10 - 2 * correccion[:, None]
    mezcla = np.minimum(segundos / .08, 1.)[:, None]
    return inicial + mezcla * (postura - inicial)


def corregir_apoyos(consigna, fase, amplitud, posicion, velocidad):
    """Corrige el lugar del próximo apoyo según el movimiento respecto a ambos pies."""
    centro, ritmo, kp_pie, kv_pie, kp_tobillo, kv_tobillo, kp_apoyo, kv_apoyo = CAPTURA
    error = posicion + .042 - centro  # El centro de la suela está 42 mm delante del tobillo.
    dv = velocidad - ritmo * amplitud
    for lado, signo in enumerate((1., -1.)):
        apoyo = 1. - np.maximum(0., signo * np.sin(fase))
        aterrizaje = .5 * (1. - signo * np.cos(fase))
        consigna[:, 3*lado] += np.clip((kp_pie*error + kv_pie*dv)*(aterrizaje + .2), -.22, .22)
        consigna[:, 3*lado] += np.clip((kp_apoyo*error + kv_apoyo*dv)*apoyo, -.16, .16)
        consigna[:, 3*lado+2] -= np.clip((kp_tobillo*error + kv_tobillo*dv)*apoyo, -.18, .18)
    return consigna


def referencias(parametros, fase, amplitud, vertical, omega, q, dq):
    """Una fila por robot. Ángulos del IMU y encoders en el marco corporal."""
    (_, paso, elevacion, flexion, sesgo, kp, kd, lateral, doble,
     postural, captura, referencia, mezcla, anticipo) = parametros.T
    cabeceo = np.rad2deg(np.arctan2(vertical[:, 1], vertical[:, 2]))
    ladeo = np.rad2deg(np.arctan2(-vertical[:, 0], vertical[:, 2]))
    q = np.rad2deg(q)
    dq = np.rad2deg(dq)
    inclinacion = cabeceo + .80 * (q[:, ::3] - .5*q[:, 1::3]).mean(axis=1)
    velocidad = np.rad2deg(omega[:, 0]) + .80*(dq[:, ::3] - .5*dq[:, 1::3]).mean(axis=1)
    tobillo_pd = np.clip(kp * (inclinacion + .8) + kd * velocidad, -8., 8.)
    balance = np.clip(lateral * (ladeo + .12*np.rad2deg(omega[:, 1])), -8., 8.)
    consigna = []
    for lado, signo in enumerate((1., -1.)):
        seno = signo * np.sin(fase)
        vuelo = np.maximum(seno, 0.)**doble * amplitud
        flex = flexion + elevacion * vuelo
        rodilla = np.clip(flex + signo * balance, 0., 65.)
        cadera = .5 * rodilla + sesgo - signo * paso * np.cos(fase) * amplitud
        pie_adelante = .5*(1.-signo*np.cos(fase))
        captura_fase = vuelo + anticipo*(pie_adelante-vuelo)
        cadera += np.clip(postural * (cabeceo - referencia + .09*np.rad2deg(omega[:,0])) * (1.-vuelo)
                          - captura * (inclinacion + .18*velocidad) * captura_fase, -14., 14.)
        apoyo = q[:, 3*lado+2] + tobillo_pd
        tobillo = (1.-vuelo)*apoyo + vuelo*(rodilla - cadera - cabeceo)
        tobillo = (1.-mezcla)*tobillo + mezcla*(rodilla-cadera+tobillo_pd)
        consigna.extend((cadera, rodilla, tobillo))
    return np.deg2rad(np.stack(consigna, axis=1))
