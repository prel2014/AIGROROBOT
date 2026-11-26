"""Cuatro zonas sensibles en la cara inferior de cada pie rígido del Xavier."""
import numpy as np
from scipy.spatial.transform import Rotation


# Metros, en el marco del centro de la caja de contacto del pie. +Y apunta a los dedos.
ANCHO = .119
LARGO = .245
Z_SUELA = -.029
TOLERANCIA = .004  # Incluye el margen de contacto y la penetración del solver.
ZONAS = ("TalonInterior", "TalonExterior", "AntepieInterior", "AntepieExterior")
LADOS = ("izq", "der")


def centro_zona(lado, indice):
    hacia_dentro = 1. if lado == "izq" else -1.
    x = hacia_dentro * (1 if indice % 2 == 0 else -1) * ANCHO/4
    y = (-1 if indice < 2 else 1) * LARGO/4
    return x, y, Z_SUELA


def integrar_contactos(acumulado, poses, muestras):
    """Suma impulsos normales [N·s] por cuadrante, en el marco de la suela de ese tick."""
    if not muestras:
        return
    pies, puntos, impulsos, normales = zip(*muestras)
    pies = np.asarray(pies)
    cuerpos = poses.reshape(-1, 7)
    rotaciones = Rotation.from_quat(cuerpos[:, 3:]).as_matrix()[pies]
    # Cada columna de R es un eje del pie expresado en coordenadas del mundo.
    local = np.einsum("ni,nij->nj", np.asarray(puntos)-cuerpos[pies, :3], rotaciones)
    normal = np.asarray(normales)
    normal_z = np.einsum("ni,ni->n", normal, rotaciones[:, :, 2])
    impulso = np.abs(np.einsum("ni,ni->n", impulsos, normal) * normal_z)
    x, y, z = local.T
    planta = ((abs(z-Z_SUELA) <= TOLERANCIA) & (abs(x) <= ANCHO/2+TOLERANCIA)
              & (abs(y) <= LARGO/2+TOLERANCIA))
    derecho = pies % 2 == 1
    exterior = np.where(derecho, x > 0., x < 0.)
    zona = 2 * (y >= 0.).astype(int) + exterior
    np.add.at(acumulado.reshape(-1, 4), (pies[planta], zona[planta]), impulso[planta])


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ejes = plt.subplots(1, 2, figsize=(7.6, 6.3), sharey=True)
    colores = ("#d1e6f1", "#f4dfbd")
    ancho, largo = 1000*ANCHO, 1000*LARGO
    for eje, lado, titulo in zip(ejes, LADOS, ("Pie izquierdo", "Pie derecho")):
        for i, nombre in enumerate(("Talón\ninterior", "Talón\nexterior", "Antepié\ninterior", "Antepié\nexterior")):
            x, y, _ = np.asarray(centro_zona(lado, i))*1000
            eje.add_patch(Rectangle((x-ancho/4, y-largo/4), ancho/2, largo/2,
                                    facecolor=colores[i%2], edgecolor="#294658", linewidth=1))
            eje.text(x, y+14, nombre, ha="center", va="center", fontsize=10)
            eje.plot(x, y, "+", color="#294658")
        eje.set(xlim=(-72, 72), ylim=(-137, 137), aspect="equal", xlabel="X local (mm)",
                xticks=[-59.5, 0, 59.5], yticks=[-122.5, -61.25, 0, 61.25, 122.5], title=titulo)
        eje.set_yticklabels(["−122,5", "−61,25", "0", "61,25", "122,5"])
        eje.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Y local (mm)  ·  +Y hacia los dedos")
    fig.suptitle("Zonas de carga del Xavier", fontsize=17, weight="bold")
    fig.subplots_adjust(left=.12, right=.96, top=.87, bottom=.17, wspace=.15)
    fig.text(.5, .055, "Vista en planta   ·   suela de 119 × 245 mm   ·   + centro de cada zona",
             ha="center", fontsize=10)
    fig.savefig(Path(__file__).with_name("plano_cargas_plantares.png"), dpi=160)
