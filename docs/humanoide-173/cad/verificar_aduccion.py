"""Mide el espacio entre las suelas con piernas extendidas y pelvis fija."""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import brentq

from cadera_xavier import HIP_X, HIP_Z


MODELO = Path(__file__).resolve().parent / "__cadgen__/models/humanoide_xavier.step.py"


def obtener_suelas():
    escena = json.loads((MODELO / "assembly.json").read_text(encoding="utf-8"))
    suelas = {}
    for pieza in escena["occurrences"]:
        if pieza["name"] != "suela_continua":
            continue
        componente = escena["components"][pieza["component"]]
        malla = trimesh.load(MODELO / componente["glb"], force="scene").to_geometry()
        matriz = np.asarray(pieza["transform"]).reshape(4, 4)
        vertices = (malla.vertices * 1000) @ matriz[:3, :3].T + matriz[:3, 3]
        lado = int(np.sign(vertices[:, 0].mean()))
        suelas[lado] = vertices
    if set(suelas) != {-1, 1}:
        raise ValueError("Se necesitan las dos suelas del modelo Xavier")
    return suelas


def borde_interno(vertices, lado, aduccion):
    angulo = np.radians(lado * aduccion)
    dx = vertices[:, 0] - lado * HIP_X
    dz = vertices[:, 2] - HIP_Z
    x = lado * HIP_X + dx * np.cos(angulo) + dz * np.sin(angulo)
    z = HIP_Z - dx * np.sin(angulo) + dz * np.cos(angulo)
    indices = np.abs(lado * x - np.min(lado * x)) < 1e-4
    return np.column_stack((x[indices], vertices[indices, 1], z[indices]))


def main():
    suelas = obtener_suelas()

    def separacion(angulo):
        izq = borde_interno(suelas[-1], -1, angulo)
        der = borde_interno(suelas[1], 1, angulo)
        return der[:, 0].min() - izq[:, 0].max()

    for grados in (0, 2, 3):
        print(f"{grados} grados por lado, espacio transversal {separacion(grados):.2f} mm")
    limite = brentq(separacion, 0, 8)
    izquierda = borde_interno(suelas[-1], -1, limite)
    derecha = borde_interno(suelas[1], 1, limite)
    distancia = np.linalg.norm(izquierda[:, None, :] - derecha[None, :, :], axis=2).min()
    if distancia > 0.001:
        raise ValueError("Los bordes coinciden en X pero están separados en Y o Z")
    print(f"Primer contacto bilateral entre suelas a {limite:.3f} grados por cadera")


if __name__ == "__main__":
    main()
