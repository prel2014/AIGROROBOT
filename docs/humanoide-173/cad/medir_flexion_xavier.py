"""Cotas de proyección de las mallas del STEP instalado, en mm."""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from cadera_xavier import HIP_X, HIP_Z


CAD = Path(__file__).resolve().parent
PAQUETE = CAD / "__cadgen__/models/cadera_xavier_instalada.step.py"


def leer():
    escena = json.loads((PAQUETE / "assembly.json").read_text(encoding="utf8"))
    motores = {n["id"]:parte["name"] for parte in escena["assembly"]["root"]["children"]
               for n in parte["children"] if n["name"].startswith("AK60-39 V3 flexion")}
    modelos = {}
    piezas = []
    for o in escena["occurrences"]:
        k = o["component"]
        if k not in modelos:
            ruta = PAQUETE / escena["components"][k]["glb"]
            modelos[k] = trimesh.load(ruta,force="scene").to_geometry()
        malla = modelos[k]
        matriz = np.array(o["transform"]).reshape(4,4)
        vertices = malla.vertices*1000 @ matriz[:3,:3].T + matriz[:3,3]
        grupo = motores.get(".".join(o["id"].split(".")[:3]),"")
        lado = (1 if "derecha" in grupo else -1) if grupo else 0
        piezas.append((vertices,malla.faces,o["name"],lado,o["color"][:3]))
    return piezas


def medidas(piezas):
    todo = np.concatenate([v for v, *_ in piezas])
    escudo = next(v for v,_,n,_,_ in piezas if n=="escudo pelvico adaptada")
    cascos = {s:np.concatenate([v for v,_,_,l,_ in piezas if l==s]) for s in (-1,1)}
    ambos = np.concatenate(list(cascos.values()))
    neutra = cascos[1][:,0].min()-cascos[-1][:,0].max()
    # Las tapas, carcasas y su placa de anclaje quedan dentro de dos esferas
    # centradas en las caderas. Su separación da una cota conservadora incluso
    # para ángulos diferentes a ambos lados. La brida móvil es axial.
    radios = []
    for lado,v in cascos.items():
        soporte = next(pts for pts,_,n,_,_ in piezas if "soporte medial" in n and lado*pts[:,0].mean()>0)
        vertices = np.vstack((v,soporte))-np.array([lado*HIP_X,0,HIP_Z])
        radios.append(np.linalg.norm(vertices[ConvexHull(vertices).vertices],axis=1).max())
    return dict(escudo=escudo, motores=ambos, completo=todo, neutra=neutra,
                separacion_envolventes=2*HIP_X-sum(radios))


if __name__=="__main__":
    m = medidas(leer())
    for clave in ("escudo","motores","completo"):
        v=m[clave]
        print(clave,"XYZ mínimo",v.min(axis=0).round(2),"máximo",v.max(axis=0).round(2))
    print("Entre motores en neutro",round(m["neutra"],2))
    print("Entre envolventes de motores y soporte en rotación",round(m["separacion_envolventes"],2))
