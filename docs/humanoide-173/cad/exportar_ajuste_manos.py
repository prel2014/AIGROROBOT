"""Exporta la geometría de ajuste en mm y comprueba cada STL después de escribirlo."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import trimesh

from manos_servo import cover,palm,wrist_parts,yoke


def write_mesh(shape,path):
    points,faces=shape.tessellate(.04,.10)
    mesh=trimesh.Trimesh(np.asarray([tuple(p) for p in points],dtype='float32'),faces)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-7))
    mesh.update_faces(mesh.unique_faces())
    # OCC can leave a sub-micrometre sliver attached along just one edge.
    # Remove only isolated tessellation fins with two free edges and a triple edge.
    degrees=np.bincount(mesh.edges_unique_inverse)[mesh.faces_unique_edges]
    fins=(np.sum(degrees==1,axis=1)==2)&np.any(degrees==3,axis=1)&(mesh.area_faces<1e-4)
    mesh.update_faces(~fins)
    mesh.remove_unreferenced_vertices()
    mesh.export(path)
    reopened=trimesh.load_mesh(path)
    if not (reopened.is_volume and reopened.body_count==1):
        raise ValueError(f'{path.name}: revisar superficie cerrada u orientación')


def main():
    cad=Path(__file__).resolve().parent
    out=cad/'ajuste_manos'
    out.mkdir(exist_ok=True)
    files=[]
    for side,suffix in [(1,'D'),(-1,'I')]:
        parts={'palma':palm(side),'base_muneca':yoke(side),'nucleo':wrist_parts(side)[0],
               'tapa_palmar':cover(side,False),'tapa_dorsal':cover(side,True)}
        for name,shape in parts.items():
            path=out/f'{name}_{suffix}.stl'
            write_mesh(shape,path)
            files.append(path)
            print(path.name,'cerrado',flush=True)
    with ZipFile(cad/'maquetas_union_manos.zip','w',ZIP_DEFLATED) as bundle:
        for path in files:bundle.write(path,path.name)
        bundle.write(cad.parent/'referencias'/'AmazingHand'/'ATRIBUCION.md','ATRIBUCION.md')
        bundle.write(cad.parent/'referencias'/'AmazingHand'/'LICENSE-Apache-2.0.txt','LICENSE-Apache-2.0.txt')


if __name__=='__main__':
    main()
