"""Exporta sólo las piezas fabricadas de la revisión de control. STL en mm sobre Z=0."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
from build123d import Pos, Rot, import_step
from exportar_ajuste_manos import write_mesh
from instalacion_control import CAD

def grounded(shape):
    shape=type(shape)(shape.wrapped)
    bounds=shape.bounding_box()
    return Pos(-bounds.center().X,-bounds.center().Y,-bounds.min.Z)*shape

def main():
    assembled=import_step(CAD/'humanoide_xavier.step')
    tray=assembled.children[1].children[0]
    parts=[('bandeja_xavier_dos_placas',tray)]
    for side,letter in [('derecho','D'),('izquierdo','I')]:
        arm=next(p for p in assembled.children if 'brazo_'+side in p.label.lower())
        housing,_,_,lid=arm.children[0].children
        hand=arm.children[-2]
        palm=next(p for p in hand.children if 'entrada_de_mazo' in p.label.lower())
        parts.extend([('carcasa_antebrazo_'+letter,housing),('tapa_dorsal_mano_'+letter,palm)])
        loom=next(p for p in assembled.children if 'cableado' in p.label.lower())
        ramal=loom.children[2 if letter=='D' else 1]
        for guide in ramal.children[-2:]:
            parts.append(('peine_'+letter+'_'+guide.label.split('_')[-1],guide))
        if letter=='D':parts.append(('tapa_registro_imprimir_dos',Rot(-90,0,0)*type(lid)(lid.wrapped)))
    output=CAD/'piezas_instalacion_control.zip'
    with TemporaryDirectory() as directory:
        tmp=Path(directory)
        for label,shape in parts:
            size=shape.bounding_box().size
            if max(size)>256:raise ValueError((label,size))
            write_mesh(grounded(shape),tmp/(label+'.stl'))
            print(label,tuple(round(v,1) for v in size),flush=True)
        with ZipFile(tmp/output.name,'w',ZIP_DEFLATED) as z:
            for f in sorted(tmp.glob('*.stl')):z.write(f,f.name)
            z.write(CAD.parent/'referencias/AmazingHand/ATRIBUCION.md','ATRIBUCION.md')
            z.write(CAD.parent/'referencias/AmazingHand/LICENSE-Apache-2.0.txt','LICENSE-Apache-2.0.txt')
        output.write_bytes((tmp/output.name).read_bytes())
        (CAD/'maquetas_xavier/bandeja_xavier.stl').write_bytes((tmp/'bandeja_xavier_dos_placas.stl').read_bytes())

if __name__=='__main__':main()
