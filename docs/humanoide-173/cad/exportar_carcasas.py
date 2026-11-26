"""Exporta los paneles y las tapas del CAD para su revisión en el laminador."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED,ZipFile
from build123d import Pos,export_stl
from tren_inferior import Leg


def build():
    root=Path(__file__).resolve().parent
    with TemporaryDirectory(dir=root/'__cadgen__') as tmp:
        folder=Path(tmp)
        for side,code in ((1,'D'),(-1,'I')):
            leg=Leg(side)
            for low,high,name in ((144,419,'pantorrilla'),(545,810,'muslo')):
                segment=leg.rails(low,high,name)
                for p in segment.children[34].children:
                    save(p,folder/f'{code}_{name}_{p.label.replace(" ","_")}.stl')
            cap=leg.motor().children[0].children[2].children[0]
            save(cap,folder/f'{code}_tapa_rodilla.stl')
        with ZipFile(folder/'carcasas_piernas_STL.zip','w',ZIP_DEFLATED) as kit:
            for p in sorted(folder.glob('*.stl')):
                kit.write(p,p.name)
        (folder/'carcasas_piernas_STL.zip').replace(root/'carcasas_piernas_STL.zip')


def save(shape,path):
    scan=shape.bounding_box()
    assert max(scan.size)<256 and len(shape.solids())==1 and shape.is_valid
    pose=Pos(-scan.center().X,-scan.center().Y,-scan.min.Z)
    if not export_stl(pose*shape,path,tolerance=.08,angular_tolerance=.16):
        raise RuntimeError(path)


def arms():
    from arnes_brazos import upper_with_pilots
    from equipo_brazos import upper_panels,elbow_motor
    from instalacion_control import cover_plate
    from interior_xavier import original,normal
    root=Path(__file__).resolve().parent
    robot=original()
    with TemporaryDirectory(dir=root/'__cadgen__') as tmp:
        folder=Path(tmp)
        for side,code,word in ((1,'D','derecho'),(-1,'I','izquierdo')):
            arm=next(o for o in robot.children if normal(o.label)=='brazo '+word)
            core=next(o for o in arm.children if normal(o.label)=='brazo')
            upper=upper_panels(side,upper_with_pilots(core,side))
            for panel,label in zip(upper.children[-2:],('anterior','dorsal')):
                save(panel.children[0],folder/f'{code}_humero_{label}.stl')
            save(cover_plate(side),folder/f'{code}_tapa_antebrazo.stl')
            motor=elbow_motor(side)
            save(motor.location*motor.children[0].children[2].children[0],folder/f'{code}_tapa_codo.stl')
        with ZipFile(folder/'carcasas_brazos_STL.zip','w',ZIP_DEFLATED) as kit:
            for p in sorted(folder.glob('*.stl')):
                kit.write(p,p.name)
        (folder/'carcasas_brazos_STL.zip').replace(root/'carcasas_brazos_STL.zip')


def left_harness():
    from equipo_brazos import left_guide,left_plug_boot
    root=Path(__file__).resolve().parent
    base,keeper=left_guide(1280).children[:2]
    parts=[(base,'base_guia_imprimir_dos.stl'),(keeper,'tapa_guia_imprimir_dos.stl'),
           (left_plug_boot().children[0],'referencia_manguito_elastomero_P2.stl')]
    with TemporaryDirectory(dir=root/'__cadgen__') as tmp:
        folder=Path(tmp)
        target=folder/'guia_traccion_izquierda_STL.zip'
        with ZipFile(target,'w',ZIP_DEFLATED) as kit:
            for shape,filename in parts:
                save(shape,folder/filename)
                kit.write(folder/filename,filename)
        target.replace(root/target.name)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    options=parser.add_mutually_exclusive_group()
    options.add_argument('--brazos',dest='export',action='store_const',const=arms)
    options.add_argument('--guia-izquierda',dest='export',action='store_const',const=left_harness)
    parser.set_defaults(export=build)
    parser.parse_args().export()
