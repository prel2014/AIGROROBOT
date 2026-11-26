"""Pruebas geométricas del cartucho y del recorrido de mantenimiento, en mm."""
from itertools import combinations
from pathlib import Path
from build123d import Compound, Pos, import_step, GeomType
from verificar_instalacion import overlap
from interior_xavier import bank, battery_lid, find_torso, modified_torso, normal, original
import cartucho_auxiliar as aux


def bare(node):
    return Compound.cast(node.wrapped)


def verify(model=None):
    if model:
        robot=import_step(model)
        chassis,bank_part,lid=robot.children[0],robot.children[2],robot.children[3]
        fixed=robot.children[-1]
        assert 'auxiliar' in fixed.label.lower()
    else:
        chassis=modified_torso(find_torso(original()))
        bank_part,lid,fixed=bank(),battery_lid(),aux.robot_loom()
    children=bank_part.children
    issues=[]
    def free(a,b,description):
        v=overlap(bare(a),bare(b))
        if v>.025:
            issues.append((description,round(v,3)))
    cells=[p for p in children if '18650' in p.label]
    port=next(p for p in children if normal(p.label).startswith('j1 cartucho'))
    protector=next(p for p in children if normal(p.label).startswith('protector y balanceador'))
    lid_top=next(p for p in children if normal(p.label).startswith('trampilla'))
    assert len(cells)==36
    for group in range(1,5):
        assert sum(p.label.replace('_',' ').endswith(f'grupo {group}') for p in cells)==9
    for p in children:
        assert p.is_valid and all(s.volume>0 for s in p.solids()),p.label
    for a,b in combinations(children,2):
        free(a,b,a.label+' / '+b.label)
    for p in [*children,*fixed.children]:
        free(p,lid,'tapa / '+p.label)
    for p in fixed.children:
        free(p,chassis,'torso / '+p.label)
        for q in children:
            free(p,q,p.label+' / '+q.label)
    # Incluye guía de entrada y brida. La tapa y el mazo externo ya se han retirado.
    for travel in (0,2,8,30,60,95,140):
        free(Pos(0,-travel,0)*bare(bank_part),chassis,f'extracción a {travel} mm')
    for travel in (0,2,8,32):
        free(Pos(0,-travel,0)*bare(fixed.children[0]),port,'desacople eléctrico')
    for travel in (1,4,15,40):
        for other in children:
            if other is not lid_top:
                free(Pos(0,0,travel)*bare(lid_top),other,'trampilla / '+other.label)
    if issues:
        for row in issues:print(*row)
        raise ValueError(f'{len(issues)} cruces por encima de 0,025 mm3')
    plate=next(p for p in protector.children if normal(p.label).startswith('pcb bms'))
    size=plate.bounding_box().size
    assert len(protector.children)==42, 'Falta un modelo en la exportación de KiCad'
    assert abs(size.X-72)<.01 and abs(size.Y-34)<.01
    centers={(round(e.arc_center.X,3),round(e.arc_center.Y,3)) for e in plate.edges()
             if e.geom_type==GeomType.CIRCLE and abs(e.radius-1.4)<.001}
    assert centers==set(aux.FIX),(centers,aux.FIX)
    headers=[p for p in protector.children if normal(p.label).startswith(('j1 ','j2 ','j3 '))]
    assert len(headers)==3
    for h in headers:
        ref=normal(h.label).split()[0]
        plug=next(p for p in children if normal(p.label).startswith(ref+' xhp'))
        assert bare(h).distance_to(bare(plug))<.01, ref+' sin acoplar'
    upper=lid_top.bounding_box().min.Z-protector.bounding_box().max.Z
    lower=min(p.bounding_box().min.Z for p in headers)-aux.FLOOR
    assert upper>0 and lower>0
    print(f'{len(cells)} celdas, 4 grupos de 9. PCB nominal {size.X:.0f} x {size.Y:.0f} x 1.6 mm, 41 componentes.')
    print(f'Cuatro taladros de 2.8 mm alineados. Tapa a placa equipada {upper:.3f} mm. Terminales al piso {lower:.3f} mm.')
    print('Cero cruces > 0,025 mm3 entre conjuntos comprobados. Extracción comprobada hasta 140 mm.')


if __name__=='__main__':
    import sys
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else None)
