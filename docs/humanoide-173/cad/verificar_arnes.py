"""Comprueba cables instalados en el STEP exportado, postura neutra."""
from itertools import combinations
from build123d import Compound, import_step
from interior_xavier import CAD,normal
from verificar_instalacion import overlap
from instalacion_control import box

def bare(node):
    return Compound.cast(node.wrapped)

def pairs(a,b,msg,issues):
    v=overlap(bare(a),bare(b))
    if v>.04: issues.append((msg,round(v,2)))

def verify(robot):
    loom=next(p for p in robot.children if 'cableado' in normal(p.label))
    assert 'cableado' in normal(loom.label)
    issues=[]
    for module in loom.children:
        for p in module.children:
            raw=bare(p)
            assert len(raw.solids())==1 and raw.is_valid and raw.volume>0,p.label
    for idx,s in [(1,'izquierdo'),(2,'derecho')]:
        branch=loom.children[idx]
        arm=next(a for a in robot.children if normal(a.label)=='brazo '+s)
        relevant=[*robot.children[0].children,*robot.children[1].children,*arm.children[0].children,*arm.children[1:]]
        relevant.append(next(a for a in robot.children if 'ak60-39 codo '+s[0] in normal(a.label)))
        for item in branch.children:
            for piece in relevant:
                pairs(item,piece,s+'/'+item.label+'/'+piece.label,issues)
        allc=branch.children
        for i,j in combinations(range(len(allc)),2):
            # Adjacent sweeps of one harness meet at a rounded junction.
            # Exposed connector faces may contact their own wire ends.
            if (i,j) in [(2,3),(6,7),(7,8),(7,9),(8,9),(3,10)]:continue
            pairs(allc[i],allc[j],s+'/'+allc[i].label+'/'+allc[j].label,issues)
    for item in loom.children[0].children:
        for piece in [*robot.children[0].children,*robot.children[1].children,robot.children[4]]:
            pairs(item,piece,'torso/'+item.label+'/'+piece.label,issues)
    dist=loom.children[0].children
    for i,j in combinations(range(len(dist)),2):
        if (i,j) in [(2,4),(6,8),(2,9),(6,9)]: continue  # splices of the CAN trunk
        pairs(dist[i],dist[j],'derivaciones/'+dist[i].label+'/'+dist[j].label,issues)
    # The posterior flange moves away from the shelf; the equipment forward of
    # it travels 150 mm through this conservative swept envelope after unplugging.
    swept=box(0,-82,1270.5,155,252,115)
    for i in (0,1,2,5,6,9):
        pairs(dist[i],swept,'retirada dorsal 150 mm/'+dist[i].label,issues)
    for msg,vol in issues: print(f'{vol:8.2f} mm3 {msg}')
    print('Interferencias',len(issues),flush=True)
    if issues: raise SystemExit(1)

if __name__=='__main__': verify(import_step(CAD/'humanoide_xavier.step'))

