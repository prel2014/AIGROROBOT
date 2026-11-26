"""Vista dorsal en servicio. Se retiran tapas, manos y piel del torso."""
from build123d import Compound
from arnes_brazos import harness, upper_with_pilots
from equipo_brazos import elbow_mount, elbow_motor, traction_branch, upper_panels
from interior_xavier import original, electronics, normal
from manos_servo import equipped_forearm

def gen_step():
    robot=original()
    items=[electronics()]
    for side,word in [(-1,'izquierdo'),(1,'derecho')]:
        arm=next(a for a in robot.children if 'brazo '+word in normal(a.label))
        fore=next(p for p in arm.children if normal(p.label)=='antebrazo')
        module=equipped_forearm(fore,side)
        module.children=tuple(module.children[:-1])
        parts=[module]
        for group in arm.children:
            label=normal(group.label)
            if label=='brazo':
                upper=upper_panels(side,upper_with_pilots(group,side))
                upper.children=upper.children[:-2]
                parts.append(upper)
            elif label in ('hombro','codo'):
                if label=='codo':group=elbow_mount(group,side)
                parts.append(group)
        motor=elbow_motor(side)
        mounting=motor.location
        case=motor.children[0]
        open_case=Compound(label=case.label,children=[p for i,p in enumerate(case.children) if i!=2])
        opened=mounting*Compound(label=motor.label,children=[open_case,*motor.children[1:]])
        parts.extend([opened,traction_branch(side)])
        items.append(Compound(label='Brazo '+word+' en servicio',children=parts))
    return Compound(label='Servicio de brazos - CAN0, tracción, CAN1 y alimentación de manos',
                    children=[*items,harness(robot)])

