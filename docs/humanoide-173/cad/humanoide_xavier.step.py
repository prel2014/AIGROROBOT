"""Humanoide Xavier con ambas manos de cinco dedos y accionamiento local."""
from build123d import Axis, Box, Compound, Cylinder, Pos, Rot, fillet, import_step
from interior_xavier import CAD, bank, battery_lid, cpu_lid, electronics, find_torso, modified_torso, named, normal, original
from manos_servo import arm_with_hand
from arnes_brazos import harness
from cartucho_auxiliar import robot_loom
from tren_inferior import Leg
from equipo_rodilla import pelvis_with_ports
from equipo_brazos import elbow_motor, shaft_pilots, traction_branch


def neck_servo():
    # Simplified body and axles at the shaft and plate datums from the bench STEP.
    plate=import_step(CAD/'soporte_st3215.step')
    plate.label='Soporte cervical de banco'
    raw=Box(45.22,24.72,31)
    case=Pos(12.5,0,19.7)*fillet(raw.edges().filter_by(Axis.Z),1.6)
    rear=Pos(0,0,2.9)*Cylinder(9.6,2.6)
    shaft=Pos(0,0,37.3)*Cylinder(4,4.2)
    body=named(case.fuse(rear,shaft),'ST3215 - envolvente y eje','#374348')
    return Compound(label='ST3215 y soporte',children=[plate,body])


def gen_step():
    robot=original()
    torso=find_torso(robot)
    parts=[modified_torso(torso,with_servo=True),electronics(),bank(),battery_lid(),cpu_lid()]
    pelvic_harness=pelvis_with_ports(parts[0],[Leg(-1),Leg(1)])
    shaft_pilots(parts[0])
    for item in robot.children:
        if item is torso:
            continue
        # Replace the exposed decorative hub covers by the actuators seen in the film.
        for group in item.children:
            if normal(group.label) in ('codo','rodilla'):
                group.children=tuple(p for p in group.children if normal(p.label) not in
                                     ('bisel exterior','anillo del eje','cabeza del pasador'))
        if normal(item.label).startswith('brazo'):
            item=arm_with_hand(item,1 if 'derecho' in normal(item.label) else -1)
        elif normal(item.label).startswith('pierna'):
            item=Leg(1 if 'derecha' in normal(item.label) else -1).assemble(item)
        parts.append(item)
    for joint in ('rodilla','codo'):
        for side in (-1,1):
            m=Leg(side).motor() if joint=='rodilla' else elbow_motor(side)
            m.label=f'AK60-39 {joint} '+('I' if side<0 else 'D')
            parts.append(m)
    servo=Pos(0,0,1492.6)*Rot(0,0,90)*neck_servo()
    servo.label='Placa y servo cervical ST3215'
    parts.append(servo)
    parts.append(harness(robot))
    parts.append(robot_loom())
    parts.append(pelvic_harness)
    parts.append(Compound(label='Tracción de los codos - CAN0 y 48 V',
                          children=[traction_branch(s) for s in (-1,1)]))
    robot=Compound(label='Humanoide 1730 - Jetson Xavier y cartucho 18650',children=parts)
    return {"shape":robot,"params":"humanoide_xavier.params.js"}

