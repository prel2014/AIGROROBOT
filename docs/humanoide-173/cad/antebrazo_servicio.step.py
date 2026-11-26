"""Antebrazo derecho en posición de servicio, tapa extraída hacia atrás."""
from build123d import Compound, Pos
from interior_xavier import original, normal
from manos_servo import equipped_forearm

def gen_step():
    arm=next(p for p in original().children if 'brazo derecho' in normal(p.label))
    shell=next(p for p in arm.children if 'antebrazo' in normal(p.label))
    module=equipped_forearm(shell,1)
    body,pcb,reg,lid=module.children
    opened=Pos(0,-68,0)*lid
    return Compound(label='Acceso posterior del antebrazo - tapa retirada',children=[body,pcb,reg,opened])
