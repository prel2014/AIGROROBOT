"""Banco abdominal en posición de inspección, con la trampilla levantada 40 mm."""
from build123d import Compound, Pos
from interior_xavier import bank


def gen_step():
    parts=list(bank().children)
    top=parts[-1]
    parts[-1]=Pos(0,0,40)*top
    parts[-1].label=top.label
    return Compound(label='Cartucho auxiliar 4S9P - vista de servicio',children=parts)
