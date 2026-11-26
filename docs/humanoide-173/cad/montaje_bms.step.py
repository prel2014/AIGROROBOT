"""Detalle de revisión a escala real del piso de servicio, PCB y conectores acoplados."""
from build123d import Compound
import cartucho_auxiliar as aux
from instalacion_control import name

def gen_step():
    deck=name(aux.service_deck(),"Piso del cartucho y cuatro apoyos","#9AAEAD")
    return Compound(label="Montaje del BMS en el piso de servicio",children=[
        deck,aux.protector(),*aux.bms_plugs(),*aux.fuse_parts(),aux.pack_port(),
        *aux.thermal(),*aux.pack_wires()])
