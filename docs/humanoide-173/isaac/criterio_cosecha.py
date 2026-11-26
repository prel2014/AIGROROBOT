"""Contrato del ensayo de prensión del Xavier. Distancias en metros y tiempo en segundos."""
from dataclasses import dataclass

DEDOS_LARGOS = frozenset(("indice", "medio", "anular", "menique"))


@dataclass
class Sosten:
    continuo: float = 0.
    maximo: float = 0.

    def medir(self, dt, elevando, altura, distancia, contactos, libre, vertical, en_apoyo):
        opuestos = bool({"pulgar", "mano"} & contactos) and bool(DEDOS_LARGOS & contactos)
        valido = elevando and altura >= .035 and distancia < .055 and opuestos and libre and vertical > .96 and en_apoyo
        self.continuo = self.continuo + dt if valido else 0.
        self.maximo = max(self.maximo, self.continuo)
        return self.continuo >= .50 - 1e-9
