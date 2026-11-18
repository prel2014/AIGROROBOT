"""Task definitions and observation dimensions shared with the PPO checkpoint."""
from dataclasses import dataclass

from .interface import ACTION_DIM
from .randomization import OBSTACLE_RADIUS, TABLE_Z

REACH, LIFT, TRANSFER, PLACE, SETTLE = range(5)
PHASE_TEXT = ("Acércate y agarra la fruta indicada.", "Levanta la fruta sin dejarla caer.",
              "Sujétala con la izquierda antes de soltarla con la derecha.",
              "Lleva la fruta a la bandeja izquierda y suéltala.",
              "Mantén la fruta estable dentro de la bandeja.")
TRAY_CENTER = (.24, .46, TABLE_Z + .017)
EPISODE_SECONDS = 12
MAX_DISTRACTORS = 3
ARM_FEATURES = 24  # palm XYZ, five tip XYZ, five contact flags, grasp flag
TASK_FEATURES = 37
OBSTACLE_FEATURES = 5
OBS_DIM = (ACTION_DIM * 3 + 2 * ARM_FEATURES + TASK_FEATURES
           + len(OBSTACLE_RADIUS) * OBSTACLE_FEATURES)


@dataclass(frozen=True)
class Objective:
    instruction: str
    lift: float
    deposit: bool = False
    handover: bool = False
    precision: bool = False


OBJECTIVES = {
    "citrus": Objective("Recoge la naranja más cercana al torso y colócala en la bandeja izquierda.", .15, deposit=True),
    "banana": Objective("Recoge el plátano más cercano con la derecha, pásalo a la izquierda y colócalo en la bandeja izquierda.",
                        .10, deposit=True, handover=True),
    "vineyard": Objective("Levanta 15 cm el racimo de uvas más cercano al torso y sostenlo. Evita la espaldera.", .15),
    "berries": Objective("Recoge el arándano más cercano al torso con pulgar e índice sin desplazar los demás.", .03,
                         precision=True),
}
