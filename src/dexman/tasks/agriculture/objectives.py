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
FORBIDDEN_IMPACT_N = 1.0
MAX_DISTRACTORS = 3
ARM_FEATURES = 24  # palm XYZ, five tip XYZ, five contact flags, grasp flag
JOINT_FEATURES = 4  # position, velocity, executed target, queued target
TASK_FEATURES = 39
OBSTACLE_FEATURES = 5
OBS_DIM = (ACTION_DIM * JOINT_FEATURES + 2 * ARM_FEATURES + TASK_FEATURES
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

# The text labels describe physical phases without changing the PPO checkpoint contract.
SUBTASKS_BY_CROP = {
    "citrus": {
        REACH: "Agarra la naranja más cercana.",
        LIFT: "Levanta la naranja 15 cm.",
        PLACE: "Deposita la naranja en la bandeja izquierda.",
        SETTLE: "Mantén la naranja estable en la bandeja.",
    },
    "banana": {
        REACH: "Agarra el plátano con la derecha.",
        LIFT: "Levanta el plátano 10 cm.",
        TRANSFER: "Sujétalo con la izquierda y luego suelta con la derecha.",
        PLACE: "Deposita el plátano en la bandeja izquierda.",
        SETTLE: "Mantén el plátano estable en la bandeja.",
    },
    "vineyard": {
        REACH: "Agarra el racimo evitando la espaldera.",
        LIFT: "Levanta el racimo 15 cm y sostenlo medio segundo.",
    },
    "berries": {
        REACH: "Pinza el arándano con pulgar e índice.",
        LIFT: "Levanta el arándano 3 cm y sostenlo sin mover los demás.",
    },
}
