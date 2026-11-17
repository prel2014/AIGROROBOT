"""Instruction and goal definitions, independent of simulator and learning backend."""
from dataclasses import asdict, dataclass
import json
import unicodedata


@dataclass(frozen=True)
class Task:
    name: str
    offset: tuple[float, float, float]
    instructions: tuple[str, ...]


# Directions refer to the robot base, never to an unspecified camera perspective.
TASKS = (
    Task("lift", (0., 0., .15), (
        "Lift the red cube 15 cm and hold it.", "Raise the red cube and keep it above the table.",
        "Levanta el cubo rojo 15 cm y sostenlo.")),
    Task("push_x_positive", (.10, 0., 0.), (
        "Push the red cube 10 cm along positive X of the robot base.",
        "Empuja el cubo rojo 10 cm hacia X positivo de la base del robot.")),
    Task("push_x_negative", (-.10, 0., 0.), (
        "Push the red cube 10 cm along negative X of the robot base.",
        "Empuja el cubo rojo 10 cm hacia X negativo de la base del robot.")),
)


def catalog():
    """JSON-compatible task semantics stored with every checkpoint."""
    return json.loads(json.dumps([asdict(task) for task in TASKS]))


def resolve_instruction(text):
    def normalize(value):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().rstrip(".!?").split())
    if not isinstance(text, str) or not text.strip():
        raise ValueError("An instruction is required")
    for index, task in enumerate(TASKS):
        if any(normalize(text) == normalize(candidate) for candidate in task.instructions):
            return index
    raise ValueError("Unsupported instruction. Choose a phrase from the task catalog; arbitrary tasks are not trained.")
