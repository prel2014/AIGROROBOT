"""Instruction and goal definitions, independent of simulator and learning backend."""
from dataclasses import asdict, dataclass
import json
import unicodedata


@dataclass(frozen=True)
class Task:
    name: str
    offset: tuple[float, float, float]
    instructions: tuple[str, ...]
    kind: str = "lift"


# Directions refer to the robot base, never to an unspecified camera perspective.
TASKS = tuple(
    Task(f"lift_{cm}cm", (0., 0., cm/100), (
        f"Levanta el objeto {cm} cm y sostenlo medio segundo.",
        f"Agarra el objeto y elévalo al menos {cm} centímetros.",
        f"Lift the object {cm} cm and hold it for half a second."))
    for cm in (5, 10, 15)
) + tuple(
    Task(f"push_{axis}_{sign}", (distance if axis=="x" else 0., distance if axis=="y" else 0., 0.), (
        f"Empuja el objeto 10 cm hacia {label}, sin levantarlo.",
        f"Desliza el objeto diez centímetros hacia {label} sobre la mesa.",
        f"Push the object 10 cm along {sign} {axis.upper()} of the robot base."), "push")
    for axis, sign, distance, label in (("x","positive",.1,"X positivo"),("x","negative",-.1,"X negativo"),
        ("y","positive",.1,"Y positivo"),("y","negative",-.1,"Y negativo"))
) + tuple(
    Task(f"place_x_{sign}", (distance, 0., .15), (
        f"Levanta el objeto 15 cm y desplázalo 10 cm hacia {label} sin soltarlo.",
        f"Transporta el objeto en el aire hacia {label}, manteniéndolo elevado 15 cm.",
        f"Lift the object 15 cm and move it 10 cm along {sign} X, keeping hold."), "transport")
    for sign, distance, label in (("positive",.1,"X positivo"),("negative",-.1,"X negativo"))
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
