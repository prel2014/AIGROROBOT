"""Reproducible, voluntary close and lift around a physically verified grasp."""
from dataclasses import dataclass
import json
import math

from .interface import ACTION_DIM, JOINTS_PER_ARM
from .objectives import OBJECTIVES


GRASP_ACTION_DIM = 2
GRASP_EPISODE_SECONDS = 6
CLOSING_SECONDS = 52 / 30
LIFTING_SECONDS = 48 / 30


def _vector(values, size):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"Se esperaban {size} valores en la calibración")
    vector = tuple(float(value) for value in values)
    if not all(map(math.isfinite, vector)):
        raise ValueError("La calibración contiene valores no finitos")
    return vector


@dataclass(frozen=True)
class GraspCalibration:
    crop: str
    side: int
    start: str
    pregrasp: tuple[float, ...]
    closed: tuple[float, ...]
    lift: tuple[float, ...]
    lift_waypoints: tuple[tuple[float, ...], ...]
    initial_joints: tuple[float, ...]
    initial_fruit_pose: tuple[float, ...]


def read_grasp_calibration(path, crop, start):
    """Accept a physically passing trial, not an optimizer-only trajectory."""
    if OBJECTIVES[crop].deposit:
        raise ValueError("El currículo de agarre directo admite cultivos cuya tarea termina al sostener la fruta")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("raised_support") is not True:
        raise ValueError("Se requiere un informe físico obtenido con --raised-support")
    group = data["plans"][crop]
    trials = [trial for trial in data["crops"][crop]["trials"] if trial.get("passed") is True]
    if not trials:
        raise ValueError(f"No hay un agarre físicamente validado para {crop}")
    trial = max(trials, key=lambda row: (row["hold_s"], row["peak_lift_m"]))
    plan = group["candidates"][trial["candidate"]]
    phase = "closed" if start == "closed" else "pregrasp"
    snapshot = trial["poses"][phase]
    side = group["side"]
    if side not in (0, 1):
        raise ValueError("La calibración exige un lado válido")
    # Reset in the requested posture. The closed start uses measured angles so
    # the first position command restores calibrated drive pressure in PhysX.
    offset = side * JOINTS_PER_ARM
    measured = _vector(snapshot["joint_position_rad"], ACTION_DIM)[offset:offset + JOINTS_PER_ARM]
    requested = _vector(plan["pregrasp"], JOINTS_PER_ARM) if start == "pregrasp" else measured
    low = _vector(group["calibration"]["low"], JOINTS_PER_ARM)
    high = _vector(group["calibration"]["high"], JOINTS_PER_ARM)
    # PhysX measurements can cross a hard joint limit by a floating-point ulp.
    initial = tuple(min(max(value, lower + 1.e-6), upper - 1.e-6) if upper - lower > 2.e-6
                    else (lower + upper) / 2 for value, lower, upper in zip(requested, low, high, strict=True))
    pregrasp, closed, lift = (_vector(plan[key], JOINTS_PER_ARM) for key in ("pregrasp", "closed", "lift"))
    waypoints = tuple(_vector(row, JOINTS_PER_ARM) for row in plan.get("lift_waypoints", ()))
    if waypoints and (len(waypoints) < 2 or waypoints[-1] != lift):
        raise ValueError("La trayectoria cartesiana debe incluir el objetivo final de elevación")
    return GraspCalibration(crop, side, start, pregrasp, closed, lift, waypoints or (closed, lift),
                            initial, _vector(snapshot["fruit_pose"], 7))


class GraspMotion:
    """Convert two requested rates into a bounded pose target for the real drives."""

    def __init__(self, calibration, home, dt):
        self.side = slice(calibration.side * JOINTS_PER_ARM, (calibration.side + 1) * JOINTS_PER_ARM)
        self.home = home
        self.open_q = home.new_tensor(calibration.pregrasp)
        self.closed_q = home.new_tensor(calibration.closed)
        self.lift_path = home.new_tensor(calibration.lift_waypoints)
        self.rates = home.new_tensor((dt / CLOSING_SECONDS, dt / LIFTING_SECONDS))
        self.start_closure = float(calibration.start == "closed")
        self.progress = home.new_zeros((len(home), GRASP_ACTION_DIM))
        self.progress[:, 0] = self.start_closure

    def reset(self, ids):
        self.progress[ids] = 0.
        self.progress[ids, 0] = self.start_closure

    def target(self, sampled_actions, may_lift):
        import torch

        invalid = ~torch.isfinite(sampled_actions).all(-1)
        actions = torch.where(invalid[:, None], 0., sampled_actions).clamp(-1., 1.)
        change = actions * self.rates
        change[:, 1] *= may_lift | (change[:, 1] < 0.)
        self.progress.copy_((self.progress + change).clamp(0., 1.))
        target = self.home.clone()
        close, lift = self.progress.split(1, dim=-1)
        position = lift[:, 0] * (len(self.lift_path) - 1)
        index = position.long().clamp_max(len(self.lift_path) - 2)
        raised = self.lift_path[index].lerp(self.lift_path[index + 1], (position - index)[:, None])
        lift_delta = torch.where(lift > 0., raised - self.closed_q, 0.)
        target[:, self.side] = self.open_q.lerp(self.closed_q, close) + lift_delta
        return target, actions, invalid
