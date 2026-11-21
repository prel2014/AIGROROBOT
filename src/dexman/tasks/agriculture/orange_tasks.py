"""Órdenes y geometría de los ensayos guiados del naranjal. Distancias en metros."""
import re

import numpy as np


FINISH = "Elévala 15 cm, sostenla medio segundo y déjala en la canasta."
ORDERS = {
    "estacionada": "Con el rover estacionado, recoge la naranja exterior por el costado.",
    "roca": "Avanza por el surco. Si ves obstáculos, esquívalos. Estaciónate y recoge la naranja exterior.",
    "izquierda": "De las dos naranjas, recoge la de la izquierda y deja la otra en la rama.",
    "derecha": "De las dos naranjas, recoge la de la derecha y deja la otra en la rama.",
}
ORDERS = {name: f"{order} {FINISH}" for name, order in ORDERS.items()}
ROCK_SCALE = 2.4
# The convex hull of field_rock.usda fits in this horizontal bounding circle.
ROCK_RADIUS = .10 * ROCK_SCALE
ROVER_HALF_WIDTH = .58
ROVER_HALF_LENGTH = 1.00
MIN_GAP = .045


def subtasks(name):
    approach = {
        "estacionada": ["Mantén el rover estacionado y orienta los ojos hacia la naranja exterior."],
        "roca": ["Recorre el surco y rodea la roca si hay espacio.", "Estaciónate frente a la rama."],
        "izquierda": ["Desde el rover estacionado, localiza la naranja izquierda y conserva la derecha en la rama."],
        "derecha": ["Desde el rover estacionado, localiza la naranja derecha y conserva la izquierda en la rama."],
    }
    return [*approach[name], "Entra de costado y cierra el pulgar frente a dos dedos.",
            "Tira gradualmente hasta desprender la naranja.", "Elévala 15 cm y sostenla medio segundo.",
            "Llévala sobre la canasta con el agarre cerrado.", "Abre la mano, retírala y espera a que la fruta repose."]


def order_task(order):
    def words(text):
        return " ".join(re.findall(r"\w+",text.casefold()))
    for name,full in ORDERS.items():
        variants = [full,full.removesuffix(FINISH)]
        if name in ("izquierda","derecha"):
            variants.extend((f"Recoge la de la {name}",f"Recoge la naranja de la {name}"))
        if any(words(order)==words(v) for v in variants):
            return name
    raise ValueError(f"Orden de naranja desconocida: {order}")


def target_index(order, positions, left_axis):
    """Resolve a Spanish command in the parked rover's coordinate frame."""
    name = order_task(order)
    if len(positions) == 1 and name in ("estacionada","roca"):
        return 0
    if len(positions) != 2 or name not in ("izquierda","derecha"):
        raise ValueError("El número de naranjas físicas no corresponde a la orden.")
    lateral = np.asarray(positions) @ np.asarray(left_axis)
    if abs(lateral[0] - lateral[1]) < .06:
        raise ValueError("Las dos naranjas no se distinguen desde el rover.")
    return int(np.argmax(lateral) if name=="izquierda" else np.argmin(lateral))


def clearance(xy, heading, rock_xy):
    """Conservative signed clearance from the oriented wheel/chassis envelope to the rock."""
    delta = np.asarray(rock_xy) - np.asarray(xy)
    c, s = np.cos(heading), np.sin(heading)
    u = abs(c*delta[..., 0] + s*delta[..., 1]) - ROVER_HALF_LENGTH
    v = abs(-s*delta[..., 0] + c*delta[..., 1]) - ROVER_HALF_WIDTH
    return np.hypot(np.maximum(u, 0), np.maximum(v, 0)) + np.minimum(np.maximum(u, v), 0) - ROCK_RADIUS


def plan_rock(route, rock_xy):
    """Try both shoulders of the row. Return an empty route if neither corridor clears the rock."""
    phase = np.linspace(0., np.pi, len(route))
    delta = np.gradient(route, axis=0)
    normal = np.column_stack((-delta[:, 1], delta[:, 0])) / np.linalg.norm(delta, axis=1)[:, None]
    for offset in (0., -.16, .16, -.26, .26, -.42, .42, -.58, .58, -.72, .72):
        candidate = route + offset * np.sin(phase)[:, None]**2 * normal
        dx, dy = np.gradient(candidate, axis=0).T
        # Leave steering reserve above the measured braking margin.
        if clearance(candidate, np.arctan2(dy, dx), rock_xy).min() >= MIN_GAP+.10:
            return candidate
    return np.empty((0, 2))


def anchor_grasp(path, knot, closed):
    """Continue from the motor targets that achieved opposition, in joint radians."""
    bias = closed-path[knot][0]
    carry = [i for i,(_,phase,_) in enumerate(path) if phase=="traslado"]
    for i in range(knot+1,len(path)):
        q,phase,ticks = path[i]
        if phase in ("apertura","retirada"):
            break
        q = q+bias
        if phase=="traslado":
            # Keep finger preload to release. Return the wrist to the surveyed basket pose.
            progress = (i-carry[0]+1)/len(carry)
            q[:8] -= bias[:8]*progress*progress*(3.-2.*progress)
        path[i] = (q,phase,ticks)
    return bias[:8]
