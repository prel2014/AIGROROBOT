"""Grounded subtask planning for teacher demonstrations, not a learned visual detector."""
from dataclasses import dataclass
import heapq
import math

def intersects_segment(a, b, rectangle):
    """Closed segment against an axis-aligned rectangle (including its boundary)."""
    lo, hi = 0., 1.
    for axis in range(2):
        delta = b[axis] - a[axis]
        low, high = rectangle[axis], rectangle[axis+2]
        if abs(delta) < 1e-12:
            if not low <= a[axis] <= high:
                return False
        else:
            first, last = sorted(((low-a[axis])/delta, (high-a[axis])/delta))
            lo, hi = max(lo, first), min(hi, last)
            if lo > hi:
                return False
    return True


def collision_free_route(start, goal, obstacles, clearance=.08, workspace=(-.95, -.6, .95, .6)):
    """Visibility graph for an inflated transported object/hand envelope in table XY.

    Simulator ground truth supplies boxes to the teacher. A student must learn from
    synchronized camera observations and recorded plans; this does not infer obstacles from RGB.
    """
    rectangles = [(x-sx/2-clearance, y-sy/2-clearance, x+sx/2+clearance, y+sy/2+clearance)
                  for x, y, sx, sy in obstacles]
    def inside(p):
        return workspace[0] <= p[0] <= workspace[2] and workspace[1] <= p[1] <= workspace[3]
    if not inside(start) or not inside(goal):
        raise ValueError("Start/goal outside the configured workspace")
    if any(intersects_segment(p, p, r) for p in (start, goal) for r in rectangles):
        raise ValueError("No safe route: start or goal intersects an inflated obstacle")
    nodes = [tuple(start), tuple(goal)]
    for x0, y0, x1, y1 in rectangles:
        nodes += [(x0-.002,y0-.002),(x0-.002,y1+.002),(x1+.002,y0-.002),(x1+.002,y1+.002)]
    distances, previous, queue = {0:0.}, {}, [(0.,0)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != distances[i]:
            continue
        if i == 1:
            route = [nodes[1]]
            while i:
                i = previous[i]
                route.append(nodes[i])
            return list(reversed(route))
        for j, point in enumerate(nodes):
            if j == i or not inside(point) or any(intersects_segment(nodes[i], point, r) for r in rectangles):
                continue
            candidate = cost + math.dist(nodes[i], point)
            if candidate < distances.get(j, math.inf):
                distances[j], previous[j] = candidate, i
                heapq.heappush(queue, (candidate,j))
    raise ValueError("No collision-free route within the workspace")


@dataclass(frozen=True)
class Subtask:
    name: str
    text: str
    target: tuple | None = None


def pick_place_plan(start, goal, obstacles=(), clearance=.08):
    route = collision_free_route(start[:2], goal[:2], obstacles, clearance)
    height = max(start[2] + .15, goal[2] + .15)
    return [Subtask("select", "Identifica el objeto indicado y no muevas los distractores."),
            Subtask("approach", "Acerca la mano al objeto."),
            Subtask("align", "Alinea el pulgar y los dedos en lados opuestos."),
            Subtask("grasp", "Cierra suavemente y verifica contactos opuestos."),
            Subtask("lift", "Eleva el objeto al menos 15 cm.", (start[0],start[1],height))] + [
            Subtask("transport", "Avanza al siguiente punto libre; si ves obstáculos, esquívalos.", (x,y,height))
            for x,y in route[1:]] + [
            Subtask("lower", "Baja el objeto hasta apoyarlo en la mesa.", tuple(goal)),
            Subtask("release", "Abre la mano sin desplazar el objeto."),
            Subtask("verify", "Comprueba que está estable en el destino.", tuple(goal))]


def lift_plan(start):
    target = (start[0],start[1],start[2]+.15)
    return pick_place_plan(start,start)[:4] + [
        Subtask("lift","Eleva el objeto al menos 15 cm.",target),
        Subtask("hold","Mantén el agarre estable durante medio segundo.",target)]


def push_plan(start, goal, obstacles=(), clearance=.08):
    route = collision_free_route(start[:2],goal[:2],obstacles,clearance)
    return [Subtask("select","Identifica el objeto indicado."),
            Subtask("approach","Acerca la mano sin tocar los demás objetos."),
            Subtask("align_push","Alinea la mano para empujar sobre la mesa.")] + [
            Subtask("slide","Empuja hacia el siguiente punto libre sin levantar el objeto.",(x,y,start[2]))
            for x,y in route[1:]] + [
            Subtask("verify","Comprueba el destino y que el objeto sigue apoyado.",tuple(goal))]
