"""Surveyed branch anchors and terminal approaches for the four field films, in metres."""
from dataclasses import dataclass
import numpy as np

from .field_layout import FIELDS


@dataclass(frozen=True)
class HarvestSite:
    fruit: str
    anchor: tuple
    reach: float
    drop: float
    local_tip_z: float
    mass: float
    break_n: float
    radius: float
    reference: tuple


# World-space points sampled from Fila2/Planta03 in each authored parcel.
# The extra fruit hangs on a corridor-facing shoot attached to that plant.
SITES = {
    "citrus": HarvestSite("la naranja",(7.9493,.05068,1.39921),.34,.15,.071,.16,4.,.045,(.30,-.12,.9355)),
    "banana": HarvestSite("el plátano",(7.43023,-.14834,2.04996),.34,.37,.086,.12,3.8,.11,(.30,-.115,.9355)),
    # Surveyed in Fila2/Planta03/CurrentSeasonShoots. The outer shoot brings
    # the hanging fruit above the rim and out to the drivable side of the trellis.
    "vineyard": HarvestSite("el racimo de uvas",(4.86516,.21799,1.43228),.48115,.11,.079,.20,4.5,.10,(.252,-.092,1.00349)),
    "berries": HarvestSite("el arándano",(2.55173,-.26025,1.19166),.15,.09,.015,.002,.065,.012,(.319996,-.080029,.934462)),
}

# Terminal points on the fruit collider meshes in their local USD frames.
ATTACHMENT_M = {
    "banana": (.06325,.08812,.01234), "vineyard": (.00682,.00026,.12694),
    "citrus": (.00089,.00045,.07584), "berries": (-.00089,.00012,.01573),
}


@dataclass(frozen=True)
class FieldSituation:
    # Lateral bend at mid-route, branch-tip height offset, dome/sun intensities and sunlight temperature.
    bend_m: float
    branch_height_m: float
    dome: float
    sun: float
    temperature_k: float


SITUATIONS = {
    "base": FieldSituation(0., 0., 400., 1750., 6500.),
    "amanecer": FieldSituation(-.09, .012, 260., 1100., 4100.),
    "tarde": FieldSituation(.10, -.015, 330., 1350., 4900.),
}

FIELD_PHASE_INSTRUCTIONS = {
    "recorrido": "Recorre el surco y esquiva los obstáculos.",
    "estacionamiento": "Frena frente al fruto y orienta los ojos hacia él.",
    "aproximacion": "Acerca la mano al fruto indicado.",
    "cierre": "Rodea el fruto y opón el pulgar a los dedos.",
    "traccion": "Separa el fruto del pedúnculo manteniendo el agarre.",
    "recuperacion": "Estabiliza el fruto después de desprenderlo.",
    "elevacion": "Levanta el fruto quince centímetros.",
    "sosten": "Sostén el fruto inmóvil durante medio segundo.",
    "traslado": "Lleva el fruto sobre la canasta sin soltarlo.",
    "apertura": "Abre los dedos dentro de la boca de la canasta.",
    "retirada": "Retira la mano y deja el fruto en reposo.",
}


def field_instruction(crop, situation):
    fruit = SITES[crop].fruit
    return {
        "base": f"Recorre el surco y estaciónate frente a la planta. Toma {fruit}. Eleva la fruta 15 cm, sostenla medio segundo y déjala en la canasta.",
        "amanecer": f"Sigue la hilera hasta {fruit}. Si ves obstáculos, esquívalos. Frena, recoge el fruto y levántalo quince centímetros. Mantenlo así medio segundo y deposítalo en la canasta.",
        "tarde": f"Avanza por el surco y detente junto a la planta. Sin golpear las ramas, sujeta {fruit}. Desprende el fruto y elévalo 15 cm. Sostén la carga medio segundo y suéltala dentro de la canasta.",
    }[situation]


def arm_occlusion(observations):
    """Fraction blocked by the arm among identified ray hits, and per-eye hit coverage."""
    hits = [[row[i] for row in observations if row[i]!="sin_impacto"] for i in (0,1)]
    blocked = [round(eye.count("brazo")/len(eye),3) if eye else None for eye in hits]
    coverage = [round(len(eye)/len(observations),3) if observations else 0. for eye in hits]
    return blocked,coverage


def binocular_visibility(pixel_rows, depth_rows):
    """Count visible fruit pixels and covered lenses per eye and phase."""
    summary = {}
    for (step, phase, eyes), (depth_step, depth_phase, depth) in zip(pixel_rows, depth_rows, strict=True):
        if (step, phase) != (depth_step, depth_phase):
            raise ValueError("Las mediciones RGB y de profundidad no están sincronizadas")
        phase_rows = summary.setdefault(phase, {"frames": 0, "both_visible": 0, "eyes": [
            {"fruit_visible_frames": 0, "covered_lens_frames": 0, "min_fruit_pixels": None} for _ in range(2)
        ]})
        phase_rows["frames"] += 1
        phase_rows["both_visible"] += int(all(pixels > 0 for pixels, _ in eyes))
        for (pixels, _), label, result in zip(eyes, depth, phase_rows["eyes"], strict=True):
            result["fruit_visible_frames"] += int(pixels > 0)
            result["covered_lens_frames"] += int(label == "visor_cubierto")
            minimum = result["min_fruit_pixels"]
            result["min_fruit_pixels"] = pixels if minimum is None else min(minimum, pixels)
    return summary


def layout(crop):
    field, site = FIELDS[crop], SITES[crop]
    anchor = np.array(site.anchor)
    field_tip = anchor + np.array([0.,-site.reach,-site.drop])
    # Sight the last section from the centre of the actual row toward the fruit.
    # The controller can roll into the final pose without a stationary skid turn.
    run_up = 3.30
    total = run_up+.81
    heading = .35
    for _ in range(5):
        entry_x = field_tip[0]-total*np.cos(heading)
        heading = np.arcsin(np.clip((field_tip[1]-field.row_y(entry_x,-.5))/total,-.85,.85))
    basis = np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
    local_xy = np.array([.13,-.015])
    offset = field_tip[:2] - basis @ local_xy
    park = np.array([-.68,0.])
    park_field = offset + basis @ park
    ground = field.height(*park_field)
    fruit = np.r_[local_xy,field_tip[2]-ground-site.local_tip_z]
    local_anchor = np.r_[basis.T @ (anchor[:2]-offset),anchor[2]-ground]
    x = park_field[0]-run_up*np.cos(heading)
    start = np.array([x, field.row_y(x,-.5)])
    # Use a shallow arc from the lane to the plant so that all four tyres roll uphill.
    chord = park_field-start
    controls = np.vstack((start,start+.33*chord-[0.,.08],park_field-.33*chord,park_field))
    t = np.linspace(0,1,55)[:,None]
    field_path = (1-t)**3*controls[0]+3*t*(1-t)**2*controls[1]+3*t*t*(1-t)*controls[2]+t**3*controls[3]
    route = (field_path-offset) @ basis
    if crop=="berries":
        # At the old terminal pose the front-left tyre climbed a 36 cm planting bed.
        # Keep all four wheels in the aisle; the left hand can reach across its edge.
        progress = t*t*(3.-2.*t)
        route += progress*np.array([-.06,-.48])
    return dict(fruit=fruit,anchor=local_anchor,route=route,ground=ground,
                start_z=field.height(*start)-ground+.06,rotation=-np.degrees(heading),translation=np.r_[-basis.T@offset,-ground])
