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
    "vineyard": HarvestSite("el racimo de uvas",(4.87463,-.02316,1.25749),.24,.11,.079,.20,4.5,.10,(.252,-.092,1.00349)),
    "berries": HarvestSite("el arándano",(2.55173,-.26025,1.19166),.15,.09,.015,.002,.065,.012,(.319996,-.080029,.934462)),
}

# Terminal points on the fruit collider meshes in their local USD frames.
ATTACHMENT_M = {
    "banana": (.06325,.08812,.01234), "vineyard": (.00682,.00026,.12694),
    "citrus": (.00089,.00045,.07584), "berries": (-.00089,.00012,.01573),
}


def arm_occlusion(observations):
    """Fraction blocked by the arm among identified ray hits, and per-eye hit coverage."""
    hits = [[row[i] for row in observations if row[i]!="sin_impacto"] for i in (0,1)]
    blocked = [round(eye.count("brazo")/len(eye),3) if eye else None for eye in hits]
    coverage = [round(len(eye)/len(observations),3) if observations else 0. for eye in hits]
    return blocked,coverage


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
        # The low outer tyre drops into the rut of the blueberry bed. Start
        # uphill of the mark and taper the correction to zero at the bush.
        route[:,1] += .20*(1-t[:,0]**2)
        start = basis@route[0]+offset
    return dict(fruit=fruit,anchor=local_anchor,route=route,ground=ground,
                start_z=field.height(*start)-ground+.06,rotation=-np.degrees(heading),translation=np.r_[-basis.T@offset,-ground])
