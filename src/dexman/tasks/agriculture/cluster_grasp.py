"""Spatial force closure and weight support for the fingers around a hanging cluster."""
import numpy as np
from scipy.optimize import linprog


def cluster_grip(forces_on_fingers, points_from_com, mass_kg, friction):
    """Test the six-dimensional grasp wrench cone and the measured normal-force budget."""
    forces = np.asarray(forces_on_fingers,dtype=np.float64)
    points = np.asarray(points_from_com,dtype=np.float64)
    loads = np.linalg.norm(forces,axis=1)
    active = (loads>=.04) & np.isfinite(points).all(axis=1) & np.isfinite(loads)
    if not active[-1] or active[:-1].sum()<2:
        return False
    normals = -forces[active]/loads[active,None]
    arms = points[active]
    if np.max(np.linalg.norm(arms[:-1]-arms[-1],axis=1))<.015:
        return False
    # Eight edges form an inscribed polygon in each static Coulomb cone.
    axes = np.eye(3)[np.argmin(abs(normals),axis=1)]
    tangent = np.cross(normals,axes)
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    angles = np.arange(8)*(2*np.pi/8)
    directions = normals[:,None,:]+friction*(tangent[:,None,:]*np.cos(angles)[None,:,None]
        +np.cross(normals,tangent)[:,None,:]*np.sin(angles)[None,:,None])
    # Scale moments by a common length for numerical conditioning.
    moments = np.cross(arms[:,None,:],directions)/max(.01,np.linalg.norm(arms,axis=1).max())
    wrench = np.concatenate((directions,moments),axis=2).reshape(-1,6).T
    if np.linalg.matrix_rank(wrench,tol=1.e-7)<6:
        return False
    edges = wrench.shape[1]
    # A full-rank positive span requires a strictly positive self-equilibrating
    # combination. It can always be scaled so that every coefficient is >= 1.
    span = linprog(np.ones(edges),A_eq=wrench,b_eq=np.zeros(6),bounds=(1.,None),method="highs")
    if not span.success:
        return False
    # Check that countering gravity at the actual COM requires no more normal
    # effort than each loaded fingertip exerted during this PhysX substep.
    budgets = np.repeat(np.eye(len(normals)),8,axis=1)
    gravity = np.array([0.,0.,mass_kg*9.81,0.,0.,0.])
    support = linprog(np.ones(edges),A_ub=budgets,b_ub=loads[active],A_eq=wrench,b_eq=gravity,
                      bounds=(0.,None),method="highs")
    return support.success
