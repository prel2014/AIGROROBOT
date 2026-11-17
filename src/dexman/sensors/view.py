"""Versioned fixed views, independent of the simulator."""
import math

MVP_POSITION = (.75, .45, .95)
MVP_ROTATION_XYZW = (.22379237413406372, .49520522356033325, .7649690508842468, .34570378065109253)
LEGACY_POSITION = (.25, 0., 1.8)
LEGACY_ROTATION_XYZW = (0., 0., -math.sqrt(.5), math.sqrt(.5))

SECONDARY_POSITION = (.55, -.55, 1.05)


def look_at_xyzw(position, target):
    """OpenGL optical frame: -Z forward, +Y up, XYZW quaternion."""
    import numpy as np
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.stack((right, up, -forward), axis=1)
    # Symmetric eigenvalue formulation avoids trace singularities.
    m = matrix
    k = np.array([[m[0,0]-m[1,1]-m[2,2], m[1,0]+m[0,1], m[2,0]+m[0,2], m[2,1]-m[1,2]],
                  [m[1,0]+m[0,1], m[1,1]-m[0,0]-m[2,2], m[2,1]+m[1,2], m[0,2]-m[2,0]],
                  [m[2,0]+m[0,2], m[2,1]+m[1,2], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], np.trace(m)]]) / 3
    values, vectors = np.linalg.eigh(k)
    q = vectors[:, values.argmax()]
    return tuple(float(x) for x in (q if q[3] >= 0 else -q))


SECONDARY_ROTATION_XYZW = look_at_xyzw(SECONDARY_POSITION, (.29, .045, .72))


def camera_signature(cfg):
    return {"position": list(cfg.offset.pos), "rotation_xyzw": list(cfg.offset.rot),
            "convention": cfg.offset.convention, "width": cfg.width, "height": cfg.height,
            "focal_length": cfg.spawn.focal_length, "horizontal_aperture": cfg.spawn.horizontal_aperture}
