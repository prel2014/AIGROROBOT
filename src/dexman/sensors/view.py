"""Versioned fixed views, independent of the simulator."""
import math

MVP_POSITION = (.75, .45, .95)
MVP_ROTATION_XYZW = (.22379237413406372, .49520522356033325, .7649690508842468, .34570378065109253)
LEGACY_POSITION = (.25, 0., 1.8)
LEGACY_ROTATION_XYZW = (0., 0., -math.sqrt(.5), math.sqrt(.5))


def camera_signature(cfg):
    return {"position": list(cfg.offset.pos), "rotation_xyzw": list(cfg.offset.rot),
            "convention": cfg.offset.convention, "width": cfg.width, "height": cfg.height,
            "focal_length": cfg.spawn.focal_length, "horizontal_aperture": cfg.spawn.horizontal_aperture}
