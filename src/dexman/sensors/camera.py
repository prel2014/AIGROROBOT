"""Fixed external RGB-D camera, with Isaac Lab 3's XYZW quaternion convention."""
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import CuboidCfg, CylinderCfg, PinholeCameraCfg, PreviewSurfaceCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from .view import MVP_POSITION, MVP_ROTATION_XYZW, LEGACY_POSITION, LEGACY_ROTATION_XYZW


def camera_cfg(legacy=False):
    # Oblique view targets (.29,.045,.675); the old overhead view hid the cube under the hand.
    position = LEGACY_POSITION if legacy else MVP_POSITION
    rotation = LEGACY_ROTATION_XYZW if legacy else MVP_ROTATION_XYZW
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation,
                                   convention="opengl"),
        spawn=PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955,
                              clipping_range=(0.05, 4.0)),
        width=224, height=224, data_types=["rgb", "distance_to_image_plane"],
        update_period=1 / 30, renderer_cfg=IsaacRtxRendererCfg(),
    )


def camera_rig_cfg(table_height):
    """Visible, non-colliding housing, lens and stand aligned to the RGB-D sensor."""
    x, y, z, w = MVP_ROTATION_XYZW
    # The OpenGL camera looks along -Z. The casing and lens sit behind its optical center.
    backward = (2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y))
    body_pos = tuple(p + .052 * direction for p, direction in zip(MVP_POSITION, backward))
    lens_pos = tuple(p + .012 * direction for p, direction in zip(MVP_POSITION, backward))
    stand_top = body_pos[2] - .025
    return (
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CameraRig/Body",
            spawn=CuboidCfg(size=(.12, .075, .064),
                            visual_material=PreviewSurfaceCfg(diffuse_color=(.95, .57, .09))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=body_pos, rot=MVP_ROTATION_XYZW),
        ),
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CameraRig/Lens",
            spawn=CylinderCfg(radius=.025, height=.022,
                              visual_material=PreviewSurfaceCfg(diffuse_color=(.025, .05, .08))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=lens_pos, rot=MVP_ROTATION_XYZW),
        ),
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CameraRig/Stand",
            spawn=CylinderCfg(radius=.014, height=stand_top - table_height,
                              visual_material=PreviewSurfaceCfg(diffuse_color=(.19, .22, .26))),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(body_pos[0], body_pos[1], (stand_top + table_height) / 2),
            ),
        ),
    )
