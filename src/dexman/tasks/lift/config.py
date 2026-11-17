"""Task rates, episode horizon, and conservative GPU defaults."""
import json
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from dexman.sensors.camera import camera_cfg, camera_rig_cfg
from .scene import LiftSceneCfg
from dexman.paths import ROOT
from dexman.robot.joints import ARM, FINGERTIPS, JOINT_NAMES

_GRASP_SEED = json.loads((ROOT / "configs/grasp_seed.json").read_text())


@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = 4
    episode_length_s = 10.0
    action_space = 26
    observation_space = 95
    state_space = 0
    sim = SimulationCfg(dt=1 / 120, render_interval=4, physics=PhysxCfg())
    scene = LiftSceneCfg(num_envs=32, env_spacing=3.5, replicate_physics=True)
    table_height = 0.60
    object_initial_height = 0.625
    lift_height = 0.15
    success_hold_steps = 15
    action_smoothing = 0.25
    instruction = "Levanta el cubo rojo al menos 15 cm y sostenlo medio segundo. Si ves obstáculos, esquívalos."
    reward_version = 1
    object_xy_range = 0.06
    arm_half_range = 1.2
    gravity_compensation = False
    action_center = None
    arm_center = None


@configclass
class VisualSceneCfg(LiftSceneCfg):
    camera = camera_cfg()


@configclass
class ContactSceneCfg(LiftSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True
        for fingertip in FINGERTIPS:
            setattr(self, fingertip, ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Hand/" + fingertip,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                track_contact_points=True, track_friction_forces=True,
                update_period=0.0,
            ))


@configclass
class CorrectedVisualSceneCfg(ContactSceneCfg):
    camera = camera_cfg()
    camera_secondary = camera_cfg(secondary=True)


@configclass
class VisualLiftEnvCfg(LiftEnvCfg):
    # Same state policy; camera outputs are collected separately, never fed to PPO implicitly.
    scene = VisualSceneCfg(num_envs=1, env_spacing=3.5, replicate_physics=True)
    # Refresh RGB after Same-Step autoreset before the next policy observation.
    num_rerenders_on_reset = 1


@configclass
class CorrectedLiftEnvCfg(LiftEnvCfg):
    reward_version = 4
    scene = ContactSceneCfg(num_envs=32, env_spacing=3.5, replicate_physics=True)
    minimum_contact_force_n = .05
    asset_shape = "box"
    asset_dimensions = (.05, .05, .05)
    scenario_name = "red_cube"
    success_criterion = "opposed_contact_minimum_lift_and_hold_v2"
    success_lift_overshoot_m = .025
    action_smoothing = 0.05
    object_xy_range = 0.02
    arm_half_range = 0.8
    arm_center = tuple(_GRASP_SEED["arm_q"])
    action_center = tuple(_GRASP_SEED["joint_center_q"])
    gravity_compensation = True
    curriculum_enabled = True
    curriculum_heights = (0.015, 0.03, 0.06, 0.10, 0.15)
    curriculum_initial_level = 0
    curriculum_window = 64
    curriculum_success_threshold = 0.6

    def __post_init__(self):
        self.scene.robot.init_state.joint_pos = {**dict(zip(ARM, self.arm_center)), **dict.fromkeys(JOINT_NAMES[6:], 0.0),
            "robot0_WRJ1": self.action_center[6], "robot0_WRJ0": self.action_center[7]}
        self.scene.object.init_state.pos = tuple(_GRASP_SEED["cube_local"])

    def play_mode(self):
        self.curriculum_enabled = False
        self.curriculum_initial_level = len(self.curriculum_heights) - 1


@configclass
class TeacherCameraEnvCfg(CorrectedLiftEnvCfg):
    scene = CorrectedVisualSceneCfg(num_envs=32, env_spacing=3.5, replicate_physics=True)
    num_rerenders_on_reset = 1

    def __post_init__(self):
        super().__post_init__()
        self.scene.camera_body, self.scene.camera_lens, self.scene.camera_stand = camera_rig_cfg(self.table_height)
        self.scene.camera_secondary_body, self.scene.camera_secondary_lens, self.scene.camera_secondary_stand = camera_rig_cfg(self.table_height, secondary=True)


@configclass
class CorrectedVisualLiftEnvCfg(TeacherCameraEnvCfg):
    scene = CorrectedVisualSceneCfg(num_envs=1, env_spacing=3.5, replicate_physics=True)
    num_rerenders_on_reset = 1
    # Evaluation and demonstrations always require the final 15 cm objective.
    curriculum_enabled = False
    curriculum_initial_level = 4


@configclass
class JointAngleVisualLiftEnvCfg(CorrectedVisualLiftEnvCfg):
    """Visual policy receives RGB/text externally and q/dq here; outputs 30 angles in rad."""
    action_space = 30
    observation_space = 60
