"""Isaac Sim scene configuration for language-conditioned tasks."""
from isaaclab.utils import configclass
from dexman.tasks.lift.config import JointAngleVisualLiftEnvCfg
from dexman.robot.joints import JOINT_NAMES
from .catalog import TASKS


@configclass
class MemoryTaskEnvCfg(JointAngleVisualLiftEnvCfg):
    observation_space = len(JOINT_NAMES)
    # q, dq, last command; four xyz vectors; task one-hot; violation and time left.
    state_space = 3 * len(JOINT_NAMES) + 12 + len(TASKS) + 2
    # Reach a specified objective within 10 s; timeout really ends this task.
    is_finite_horizon = True
    instruction_override = None
    curriculum_enabled = False
