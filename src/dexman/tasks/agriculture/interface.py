"""Shared ordering for simulation, datasets and the bimanual policy."""
from dexman.robot.joints import JOINT_NAMES

ARM_KEYS = ("left_arm", "right_arm")
EYE_KEYS = ("left_eye", "right_eye")
JOINT_ORDER = tuple(f"{side}/{joint}" for side in ARM_KEYS for joint in JOINT_NAMES)
JOINTS_PER_ARM = len(JOINT_NAMES)
ACTION_DIM = len(JOINT_ORDER)
STATE_DIM = 2 * ACTION_DIM
