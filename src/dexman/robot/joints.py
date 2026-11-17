"""Canonical policy action order, independent of physics joint enumeration."""
ARM = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
HAND_DIRECT = tuple("robot0_" + name for name in (
    "WRJ1", "WRJ0", "FFJ3", "FFJ2", "MFJ3", "MFJ2", "RFJ3", "RFJ2",
    "LFJ4", "LFJ3", "LFJ2", "THJ4", "THJ3", "THJ2", "THJ1", "THJ0"))
COUPLED = tuple((f"robot0_{finger}J1", f"robot0_{finger}J0") for finger in ("FF", "MF", "RF", "LF"))
ACTION_NAMES = ARM + HAND_DIRECT + tuple(f"{f}_flexion_sum" for f in ("FF", "MF", "RF", "LF"))
JOINT_NAMES = ARM + HAND_DIRECT + tuple(name for pair in COUPLED for name in pair)
PALM = "robot0_palm"
FINGERTIPS = tuple(f"robot0_{finger}distal" for finger in ("ff", "mf", "rf", "lf", "th"))
HOME = dict(zip(ARM, (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)))
