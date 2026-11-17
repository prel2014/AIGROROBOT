"""Register tasks without constructing an environment or starting simulation."""
import gymnasium as gym

for task_id, config in [("Dexman-UR10e-Shadow-Lift-v0", "LiftEnvCfg"),
                        ("Dexman-UR10e-Shadow-Lift-RGBD-v0", "VisualLiftEnvCfg")]:
    gym.register(id=task_id, entry_point="dexman.tasks.lift.environment:LiftEnv",
        disable_env_checker=True, kwargs={
            "env_cfg_entry_point": f"dexman.tasks.lift.config:{config}",
            "rsl_rl_cfg_entry_point": "dexman.agents.ppo:LiftPPORunnerCfg",
            "default_agent": "rsl_rl",
        })

for task_id, config in [("Dexman-UR10e-Shadow-Lift-v1", "CorrectedLiftEnvCfg"),
                        ("Dexman-UR10e-Shadow-Lift-RGBD-v1", "CorrectedVisualLiftEnvCfg")]:
    gym.register(id=task_id, entry_point="dexman.tasks.lift.environment:LiftEnv",
        disable_env_checker=True, kwargs={
            "env_cfg_entry_point": f"dexman.tasks.lift.config:{config}",
            "rsl_rl_cfg_entry_point": "dexman.agents.ppo:CorrectedLiftPPORunnerCfg",
            "default_agent": "rsl_rl"})

gym.register(id="Dexman-UR10e-Shadow-Lift-VLA-v2",
    entry_point="dexman.tasks.lift.visual_policy_environment:JointAngleLiftEnv",
    disable_env_checker=True, kwargs={
        "env_cfg_entry_point": "dexman.tasks.lift.config:JointAngleVisualLiftEnvCfg"})
