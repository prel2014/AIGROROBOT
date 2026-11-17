"""Pass/fail physical grasp test for the corrected task; no learned policy or object attachment."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="captures/corrected_control_verification.json")
    add_launcher_args(parser)
    args = parser.parse_args()
    from dexman.tasks.lift.config import CorrectedLiftEnvCfg
    from dexman.paths import ROOT
    cfg = CorrectedLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.object_xy_range = 0.
    cfg.curriculum_enabled = False
    cfg.curriculum_initial_level = 4
    seed = json.loads((ROOT / "configs/grasp_seed.json").read_text())
    with launch_simulation(cfg, args):
        import torch
        import gymnasium as gym
        import dexman.tasks  # noqa: F401
        from dexman.robot.control import encode
        env = gym.make("Dexman-UR10e-Shadow-Lift-v1", cfg=cfg)
        report = {"object_attached": False, "policy": "scripted physical verification", "success": False}
        try:
            with torch.inference_mode():
                env.reset()
                raw = env.unwrapped
                close_q = raw.action_center[None].clone()
                start_q = raw.targets.clone()
                lift_arm = torch.tensor(seed["lift_arm_q"], device=raw.device)
                max_lift = 0.
                for step in range(290):
                    q = start_q.lerp(close_q, min(max((step-20)/60, 0), 1))
                    if step >= 120:
                        q[:, :6] = close_q[:, :6].lerp(lift_arm[None], min((step-120)/90, 1))
                    _, _, _, _, extras = env.step(encode(q, raw.limits, raw.action_center))
                    height = float((raw.object.data.root_pos_w.torch[0,2]-raw.scene.env_origins[0,2]-.625).item())
                    max_lift = max(max_lift, height)
                    if bool(extras["success"][0].item()):
                        report.update(success=True, steps=step+1, max_lift_m=max_lift,
                                      target_lift_m=.15, required_hold_steps=cfg.success_hold_steps)
                        break
                else:
                    report.update(steps=290, max_lift_m=max_lift)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report), flush=True)
            if not report["success"]:
                raise RuntimeError("Scripted grasp did not pass the corrected success criterion; inspect report")
        finally:
            env.close()


if __name__ == "__main__":
    main()
