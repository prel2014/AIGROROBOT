"""Render one situation and both RGB-D views. No policy, optimizer or training."""
import argparse
from pathlib import Path


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.manipulation.scenarios import SITUATIONS, apply_situation, instructions
    from dexman.tasks.lift.config import TeacherCameraEnvCfg
    from dexman.sensors.view import look_at_xyzw
    from dexman.sensors.camera import camera_cfg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[s.name for s in SITUATIONS], required=True)
    parser.add_argument("--output", default="assets/previews")
    add_launcher_args(parser)
    args = parser.parse_args()
    output = Path(args.output) / args.scenario
    output.mkdir(parents=True, exist_ok=True)
    cfg = TeacherCameraEnvCfg()
    cfg.scene.num_envs, cfg.object_xy_range = 1, 0.
    situation = apply_situation(cfg, args.scenario)
    cfg.scene.overview = camera_cfg()
    cfg.scene.overview.prim_path = "{ENV_REGEX_NS}/Overview"
    cfg.scene.overview.width, cfg.scene.overview.height = 800, 600
    cfg.scene.overview.offset.pos = (1.7,1.7,1.7)
    cfg.scene.overview.offset.rot = look_at_xyzw((1.7,1.7,1.7),(-.05,0.,.72))
    cfg.scene.overview.spawn.clipping_range = (.05, 10.)
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image
        import dexman.tasks  # noqa: F401
        from dexman.experiments import write_json
        env = gym.make("Dexman-UR10e-Shadow-Lift-v1", cfg=cfg)
        try:
            env.reset(seed=42)
            raw = env.unwrapped
            with torch.inference_mode():
                for _ in range(8):
                    env.step(raw.home_action[None])
            for name in ("overview","camera","camera_secondary"):
                data = raw.scene[name].data.output
                rgb = data["rgb"].torch[0,...,:3].cpu().numpy().astype(np.uint8)
                Image.fromarray(rgb).save(output / f"{name}.png")
                depth = data["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy()
                valid = np.isfinite(depth) & (depth>0)
                gray = np.zeros(depth.shape,dtype=np.uint8)
                gray[valid] = (255*np.clip((2.-depth[valid])/2.,0,1)).astype(np.uint8)
                Image.fromarray(gray).save(output / f"{name}_depth.png")
            write_json(output/"scene.json", {"scenario":args.scenario,"theme":situation.theme,
                "orders":instructions(situation),"training":False,"policy_loaded":False,
                "physics_steps":8,"target_shape":situation.shape,"target_dimensions":situation.dimensions,
                "minimum_contact_force_n":cfg.minimum_contact_force_n,
                "cameras_for_vla":["camera","camera_secondary"]})
            print(f"SCENE_CAPTURED {args.scenario} {output}",flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
