"""Render a saved PPO checkpoint in a separate evaluation; never attach to training."""
import argparse
import json
import shutil
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for images and metadata")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--task-version", choices=["v0", "v1"], default="v1")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 30, 90])
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Total evaluation steps; must include every capture step")
    add_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 4 or min(args.steps) < 0:
        parser.error("Use 1-4 evaluation robots and nonnegative capture steps")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.checkpoint).resolve(strict=True)
    checkpoint = output / source.name
    shutil.copy2(source, checkpoint)

    from dexman.tasks.lift.config import VisualLiftEnvCfg, CorrectedVisualLiftEnvCfg
    cfg = CorrectedVisualLiftEnvCfg() if args.task_version == "v1" else VisualLiftEnvCfg()
    if args.task_version == "v1":
        from dexman.agents.checkpoint import validate_teacher
        validate_teacher(source, cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.camera.width, cfg.scene.camera.height = 640, 480
    cfg.scene.camera.data_types = ["rgb"]
    cfg.scene.camera.spawn.clipping_range = (0.05, 10.0)
    capture_steps = sorted(set(args.steps))
    rollout_steps = capture_steps[-1] if args.rollout_steps is None else args.rollout_steps
    if rollout_steps < capture_steps[-1]:
        parser.error("rollout-steps must be at least the last capture step")
    metadata = {"source_checkpoint": str(source), "checkpoint_copy": str(checkpoint),
                "kind": "separate checkpoint evaluation, not live training frames",
                "num_envs": args.num_envs, "seed": cfg.seed, "frames": []}
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import dexman.tasks  # noqa: F401
        from dexman.agents.ppo import LiftPPORunnerCfg, CorrectedLiftPPORunnerCfg
        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)

        env = gym.make(f"Dexman-UR10e-Shadow-Lift-RGBD-{args.task_version}", cfg=cfg)
        tiles = {}
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
            agent_cfg = CorrectedLiftPPORunnerCfg() if args.task_version == "v1" else LiftPPORunnerCfg()
            agent = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
            agent.device = cfg.sim.device
            runner = create_rsl_rl_runner(wrapped, agent)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=cfg.sim.device)
            obs, _ = wrapped.reset()
            raw = env.unwrapped
            camera = raw.scene["camera"]
            origins = raw.scene.env_origins
            eyes = origins + torch.tensor([2.1, -2.5, 2.2], device=cfg.sim.device)
            targets = origins + torch.tensor([-0.15, 0.0, 1.0], device=cfg.sim.device)
            camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            # Populate renderer buffers after positioning the diagnostic cameras.
            for _ in range(5):
                raw.sim.render()
            camera.update(raw.step_dt, force_recompute=True)
            max_lift = torch.zeros(args.num_envs, device=cfg.sim.device)
            completed_episodes = []
            with torch.inference_mode():
                for step in range(rollout_steps + 1):
                    positions = raw.object.data.root_pos_w.torch - origins
                    max_lift = torch.maximum(max_lift, positions[:, 2] - cfg.object_initial_height)
                    if step in capture_steps:
                        rgb = camera.data.output["rgb"].torch[..., :3].to(torch.uint8).cpu().numpy().copy()
                        positions = raw.object.data.root_pos_w.torch - origins
                        for i in range(args.num_envs):
                            image = Image.fromarray(rgb[i])
                            filename = f"robot_{i + 1:02d}_step_{step:04d}.png"
                            image.save(output / filename)
                            tiles[i, step] = image
                            metadata["frames"].append({"file": filename, "robot": i + 1,
                                "simulation_time_s": step * raw.step_dt,
                                "episode_step": int(raw.episode_length_buf[i].item()),
                                "object_position_local_m": positions[i].cpu().tolist(),
                                "image_std": float(np.std(rgb[i]))})
                        print(f"Captured step {step}", flush=True)
                    if step < rollout_steps:
                        obs, _, dones, extras = wrapped.step(policy(obs))
                        for i in dones.nonzero(as_tuple=False).flatten().cpu().tolist():
                            completed_episodes.append({"robot": i + 1, "step": step + 1,
                                "success": bool(extras["success"][i].item())})
                        policy.reset(dones)
            metadata["evaluation"] = {"steps_per_robot": rollout_steps,
                "max_object_lift_m_by_robot": max_lift.cpu().tolist(),
                "completed_episodes": completed_episodes,
                "success_count": sum(int(episode["success"]) for episode in completed_episodes)}
            print(f"Evaluation: {metadata['evaluation']}", flush=True)
            # Preserve original pixels, adding captions around the captured frames.
            header, label = 64, 26
            sheet = Image.new("RGB", (640 * len(capture_steps), header + (480 + label) * args.num_envs), "#14202d")
            draw = ImageDraw.Draw(sheet)
            draw.text((16, 12), f"UR10e + Shadow Hand | {source.name} | PPO checkpoint replay", fill="white")
            draw.text((16, 32), "Separate evaluation. Columns: simulated time. Rows: independent robot replicas.", fill="#d1d9e0")
            for i in range(args.num_envs):
                for column, step in enumerate(capture_steps):
                    x, y = column * 640, header + i * (480 + label)
                    draw.text((x + 10, y + 6), f"Robot {i + 1} | t={step * raw.step_dt:.1f}s", fill="white")
                    sheet.paste(tiles[i, step], (x, y + label))
            sheet.save(output / "robots.png")
            # Kit may terminate the interpreter during launch_simulation's exit;
            # persist all artifacts before closing the simulator.
            (output / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"Saved robot images to {output}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
