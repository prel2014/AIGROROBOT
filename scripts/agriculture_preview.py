"""Open an agricultural scene in Isaac Sim and capture both torso RGB-D eyes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def capture(scene, output):
    import numpy as np
    from PIL import Image
    from dexman.tasks.agriculture.interface import EYE_KEYS

    views = {}
    for name in ("overview", *EYE_KEYS):
        sensor = scene[name]
        data = sensor.data
        rgb = data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = data.output["distance_to_image_plane"].torch[0, ..., 0].cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0.)
        if not valid.any() or rgb.std() < 1.:
            raise RuntimeError(f"No usable RGB-D frame from {name}")
        Image.fromarray(rgb).save(output / f"{name}.png")
        # Preserve millimetres in the PNG. Zero means an invalid or absent return.
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(np.rint(depth[valid]*1000), 1, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(output / f"{name}_depth_mm.png")
        preview = np.zeros(depth.shape, dtype=np.uint8)
        preview[valid] = np.rint(255*np.clip((3.-depth[valid])/2.8, 0, 1)).astype(np.uint8)
        Image.fromarray(preview).save(output / f"{name}_depth.png")
        views[name] = {"shape": list(rgb.shape), "valid_depth_pixels": int(valid.sum()),
                       "position_world_m": data.pos_w.torch[0].cpu().tolist(),
                       "orientation_ros_xyzw": data.quat_w_ros.torch[0].cpu().tolist(),
                       "intrinsics": data.intrinsic_matrices.torch[0].cpu().tolist()}
    return views


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.interface import ARM_KEYS, EYE_KEYS, JOINT_ORDER
    from dexman.tasks.agriculture.layouts import LAYOUTS, layout_variation
    from dexman.tasks.agriculture.randomization import OBSTACLE_LAYOUT_VERSION, apply_variations, sample_episode
    from dexman.tasks.agriculture.scene import CROPS, make_preview

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenario", choices=CROPS)
    selection.add_argument("--layout", choices=LAYOUTS, help="Select a fixed physical task layout")
    parser.add_argument("--output", type=Path, default=Path("assets/agriculture/previews"))
    parser.add_argument("--live", action="store_true", help="Keep the scene and eye panels open in Kit")
    parser.add_argument("--randomize", action="store_true", help="Apply a seeded episode layout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0, help="Episode index for --randomize")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.live and "kit" not in (args.visualizer or ()):
        parser.error("--live requires --viz kit")
    if min(args.seed, args.episode) < 0:
        parser.error("--seed and --episode must be nonnegative")
    layout = LAYOUTS[args.layout] if args.layout else None
    scenario = layout.crop if layout else args.scenario
    output = args.output / (args.layout or scenario)
    if layout:
        variation = layout_variation(layout, args.seed, args.episode, randomize=args.randomize)
    else:
        variation = sample_episode(scenario, args.seed, 0, args.episode) if args.randomize else None
    if args.randomize:
        output /= f"layout_{OBSTACLE_LAYOUT_VERSION}_seed_{args.seed}_episode_{args.episode}"
    cfg = make_preview(scenario, randomize=args.randomize, layout=layout)
    crop = CROPS[scenario]
    instruction = layout.objective if layout else crop.instruction
    inspection_target = layout.inspection_target if layout else (.37, .05, 1.12)
    with launch_simulation(cfg, args):
        import torch
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from dexman.robot.joints import JOINT_NAMES, PALM

        sim = SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene)
        sim.set_camera_view(cfg.scene.overview.offset.pos, inspection_target)
        sim.reset()
        scene.reset()
        if variation is not None:
            apply_variations(scene, {0: variation})
        arms = [scene[key] for key in ARM_KEYS]
        for arm in arms:
            ids, names = arm.find_joints(list(JOINT_NAMES), preserve_order=True)
            if len(arm.joint_names) != 30 or tuple(names) != JOINT_NAMES or len(ids) != 30:
                raise RuntimeError(f"Unexpected arm joint map: {names}")
            arm.write_joint_position_to_sim_index(position=arm.data.default_joint_pos.torch)
            arm.write_joint_velocity_to_sim_index(velocity=arm.data.default_joint_vel.torch)
            arm.actuators.target_command.set_position_index(value=arm.data.default_joint_pos.torch)

        def step(count):
            for index in range(count):
                for arm in arms:
                    arm.actuators.target_command.set_effort_index(
                        value=arm.data.gravity_compensation_forces.torch)
                scene.write_data_to_sim()
                sim.step(render=(index+1) % cfg.sim.render_interval == 0)
                scene.update(cfg.sim.dt)

        step(48)
        output.mkdir(parents=True, exist_ok=True)
        views = capture(scene, output)
        arm_state = {}
        for key, arm in zip(ARM_KEYS, arms, strict=True):
            palm_ids, _ = arm.find_bodies(PALM)
            actual = arm.data.joint_pos.torch
            if not torch.isfinite(actual).all():
                raise RuntimeError(f"Non-finite joint state on {key}")
            arm_state[key] = {"joints": len(arm.joint_names),
                "palm_world_m": arm.data.body_pos_w.torch[0, palm_ids[0]].cpu().tolist(),
                "max_tracking_error_rad": float((actual-arm.data.default_joint_pos.torch).abs().max()),
                "max_error_joint": arm.joint_names[int((actual-arm.data.default_joint_pos.torch).abs().argmax())]}
        fruits = {key: obj.data.root_pos_w.torch[0].cpu().tolist()
                  for key, obj in scene.rigid_objects.items() if key.startswith("fruit_")}
        report = {"scenario": scenario, "instruction": instruction, "plant": crop.plant,
                  "arms": arm_state, "joint_order": list(JOINT_ORDER), "views": views,
                  "fruit_positions_m": fruits, "eye_order": list(EYE_KEYS),
                  "physics_steps": 48, "training": False}
        if layout:
            report["layout"] = {"name": args.layout, **asdict(layout)}
        if variation is not None:
            report["variation"] = {"layout_version": OBSTACLE_LAYOUT_VERSION,
                                   "seed": args.seed, "replica_id": 0, "episode_id": args.episode,
                                   "seeded": args.randomize,
                                   **asdict(variation)}
            report["obstacle_positions_m"] = {
                prop.kind: scene[f"obstacle_{prop.kind}"].data.root_pos_w.torch[0].cpu().tolist()
                for prop in variation.obstacles
            }
        (output / "scene.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(f"AGRICULTURE_CAPTURED {args.layout or scenario} {output} {arm_state} {fruits}", flush=True)

        if args.live:
            from dexman.sensors.teacher_camera_panel import TeacherCameraPanel
            panels = [TeacherCameraPanel(scene[key], 1, instruction, title)
                      for key, title in zip(EYE_KEYS, ("Ojo izquierdo RGB-D", "Ojo derecho RGB-D"), strict=True)]
            try:
                while sim.is_running():
                    step(cfg.sim.render_interval)
                    for panel in panels:
                        panel.refresh()
            finally:
                for panel in panels:
                    panel.close()


if __name__ == "__main__":
    main()
