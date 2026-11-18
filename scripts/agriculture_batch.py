"""Validate parallel agricultural replicas in Isaac Sim without training."""
import argparse
import json


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    from dexman.tasks.agriculture.batch import AgriculturalBatch, DEFAULT_REPLICAS, make_batch_config
    from dexman.tasks.agriculture.interface import ACTION_DIM, STATE_DIM
    from dexman.tasks.agriculture.randomization import GROUND_Z, INACTIVE_Z, OBSTACLE_RADIUS, TABLE_Z
    from dexman.tasks.agriculture.scene import CROPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--steps", type=int, default=150, help="Number of 30 Hz control ticks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgbd", action="store_true", help="Render both 224x224 torso cameras per replica")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < len(CROPS) or args.steps < 1 or args.seed < 0:
        parser.error(f"Expected at least {len(CROPS)} replicas, positive steps and a nonnegative seed")
    cfg = make_batch_config(args.num_envs, rgbd=args.rgbd)

    with launch_simulation(cfg, args):
        import torch
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(cfg.sim)
        batch = AgriculturalBatch(sim, cfg, seed=args.seed)
        sim.set_camera_view((23., -25., 36.), (0., 0., 0.))
        batch.initialize()
        state = batch.angular_state()
        if state.shape != (batch.num_envs, STATE_DIM) or not torch.isfinite(state).all():
            raise RuntimeError(f"Invalid initial angular state: {state.shape}")
        targets = state[:, :ACTION_DIM].clone()
        for _ in range(args.steps):
            batch.step(targets, render=args.rgbd or "kit" in (args.visualizer or ()))
        state = batch.angular_state()
        if not torch.isfinite(state).all():
            raise RuntimeError("Non-finite angular state after advancing the batch")
        error = (state[:, :ACTION_DIM] - targets).abs().max().item()
        for group in batch.groups:
            for name, fruit in group.scene.rigid_objects.items():
                if name.startswith("fruit_"):
                    pos = fruit.data.root_pos_w.torch - group.origins
                    if not (torch.isfinite(pos).all() and (pos[:, 2] > TABLE_Z - .05).all()
                            and (pos[:, :2].abs() < 1.5).all()):
                        raise RuntimeError(f"Fruit outside its local scene: {group.crop}/{name}")
            layouts = [episode.layout for episode in batch.dynamics[group.start:group.stop]]
            props = [{item.kind: item for item in layout.obstacles} for layout in layouts]
            for kind in OBSTACLE_RADIUS:
                obstacle = group.scene[f"obstacle_{kind}"]
                position = obstacle.data.root_pos_w.torch - group.origins
                expected = []
                for items in props:
                    prop = items.get(kind)
                    if prop is None:
                        expected.append((0., 0., INACTIVE_Z))
                    else:
                        z = TABLE_Z if prop.surface == "bench" else GROUND_Z
                        expected.append((*prop.xy, z + .001))
                if not torch.allclose(position, position.new_tensor(expected), atol=.001):
                    raise RuntimeError(f"Obstacle poses do not match the active layouts: {group.crop}/{kind}")
        added = sum(episode.added for episode in batch.dynamics)
        removed = sum(episode.removed for episode in batch.dynamics)
        report = {
            "replicas": batch.num_envs,
            "replicas_per_crop": {group.crop: group.scene.num_envs for group in batch.groups},
            "angular_state": list(state.shape),
            "joint_targets": list(targets.shape),
            "max_initial_pose_error_rad": error,
            "active_obstacles": sum(len(episode.layout.obstacles) for episode in batch.dynamics),
            "obstacles_added": added,
            "obstacles_removed": removed,
            "lights_changed": sum(bool(episode.layout.light_intensity != episode.initial.light_intensity)
                                  for episode in batch.dynamics),
            "control_steps": args.steps,
            "training": False,
        }
        if args.rgbd:
            rgb, depth = batch.rgbd()
            expected = (batch.num_envs, 2, 224, 224)
            valid = torch.isfinite(depth) & (depth > 0.)
            if (rgb.shape != (*expected, 3) or depth.shape != expected or not valid.flatten(2).any(-1).all()
                    or not (rgb.flatten(2).amax(-1) > rgb.flatten(2).amin(-1)).all()):
                raise RuntimeError("Invalid RGB-D frames from the batched torso cameras")
            report.update(rgb=list(rgb.shape), metric_depth=list(depth.shape))
        selected = (0, batch.num_envs - 1)
        batch.reset(selected)
        if tuple(batch.episode_ids) != tuple(int(i in selected) for i in range(batch.num_envs)):
            raise RuntimeError("Indexed episode reset touched an unexpected replica")
        report["indexed_reset"] = selected
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
