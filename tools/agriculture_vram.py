"""Measure physical GPU peak for an agricultural rollout and PPO-shaped backward pass."""
import argparse
from pathlib import Path

from dexman.experiments import write_json


def probe(cfg, *, rgbd):
    import torch
    from isaaclab.sensors import SensorBase
    from isaaclab.sim import SimulationContext
    from dexman.agents.agriculture import BimanualTeacherModel, GROUPS, ppo_config
    from dexman.tasks.agriculture.environment import AgriculturalPPOEnv
    from dexman.tasks.agriculture.interface import ACTION_DIM, EYE_KEYS

    sim = SimulationContext(cfg.sim)
    env = AgriculturalPPOEnv(sim, cfg)
    settings = ppo_config(25)
    obs = env.get_observations()
    actor = BimanualTeacherModel(obs, GROUPS, "actor", ACTION_DIM).to(env.device)
    critic = BimanualTeacherModel(obs, GROUPS, "critic", 1).to(env.device)
    print(f"VRAM_STAGE initialized {env.num_envs} replicas, rgbd={rgbd}", flush=True)
    steps = settings["num_steps_per_env"]
    storage = obs["policy"].new_empty((steps, *obs["policy"].shape))
    eyes = [group.scene[key] for group in env.batch.groups for key in EYE_KEYS] if rgbd else []
    frames = 0
    with torch.inference_mode():
        for t in range(steps):
            storage[t].copy_(obs["policy"])
            obs, _, _, _ = env.step(actor(obs))
            if rgbd and (t + 1) % 4 == 0:
                SensorBase.update_batch(eyes, 0.)
                rgb, depth = env.batch.rgbd()
                frames += 1
                if frames == 1:
                    print(f"VRAM_STAGE cameras {tuple(rgb.shape)} {tuple(depth.shape)}", flush=True)
    print("VRAM_STAGE rollout", flush=True)
    minibatch = steps * env.num_envs // settings["algorithm"]["num_mini_batches"]
    sample = {"policy": storage.flatten(0, 1)[:minibatch]}
    # Reserve the two running-moment buffers for each Adam parameter. The
    # backward pass allocates activations and gradients without any update.
    moments = [torch.empty_like(parameter) for model in (actor, critic)
               for parameter in model.parameters() for _ in range(2)]
    (actor(sample).square().mean() + critic(sample).square().mean()).backward()
    torch.cuda.synchronize(env.device)
    print("VRAM_STAGE backward", flush=True)
    return {"mode": "rgbd" if rgbd else "headless", "replicas": env.num_envs, "rollout_steps": steps,
            "captured_instants": frames, "minibatch": minibatch,
            "adam_buffer_mib": sum(x.numel() * x.element_size() for x in moments) / 2**20,
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(env.device) / 2**20}


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from dexman.tasks.agriculture.batch import make_batch_config
    from dexman.tasks.agriculture.interface import EYE_KEYS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.rgbd and args.num_envs % 4:
        parser.error("RGB-D requiere igual número de réplicas en los cuatro grupos")
    device = args.device or "cuda:0"
    if not device.startswith("cuda"):
        parser.error("La prueba requiere una GPU CUDA")
    cfg = make_batch_config(args.num_envs, contacts=True, rgbd=args.rgbd, nominal=True)
    if args.rgbd:
        for scene in vars(cfg.scenes).values():
            for eye in EYE_KEYS:
                getattr(scene, eye).update_period = 0.
    with launch_simulation(cfg, args):
        result = probe(cfg, rgbd=args.rgbd)
        write_json(args.output, result)
        print(f"AGRICULTURE_VRAM {result}", flush=True)


if __name__ == "__main__":
    main()
