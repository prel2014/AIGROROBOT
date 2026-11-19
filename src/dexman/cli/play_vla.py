"""Evaluate a visual policy in parallel simulated robots, with or without a Kit window."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20, help="Total completed episodes across replicas")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--execute-steps", type=int, default=1, help="Chunk actions before replanning; default 1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--crop", choices=("berries", "banana", "citrus", "vineyard"),
                        help="Evaluar la tarea agrícola de este cultivo con el estudiante bimanual")
    parser.add_argument("--phase-hints", action="store_true",
                        help="Añadir a la orden la subtarea indicada por el simulador agrícola")
    parser.add_argument("--grasp-calibration", type=Path, help="Evaluar desde el preagarre agrícola calibrado")
    parser.add_argument("--exercise", choices=("pinch", "return"), help="Selecciona el objetivo físico alternativo")
    parser.add_argument("--output", default="logs/vla_eval.json")
    from dexman.tasks.manipulation.scenarios import add_scenario_argument, configure_scenario
    add_scenario_argument(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.episodes, args.max_steps, args.execute_steps) < 1:
        parser.error("num_envs, episodes, max-steps and execute-steps must be positive")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new evaluation filename")
    if args.crop:
        if args.scenario != "red_cube" or args.execute_steps != 1:
            parser.error("En agricultura se usa --crop y se reobservan ambos ojos en cada paso")
        if args.instruction is not None and not args.instruction.strip():
            parser.error("La orden debe contener texto")
        from dexman.evaluation.agriculture_vla import play
        return play(args, checkpoint, output)
    if args.phase_hints or args.grasp_calibration or args.exercise:
        parser.error("Las opciones de evaluación agrícola requieren --crop")
    import numpy as np
    from dexman.evaluation.metrics import REASONS
    from dexman.evaluation.policies import VisualPolicy
    from dexman.tasks.lift.config import JointAngleVisualLiftEnvCfg
    from dexman.vla.contract import VLA_TASK, check_environment_contract

    cfg = JointAngleVisualLiftEnvCfg()
    configure_scenario(cfg, args.scenario)
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    if args.instruction is not None:
        if not args.instruction.strip():
            parser.error("instruction must contain text")
        cfg.instruction = args.instruction
    # Validate checkpoint semantics on CPU before allocating a second GPU simulation.
    policy = VisualPolicy(checkpoint, "cpu", args.num_envs, args.execute_steps)
    contract = policy.predictor.robot_contract
    check_environment_contract(contract, cfg)
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import torch
        import dexman.tasks  # noqa: F401
        policy.predictor.device = cfg.sim.device
        policy.predictor.model.to(cfg.sim.device)
        env = gym.make(VLA_TASK, cfg=cfg)
        records = []
        lengths = np.zeros(args.num_envs, dtype=np.int64)
        returns = np.zeros(args.num_envs)
        try:
            env.reset(seed=args.seed)
            raw = env.unwrapped
            with torch.inference_mode():
                for _ in range(args.max_steps):
                    _, rewards, terminated, truncated, extras = env.step(policy.act(raw, None))
                    dones = (terminated | truncated).cpu().numpy()
                    lengths += 1
                    returns += rewards.cpu().numpy()
                    policy.reset(dones)
                    for i in np.flatnonzero(dones):
                        reason = int(extras["terminal_reason"][i].item())
                        record = {"replica": int(i), "success": bool(extras["success"][i].item()),
                                  "reason": REASONS.get(reason, "unknown"),
                                  "steps": int(lengths[i]), "return": float(returns[i])}
                        records.append(record)
                        print(record, flush=True)
                        lengths[i], returns[i] = 0, 0
                        if len(records) >= args.episodes:
                            break
                    if len(records) >= args.episodes:
                        break
        finally:
            env.close()
        summary = {"checkpoint": str(checkpoint), "seed": args.seed, "num_envs": args.num_envs,
                   "task": VLA_TASK, "output_space": contract["output_space"],
                   "instruction": cfg.instruction, "execute_steps": args.execute_steps,
                   "completed": len(records), "requested": args.episodes,
                   "success_rate": sum(r["success"] for r in records) / len(records) if records else None,
                   "episodes": records}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Evaluation: {output}; success_rate={summary['success_rate']}", flush=True)
        if len(records) < args.episodes:
            raise RuntimeError("max-steps reached before completing requested episodes; see evaluation report")
