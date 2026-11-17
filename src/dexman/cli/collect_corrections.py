"""Collect teacher corrections on student rollouts, including recoveries and failures."""
import argparse
from pathlib import Path

from dexman.experiments import provenance, seed_everything, sha256, write_json


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True, help="New collection directory")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--beta", type=float, default=.2, help="Probability of executing the teacher action")
    parser.add_argument("--seed", type=int, default=4242)
    add_launcher_args(parser)
    args = parser.parse_args()
    if min(args.episodes, args.num_envs) < 1 or not 0 <= args.beta <= 1:
        parser.error("Counts must be positive and beta in [0,1]")
    teacher_path, student_path = Path(args.teacher).resolve(strict=True), Path(args.student).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Collection output already exists")
    from dexman.agents.checkpoint import validate_teacher
    from dexman.data.dagger import collect_corrections
    from dexman.data.metadata import teacher_metadata
    from dexman.data.validation import audit_dataset
    from dexman.data.writer import EpisodeWriter
    from dexman.evaluation.policies import TeacherPolicy, VisualPolicy
    from dexman.tasks.lift.config import JointAngleVisualLiftEnvCfg
    from dexman.vla.contract import VLA_TASK, check_environment_contract
    cfg = JointAngleVisualLiftEnvCfg()
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    validate_teacher(teacher_path, cfg)
    seed_everything(args.seed)
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import dexman.tasks  # noqa: F401
        env = gym.make(VLA_TASK, cfg=cfg)
        writers = []
        try:
            teacher = TeacherPolicy(teacher_path, cfg.sim.device)
            student = VisualPolicy(student_path, cfg.sim.device, args.num_envs)
            check_environment_contract(student.predictor.robot_contract, cfg)
            metadata = {**teacher_metadata(teacher_path, cfg, "teacher_correction"),
                        "student_sha256": sha256(student_path), "beta": args.beta, "label_horizon": 1}
            output.mkdir(parents=True)
            write_json(output / "run.json", {"arguments": vars(args), "provenance": provenance(), "metadata": metadata})
            writers = [EpisodeWriter(output / f"env_{index:03d}", {**metadata, "replica": index}, success_only=False)
                       for index in range(args.num_envs)]
            summary = collect_corrections(env, teacher, student, writers, args.episodes, args.episodes, args.beta)
            write_json(output / "collection.json", summary)
        finally:
            for writer in writers:
                writer.abort()
            env.close()
        report = audit_dataset(output)
        write_json(output / "audit.json", report)
        if not report["valid"]:
            raise RuntimeError("Corrective data did not pass validation; see audit.json")
