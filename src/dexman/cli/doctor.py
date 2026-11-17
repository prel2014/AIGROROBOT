"""Report runtime, asset and data readiness without launching or changing the simulator."""
import argparse
import importlib.metadata
import importlib.util
import json
import sys

from dexman.experiments import provenance, sha256, write_json
from dexman.paths import ROOT, ROBOT_USD, JOINT_MANIFEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--stage", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--dataset", default="datasets/demos")
    args = parser.parse_args()
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Use Isaac Sim's Python 3.12")
    versions = {}
    for line in (ROOT / "configs/runtime-constraints.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        package, expected = line.split("==")
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[package] = {"expected": expected, "actual": actual, "matches": bool(actual and actual.split("+")[0] == expected)}
        if not versions[package]["matches"]:
            errors.append(f"{package}: expected {expected}, found {actual}")
    for module in ("isaaclab", "isaaclab_rl", "isaaclab_physx"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing runtime package: {module}")
    assets = {}
    for path in (ROBOT_USD, JOINT_MANIFEST, ROOT / "configs/grasp_seed.json"):
        assets[path.relative_to(ROOT).as_posix()] = sha256(path) if path.is_file() else None
        if not path.is_file():
            errors.append(f"Missing asset: {path}")
    data = None
    if args.stage == "student":
        from dexman.data.validation import audit_dataset
        try:
            data = audit_dataset(args.dataset)
            if not data["valid"] or data["successes"] < 3:
                errors.append("Student needs at least 3 validated, complete successful demonstrations")
        except (ValueError, OSError) as error:
            errors.append(str(error))
    result = {"stage": args.stage, "ready": not errors, "errors": errors, "versions": versions,
              "assets": assets, "data": data, "provenance": provenance()}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)
