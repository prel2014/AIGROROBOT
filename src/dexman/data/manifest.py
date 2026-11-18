"""Content-verified train/validation/test partitions; corrective data is train-only."""
import json
import random
from pathlib import Path

from dexman.data.validation import audit_dataset
from dexman.experiments import sha256


def create_manifest(root, seed=42, validation_fraction=.2, test_fraction=.2):
    if min(validation_fraction, test_fraction) <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    audit = audit_dataset(root)
    if not audit["valid"]:
        raise ValueError(f"Dataset audit failed: {audit['errors'] or 'no complete episodes'}")
    demos = [row for row in audit["episodes"] if row["success"] and row["sample_kind"] == "teacher_demonstration"]
    corrections = [row for row in audit["episodes"] if row["sample_kind"] == "teacher_correction"]
    if len(demos) < 3:
        raise ValueError("At least 3 successful demonstrations are required for independent train/validation/test")
    hashes = [row["sha256"] for row in demos + corrections]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate episode contents: remove duplicate copies before splitting")
    rng = random.Random(seed)
    per_crop = {}
    for row in demos:
        per_crop.setdefault(row.get("crop"), []).append(row)
    groups = per_crop.values() if all(len(rows) >= 3 for rows in per_crop.values()) else (demos,)
    splits = {"test": [], "validation": [], "train": []}
    for rows in groups:
        rng.shuffle(rows)
        test_count = min(len(rows) - 2, max(1, round(len(rows) * test_fraction)))
        val_count = min(len(rows) - test_count - 1, max(1, round(len(rows) * validation_fraction)))
        splits["test"].extend(rows[:test_count])
        splits["validation"].extend(rows[test_count:test_count + val_count])
        splits["train"].extend(rows[test_count + val_count:])
    splits["train"].extend(corrections)
    return {"version": 1, "dataset_root": str(Path(root).resolve()), "seed": seed, "split_unit": "episode",
            "robot_contract": demos[0]["contract"], "partial_files_ignored": audit["partial_files"],
            "splits": splits}


def resolve_manifest(manifest, root=None):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported dataset manifest")
    root = Path(root or manifest["dataset_root"]).resolve(strict=True)
    splits, seen_paths, seen_hashes = {}, set(), set()
    for name in ("train", "validation", "test"):
        rows = manifest.get("splits", {}).get(name, [])
        if not rows:
            raise ValueError(f"Empty partition: {name}")
        splits[name] = []
        for row in rows:
            path = (root / row["path"]).resolve(strict=True)
            if not path.is_relative_to(root) or path in seen_paths or row["sha256"] in seen_hashes:
                raise ValueError("Dataset paths or contents leak across partitions")
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Dataset changed since split was created: {path}")
            if name != "train" and (not row["success"] or row["sample_kind"] == "teacher_correction"):
                raise ValueError("Corrections and failures must never enter validation/test")
            seen_paths.add(path)
            seen_hashes.add(row["sha256"])
            splits[name].append(path)
    return splits


def read_manifest(path, root=None):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest, resolve_manifest(manifest, root)


def add_corrections(manifest, correction_root):
    """Keep held-out partitions fixed while adding DAgger rounds from a sibling directory."""
    from copy import deepcopy
    result = deepcopy(manifest)
    old_root = Path(result["dataset_root"])
    correction_root = Path(correction_root).resolve()
    import os
    common = Path(os.path.commonpath([old_root, correction_root]))
    # Rebase entries instead of copying datasets or changing their content.
    for rows in result["splits"].values():
        for row in rows:
            row["path"] = (old_root / row["path"]).relative_to(common).as_posix()
    audit = audit_dataset(correction_root)
    if not audit["valid"] or not audit["episodes"]:
        raise ValueError(f"Invalid corrections: {audit['errors']}")
    for row in audit["episodes"]:
        if row["sample_kind"] != "teacher_correction" or row["contract"] != result["robot_contract"]:
            raise ValueError("Expected teacher corrections with the original control contract")
        row["path"] = (correction_root / row["path"]).relative_to(common).as_posix()
        result["splits"]["train"].append(row)
    result["dataset_root"] = str(common)
    resolve_manifest(result)
    return result
