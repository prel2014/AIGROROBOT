"""Reproducible run metadata and atomic local artifacts; no simulator imports."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dexman.paths import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def provenance():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "h5py", "gymnasium", "transformers", "rsl-rl-lib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = hashlib.sha256()
    for path in sorted((ROOT / "src/dexman").rglob("*.py")):
        source.update(path.relative_to(ROOT).as_posix().encode())
        source.update(path.read_bytes())
    try:
        revision = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                                  cwd=ROOT, capture_output=True, text=True, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    return {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages, "source_sha256": source.hexdigest(),
            "git_commit": commit,
            "runtime": json.loads((ROOT / "configs/runtime.lock.json").read_text(encoding="utf-8"))}


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(source, root, pattern):
    """Resolve a checkpoint file, a run directory, or the most recently saved run."""
    directory = Path(root if source == "latest" else source).expanduser().resolve(strict=True)
    if directory.is_file():
        return directory
    candidates = [path for path in directory.rglob(pattern) if path.is_file()]
    if not candidates:
        raise ValueError(f"No checkpoints matching {pattern} in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def random_state():
    import random
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    import random
    import numpy as np
    import torch
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    rng = state["numpy_rng"]
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def save_training_checkpoint(path, state):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
