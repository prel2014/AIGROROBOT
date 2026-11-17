"""Mirror the two official USD assets and every referenced local dependency."""
import concurrent.futures
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from usd_python import configure_usd

configure_usd()
from pxr import Sdf  # noqa: E402

ROOT = (Path(__file__).resolve().parents[1] / "assets" / "vendor").resolve()
BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/"
SEEDS = ["Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
         "Isaac/Robots/ShadowRobot/ShadowHand/shadow_hand.usd"]


def fetch(relative):
    target = Path(os.path.abspath(ROOT / relative))
    if not target.is_relative_to(ROOT):
        raise ValueError(f"Asset outside mirror: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = BASE + urllib.parse.quote(relative, safe="/")
        temporary = target.with_name(target.name + ".part")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                temporary.replace(target)
                break
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f"Download failed: {url}") from error
    return relative, target


def dependencies(relative, target):
    if target.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        return set()
    layer = Sdf.Layer.FindOrOpen(str(target))
    if not layer:
        raise RuntimeError(f"Unreadable USD: {target}")
    # Asset-valued attributes include textures; composition references alone do not.
    references = set(re.findall(r"@([^@]+)@", layer.ExportToString()))
    result = set()
    for ref in references:
        if ref in {"OmniPBR.mdl", "OmniGlass.mdl"}:
            continue  # Built-in Isaac Sim material modules.
        if "<UDIM>" in ref:
            raise ValueError(f"UDIM needs an explicit tile manifest: {ref}")
        url = urllib.parse.urljoin(BASE + relative, ref)
        if url.startswith(BASE):
            result.add(urllib.parse.unquote(url[len(BASE):]))
        elif not ref.endswith(".mdl"):
            raise ValueError(f"External dependency outside NVIDIA mirror: {ref}")
        # Built-in NVIDIA MDL modules are supplied by Isaac Sim, not this mirror.
    return result


def main():
    pending, seen = set(SEEDS), set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while pending:
            batch = sorted(pending - seen)
            pending = set()
            for relative, target in pool.map(fetch, batch):
                seen.add(relative)
                pending.update(dependencies(relative, target) - seen)
            print(f"Downloaded/checked {len(seen)} assets; next {len(pending)}", flush=True)
    manifest = {name: {"url": BASE + urllib.parse.quote(name, safe="/"),
                       "bytes": (ROOT / name).stat().st_size,
                       "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                for name in sorted(seen)}
    (ROOT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {len(manifest)} files", flush=True)


if __name__ == "__main__":
    main()
