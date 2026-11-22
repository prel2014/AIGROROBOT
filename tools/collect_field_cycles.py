"""Record and validate one mobile field dataset for the four crops, without training."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from dexman.tasks.agriculture.harvest_route import SITES, SITUATIONS


def complete(root, crop, situation):
    import h5py
    for path in (root/crop).glob("*/robot_*/*.h5"):
        with h5py.File(path) as episode:
            if episode.attrs.get("situation","base")==situation and episode.attrs["success"]:
                return True
    return False


def acquisition_profile(crop, situation, attempt):
    if crop=="citrus":
        options = ["--orange-task","roca"]
        if attempt:
            options += ["--rock-offset-m",str({"base":.82,"amanecer":.84,"tarde":.92}[situation]+.02*attempt),
                        "--orange-offset-m","-.008",str(.020-.002*attempt),".003",
                        "--orange-offset-m","-.008",".020",".003"]
        if attempt==1 and situation!="base":
            options += ["--orange-closure","contacto"]
        return options
    if crop=="banana":
        return ["--banana-grasps","derecha_calibrada","derecha_despejada"] if attempt else []
    options = ["--height-servo","--extended-lift"]
    if attempt:
        # Adjacent physical grasps bracket the centre of the suspended fruit.
        lateral = .020 if crop=="vineyard" else -.006
        deviation = .003*attempt
        for sign in (0,(-1)**attempt):
            options += ["--field-grasp","derecha","-.008",str(lateral+sign*deviation),".003"]
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"datasets/agriculture/cosecha_surcos_v1")
    parser.add_argument("--evidence",type=Path,default=ROOT/"captures/dataset_ciclo_v1")
    parser.add_argument("--crop",nargs="+",choices=SITES,default=["banana","citrus","vineyard","berries"])
    parser.add_argument("--situation",nargs="+",choices=SITUATIONS,default=list(SITUATIONS))
    parser.add_argument("--attempts",type=int,choices=(1,2,3),default=3,
                        help="Pasadas por condición, con otro reinicio y otra aproximación si falta una demostración")
    args = parser.parse_args()
    root, evidence = args.dataset.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    interpreter = sys.executable
    for attempt in range(args.attempts):
        for situation,crop in ((s,c) for s in args.situation for c in args.crop):
            if complete(root,crop,situation):
                continue
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out = evidence/f"{crop}_{situation}_{stamp}"
            seed = {"base":751,"amanecer":821,"tarde":977}[situation]+attempt
            command = [interpreter,str(ROOT/"tools/film_field_cycle.py"),"--viz","none","--crop",crop,
                       "--situation",situation,"--seed",str(seed),"--output",str(out),"--dataset-root",str(root)]
            command += acquisition_profile(crop,situation,attempt)
            print(f"{crop} / {situation}, intento {attempt+1}: {out}",flush=True)
            with out.with_suffix(".log").open("xb") as stream:
                run = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            if run.returncode:
                print(f"Error de captura: {out.with_suffix('.log')}",flush=True)
            elif not complete(root,crop,situation):
                report = json.loads((out/"resultado.json").read_text(encoding="utf-8"))
                print([(r["robot"],r.get("motion_stop_reason"),r.get("stable_hold_s"),r.get("deposit"))
                       for r in report["robots"]],flush=True)
    missing = [(crop,situation) for crop in SITES for situation in SITUATIONS if not complete(root,crop,situation)]
    unrecorded = [(c,s) for c,s in missing if c in args.crop and s in args.situation]
    if unrecorded:
        raise RuntimeError(f"Faltan demostraciones verificadas en las condiciones solicitadas: {unrecorded}")
    if missing:
        print(f"Condiciones pendientes para publicar el dataset completo: {missing}",flush=True)
        return
    from dexman.data.field_release import field_manifest
    from dexman.experiments import write_json
    manifest = field_manifest(root,evidence)
    write_json(root/"manifest.json",manifest)
    print(root/"manifest.json")


if __name__=="__main__":
    main()
