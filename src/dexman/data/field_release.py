"""Verify complete field cycles and freeze whole scene conditions into a portable release."""
import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from dexman.experiments import sha256
from dexman.tasks.agriculture.harvest_route import FIELD_PHASE_INSTRUCTIONS, SITES
from dexman.tasks.agriculture.rover_basket_task import complete_field_cycle
from .manifest import resolve_manifest
from .validation import inspect_episode


PARTITION = {"base": "train", "amanecer": "validation", "tarde": "test"}
REQUIRED_PHASES = {"recorrido", "estacionamiento", "aproximacion", "cierre", "elevacion",
                   "sosten", "traslado", "apertura", "retirada"}


def video_frames(path):
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("Se requiere ffprobe para verificar la duración de las dos vistas")
    shim = Path(executable).with_suffix(".shim")
    if shim.is_file():
        for line in shim.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip()=="path":
                executable = value.strip().strip('"')
                break
    result = subprocess.run([executable,"-v","error","-select_streams","v:0",
                             "-show_entries","stream=codec_name,avg_frame_rate,nb_frames",
                             "-of","json",str(path)],capture_output=True,text=True,check=True)
    stream, = json.loads(result.stdout)["streams"]
    if stream["codec_name"]!="h264" or stream["avg_frame_rate"]!="15/1":
        raise ValueError(f"Vídeo fuera del perfil H.264 a 15 Hz: {path}")
    return int(stream["nb_frames"])


def verify_detachment(source, index, report):
    rows = json.loads((source/"mediciones.json").read_text(encoding="utf-8"))
    observations = [(tick,phase,robots[index]) for tick,phase,robots in rows
                    if phase not in ("recorrido","estacionamiento","detenido")]
    columns = report["columns"]["manipulacion"]
    stem, held = columns.index("stem_attached"), columns.index("held")
    first = next(i for i,(_,_,v) in enumerate(observations) if not v[stem])
    if first<3 or not all(v[held] for _,_,v in observations[first-3:first]):
        raise ValueError(f"El fruto se desprendió sin agarre previo de la réplica {index}: {source}")
    released = next(t for t,phase,_ in observations if phase=="apertura")
    start = observations[first][0]
    if any(not v[held] for t,_,v in observations if start<=t<released):
        raise ValueError(f"Se perdió el agarre antes de abrir la mano: {source}, réplica {index}")
    sub = json.loads((source/"subpasos.json").read_text(encoding="utf-8"))
    carrying = [(t,values) for t,_,values in sub["samples"] if start<=t<released]
    decimation = round(report["physics_hz"]/report["control_hz"])
    if ([t for t,_ in carrying]!=list(range(start,released)) or not carrying
            or any(len(values)!=decimation for _,values in carrying)):
        raise ValueError(f"Faltan pasos físicos entre el desprendimiento y la apertura: {source}")
    for t,values in carrying:
        for sample in values:
            measured = sample[index]
            if not measured[0] and ("stem_attached" not in sub["columns"] or not measured[sub["columns"].index("stem_attached")]):
                raise ValueError(f"Se perdió oposición en el paso físico {t} de {source}, réplica {index}")
    if report["crop"]=="vineyard":
        from dexman.tasks.agriculture.cluster_grasp import cluster_grip
        from dexman.tasks.agriculture.harvest_route import SITES
        with np.load(source/"contactos_racimo.npz") as native:
            ticks = np.asarray(native["control_step"])
            all_pairs = native["forces_and_points"]
            expected_ticks = [t for t,_,values in sub["samples"] for _ in values]
            if (not np.array_equal(ticks,expected_ticks) or all_pairs.shape[:2]!=(len(ticks),len(report["robots"]))
                    or all_pairs.shape[2:]!=(5,6) or float(native["mass_kg"])!=SITES["vineyard"].mass
                    or float(native["friction"])!=.60):
                raise ValueError(f"Las fuerzas del racimo no corresponden a los pasos físicos: {source}")
            attached = [v[index][sub["columns"].index("stem_attached")] for _,_,values in sub["samples"] for v in values]
            carry = (ticks>=start) & (ticks<released) & ~np.asarray(attached,dtype=bool)
            for row in all_pairs[carry,index]:
                if not cluster_grip(row[:,:3],row[:,3:],SITES["vineyard"].mass,.60):
                    raise ValueError(f"Los tres dedos no equilibran el racimo desprendido: {source}")


def field_manifest(root, evidence, *, snapshot=False):
    root, evidence = Path(root).resolve(strict=True), Path(evidence).resolve(strict=True)
    if any(root.rglob("*.partial")):
        raise ValueError("Hay una adquisición incompleta en el dataset")
    splits = {part: [] for part in PARTITION.values()}
    coverage = {part: set() for part in splits}
    contracts, provenance = {}, {}
    media = {}
    recoveries = []
    for path in sorted(root.rglob("*.h5")):
        row = inspect_episode(path)
        run = path.parent.parent
        source = evidence / run.name
        report = json.loads((source / "resultado.json").read_text(encoding="utf-8"))
        with h5py.File(path, "r") as episode:
            crop, index = str(episode.attrs["crop"]), int(episode.attrs["replica_id"])
            situation = str(episode.attrs.get("situation", "base"))
            robot = next(r for r in report["robots"] if r["robot"] == index)
            recovery = path.relative_to(root).parts[0]=="recuperaciones"
            phase_ids = episode["phase"][:]
            names = list(FIELD_PHASE_INSTRUCTIONS)
            phases = {names[i] for i in np.unique(phase_ids)}
            boundaries = np.r_[0,1+np.flatnonzero(phase_ids[1:]!=phase_ids[:-1]),len(phase_ids)]
            orders = json.loads(episode.attrs["subtasks_by_phase"])
            row["instruction_es"] = str(episode.attrs["instruction"])
            row["subtasks"] = [{"phase":names[phase_ids[a]],"instruction_es":orders[str(phase_ids[a])],
                               "start_sample":int(a),"stop_sample_exclusive":int(b)}
                              for a,b in zip(boundaries[:-1],boundaries[1:],strict=True)]
            wheel = episode["wheel_command_rad_s"][:]
            if (row["success"]==recovery or row["exercise"] != "cosecha_en_surco" or crop != report["crop"]
                    or situation != report.get("situation", "base") or episode.attrs["instruction"] != report["instruction"]
                    or row["seed"] != report["seed"] or json.loads(episode.attrs["physical_result"]) != robot
                    or not REQUIRED_PHASES <= phases or wheel.shape != (row["samples"],4)
                    or not np.isfinite(wheel).all() or not np.any(np.abs(wheel[episode["phase"][:]==0]) > .1)):
                raise ValueError(f"Trayectoria, sincronización u orden incompatible en {path}")
            if not complete_field_cycle(robot):
                raise ValueError(f"Cosecha o visión binocular sin verificar en {path}")
            surface = robot["surface_audit"]
            row["measurements"] = {key:surface[key] for key in ("peak_lift_m","stable_hold_s",
                "held_delivery_ticks","delivery_ticks","final_basket_rest_s")}
            row["measurements"]["travel_m"] = robot["travel_m"]
            if recovery:
                ticks = episode.attrs["recovery_ticks"].tolist()
                observations = json.loads((source/"mediciones.json").read_text(encoding="utf-8"))
                lost = [(t,phase,v[index]) for t,phase,v in observations if len(v[index])>10 and not v[index][12]
                        and phase not in ("apertura","retirada") and not v[index][1]]
                if (episode.attrs.get("outcome")!="recovery_before_lift" or not lost or [t for t,_,_ in lost]!=ticks
                        or not all(phase=="recuperacion" and v[9] for _,phase,v in lost)):
                    raise ValueError(f"Recuperación sin evidencia de contacto: {path}")
                row["recovery_ticks"] = ticks
            else:
                verify_detachment(source,index,report)
        # All replicas from the same geometric/lighting condition stay together, including retries.
        row["path"] = path.relative_to(root).as_posix()
        row["scene_reset"] = [crop, situation]
        row["situation"] = situation
        part = PARTITION[situation]
        if recovery:
            recoveries.append(row)
        else:
            splits[part].append(row)
            coverage[part].add(crop)
        contracts[json.dumps(row["contract"], sort_keys=True)] = row["contract"]
        artifacts = ["resultado.json", "mediciones.json", "contactos.json", "subpasos.json", "pares.json",
                     "poses_fisicas.npz", "visibilidad.json", "rayos.json", "superficie_frente_al_fruto.json",
                     "pixeles.json", f"robot_{index:03d}.mp4", f"robot_{index:03d}_ojos.mp4"]
        if report.get("orange_task")=="roca":
            artifacts.append("ruta.json")
        if crop=="vineyard":
            artifacts.append("contactos_racimo.npz")
        for calibration in sorted((source/"calibraciones").glob("*.json")):
            if sha256(calibration)!=report["source_sha256"].get(calibration.name):
                raise ValueError(f"La calibración difiere de la utilizada en el ensayo: {calibration}")
            artifacts.append(calibration.relative_to(source).as_posix())
        for name in artifacts:
            src = source/name
            if not src.is_file() or src.stat().st_size==0:
                raise ValueError(f"Falta evidencia del episodio: {src}")
            if src.suffix==".mp4" and not (video_frames(src)==report["frames"]==(row["samples"]+1)//2):
                raise ValueError(f"El MP4 y las muestras de control corresponden a duraciones distintas: {src}")
            dest = run/"evidencia"/name
            media[dest] = src
        provenance[row["path"]] = [(run/"evidencia"/a).relative_to(root).as_posix() for a in artifacts]
    missing = {part: sorted(set(SITES)-seen) for part,seen in coverage.items() if seen!=set(SITES)}
    if (missing and not snapshot) or len(contracts)!=1:
        raise ValueError(f"Dataset sin cobertura completa en las particiones: {missing}")
    manifest = dict(version=1,dataset_root=str(root),complete=not bool(missing),missing_crops_by_split=missing,
                    split_unit="crop_and_field_situation",splits=splits,
                    robot_contract=next(iter(contracts.values())),evidence=provenance,recovery_examples=recoveries)
    resolve_manifest(manifest)
    manifest["evidence_sha256"] = {}
    for dest,src in media.items():
        dest.parent.mkdir(parents=True,exist_ok=True)
        digest = sha256(src)
        if not dest.exists():
            temporary = dest.with_name(f".{dest.name}.{uuid4().hex}.tmp")
            try:
                shutil.copy2(src,temporary)
                if sha256(temporary)!=digest:
                    raise ValueError(f"Copia de evidencia incompleta: {src}")
                temporary.replace(dest)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(dest)!=digest:
            raise ValueError(f"La evidencia publicada difiere del original: {dest}")
        manifest["evidence_sha256"][dest.relative_to(root).as_posix()] = digest
    return manifest
