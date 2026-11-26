"""Comprueba el movimiento independiente de las diez almohadillas distales en el banco."""
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, UsdGeom

from controlar import DEDOS
from mostrar_recorrido import fases_dedos, sample


def verificar(stage, robot, step):
    names = list(robot.shared_metatype.dof_names)
    links = list(robot.shared_metatype.link_names)
    keys = [f"{digit}_{side}" for side in ("der", "izq") for digit in DEDOS]
    limits = robot.get_dof_limits()[0]
    motors = np.array([[names.index(f"{key}_m{m}") for m in (1, 2)] for key in keys])
    palms = [links.index("mano_" + key.split("_")[1]) for key in keys]
    distal = [links.index(key + "_distal") for key in keys]
    offsets = []
    for key in keys:
        mesh = UsdGeom.Mesh.Get(stage, f"/World/Humanoide/{key}_distal/Contacto")
        center = Gf.Vec3d(*np.mean(mesh.GetExtentAttr().Get(), axis=0).tolist())
        offsets.append(mesh.GetLocalTransformation().Transform(center))

    def pad_centers():
        poses = robot.get_link_transforms()[0]
        pads = poses[distal, :3] + Rotation.from_quat(poses[distal, 3:]).apply(offsets)
        return Rotation.from_quat(poses[palms, 3:]).inv().apply(pads - poses[palms, :3])

    ids = np.array([0], dtype=np.uint32)
    step(192)  # Asentar las manos abiertas bajo gravedad antes de tomar la referencia.
    passed = 0
    for key, phase in fases_dedos(names, limits):
        i = keys.index(key)
        others = np.arange(len(keys)) != i
        frames = phase[-1]
        reference = pad_centers()
        rest = robot.get_dof_positions()[0, motors].copy()
        max_other_mm = max_other_deg = 0.
        close_error = close_mm = float("nan")
        # Muestreo a 30 Hz con ocho pasos PhysX de 1/240 s entre lecturas.
        steps = round(frames[-1][0] * 240)
        closed_tick = round(frames[2][0] * 240)
        for tick in range(0, steps, 8):
            count = min(8, steps-tick)
            target = sample(frames, (tick + count/2) / 240)
            if np.any(target[motors[others]] != 0.):
                raise AssertionError(f"{key}: la secuencia envía consignas a otros dedos")
            robot.set_dof_position_targets(target[None, :], ids)
            step(count)
            delta_mm = 1000 * np.linalg.norm(pad_centers() - reference, axis=1)
            actual = robot.get_dof_positions()[0, motors]
            if not np.isfinite(delta_mm).all() or not np.isfinite(actual).all():
                raise RuntimeError(f"{key}: lectura física no finita")
            max_other_mm = max(max_other_mm, delta_mm[others].max())
            max_other_deg = max(max_other_deg, np.rad2deg(np.abs(actual[others]-rest[others])).max())
            if tick + count == closed_tick:
                close_error = np.rad2deg(abs(actual[i] - frames[2][1][motors[i]])).max()
                close_mm = delta_mm[i]
        return_mm = delta_mm[i]
        open_error = np.rad2deg(abs(actual[i])).max()
        ok = (close_mm > 3 and close_error < 3 and open_error < 2 and return_mm < 1
              and max_other_mm < 1 and max_other_deg < 1)
        passed += int(ok)
        print(f"{key}: {'OK' if ok else 'REVISAR'} | almohadilla {close_mm:.2f} mm"
              f" | otros nueve {max_other_mm:.3f} mm / {max_other_deg:.3f} grados"
              f" | retorno {return_mm:.3f} mm | error de servos al cerrar/abrir "
              f"{close_error:.2f}/{open_error:.2f} grados", flush=True)
    if passed != 10:
        raise RuntimeError(f"Superaron la prueba de independencia {passed}/10 dedos")
    print("Independencia verificada en los 10 dedos.", flush=True)
