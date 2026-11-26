"""Control de posición en el banco de Isaac Sim. Las consignas de la CLI se dan en grados."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
DEDOS = {"indice": "Índice", "medio": "Medio", "anular": "Anular", "menique": "Meñique", "pulgar": "Pulgar"}


def cierre_dedo(nombre, porcentaje, limites):
    """Consigna de las dos palancas en las unidades de los límites de entrada."""
    dedo, _, lado = nombre.rpartition("_")
    if dedo not in DEDOS or lado not in ("der", "izq"):
        raise ValueError(f"Dedo desconocido: {nombre}. Ejemplo: indice_der.")
    if not 0 <= porcentaje <= 100:
        raise ValueError(f"{nombre}: el cierre debe estar entre 0 y 100 %.")
    result = {}
    for motor in (1, 2):
        eje = f"{nombre}_m{motor}"
        bajo, alto = limites[eje]
        result[eje] = (bajo if abs(bajo) > abs(alto) else alto) * porcentaje / 100
    return result


@contextmanager
def simulation(stage):
    import omni.physics.tensors as tensors
    from omni.physx import get_physx_simulation_interface
    from pxr import UsdUtils

    cache = UsdUtils.StageCache.Get()
    handle = cache.Insert(stage)
    stage_id = handle.ToLongInt()
    physics = get_physx_simulation_interface()
    physics.attach_stage(stage_id)
    dt = 1 / 240
    time = 0.

    def step(count):
        nonlocal time
        for _ in range(count):
            physics.simulate(dt, time)
            physics.fetch_results()
            time += dt

    try:
        step(1)
        sim = tensors.create_simulation_view("numpy", stage_id=stage_id, backend="physx")
        robot = sim.create_articulation_view("/World/Humanoide/*")
        if robot.count != 1:
            raise RuntimeError(f"Se esperaba una articulación y PhysX encontró {robot.count}")
        yield robot, step
    finally:
        physics.detach_stage()
        cache.Erase(handle)


def prueba_base_libre():
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    bench_layer = Sdf.Layer.FindOrOpen(str(HERE / "banco_xavier.usda"))
    for path in ("/World/Fisica", "/World/Suelo"):
        Sdf.CopySpec(bench_layer, path, stage.GetRootLayer(), path)
    prim = UsdGeom.Xform.Define(stage, "/World/Humanoide")
    prim.GetPrim().GetReferences().AddReference(str(HERE / "humanoide_xavier_isaac.usd"))
    prim.AddTranslateOp().Set((0, 0, .035))
    with simulation(stage) as (robot, step):
        assert not robot.shared_metatype.fixed_base
        ids = [robot.shared_metatype.link_names.index("pie_" + side) for side in ("izq", "der")]
        z0 = robot.get_link_transforms()[0, ids, 2].copy()
        step(120)
        z = robot.get_link_transforms()[0, ids, 2]
        print("Base libre, centros de los pies tras 0,5 s (m):", z.tolist(), flush=True)
        if not np.all((z0-z > .02) & (abs(z-.029) < .006)):
            raise RuntimeError("La caída y el contacto de los pies difieren de las cotas esperadas.")


def controlar(args, stage, robot, step):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from pxr import UsdPhysics

    assert robot.shared_metatype.fixed_base
    names = list(robot.shared_metatype.dof_names)
    indices = np.array([0], dtype=np.uint32)
    drives = {}
    radians = robot.get_dof_limits()[0]
    speeds = robot.get_dof_max_velocities()[0]
    efforts = robot.get_dof_max_forces()[0]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            name = prim.GetName()
            i = names.index(name)
            joint = UsdPhysics.RevoluteJoint(prim)
            limits = np.array([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
            np.testing.assert_allclose(np.rad2deg(radians[i]), limits, atol=.001)
            np.testing.assert_allclose(speeds[i], np.deg2rad(prim.GetAttribute("physxJoint:maxJointVelocity").Get()), rtol=1e-5)
            np.testing.assert_allclose(efforts[i], UsdPhysics.DriveAPI(prim, "angular").GetMaxForceAttr().Get(), rtol=1e-5)
            drives[name] = (i, *limits)
    print(f"PhysX: {robot.max_links} cuerpos y {len(drives)} ejes con mando.", flush=True)

    def send(commands):
        targets = robot.get_dof_position_targets().copy()
        for name, degrees in commands.items():
            if name not in drives:
                raise ValueError(f"Eje desconocido: {name}. Usa --listar.")
            i, low, high = drives[name]
            if not np.isfinite(degrees) or not low <= degrees <= high:
                raise ValueError(f"{name}: {degrees} fuera de [{low:g}, {high:g}] grados.")
            targets[0, i] = np.deg2rad(degrees)
        robot.set_dof_position_targets(targets, indices)

    if args.listar:
        for name, (i, low, high) in drives.items():
            print(f"{name:21} [{low:5g}, {high:5g}] grados   {efforts[i]:6.2f} N·m")
        return
    if args.verificar_dedos:
        from verificar_dedos import verificar
        verificar(stage, robot, step)
        return
    if args.verificar:
        commands = {name: (high if high > 0 else low) * .40 for name, (_, low, high) in drives.items()}
        send(commands)
        step(480)
        q = np.rad2deg(robot.get_dof_positions()[0])
        errors = {name: round(float(q[drives[name][0]] - target), 2) for name, target in commands.items()}
        print("Error de seguimiento en grados:", errors, flush=True)
        if not np.isfinite(robot.get_link_transforms()).all() or max(map(abs, errors.values())) > 6:
            raise RuntimeError("Error de seguimiento o estado no finito.")
        poses = dict(zip(robot.shared_metatype.link_names, robot.get_link_transforms()[0], strict=True))
        closure_errors = []
        for prim in stage.Traverse():
            joint = UsdPhysics.Joint(prim)
            if not joint or not joint.GetExcludeFromArticulationAttr().Get():
                continue
            anchors = []
            for body, pos in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                              (joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
                pose = poses[body.GetTargets()[0].name]
                anchors.append(pose[:3] + Rotation.from_quat(pose[3:]).apply(pos.Get()))
            closure_errors.append(float(np.linalg.norm(anchors[1] - anchors[0])))
        print(f"Error máximo de los {len(closure_errors)} cierres: {1000*max(closure_errors):.3f} mm.", flush=True)
        if len(closure_errors) != 30 or max(closure_errors) > .001:
            raise RuntimeError("La separación entre anclajes del mecanismo supera 1 mm.")
        # Saltar el filtro de consignas únicamente en esta prueba del tope del solver.
        i, low, high = drives["cuello"]
        for outside, stop in ((high + 25, high), (low - 25, low)):
            targets = robot.get_dof_position_targets().copy()
            targets[0, i] = np.deg2rad(outside)
            robot.set_dof_position_targets(targets, indices)
            step(360)
            actual = np.rad2deg(robot.get_dof_positions()[0, i])
            print(f"Tope de cuello: orden {outside:.0f}, lectura {actual:.3f} grados.", flush=True)
            if abs(actual - stop) > .5:
                raise RuntimeError("El cuello excede la tolerancia del tope.")
    else:
        commands = {}
        limites = {nombre: (bajo, alto) for nombre, (_, bajo, alto) in drives.items()}
        for side in ("izq", "der"):
            percent = getattr(args, "mano_" + side)
            if percent is not None:
                for dedo in DEDOS:
                    commands.update(cierre_dedo(f"{dedo}_{side}", percent, limites))
        for pair in args.dedo:
            nombre, valor = pair.split("=", 1)
            commands.update(cierre_dedo(nombre, float(valor), limites))
        for pair in args.eje:
            name, value = pair.split("=", 1)
            commands[name] = float(value)
        commands = commands or {"codo_der": 30.}
        send(commands)
        step(480)
        q = np.rad2deg(robot.get_dof_positions()[0])
        for name, target in commands.items():
            print(f"{name}: consigna {target:.2f}°, lectura {q[drives[name][0]]:.2f}°")


def main(args):
    from omni.physx import get_physx_interface
    from pxr import Usd
    stage = Usd.Stage.Open(str(HERE / "banco_xavier.usda"))
    get_physx_interface().set_thread_count(1)
    with simulation(stage) as (robot, step):
        controlar(args, stage, robot, step)
    if args.verificar:
        prueba_base_libre()
        print("Verificación dinámica completada.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eje", action="append", default=[], metavar="NOMBRE=GRADOS")
    parser.add_argument("--mano-izq", type=float, metavar="PORCENTAJE")
    parser.add_argument("--mano-der", type=float, metavar="PORCENTAJE")
    parser.add_argument("--dedo", action="append", default=[], metavar="DEDO_LADO=PORCENTAJE")
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument("--verificar-dedos", action="store_true")
    args, kit_args = parser.parse_known_args()
    sys.argv[1:] = kit_args
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False, "disable_viewport_updates": True})
    status = 0
    try:
        main(args)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    sys.exit(status)
