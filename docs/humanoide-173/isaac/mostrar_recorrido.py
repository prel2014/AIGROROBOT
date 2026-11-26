"""Demostración visible del banco Xavier. Ejecutar con el python.bat de Isaac Sim."""
from pathlib import Path
import sys
import traceback

import numpy as np
from controlar import DEDOS, cierre_dedo

BANK = Path(__file__).resolve().with_name("banco_xavier.usda")
ARMS = {"hombro", "codo", "muneca"}
LEGS = {"cadera", "rodilla", "tobillo"}


def fases_dedos(names, limits):
    por_eje = dict(zip(names, limits, strict=True))
    for side, title, sign in (("der", "Mano derecha", 1), ("izq", "Mano izquierda", -1)):
        for digit, label in DEDOS.items():
            key = f"{digit}_{side}"
            neutral = np.zeros(len(names), dtype=np.float32)
            closed = neutral.copy()
            for motor, angle in cierre_dedo(key, 100, por_eje).items():
                closed[names.index(motor)] = angle
            frames = [(0., neutral), (1.2, closed), (1.6, closed), (2.8, neutral), (3.2, neutral)]
            yield key, (f"{title} · {label}", (sign*.236, .025, .875),
                        (sign*2.0, .65, .1), .32, frames)


def choreography(robot):
    names = list(robot.shared_metatype.dof_names)
    limits = robot.get_dof_limits()[0]
    neutral = np.zeros(len(names), dtype=np.float32)
    hand_views = dict(fases_dedos(names, limits))
    phases = list(hand_views.values())

    def pose(group, side):
        target = neutral.copy()
        for i, name in enumerate(names):
            tokens = name.split("_")
            if tokens[0] not in group:
                continue
            lo, hi = limits[i]
            target[i] = hi if tokens[1] == side else lo
        return target

    def alternate(title, group, look_at, height):
        right, left = pose(group, "der"), pose(group, "izq")
        phases.append((title, look_at, (1.4, 1.7, .35), height,
                       [(0, neutral), (2.5, right), (2.9, right), (6.2, left),
                        (6.6, left), (9.1, neutral)]))

    alternate("Brazos · hombros, codos y muñecas", ARMS, (0, .07, 1.04), 2.05)
    alternate("Piernas · caderas, rodillas y tobillos", LEGS, (0, .035, .66), 1.32)
    active = [i for i, name in enumerate(names) if name.split("_")[0] in ARMS | LEGS
              or (name.split("_")[0] in DEDOS and name.endswith(("_m1", "_m2")))]
    if len(active) != 32:
        raise RuntimeError(f"Se esperaban 32 ejes en las extremidades, se encontraron {len(active)}")
    return phases, active, hand_views


def sample(frames, phase_time):
    for (a, start), (b, end) in zip(frames, frames[1:]):
        if phase_time <= b:
            t = (phase_time-a)/(b-a)
            return start + (end-start) * (t*t*(3-2*t))
    return frames[-1][1]


class PanelDedos:
    def __init__(self):
        import omni.ui as ui
        self.manual = False
        self.restart = False
        self.reflecting = False
        self.requested = {}
        self.selected = None
        self.models = {}
        self.window = ui.Window("Control de dedos", width=430, height=196)
        self.window.deferred_dock_in("Content", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        with self.window.frame:
            with ui.VStack(spacing=3):
                with ui.HStack(height=24, spacing=6):
                    ui.Button("Control manual", clicked_fn=self.freeze)
                    ui.Button("Reproducir secuencia", clicked_fn=self.play)
                with ui.HStack(height=18):
                    ui.Spacer(width=72)
                    for title in ("Derecha (%)", "Izquierda (%)"):
                        ui.Label(title, alignment=ui.Alignment.CENTER)
                for digit, label in DEDOS.items():
                    with ui.HStack(height=20, spacing=5):
                        ui.Label(label, width=70)
                        for side in ("der", "izq"):
                            key = f"{digit}_{side}"
                            model = ui.IntSlider(min=0, max=100).model
                            self.models[key] = model
                            model.add_value_changed_fn(lambda m, k=key: self.changed(k, m.as_int))

    def freeze(self):
        self.manual = True

    def play(self):
        self.manual = False
        self.restart = True
        self.selected = None
        self.requested.clear()

    def changed(self, key, value):
        if not self.reflecting:
            self.manual = True
            self.selected = key
            self.requested[key] = value

    def reflect(self, pose, first_motor):
        self.reflecting = True
        try:
            for key, (i, end) in first_motor.items():
                value = round(float(100 * pose[i]/end))
                if value != self.models[key].as_int:
                    self.models[key].set_value(value)
        finally:
            self.reflecting = False


def run(app):
    import omni.physics.tensors as tensors
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.kit.viewport.utility import get_active_viewport_window
    from omni.physx import get_physx_interface
    from pxr import Gf, Sdf, UsdGeom

    context = omni.usd.get_context()
    stage = context.get_stage()
    if not stage or not stage.GetPrimAtPath("/World/Humanoide/Anclaje"):
        raise RuntimeError("La ventana de Isaac Sim no cargó la escena de banco")
    stage_id = context.get_stage_id()
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.SetStartTimeCode(0.)
    stage.SetEndTimeCode(3600. * stage.GetTimeCodesPerSecond())
    SimulationManager.set_default_physics_scene("/World/Fisica")
    SimulationManager.set_device("cpu")
    get_physx_interface().set_thread_count(1)
    viewport_window = get_active_viewport_window()
    viewport = viewport_window.viewport_api
    camera = UsdGeom.Camera.Define(stage, "/World/CamaraRecorrido")
    camera.CreateClippingRangeAttr((.015, 100.))
    camera.CreateFocalLengthAttr(28.)
    camera.CreateHorizontalApertureAttr(36.)
    transform = camera.AddTransformOp()
    panel = PanelDedos()
    with viewport_window.get_frame("recorrido_xavier"):
        with ui.VStack():
            ui.Spacer()
            with ui.HStack(height=32):
                ui.Spacer()
                with ui.ZStack(width=322):
                    ui.Rectangle(style={"background_color": 0xD5272421, "border_radius": 5})
                    label = ui.Label("Humanoide Xavier", alignment=ui.Alignment.CENTER,
                                     style={"font_size": 17, "color": 0xFFFFFFFF})
                ui.Spacer()
            ui.Spacer(height=12)

    def frame_camera(phase):
        title, center, direction, height, _ = phase
        w, h = viewport.get_texture_resolution()
        aspect = w / h if w > 0 and h > 0 else 16/9
        camera.CreateVerticalApertureAttr(36./aspect)
        aim = np.asarray(center)
        delta = np.asarray(direction)
        eye = aim + delta / np.linalg.norm(delta) * (height * 28. * aspect / 36.)
        transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*aim), Gf.Vec3d.ZAxis()).GetInverse())
        viewport.camera_path = Sdf.Path("/World/CamaraRecorrido")
        label.text = title

    elapsed = 0.
    def clock(dt):
        nonlocal elapsed
        elapsed += dt

    subscription = get_physx_interface().subscribe_physics_step_events(clock)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(3600.)
    timeline.set_looping(True)
    timeline.play()
    run_id = np.array([0], dtype=np.uint32)
    robot = None
    last_phase = None
    measured = False
    while app.is_running():
        app.update()
        if context.get_stage_id() != stage_id:
            break
        if timeline.is_stopped():
            robot, last_phase, elapsed = None, None, 0.
            continue
        if not timeline.is_playing() or elapsed == 0:
            continue
        if robot is None:
            sim = tensors.create_simulation_view("numpy", stage_id=stage_id, backend="physx")
            robot = sim.create_articulation_view("/World/Humanoide/*")
            if robot.count != 1 or not robot.shared_metatype.fixed_base:
                raise RuntimeError("PhysX no encontró la articulación del banco")
            phases, active, hand_views = choreography(robot)
            names = list(robot.shared_metatype.dof_names)
            limits = dict(zip(names, robot.get_dof_limits()[0], strict=True))
            indices = {name: i for i, name in enumerate(names)}
            first_motor = {}
            for key in panel.models:
                name = key + "_m1"
                first_motor[key] = (indices[name], cierre_dedo(key, 100, limits)[name])
            speeds = robot.get_dof_max_velocities()[0, active]
            sent = robot.get_dof_position_targets()[0].copy()
            last_time = elapsed
            ends = np.cumsum([phase[-1][-1][0] for phase in phases])
            qmin = robot.get_dof_positions()[0, active].copy()
            qmax = qmin.copy()
            start_time = elapsed
            scene = stage.GetPrimAtPath("/World/Fisica")
            broadphase = scene.GetAttribute("physxScene:broadphaseType").Get()
            if scene.GetAttribute("physxScene:enableGPUDynamics").Get() or not SimulationManager.get_device().is_cpu:
                raise RuntimeError("El banco requiere PhysX en CPU")
            print(f"XAVIER: PhysX en CPU, broadphase {broadphase}, {len(names)} ejes en el banco", flush=True)
        if panel.restart:
            start_time = elapsed
            last_phase = None
            panel.restart = False
        cycle_time = elapsed - start_time
        t = cycle_time % ends[-1]
        which = int(np.searchsorted(ends, t, side="right"))
        phase = phases[which]
        if panel.manual:
            # Conservar todos los objetivos que el usuario no haya cambiado.
            pose = sent.copy()
            for key, percent in panel.requested.items():
                for name, target in cierre_dedo(key, percent, limits).items():
                    pose[indices[name]] = target
            if last_phase != ("manual", panel.selected):
                if panel.selected is not None:
                    manual_phase = hand_views[panel.selected]
                    frame_camera(manual_phase)
                    label.text = "Manual · " + manual_phase[0]
                else:
                    label.text = "Control manual de dedos"
                last_phase = ("manual", panel.selected)
        else:
            if which != last_phase:
                frame_camera(phase)
                if not measured:
                    print("XAVIER:", phase[0], flush=True)
                last_phase = which
            pose = sample(phase[-1], t - (ends[which-1] if which else 0.))
        max_step = speeds * max(0., elapsed-last_time)
        sent[active] += np.clip(pose[active]-sent[active], -max_step, max_step)
        last_time = elapsed
        robot.set_dof_position_targets(sent[None, :], run_id)
        if not panel.manual:
            panel.reflect(sent, first_motor)
        if not panel.manual and not measured:
            actual = robot.get_dof_positions()[0, active]
            qmin = np.minimum(qmin, actual)
            qmax = np.maximum(qmax, actual)
            if cycle_time >= ends[-1]:
                ranges = dict(zip((robot.shared_metatype.dof_names[i] for i in active),
                                  np.rad2deg(qmax-qmin).round(1).tolist()))
                print("XAVIER: recorrido medido en grados:", ranges, flush=True)
                measured = True
    subscription = None
    panel.window.destroy()
    if context.get_stage_id() != stage_id:
        while app.is_running():
            app.update()


if __name__ == "__main__":
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False, "hide_ui": False, "multi_gpu": False,
                         "width": 960, "height": 640, "window_width": 1480, "window_height": 900,
                         "open_usd": str(BANK)})
    status = 0
    try:
        run(app)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    sys.exit(status)
