from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from build123d import Axis, Location
from cadgen.assembly import AssemblyHelper
from cadgen.step_scene import import_step

HERE = Path(__file__).resolve().parent
SUPPORT_SPEC = spec_from_file_location("soporte_ak60_39", HERE / "soporte_ak60_39.step.py")
SUPPORT = module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)
MOTOR_STEP = HERE.parent / "referencias" / "AK60-39-V3.step"
ORIGINAL_FRONT_MOUNT_Z = -10.5


def gen_step():
    assembly = AssemblyHelper("Prueba de encaje AK60-39 V3.0 KV80")
    plate = assembly.add(SUPPORT.gen_step(), "plantilla PLA")
    motor = import_step(MOTOR_STEP).rotate(Axis.X, 180)
    motor = assembly.add(motor, "actuador AK60-39 V3.0 STEP CubeMars")
    seat = assembly.rigid_frame(plate, "cara de asiento del actuador", Location())
    motor_mount = assembly.rigid_frame(
        motor, "cara frontal estacionaria y eje de salida",
        Location((0, 0, -ORIGINAL_FRONT_MOUNT_Z)),
    )
    assembly.connect(seat, motor_mount)
    return assembly.build()
