from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from build123d import Axis, Location
from cadgen.assembly import AssemblyHelper
from cadgen.step_scene import import_step

HERE = Path(__file__).resolve().parent
SUPPORT_SPEC = spec_from_file_location("soporte_st3215", HERE / "soporte_st3215.step.py")
SUPPORT = module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)
SERVO_STEP = HERE.parent / "referencias" / "ST3215-Waveshare.step"
ORIGINAL_SHAFT_X = -25.5
ORIGINAL_REAR_SEAT_Y = -25.6


def gen_step():
    assembly = AssemblyHelper("Prueba de encaje ST3215 SKU 22414")
    plate = assembly.add(SUPPORT.gen_step(), "soporte PLA")
    servo = import_step(SERVO_STEP).rotate(Axis.X, 90)
    servo = assembly.add(servo, "servomotor ST3215 STEP Waveshare")
    seat = assembly.rigid_frame(plate, "cara superior del soporte",
                                Location((0, 0, SUPPORT.PLATE_SIZE[2])))
    rear_mount = assembly.rigid_frame(
        servo, "cara posterior del servo y eje de salida",
        Location((ORIGINAL_SHAFT_X, 0, ORIGINAL_REAR_SEAT_Y)),
    )
    assembly.connect(seat, rear_mount)
    return assembly.build()
