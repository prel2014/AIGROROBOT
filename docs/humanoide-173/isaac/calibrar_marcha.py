"""Ajusta las ganancias de las piernas con poblaciones de 16 Xavier en PhysX."""
import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from control_marcha import GANANCIAS_MARCHA, GANANCIAS_REPOSO, MARCHA, corregir_apoyos, referencias
from entrenar_marcha import ARTICULACIONES, DT, PISTA, SUBPASOS_EQUILIBRIO


class Banco:
    def __init__(self):
        import omni.physics.tensors as tensors
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from pxr import Usd, UsdUtils

        self.stage = Usd.Stage.Open(str(PISTA))
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.stage.GetPrimAtPath('/World/Suelo').GetAttribute('xformOp:scale').Set((100., 100., .07))
        self.cache = UsdUtils.StageCache.Get()
        self.stage_id = self.cache.Insert(self.stage)
        self.fisica = get_physx_simulation_interface()
        get_physx_interface().set_thread_count(2)
        self.fisica.attach_stage(self.stage_id.ToLongInt())
        self.reloj = 0.
        self.avanzar(1)
        self.vista = tensors.create_simulation_view('numpy', stage_id=self.stage_id.ToLongInt(), backend='physx')
        self.robot = self.vista.create_articulation_view('/World/Xavier_*/pelvis')
        r = self.robot
        if r.count != 16 or r.shared_metatype.fixed_base:
            raise RuntimeError('La calibración requiere 16 Xavier de base libre.')
        self.ids = np.arange(r.count, dtype=np.uint32)
        self.juntas = [r.shared_metatype.dof_names.index(n) for n in ARTICULACIONES]
        self.pies = [r.shared_metatype.link_names.index('pie_'+lado) for lado in ('izq', 'der')]
        self.neutro = r.get_dof_position_targets().copy()
        self.inicio = r.get_root_transforms().copy()
        self.limites = r.get_dof_limits()[0, self.juntas]
        self.velocidad = r.get_dof_max_velocities()[0, self.juntas]
        self.kp = r.get_dof_stiffnesses().copy()
        self.kd = r.get_dof_dampings().copy()

    def avanzar(self, pasos):
        for _ in range(pasos):
            self.fisica.simulate(DT, self.reloj)
            self.fisica.fetch_results()
            self.reloj += DT

    def ganancias(self, kp, kd):
        self.kp[:, self.juntas] = np.tile(kp, (1, 2))
        self.kd[:, self.juntas] = np.tile(kd, (1, 2))
        self.robot.set_dof_stiffnesses(self.kp, self.ids)
        self.robot.set_dof_dampings(self.kd, self.ids)

    def medir(self, candidatos, duracion, espera, semilla):
        r = self.robot
        rng = np.random.default_rng(semilla)
        yaw = (self.ids % 4) * np.pi / 2
        frente = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        lado = frente[:, ::-1] * [1., -1.]
        root = self.inicio.copy()
        root[:, :2] = (frente * (4.5 + rng.uniform(-.14, .14, (16, 1)))
                       + lado * ((self.ids.astype(float)//4 - 1.5)*1.45)[:, None])
        root[:, 2] = 1.173
        root[:, 3:] = Rotation.from_euler('z', yaw).as_quat()
        self.ganancias(*GANANCIAS_REPOSO)
        r.set_root_transforms(root, self.ids)
        r.set_root_velocities(np.zeros((16, 6), np.float32), self.ids)
        r.set_dof_positions(self.neutro, self.ids)
        r.set_dof_velocities(np.zeros_like(self.neutro), self.ids)
        orden = self.neutro.copy()
        r.set_dof_position_targets(orden, self.ids)
        self.avanzar(round(espera / DT))
        self.ganancias(candidatos[:, :3], candidatos[:, 3:])
        posicion_inicial = r.get_link_transforms()[:, self.pies, :2].mean(axis=1)
        fase = np.zeros(16)
        vivos = np.ones(16, bool)
        tiempo = np.zeros(16)
        avance = np.zeros(16)
        intervalo = SUBPASOS_EQUILIBRIO * DT
        parametros = np.tile(MARCHA, (16, 1))
        for i in range(round(duracion / intervalo)):
            t = i * intervalo
            if i % round(.05/intervalo) == 0:
                accion = np.tanh(rng.normal(0., .56, (16, 7)))
            ajuste = np.clip(t - 2., 0., 1.)
            fase += 2*np.pi*MARCHA[0]*intervalo*(1.+.012*ajuste*accion[:, 6])
            pose = r.get_root_transforms()
            vel = r.get_root_velocities()
            angulo = Rotation.from_quat(pose[:, 3:])
            vertical = angulo.as_matrix()[:, 2, :]
            pies = r.get_link_transforms()[:, self.pies, :2].mean(axis=1)
            amplitud = np.full(16, min(t/.7, 1.))
            orden_piernas = referencias(parametros, fase, amplitud, vertical, angulo.inv().apply(vel[:, 3:]),
                                        r.get_dof_positions()[:, self.juntas], r.get_dof_velocities()[:, self.juntas])
            corregir_apoyos(orden_piernas, fase, amplitud, ((pose[:, :2]-pies)*frente).sum(axis=1),
                            (vel[:, :2]*frente).sum(axis=1))
            orden_piernas += np.deg2rad(.14 * ajuste * accion[:, :6])
            deseado = np.clip(orden_piernas, self.limites[:, 0], self.limites[:, 1])
            orden[:, self.juntas] += np.clip(deseado-orden[:, self.juntas],
                                              -self.velocidad*intervalo, self.velocidad*intervalo)
            r.set_dof_position_targets(orden, self.ids)
            self.avanzar(SUBPASOS_EQUILIBRIO)
            pose = r.get_root_transforms()
            pies = r.get_link_transforms()[:, self.pies, :2].mean(axis=1)
            delta = pies - posicion_inicial
            vivos &= ((pose[:, 2] > .85) & (Rotation.from_quat(pose[:, 3:]).as_matrix()[:, 2, 2] > .8)
                      & (abs((delta*lado).sum(axis=1)) < .65))
            tiempo += intervalo * vivos
            avance[vivos] = (delta*frente).sum(axis=1)[vivos]
            if not vivos.any():
                break
        print(f'Espera {espera:g} s · en pie {vivos.sum()}/16 · avance mínimo {avance.min():.2f} m '
              f'· mediana {np.median(avance):.2f} m', flush=True)
        return tiempo + 2*vivos + avance

    def cerrar(self):
        self.fisica.detach_stage()
        self.cache.Erase(self.stage_id)


def ajustar(banco, args):
    rng = np.random.default_rng(args.semilla)
    media = np.concatenate(GANANCIAS_MARCHA)
    dispersion = np.array([40., 40., 22., 1.7, 1.4, .9])
    desv = dispersion.copy()
    for g in range(args.generaciones):
        lote = np.tile(media, (16, 1))
        if not args.validar:
            lote[1:] = np.clip(rng.normal(media, desv, (15, 6)), [250, 250, 120, 4, 3, 2],
                               [1200, 1100, 700, 45, 38, 22])
        resultados = [banco.medir(lote, args.segundos, espera, args.semilla + 2*g + i)
                      for i, espera in enumerate((.7, 12.3))]
        if not args.validar:
            notas = .8*np.min(resultados, axis=0) + .2*np.mean(resultados, axis=0)
            mejores = lote[np.argsort(notas)[-4:]]
            media = mejores.mean(axis=0)
            desv = np.maximum(.12*dispersion, .3*desv + .7*mejores.std(axis=0))
            print(f'Generación {g+1} · rigidez y amortiguación {media.round(2).tolist()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generaciones', type=int, default=8)
    parser.add_argument('--segundos', type=float, default=12.)
    parser.add_argument('--semilla', type=int, default=741)
    parser.add_argument('--validar', action='store_true', help='Mide las ganancias guardadas en los 16 robots.')
    args, kit_args = parser.parse_known_args()
    if args.generaciones < 1 or args.segundos <= 0:
        parser.error('Se requieren duración y número de generaciones positivos.')
    sys.argv[1:] = kit_args
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False, 'disable_viewport_updates': True})
    try:
        banco = Banco()
        try:
            ajustar(banco, args)
        finally:
            banco.cerrar()
    finally:
        app.close()
