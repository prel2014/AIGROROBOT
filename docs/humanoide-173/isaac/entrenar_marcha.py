"""Entrena o evalúa en Isaac Sim la marcha de los 16 Xavier con visión e IMU."""
import argparse
from functools import lru_cache
from pathlib import Path
import sys
import traceback
from time import perf_counter

import numpy as np
from control_marcha import GANANCIAS_MARCHA, GANANCIAS_REPOSO, MARCHA, referencias, corregir_apoyos, frenar_con_apoyo
from control_marcha import ARTICULACIONES, EJES_CONTROL, ACCIONES_MARCHA
from control_marcha import APERTURA_MARCHA_GRADOS, REALIMENTACION_LATERAL, SESGO_FLEXION_GRADOS, SESGO_TOBILLO_GRADOS
from control_marcha import GANANCIAS_CADERA_TRANSVERSAL, RESIDUO_SAGITAL, RESIDUO_GIRO, RESIDUO_APERTURA, RESIDUO_CADENCIA
from control_marcha import punto_captura, postura_de_parada, UMBRAL_CAPTURA_M, ESPERA_EN_META_S
from cargas_plantares import LADOS, ZONAS, integrar_contactos
from situaciones import Secuencias

HERE = Path(__file__).resolve().parent
PISTA = HERE / "pista_marcha_16.usda"
DT = 1 / 120
SUBPASOS = 6
SUBPASOS_EQUILIBRIO = 2
PERIODO = DT * SUBPASOS
RES_CAMARA = 256


class Pista:
    def __init__(self, app, args):
        import omni.physics.tensors as tensors
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.rtx import TiledCameraSensor
        from omni.physx import get_physx_simulation_interface, get_physx_interface
        from pxr import Sdf, PhysicsSchemaTools
        from pista_marcha import recorrido, N

        self.rng = np.random.default_rng(args.semilla)
        self.escenario = args.escenario
        self.dt_control = PERIODO
        self.app = app
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        SimulationManager.set_default_physics_scene("/World/Fisica")
        self.sim = SimulationContext(physics_dt=DT, rendering_dt=PERIODO,
                                     physics_prim_path="/World/Fisica", device="cpu",
                                     stage_units_in_meters=1.)
        # Mantener lectura física en CPU para los eventos de contacto.
        # El solver es independiente del dispositivo de sus buffers de lectura.
        physics = self.sim.get_physics_context()
        if args.fisica == "gpu":
            physics.enable_gpu_dynamics(True)
            physics.set_broadphase_type("GPU")
        SimulationManager.enable_fabric(True)
        scene = stage.GetPrimAtPath("/World/Fisica")
        gpu = scene.GetAttribute("physxScene:enableGPUDynamics").Get()
        broadphase = scene.GetAttribute("physxScene:broadphaseType").Get()
        print(f"MARCHA PhysX {'GPU' if gpu else 'CPU'}, broadphase {broadphase}, "
              f"Fabric {SimulationManager.is_fabric_enabled()}", flush=True)
        get_physx_interface().set_thread_count(4)
        self.paths = [f"/World/Xavier_{i:02}" for i in range(N)]
        self.focales = [stage.GetPrimAtPath(p + "/cabeza/Camara").GetAttribute("focalLength") for p in self.paths]
        self.focal_normal = self.focales[0].Get()
        self.politica_paso = None
        self.ruta_paso = getattr(args, "modelo_paso", HERE / "paso_visual_campus.pt")
        if not all(stage.GetPrimAtPath(p + "/pelvis/IMU") for p in self.paths):
            raise RuntimeError("Faltan robots o sensores en la pista de marcha.")
        if not all(stage.GetPrimAtPath(f"{p}/pie_{lado}/Cargas/{zona}")
                   for p in self.paths for lado in LADOS for zona in ZONAS):
            raise RuntimeError("Faltan las zonas plantares. Regenera la pista con pista_marcha.py.")
        self.vision = TiledCameraSensor([p + "/cabeza/Camara" for p in self.paths],
                                        resolution=(RES_CAMARA, RES_CAMARA), annotators=["rgb"])
        self.contactos = set()
        self.contactos_entorno = set()
        entorno = stage.GetPrimAtPath("/World/Escenario")
        self.obstaculos = {}
        if entorno:
            from pxr import Usd
            for prim in Usd.PrimRange(entorno):
                if prim.GetAttribute("entorno:tipoContacto").Get() == "obstaculo":
                    self.obstaculos[PhysicsSchemaTools.sdfPathToInt(prim.GetPath())] = str(prim.GetPath())
        self.indice_pie = {}
        self.muestras_pies = []
        self.sub = get_physx_simulation_interface().subscribe_contact_report_events(self.contact_report)
        self.sim.play()
        self.sim.step(render=False)
        self.tensor_sim = tensors.create_simulation_view("numpy", stage_id=ctx.get_stage_id(), backend="physx")
        self.robot = self.tensor_sim.create_articulation_view("/World/Xavier_*/pelvis")
        if self.robot.count != N or self.robot.shared_metatype.fixed_base:
            raise RuntimeError(f"Se esperaban 16 robots de base libre. Encontrados: {self.robot.count}")
        order = [str(Sdf.Path(p).GetParentPath()) for p in self.robot.prim_paths]
        self.camera_order = [self.vision.camera.paths.index(p + "/cabeza/Camara") for p in order]
        self.contact_index = {p: i for i, p in enumerate(order)}
        ids = [self.paths.index(p) for p in order]
        self.numeros = np.asarray(ids)
        tracks = [recorrido(i) for i in ids]
        self.start, self.heading, self.goal = (np.stack(x) for x in zip(*tracks))
        self.joints = [self.robot.shared_metatype.dof_names.index(k) for k in EJES_CONTROL]
        self.ids = np.arange(N, dtype=np.uint32)
        self.neutral = self.robot.get_dof_position_targets().copy()
        self.rest = self.robot.get_root_transforms().copy()
        self.limits = self.robot.get_dof_limits()[0, self.joints]
        self.speeds = self.robot.get_dof_max_velocities()[0, self.joints].copy()
        # Velocidad de consigna al caminar. El límite físico del actuador sigue en el USD.
        caderas = [i for i,eje in enumerate(EJES_CONTROL) if "cadera" in eje]
        self.speeds[caderas] = np.minimum(self.speeds[caderas], np.deg2rad(45.))
        self.codos = [self.robot.shared_metatype.dof_names.index("codo_" + lado) for lado in LADOS]
        self.limites_codo = self.robot.get_dof_limits()[0, self.codos]
        self.vel_codo = self.robot.get_dof_max_velocities()[0, self.codos]
        self.kp = self.robot.get_dof_stiffnesses().copy()
        self.kd = self.robot.get_dof_dampings().copy()
        self.ganancias_originales_transversales = [matriz[:, self.joints[6:]].copy() for matriz in (self.kp, self.kd)]
        self.parametros = np.tile(MARCHA, (N, 1))
        self.etapa = args.etapa or "equilibrio"
        self.foot_ids = [self.robot.shared_metatype.link_names.index("pie_" + s) for s in ("izq", "der")]
        self.indice_pie = {PhysicsSchemaTools.sdfPathToInt(Sdf.Path(f"{p}/pie_{lado}")): 2*i+j
                           for i, p in enumerate(order) for j, lado in enumerate(LADOS)}
        self.impulso_plantar = np.zeros((N, 2, 4))
        masas = self.robot.get_masses().copy()
        self.peso = masas.sum(axis=1) * 9.81
        self.fraccion_masa = (masas / masas.sum(axis=1, keepdims=True))[:, :, None]
        self.centros_locales = self.robot.get_coms()[:, :, :3].copy()
        self.rgb = None
        self.monitor = None
        if args.visible:
            from monitor_camaras import MonitorCamaras
            self.monitor = MonitorCamaras([self.contact_index[p] for p in self.paths])
        self.seg_fisica = self.seg_imagen = 0.

    @staticmethod
    @lru_cache(maxsize=4096)
    def propietario(actor):
        from pxr import PhysicsSchemaTools
        return str(PhysicsSchemaTools.intToSdfPath(actor).GetParentPath())

    def contact_report(self, headers, data):
        from omni.physx.bindings._physx import ContactEventType
        for report in headers:
            if report.type == ContactEventType.CONTACT_LOST:
                continue
            a, b = self.propietario(report.actor0), self.propietario(report.actor1)
            if a != b and a in self.paths and b in self.paths:
                self.contactos.add(tuple(sorted((a, b))))
            elif self.obstaculos:
                for robot, collider in ((a, report.collider1), (b, report.collider0)):
                    if robot in self.paths and collider in self.obstaculos:
                        self.contactos_entorno.add((robot, self.obstaculos[collider]))
            if a == b:
                continue
            for actor in (report.actor0, report.actor1):
                pie = self.indice_pie.get(actor)
                if pie is None:
                    continue
                for k in range(report.contact_data_offset, report.contact_data_offset + report.num_contact_data):
                    punto = data[k]
                    self.muestras_pies.append((pie, tuple(punto.position), tuple(punto.impulse), tuple(punto.normal)))

    def imagen(self):
        inicio = perf_counter()
        self.sim.render()
        frame, _ = self.vision.get_data("rgb")
        if frame is None:
            raise RuntimeError("La cámara RTX no entregó el lote de imágenes.")
        rgb = frame.numpy()[self.camera_order]
        if rgb.shape != (16, RES_CAMARA, RES_CAMARA, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Formato RGB inesperado: {rgb.shape}, {rgb.dtype}")
        if self.monitor:
            self.monitor.actualizar(rgb, PERIODO, self.cargas)
        self.seg_imagen += perf_counter() - inicio
        return rgb

    def reset(self):
        # Preparar toda la pista a la vez evita reaparecer sobre otro participante.
        root = self.rest.copy()
        from paso_visual import itinerarios_cruce, FOCAL_CRUCE_MM
        from scipy.spatial.transform import Rotation
        focal = FOCAL_CRUCE_MM if self.etapa == "cruce" else self.focal_normal
        if abs(self.focales[0].Get() - focal) > 1e-5:
            for atributo in self.focales:
                atributo.Set(focal)
        if self.etapa == "cruce":
            self.start, self.heading, self.goal, _ = itinerarios_cruce(self.rng, self.numeros)
        if self.etapa != "cruce":
            self.start = np.column_stack(((self.numeros % 4 - 1.5)*3.4, (self.numeros // 4 - 1.5)*3.4))
            if self.escenario != "pista":
                self.start[:, 0] = np.array([-3.72, -2.68, 2.68, 3.72])[self.numeros % 4]
            self.heading = np.tile((0., 1.), (16, 1))
            self.goal = self.start + self.heading * (1.3 if self.etapa == "marcha" else 0.)
        root[:, :2] = self.start + self.heading * self.rng.uniform(-.14, .14, (16, 1))
        self.secuencia = Secuencias(self.rng, len(self.ids)) if self.etapa == "situaciones" else None
        if self.secuencia:
            self.goal = root[:, :2] + self.heading * self.secuencia.recorrido[:, None]
        root[:, 3:] = Rotation.from_rotvec(np.column_stack((np.zeros((16, 2)),
                                      -np.arctan2(self.heading[:, 0], self.heading[:, 1])))).as_quat()
        root[:, 2] = 1.173
        self.robot.set_root_transforms(root, self.ids)
        self.robot.set_root_velocities(np.zeros((16, 6), np.float32), self.ids)
        self.robot.set_dof_positions(self.neutral, self.ids)
        self.robot.set_dof_velocities(np.zeros_like(self.neutral), self.ids)
        self.target = self.neutral.copy()
        self.apoyo_final = np.zeros(16, bool)
        self.modo_movil = np.ones(16, bool)
        self.ajustar_ganancias(np.zeros(16, bool))
        self.robot.set_dof_position_targets(self.target, self.ids)
        self.avanzar(SUBPASOS)
        self.impulso_plantar.fill(0.)
        self.avanzar(SUBPASOS)
        self.cargas = self.impulso_plantar / PERIODO
        self.cargas_maximas = self.cargas.copy()
        self.phase = np.zeros(16)
        self.en_marcha = np.zeros(16)
        self.tiempo_freno = np.zeros(16)
        self.tiempo_aproximacion = np.zeros(16)
        self.segundos_estables = np.zeros(16)
        self.parada_premiada = np.zeros(16, bool)
        self.inicio_freno = self.neutral[:, self.joints].copy()
        self.eventos = []
        self.parejas = set()
        self.en_meta = np.zeros(16, bool)
        self.caido = np.zeros(16, bool)
        self.chocado = np.zeros(16, bool)
        self.chocado_entorno = np.zeros(16, bool)
        self.vista_libre = np.zeros(16)
        self.referencia_visual = None
        self.ultimo_muestreo_visual = -1.
        self.movimiento_derecha = np.zeros(16)
        self.vio_trafico = np.zeros(16, bool)
        self.en_travesia = np.zeros(16, bool)
        self.salida_s = np.full(16, np.nan)
        self.prev_action = np.zeros((16, len(ACCIONES_MARCHA) + (2 if self.secuencia else 0)), np.float32)
        self.mando_marcha = np.zeros((16,len(ACCIONES_MARCHA)))
        self.active = np.ones(16, bool)
        self.time = 0.
        self.horizonte = {"equilibrio": 12., "marcha": 24., "cruce": 65., "situaciones": 14.}[self.etapa]
        self.contactos.clear()
        self.contactos_entorno.clear()
        self.distance = self.distancia()
        self.falls = self.hits = self.arrivals = 0
        self.avance = np.zeros(16)
        self.separacion_minima = np.full(16, np.inf)
        self.last_support = np.zeros(16, np.int8)
        self.steps = np.zeros(16, int)
        # Tras teletransportar los 16 cuerpos, RTX aún conserva vistas del episodio anterior.
        # Estabilizar su historial temporal sin avanzar la física ni la política.
        for _ in range(10):
            self.sim.render()
        for _ in range(100):
            self.sim.render()
            frame, _ = self.vision.get_data("rgb")
            if frame is not None and np.all(frame.numpy().reshape(16, -1).std(axis=1) > 1):
                break
        else:
            raise RuntimeError("Alguna de las 16 cámaras no produjo una imagen con contenido.")
        return self.observar()

    def ajustar_ganancias(self, movil):
        if np.array_equal(self.modo_movil, movil):
            return
        for indice, matriz, enviar in (
            (0, self.kp, self.robot.set_dof_stiffnesses),
            (1, self.kd, self.robot.set_dof_dampings),
        ):
            matriz[:, self.joints[:6]] = np.where(movil[:, None], GANANCIAS_MARCHA[indice] * 2, GANANCIAS_REPOSO[indice] * 2)
            if matriz is self.kd:
                matriz[np.ix_(self.apoyo_final, self.joints[:6])] = 25.
            transversal = self.ganancias_originales_transversales[indice] if self.secuencia else GANANCIAS_CADERA_TRANSVERSAL[indice] * 2
            matriz[:, self.joints[6:]] = transversal
            enviar(matriz, self.ids)
        self.modo_movil = movil.copy()

    def decidir_salida(self):
        if self.secuencia:
            return self.secuencia.caminar()
        if self.etapa == "equilibrio":
            return np.zeros(16, bool)
        if self.etapa == "marcha":
            return np.ones(16, bool)
        from paso_visual import PasoVisual
        if self.politica_paso is None:
            self.politica_paso = PasoVisual(self.ruta_paso)
        mirando = self.active & ~self.en_travesia
        if mirando.any():
            riesgos = self.politica_paso.riesgo(self.rgb[mirando])
            libre = riesgos.max(axis=1) < self.politica_paso.umbral
            if getattr(self.politica_paso, "umbral_movimiento", None) is not None:
                from paso_visual import vista_lateral, movimiento_lateral
                self.vio_trafico[mirando] |= riesgos[:, 1] > .55
                if self.time - self.ultimo_muestreo_visual >= .4 - 1e-6:
                    lateral = vista_lateral(self.rgb)
                    if self.referencia_visual is not None:
                        self.movimiento_derecha = movimiento_lateral(self.referencia_visual, lateral)
                    self.referencia_visual = lateral
                    self.ultimo_muestreo_visual = self.time
                libre &= ~(self.vio_trafico[mirando] & (
                    self.movimiento_derecha[mirando] >= self.politica_paso.umbral_movimiento))
            self.vista_libre[mirando] = np.where(libre, self.vista_libre[mirando] + PERIODO, 0.)
            entrar = mirando & (self.vista_libre >= self.politica_paso.confirmacion_s) & (self.time >= .6)
            self.salida_s[entrar] = self.time
            self.en_travesia |= entrar
        return self.en_travesia.copy()

    def terminado(self):
        if not self.active.any() or self.time + 1e-6 >= self.horizonte:
            return True
        if self.etapa == 'equilibrio':
            return False
        pendientes = self.active & (~self.en_meta | (self.segundos_estables < ESPERA_EN_META_S))
        if self.secuencia:
            pendientes = self.active & ~self.en_meta
        return not pendientes.any()

    def verificar_sensores_y_contacto(self):
        from scipy.spatial.transform import Rotation

        self.reset()
        if len({img.tobytes() for img in self.rgb}) != 16:
            raise RuntimeError("Hay cámaras duplicadas en la pista de prueba.")
        gravity = self.imu[:, 2]
        if not np.all((gravity > .8) & (gravity < 1.2)) or self.contactos:
            raise RuntimeError("El reposo inicial no coincide con las lecturas o contactos esperados.")
        # Promediar después de que se asientan las suelas evita contar el impacto inicial.
        self.avanzar(48)
        self.impulso_plantar.fill(0.)
        self.avanzar(24)
        por_pie = self.impulso_plantar.sum(axis=2) / (24 * DT)
        fraccion = por_pie.sum(axis=1) / self.peso
        print(f"MARCHA carga estática media por pie {por_pie.mean(axis=0).round(1).tolist()} N, "
              f"suma/peso {fraccion.min():.3f}..{fraccion.max():.3f}", flush=True)
        if not (np.all(abs(fraccion-1.) < .10) and np.all(por_pie > .15*self.peso[:,None])):
            raise RuntimeError("Las cargas plantares en reposo no recuperan el peso en los 16 robots.")
        a, b = (self.contact_index[p] for p in self.paths[:2])
        elevado = self.robot.get_root_transforms().copy()
        elevado[a, 2] += .3
        self.robot.set_root_transforms(elevado, self.ids[a:a+1])
        self.impulso_plantar.fill(0.)
        self.avanzar(SUBPASOS)
        if self.impulso_plantar[a].any() or np.any(self.impulso_plantar.sum(axis=(1,2))[self.ids != a] <= 0.):
            raise RuntimeError("Fallo de descarga o atribución de las celdas al levantar un Xavier.")
        # Un eje por robot en vuelo libre, para aislarlo de la reacción del suelo.
        ejes = ["cadera_izq","cadera_der",*EJES_CONTROL[6:]]
        columnas = [self.robot.shared_metatype.dof_names.index(n) for n in ejes]
        root = self.robot.get_root_transforms().copy()
        root[:,2] += .60
        self.robot.set_root_transforms(root,self.ids)
        self.robot.set_root_velocities(np.zeros((16,6),np.float32),self.ids)
        inicial = self.robot.get_dof_positions().copy()  # PhysX reutiliza este búfer de lectura.
        orden = inicial.copy()
        filas = np.arange(len(ejes))
        orden[filas,columnas] += np.deg2rad(5.)
        self.robot.set_dof_position_targets(orden,self.ids)
        self.avanzar(24)
        cambio = np.rad2deg((self.robot.get_dof_positions()-inicial)[filas,columnas])
        if not np.all(np.isfinite(cambio) & (cambio > 1.)):
            raise RuntimeError(f"Mandos de cadera sin respuesta física: {dict(zip(ejes,cambio))}")
        print(f"MARCHA seis ejes de cadera en PhysX, respuesta a +5 grados: {dict(zip(ejes,cambio.round(2).tolist()))}",flush=True)
        self.reset()
        root = self.robot.get_root_transforms().copy()
        root[b] = root[a]
        root[b, :3] += Rotation.from_quat(root[a, 3:]).apply([.51, 0., 0.])
        self.robot.set_root_transforms(root, self.ids)
        self.distance = self.distancia()
        _, reward = self.step(np.zeros_like(self.prev_action))
        if self.active.sum() != 14 or self.active[[a, b]].any() or not (reward[[a, b]] < -6).all():
            raise RuntimeError(f"Fallo al atribuir el contacto de prueba: {self.contactos}, {reward}")
        print("MARCHA prueba: 16 imágenes distintas, 16 IMU en reposo y contacto entre Xavier_00 y "
              "Xavier_01 atribuido a ambos. Los otros 14 siguen activos.", flush=True)
        if self.obstaculos:
            self.verificar_contacto_entorno(a)

    def verificar_contacto_entorno(self, i):
        from pxr import Usd, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        cajas = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for ruta in self.obstaculos.values():
            limites = cajas.ComputeWorldBound(stage.GetPrimAtPath(ruta)).ComputeAlignedRange()
            dimension = limites.GetSize()
            if limites.GetMin()[2] < .3 and limites.GetMax()[2] > 2. and min(dimension[:2]) < .4:
                break
        else:
            raise RuntimeError("Falta una pared o un poste para probar el contacto con el escenario.")
        self.reset()
        pose = self.robot.get_root_transforms().copy()
        pose[i,:2] = tuple(limites.GetMidpoint())[:2]
        self.robot.set_root_transforms(pose, self.ids)
        _, reward = self.step(np.zeros_like(self.prev_action))
        if self.chocado_entorno.sum() != 1 or not self.chocado_entorno[i] or reward[i] > -6:
            raise RuntimeError(f"No se atribuyó el contacto con {ruta} al Xavier de prueba.")
        print(f"MARCHA prueba de obstáculo: Xavier_{self.numeros[i]:02} en contacto con {ruta}.", flush=True)

    def distancia(self):
        # Medir desde los apoyos impide ganar avance echando el torso hacia la meta al caer.
        pies = self.robot.get_link_transforms()[:, self.foot_ids, :2].mean(axis=1)
        return np.linalg.norm(self.goal - pies, axis=1)

    def avanzar(self, pasos):
        for k in range(pasos):
            if k == pasos - 1:
                self.vel_previa = self.robot.get_root_velocities()[:, :3].copy()
            self.muestras_pies.clear()
            self.sim.step(render=False)
            if self.muestras_pies:
                poses = self.robot.get_link_transforms()[:, self.foot_ids]
                integrar_contactos(self.impulso_plantar, poses, self.muestras_pies)

    def observar(self):
        from scipy.spatial.transform import Rotation
        self.rgb = self.imagen()
        pose = self.robot.get_root_transforms()
        orientation = Rotation.from_quat(pose[:, 3:])
        # El marco IMU coincide con el centro de masa de la pelvis en los 16 USD.
        # A partir del último par de muestras físicas, f = Rᵀ(a - g), ω_local = Rᵀω.
        vel = self.robot.get_root_velocities()
        force = (vel[:, :3] - self.vel_previa) / DT + [0., 0., 9.81]
        self.imu = np.column_stack((orientation.inv().apply(force) / 9.81,
                                    orientation.inv().apply(vel[:, 3:])))
        self.vertical = orientation.as_matrix()[:, 2, :]
        direction = self.goal - pose[:, :2]
        local = orientation.inv().apply(np.c_[direction, np.zeros(16)])
        q = self.robot.get_dof_positions()[:, self.joints]
        dq = self.robot.get_dof_velocities()[:, self.joints]
        state = np.column_stack((self.imu, self.vertical, q[:,:6], dq[:,:6], self.mando_marcha[:, :7], np.sin(self.phase),
                                 np.cos(self.phase), local[:, :2], self.cargas.reshape(16, 8)/self.peso[:, None],
                                 q[:,6:], dq[:,6:], self.mando_marcha[:,7:11]))
        if self.secuencia:
            state = np.column_stack((state, self.secuencia.consignas(), self.robot.get_dof_positions()[:, self.codos],
                                     self.secuencia.caminar(), self.prev_action[:, 11:]))
        if not np.isfinite(state).all():
            raise RuntimeError("Estado físico o sensorial no finito.")
        return self.rgb, np.clip(state, -10, 10).astype(np.float32)

    def medir_captura(self, eslabones, pies):
        """Centro de masa de los 18 eslabones, proyectado en el eje del carril."""
        from scipy.spatial.transform import Rotation
        orientaciones = Rotation.from_quat(eslabones[:, :, 3:].reshape(-1, 4))
        offsets = orientaciones.apply(self.centros_locales.reshape(-1, 3)).reshape(self.centros_locales.shape)
        centro = (self.fraccion_masa * (eslabones[:, :, :3] + offsets)).sum(axis=1)
        velocidad = (self.fraccion_masa * self.robot.get_link_velocities()[:, :, :3]).sum(axis=1)
        return punto_captura(centro, velocidad, pies, self.heading), (velocidad[:, :2] * self.heading).sum(axis=1)

    def motores(self, action, mover, intervalo):
        from scipy.spatial.transform import Rotation
        pose = self.robot.get_root_transforms()
        rot = Rotation.from_quat(pose[:, 3:])
        velocidad_raiz = self.robot.get_root_velocities()
        eslabones = self.robot.get_link_transforms()
        pies = eslabones[:, self.foot_ids, :3].mean(axis=1)
        if not self.secuencia:
            llegando = self.en_meta & self.active
            if llegando.any():
                captura, velocidad_centro = self.medir_captura(eslabones, pies)
                buscando = llegando & ~self.apoyo_final
                self.apoyo_final |= buscando & (captura <= UMBRAL_CAPTURA_M)
                self.tiempo_aproximacion += intervalo * buscando
            mover = mover & ~self.apoyo_final
            self.ajustar_ganancias(mover)
        # Constante de 0,20 s para que la exploración a 20 Hz no sacuda los apoyos.
        self.mando_marcha += -np.expm1(-intervalo/.20)*(action[:,:11]-self.mando_marcha)
        mando = self.mando_marcha
        cadencia = .12 if self.secuencia else RESIDUO_CADENCIA
        self.phase += 2*np.pi*self.parametros[:,0]*intervalo*(1.+cadencia*mando[:, 6])*mover
        q = self.robot.get_dof_positions()[:, self.joints]
        dq = self.robot.get_dof_velocities()[:, self.joints]
        amplitud = np.minimum(self.en_marcha/.7, 1.)
        desired = referencias(self.parametros, self.phase, amplitud,
                              rot.as_matrix()[:, 2, :], rot.inv().apply(velocidad_raiz[:, 3:]), q[:,:6], dq[:,:6])
        separacion = ((pose[:, :2]-pies[:, :2])*self.heading).sum(axis=1)
        velocidad = (velocidad_raiz[:, :2]*self.heading).sum(axis=1)
        corregir_apoyos(desired, self.phase, amplitud, separacion, velocidad)
        desired[:,0::3] += np.deg2rad(SESGO_FLEXION_GRADOS)
        desired[:,2::3] += np.deg2rad(SESGO_TOBILLO_GRADOS)
        if not self.secuencia:
            # Reduce el impulso hacia delante durante los últimos apoyos.
            desired[:,2::3] -= np.deg2rad(5.) * np.minimum(self.tiempo_aproximacion / .55, 1.)[:,None]
        if self.secuencia:
            ajuste = (1.5 + np.clip(self.en_marcha - 2., 0., 1.))[:, None]
            desired += np.deg2rad(ajuste * mando[:, :6])
            desired[~mover] = np.deg2rad(.45 * mando[~mover, :6])
        else:
            desired += np.deg2rad(RESIDUO_SAGITAL * mando[:, :6])
            desired[~mover] = 0.
        frenar = self.apoyo_final & self.active
        if self.secuencia:
            frenar |= (self.secuencia.tipo == 3) & ~mover & (self.en_marcha > 0.) & self.active
        if frenar.any():
            inicio = frenar & (self.tiempo_freno == 0.)
            self.inicio_freno[inicio] = q[inicio]
            self.tiempo_freno += intervalo * frenar
            if self.secuencia:
                desired[frenar] = frenar_con_apoyo(self.inicio_freno[frenar,:6], q[frenar,:6],
                    separacion[frenar], velocidad[frenar], self.tiempo_freno[frenar])
                desired[frenar] += np.deg2rad(2. * mando[frenar, :6])
            else:
                desired[frenar] = postura_de_parada(self.inicio_freno[frenar,:6], captura[frenar], velocidad_centro[frenar])
        # Las convenciones anatómicas se espejan en el USD. Dos consignas independientes
        # por pierna mantienen el equilibrio lateral y orientan el siguiente apoyo.
        vertical = rot.as_matrix()[:,2,:]
        balance = np.clip(REALIMENTACION_LATERAL*(-.35*np.arctan2(vertical[:,0],vertical[:,2])
                          + .07*rot.inv().apply(velocidad_raiz[:,3:])[:,1]), -.08,.08)
        apertura = np.deg2rad(APERTURA_MARCHA_GRADOS)*mover
        lateral = np.column_stack((np.zeros(16),apertura+balance,np.zeros(16),apertura-balance))
        escala = np.array([RESIDUO_GIRO, RESIDUO_APERTURA] * 2) * mover[:, None]
        if self.secuencia:
            escala = np.where(mover[:, None], [10., 6., 10., 6.], [1.5, 1., 1.5, 1.])
        lateral += np.deg2rad(escala * mando[:,7:11])
        desired = np.column_stack((desired,lateral))
        self.en_marcha += mover*intervalo
        desired = np.clip(desired, self.limits[:, 0], self.limits[:, 1])
        desired[~self.active] = 0.
        previous = self.target[:, self.joints]
        self.target[:, self.joints] = previous + np.clip(desired - previous, -self.speeds * intervalo, self.speeds * intervalo)
        if self.secuencia:
            previa = self.target[:, self.codos]
            meta = self.secuencia.consignas() + np.deg2rad(18.) * action[:, 11:]
            meta[~self.active] = 0.
            meta = np.clip(meta, self.limites_codo[:, 0], self.limites_codo[:, 1])
            delta = np.minimum(self.vel_codo, np.deg2rad(85.)) * intervalo
            self.target[:, self.codos] = previa + np.clip(meta - previa, -delta, delta)
        self.robot.set_dof_position_targets(self.target, self.ids)

    def step(self, action):
        from scipy.spatial.transform import Rotation
        detenido = self.en_meta if self.secuencia else self.apoyo_final
        mover = self.decidir_salida() & self.active & ~detenido & (self.time >= .6)
        self.ajustar_ganancias(mover)
        self.contactos.clear()
        self.contactos_entorno.clear()
        self.impulso_plantar.fill(0.)
        inicio = perf_counter()
        regular_apoyos = mover.any() or (self.apoyo_final & self.active).any() or self.secuencia
        bloque = SUBPASOS_EQUILIBRIO if regular_apoyos else SUBPASOS
        for _ in range(SUBPASOS // bloque):
            self.motores(action, mover, DT * bloque)
            self.avanzar(bloque)
        self.cargas = self.impulso_plantar / PERIODO
        self.cargas_maximas = np.maximum(self.cargas_maximas, self.cargas)
        self.seg_fisica += perf_counter() - inicio
        pose = self.robot.get_root_transforms()
        upright = Rotation.from_quat(pose[:, 3:]).as_matrix()[:, 2, 2]
        feet = self.robot.get_link_transforms()[:, self.foot_ids]
        desplazamiento = self.goal - feet[:, :, :2].mean(axis=1)
        restante = (desplazamiento * self.heading).sum(axis=1)
        distance = np.linalg.norm(desplazamiento, axis=1)
        gain = (self.distance - distance) * ~self.en_meta
        self.distance = distance
        fallen = (pose[:, 2] < .85) | (upright < .69)
        hit = np.zeros(16, bool)
        for a, b in self.contactos:
            indices = [self.contact_index[a], self.contact_index[b]]
            participantes = [i for i in indices if self.active[i]]
            hit[participantes] = True
            if (a,b) not in self.parejas:
                self.parejas.add((a,b))
                self.eventos.append(dict(segundos=round(self.time + PERIODO, 2), tipo='contacto',
                    robots=self.numeros[indices].tolist(), xy=pose[indices, :2].round(3).tolist()))
        for robot, obstaculo in sorted(self.contactos_entorno):
            i = self.contact_index[robot]
            if not self.active[i]:
                continue
            hit[i] = True
            if not self.chocado_entorno[i]:
                self.eventos.append(dict(segundos=round(self.time + PERIODO, 2), tipo="obstaculo",
                    robot=int(self.numeros[i]), objeto=obstaculo))
                self.chocado_entorno[i] = True
        for i in np.flatnonzero(fallen & ~self.caido):
            self.eventos.append(dict(segundos=round(self.time + PERIODO, 2), tipo='caida', robot=int(self.numeros[i]),
                                      marcha_s=round(float(self.en_marcha[i]), 2)))
        lateral = np.abs(desplazamiento[:, 0]*self.heading[:, 1] - desplazamiento[:, 1]*self.heading[:, 0])
        arrived = ((self.etapa != 'equilibrio') & (restante <= 0.) & (lateral < .35)
                   & (upright > .96) & (self.steps >= 2) & self.active & ~self.en_meta & ~fallen & ~hit)
        distances = np.linalg.norm(pose[:, None, :2] - pose[None, :, :2], axis=2)
        np.fill_diagonal(distances, np.inf)
        vecino = distances.min(axis=1)
        self.separacion_minima = np.where(self.active, np.minimum(self.separacion_minima, vecino), self.separacion_minima)
        near = np.clip((.78 - vecino) / .38, 0, 1)
        izq, der = (self.cargas.sum(axis=2) / self.peso[:, None]).T
        support = np.where((izq > .20) & (der < .04), 1, np.where((der > .20) & (izq < .04), -1, 0))
        alternated = (support != 0) & (support == -self.last_support)
        self.last_support = np.where(support, support, self.last_support)
        self.steps += alternated & self.active
        premio_tarea = 0.
        if self.secuencia:
            fase_antes = self.secuencia.fase.copy()
            arrived, premio_tarea = self.secuencia.actualizar(
                PERIODO, self.active & ~hit & ~fallen & (self.time >= .6),
                (restante <= .06) & (distance < .22), self.steps, feet[:, :, :2].mean(axis=1), upright,
                np.linalg.norm(self.robot.get_root_velocities()[:, :2], axis=1),
                self.robot.get_dof_positions()[:, self.codos], izq + der)
            for i in np.flatnonzero(self.secuencia.fase != fase_antes):
                self.eventos.append(dict(tipo="fase", robot=int(self.numeros[i]), segundos=round(self.time + PERIODO, 2),
                                         fase_completada=int(fase_antes[i])))
        self.en_meta |= arrived
        velocidad_horizontal = np.linalg.norm(self.robot.get_root_velocities()[:, :2], axis=1)
        estable = self.apoyo_final & self.active & (upright > .96) & (velocidad_horizontal < .12)
        self.segundos_estables = np.where(estable, self.segundos_estables + PERIODO, 0.)
        premio_llegada = 8. * arrived
        if not self.secuencia:
            parada = (self.segundos_estables >= ESPERA_EN_META_S) & ~self.parada_premiada & ~fallen & ~hit
            premio_llegada = 2. * arrived + 6. * parada
            self.parada_premiada |= parada
        reward = (12 * gain * mover + .04 * upright - .15*(1.-upright) - .035 * near
                  - .009 * np.square(action - self.prev_action).mean(axis=1)
                  + .08 * alternated * (gain > 0) + premio_llegada + premio_tarea)
        reward[fallen] = -4.
        reward[hit] = -7.
        self.prev_action = action.copy()
        self.avance += gain * self.active
        self.caido |= fallen
        self.chocado |= hit
        self.falls = int(self.caido.sum())
        self.hits = int(self.chocado.sum())
        self.arrivals = int(self.en_meta.sum())
        reward *= self.active
        self.active &= ~(hit | fallen)
        if self.secuencia:
            self.active &= ~arrived
        self.time += PERIODO
        return self.observar(), reward.astype(np.float32)

    def resultado(self):
        if self.secuencia:
            return self.en_meta.copy()
        exito = self.active if self.etapa == "equilibrio" else (
            self.en_meta & (self.steps >= 2) & (self.segundos_estables >= ESPERA_EN_META_S))
        return exito & ~self.caido & ~self.chocado


def ejecutar(args, app):
    import torch
    from politica_marcha import aprender
    torch.set_num_threads(2)
    torch.manual_seed(args.semilla)
    env = Pista(app, args)
    def ejecutar_episodios(config, video):
        from mostrar_cruce_visual import GrabadoraCruce
        grabadora = GrabadoraCruce(env, video, args.robot_video, config.semilla) if video else None
        try:
            return aprender(config, env, grabadora)
        finally:
            if grabadora:
                grabadora.cerrar()

    if args.verificar:
        env.verificar_sensores_y_contacto()
    if not args.validar_marcha:
        ejecutar_episodios(args, args.video)
        return
    import hashlib
    import json
    from argparse import Namespace
    digest = hashlib.sha256(args.modelo.read_bytes()).hexdigest()
    digest_paso = hashlib.sha256(args.modelo_paso.read_bytes()).hexdigest()
    resultado = dict(modelo=args.modelo.name, sha256=digest, modelo_paso=args.modelo_paso.name,
                     sha256_paso=digest_paso, pruebas=[], aprobado=False)
    for semilla in args.validar_marcha:
        for etapa in ((args.etapa,) if args.etapa else ("marcha", "cruce")):
            prueba = Namespace(**vars(args))
            prueba.etapa, prueba.semilla, prueba.evaluar, prueba.informe = etapa, semilla, 1, None
            env.rng = np.random.default_rng(semilla)
            video = args.video if not resultado["pruebas"] and etapa == "cruce" else None
            informe = ejecutar_episodios(prueba, video)
            if (hashlib.sha256(args.modelo.read_bytes()).hexdigest() != digest or
                    hashlib.sha256(args.modelo_paso.read_bytes()).hexdigest() != digest_paso):
                raise RuntimeError("El checkpoint cambió durante la evaluación.")
            resultado["pruebas"].append(informe)
            args.informe.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf8")
    resultado["aprobado"] = all(
        e["aprobados"] == e["llegadas"] == 16 and e["caidas"] == e["robots_con_choque"] == 0
        for bloque in resultado["pruebas"] for e in bloque["episodios"])
    args.informe.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf8")
    if not resultado["aprobado"]:
        raise RuntimeError("Al menos un Xavier no completó su recorrido sin incidencias.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteraciones", type=int, default=300)
    parser.add_argument("--pasos", type=int, default=64, help="Decisiones por robot y actualización PPO.")
    parser.add_argument("--semilla", type=int, default=17)
    parser.add_argument("--modelo", type=Path)
    parser.add_argument("--modelo-paso", type=Path, default=HERE / "paso_visual_campus.pt", help="Red de paso/espera a partir del RGB local.")
    parser.add_argument("--inicial", type=Path, help="Transfiere una política de marcha a una nueva ejecución de PPO.")
    parser.add_argument("--etapa", choices=("equilibrio", "marcha", "cruce", "situaciones"))
    parser.add_argument("--escenario", choices=("campus", "calle", "vivienda", "pista"), default="campus",
                        help="Entorno físico y visual compartido por los 16 robots (por defecto, campus).")
    parser.add_argument("--reanudar", action="store_true")
    parser.add_argument("--evaluar", type=int, metavar="EPISODIOS")
    parser.add_argument("--validar-marcha", type=int, nargs="+", metavar="SEMILLA",
                        help="Evalúa los 16 robots en recta y cruce con el mismo checkpoint para cada semilla.")
    parser.add_argument("--informe", type=Path, help="Resultados de evaluación por robot en JSON.")
    parser.add_argument("--video", type=Path, help="Graba la primera prueba de cruce con una cámara individual.")
    parser.add_argument("--robot-video", type=int, default=0, choices=range(16))
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--fisica", choices=("cpu", "gpu"), default="cpu", help="Dispositivo del solver PhysX.")
    parser.add_argument("--verificar", action="store_true", help="Prueba las cámaras, las IMU y un choque controlado antes de empezar.")
    parser.add_argument("--captura", type=Path, help="Mosaico PNG de las 16 cámaras al empezar.")
    args, kit_args = parser.parse_known_args()
    if args.video and (args.etapa != "cruce" or not (args.validar_marcha or args.evaluar == 1)):
        parser.error("--video requiere --etapa cruce y --evaluar 1 o --validar-marcha.")
    if args.modelo is None:
        if args.etapa == "situaciones":
            args.modelo = HERE / "situaciones_cadera_3ejes.pt"
        elif args.evaluar or args.validar_marcha:
            args.modelo = HERE / "marcha_campus_validada.pt"
        else:
            args.modelo = HERE / "marcha_cadera_estable.pt"
    if args.validar_marcha:
        if not args.informe or args.evaluar or args.reanudar or args.inicial or args.etapa not in (None, "cruce"):
            parser.error("--validar-marcha requiere --informe y acepta --etapa cruce para evaluar sólo los cruces.")
        args.evaluar = 1
    if args.inicial and (args.evaluar or args.reanudar or
                         not args.inicial.is_file() or args.inicial.resolve() == args.modelo.resolve()):
        parser.error("--inicial requiere una política de marcha existente, una salida distinta y una ejecución nueva.")
    if args.iteraciones < 1 or args.pasos < 2 or (args.evaluar is not None and args.evaluar < 1):
        parser.error("Se requieren iteraciones y episodios positivos y al menos dos pasos por actualización.")
    if (args.reanudar or args.evaluar) and not args.modelo.is_file():
        parser.error(f"No se encontró el modelo {args.modelo}.")
    sys.argv[1:] = kit_args
    mundo = PISTA if args.escenario == "pista" else HERE / "escenarios" / (args.escenario + ".usda")
    if not mundo.is_file():
        parser.error(f"Falta el escenario {mundo}. Genéralo con escenarios_marcha.py.")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not args.visible, "multi_gpu": False, "width": 800,
                         "height": 600, "open_usd": str(mundo), "hide_ui": not args.visible,
                         "extra_args": ["--/app/hydraEngine/waitIdle=true"]})
    status = 0
    try:
        ejecutar(args, app)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        app.close(exit_code=status)
    sys.exit(status)
