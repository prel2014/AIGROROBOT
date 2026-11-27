"""Física y sensores para lotes de Xavier en estaciones aisladas de Isaac Sim."""
import numpy as np
from scipy.spatial.transform import Rotation
from curriculo_xavier import ACCIONES, PLANES, DT, Episodio, Medida
from cargas_plantares import integrar_contactos
from estaciones_xavier import base, construir, silla_z


class Estaciones:
    def __init__(self, app, tarea, nivel, entorno, cantidad, semilla):
        import omni.usd
        import omni.physics.tensors as tensors
        from omni.physx import get_physx_simulation_interface
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.rtx import TiledCameraSensor
        self.tarea, self.nivel, self.n = tarea, nivel, cantidad
        self.rng = np.random.default_rng(semilla)
        ctx = omni.usd.get_context()
        ctx.new_stage()
        self.stage = ctx.get_stage()
        construir(self.stage, tarea, nivel, entorno, cantidad)
        SimulationManager.set_default_physics_scene("/World/Fisica")
        self.sim = SimulationContext(physics_dt=DT/6, rendering_dt=DT, physics_prim_path="/World/Fisica",
                                     stage_units_in_meters=1., device="cpu")
        self.sim.get_physics_context().set_physics_dt(DT/6)
        SimulationManager.enable_fabric(True)
        rutas = [f"/World/E{i:02}/Xavier/cabeza/Camara" for i in range(cantidad)]
        self.sensor = TiledCameraSensor(rutas, resolution=(256,256), annotators=["rgb"])
        self.sim.play()
        self.sim.step(render=False)
        self.tensor = tensors.create_simulation_view("numpy", stage_id=ctx.get_stage_id(), backend="physx")
        self.robot = self.tensor.create_articulation_view("/World/E*/Xavier/pelvis")
        if self.robot.count != cantidad or self.robot.shared_metatype.fixed_base:
            raise RuntimeError("Se requieren articulaciones de base libre para todas las estaciones")
        self.rutas = [p.split("/Xavier/")[0] for p in self.robot.prim_paths]
        self.indices = {p:i for i,p in enumerate(self.rutas)}
        self.bases = np.array([base(int(p[-2:])) for p in self.rutas])
        self.camaras = [self.sensor.camera.paths.index(p+"/Xavier/cabeza/Camara") for p in self.rutas]
        self.ids = np.arange(cantidad,dtype=np.uint32)
        self.obj = None
        if PLANES[tarea].objeto:
            self.obj = self.tensor.create_rigid_body_view("/World/E*/Objeto")
            self.orden_obj = [list(self.obj.prim_paths).index(p+"/Objeto") for p in self.rutas]
        meta = self.robot.shared_metatype
        self.j = [meta.dof_names.index(j) for j in ACCIONES]
        self.links = lambda nombres: [meta.link_names.index(j) for j in nombres]
        self.pies = self.links(("pie_izq","pie_der"))
        self.manos = self.links(("mano_izq","mano_der"))
        self.limites = self.robot.get_dof_limits()[:,self.j]
        self.rapidez = self.robot.get_dof_max_velocities()[:,self.j]
        self.reposo = self.robot.get_dof_positions().copy()
        self.peso = self.robot.get_masses().sum(1)*9.81
        self.suscripcion = get_physx_simulation_interface().subscribe_contact_report_events(self.contactos)
        self.limpiar()

    def limpiar(self):
        self.eventos = [set() for _ in self.ids]
        self.plantares = np.zeros((self.n,2,4))
        self.muestras = []

    def contactos(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        for h in headers:
            if h.type == ContactEventType.CONTACT_LOST or not h.num_contact_data:
                continue
            a,b = [str(PhysicsSchemaTools.intToSdfPath(k)) for k in (h.collider0,h.collider1)]
            padre = "/".join(a.split("/")[:3])
            i = self.indices.get(padre)
            if i is None or not b.startswith(padre+"/"):
                continue
            a,b = a[len(padre)+1:], b[len(padre)+1:]
            if a.startswith("Xavier/") and b.startswith("Xavier/"):
                continue
            for link,otro in ((a,b),(b,a)):
                if link.startswith("Objeto/"):
                    if otro in ("Destino","Fuente","Suelo") or otro.startswith("Decorado/"):
                        self.eventos[i].add(("objeto",otro))
                if not link.startswith("Xavier/"):
                    continue
                cuerpo = link.split("/")[1]
                self.eventos[i].add((cuerpo,otro))
                if cuerpo in ("pie_izq","pie_der"):
                    pie = 2*i+int(cuerpo=="pie_der")
                    for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                        punto = data[k]
                        self.muestras.append((pie, tuple(punto.position), tuple(punto.impulse), tuple(punto.normal)))

    def fisica(self):
        self.limpiar()
        for _ in range(6):
            self.muestras.clear()
            self.vel_anterior = self.robot.get_root_velocities()[:,:3].copy()
            self.sim.step(render=False)
            if self.muestras:
                poses = self.robot.get_link_transforms()[:,self.pies]
                integrar_contactos(self.plantares, poses, self.muestras)
        self.carga = self.plantares / (DT*self.peso[:,None,None])

    def reset(self):
        self.episodios = [Episodio(self.tarea,self.nivel) for _ in self.ids]
        self.objetivos_gesto = np.zeros((self.n, 3, 6))
        for i in self.ids:
            # Tres poses de un demostrador frontal. Los objetivos sólo llegan al árbitro.
            lado = int(self.rng.integers(2))
            amp = self.rng.uniform(.55,1.0)
            self.objetivos_gesto[i,0,lado*2:lado*2+2] = [.35,amp]
            self.objetivos_gesto[i,1,(1-lado)*2:(1-lado)*2+2] = [.4,amp]
            if self.nivel == 0:
                self.objetivos_gesto[i,1,lado*2+1] = .38
            if self.nivel >= 1:
                self.objetivos_gesto[i,2,4] = -.065
            if self.nivel == 2:
                self.objetivos_gesto[i,2,5] = self.rng.choice([-.20,.20])
        root = self.robot.get_root_transforms().copy()
        root[:,:3] = self.bases+[0.,0.,1.173+(.024 if self.tarea=="pirueta" else 0)]
        root[:,3:] = [0,0,0,1]
        self.target = self.reposo.copy()
        if self.tarea in ("levantarse","mision"):
            if self.tarea == "levantarse" and self.nivel==2:
                root[:,2] = .14
                root[:,3:] = Rotation.from_euler("x",90,degrees=True).as_quat()
            else:
                root[:,2] = silla_z(self.nivel)+.316
                for lado in ("izq","der"):
                    for nombre,angulo in (("cadera",34),("rodilla",45),("tobillo",11)):
                        self.target[:,self.j[ACCIONES.index(nombre+"_"+lado)]] = np.deg2rad(angulo)
        root[:,:2] += self.rng.uniform(-.025,.025,(self.n,2))
        rumbo = Rotation.from_euler("z",self.rng.uniform(-.035,.035,self.n))
        root[:,3:] = (rumbo*Rotation.from_quat(root[:,3:])).as_quat()
        self.rumbo_inicial = rumbo
        self.robot.set_root_transforms(root,self.ids)
        self.robot.set_root_velocities(np.zeros((self.n,6),np.float32),self.ids)
        self.robot.set_dof_positions(self.target,self.ids)
        self.robot.set_dof_velocities(np.zeros_like(self.target),self.ids)
        self.robot.set_dof_position_targets(self.target,self.ids)
        if self.obj is not None:
            pose = self.obj.get_transforms().copy()
            for i,k in enumerate(self.orden_obj):
                pose[k,:3] = self.bases[i]+self.episodios[i].origen
                amplitud = (.18,.70,1.3)[self.nivel] if self.tarea=="tijeras" else .06
                yaw = self.rng.uniform(-amplitud,amplitud)
                pose[k,3:] = Rotation.from_euler("z",yaw).as_quat()
            self.obj.set_transforms(pose,self.ids)
            self.obj.set_velocities(np.zeros((self.n,6),np.float32),self.ids)
        self.anterior_accion = np.zeros((self.n,len(self.j)),np.float32)
        self.actualizar_modelo()
        self.fisica()
        for i,e in enumerate(self.episodios):
            if e.fase=="incorporar" and not e.suelo_inicial:
                e.inicio_valido = any(b=="Asiento" for _,b in self.eventos[i])
        self.rot_previa = Rotation.from_quat(self.robot.get_root_transforms()[:,3:])
        self.inicio_pies = self.robot.get_link_transforms()[:,self.pies,:2].mean(1)
        for _ in range(12):
            self.sim.render()
        return self.observar()

    def actualizar_modelo(self):
        if self.tarea!="imitar":
            return
        from pxr import UsdGeom,Gf
        for i,ruta in enumerate(self.rutas):
            e = self.episodios[i]
            fase = min(2,e.indice)
            inicial = self.objetivos_gesto[i,fase-1] if fase else np.zeros(6)
            f = min(1.,e.ticks_fase*DT/.65)
            objetivo = (1-f)*inicial + f*self.objetivos_gesto[i,fase]
            figura = UsdGeom.Xformable(self.stage.GetPrimAtPath(ruta+"/Modelo"))
            ops = figura.GetOrderedXformOps()
            op = ops[0] if ops else figura.AddTranslateOp()
            op.Set(Gf.Vec3d(float(-objetivo[5]),0.,float(objetivo[4])))
            for j in range(2):
                angulo_pierna = np.rad2deg(np.arccos(np.clip(1.+objetivo[4],0,1)))
                pierna = ruta+f"/Modelo/Pierna{j}"
                self.stage.GetPrimAtPath(pierna).GetAttribute("xformOp:rotateX").Set(float(-angulo_pierna))
                self.stage.GetPrimAtPath(pierna+"/Rodilla").GetAttribute("xformOp:rotateX").Set(float(2*angulo_pierna))
                brazo = ruta+f"/Modelo/Brazo{j}"
                for segmento,angulo in ((brazo,objetivo[2*j]),(brazo+"/Codo",objetivo[2*j+1])):
                    self.stage.GetPrimAtPath(segmento).GetAttribute("xformOp:rotateX").Set(float(-np.rad2deg(angulo)))

    def observar(self):
        self.sim.render()
        rgb,_ = self.sensor.get_data("rgb")
        if rgb is None:
            raise RuntimeError("No hay fotogramas RTX")
        imagen = rgb.numpy()[self.camaras]
        if imagen.shape != (self.n,256,256,3):
            raise RuntimeError(f"Formato RGB inesperado {imagen.shape}")
        pose = self.robot.get_root_transforms()
        rot = Rotation.from_quat(pose[:,3:])
        vel = self.robot.get_root_velocities()
        imu = rot.inv().apply((vel[:,:3]-self.vel_anterior)*6/DT+[0.,0.,9.81])/9.81
        meta = np.stack([e.meta() for e in self.episodios])-(pose[:,:2]-self.bases[:,:2])
        fase = np.eye(6)[[min(e.indice,5) for e in self.episodios]]
        estado = np.column_stack((imu,rot.inv().apply(vel[:,3:]),rot.as_matrix()[:,2,:],
            self.carga.reshape(self.n,8),self.robot.get_dof_positions()[:,self.j],
            self.robot.get_dof_velocities()[:,self.j],self.anterior_accion,
            rot.inv().apply(np.c_[meta,np.zeros(self.n)])[:,:2],fase))
        if not np.isfinite(estado).all():
            raise ValueError("Lectura de sensor no finita")
        return imagen,np.clip(estado,-20,20).astype(np.float32)

    def step(self,accion):
        self.anterior_accion = accion
        anterior = self.target[:,self.j]
        self.target[:,self.j] = np.clip(anterior + DT*np.minimum(self.rapidez,4.)*accion,
                                         self.limites[:,:,0],self.limites[:,:,1])
        self.robot.set_dof_position_targets(self.target,self.ids)
        self.fisica()
        pose = self.robot.get_root_transforms()
        rot = Rotation.from_quat(pose[:,3:])
        dr = (self.rot_previa.inv()*rot).as_rotvec()
        self.rot_previa = rot
        vel = self.robot.get_root_velocities()
        links = self.robot.get_link_transforms()[:,:,:3]
        q = self.robot.get_dof_positions()
        if self.obj is not None:
            objetos = self.obj.get_transforms()[self.orden_obj,:3]-self.bases
            velocidades = self.obj.get_velocities()[self.orden_obj,:3]
        premios = np.zeros(self.n,np.float32)
        cambiado = False
        for i,e in enumerate(self.episodios):
            if e.hecho:
                continue
            xyz = pose[i,:3]-self.bases[i]
            ev = self.eventos[i]
            m = Medida(xyz,float(rot.as_matrix()[i,2,2]),np.linalg.norm(vel[i,:3]),self.carga[i].sum(1))
            m.asiento = any(b=="Asiento" for _,b in ev)
            m.pies_sin_contacto = (not any(a.startswith("pie_") for a,_ in ev)
                                   and np.min(links[i,self.pies,2])>.040)
            m.velocidad_horizontal = np.linalg.norm(vel[i,:2])
            m.contacto_cuerpo = any(a!="objeto" and not a.startswith("pie_") and b in ("Suelo","Colchoneta") for a,b in ev)
            m.obstaculo = any(a!="objeto" and (b.startswith(("Puerta","Decorado/","Aro/","Pedestal")) or b in ("Bordillo","MesaFinal","Silla","Tablero")) for a,b in ev)
            m.avance_pies = np.linalg.norm(links[i,self.pies,:2].mean(0)-self.inicio_pies[i])
            if self.obj is not None:
                m.objeto = objetos[i]
                m.rapidez_objeto = np.linalg.norm(velocidades[i])
                m.velocidad_vertical_objeto = velocidades[i,2]
                manos = [(a,b) for a,b in ev if b.startswith("Objeto/") and
                         a.split("_")[0] in ("mano","pulgar","indice","medio","anular","menique")]
                lados = {s for a,b in manos for s in ("izq","der") if s in a.split("_")}
                m.toca_mano = bool(manos)
                m.agarre = len(lados)==2
                if self.tarea=="encestar":
                    m.agarre = bool(manos)
                if self.tarea=="tijeras":
                    dedos = {a.split("_")[0] for a,b in manos if "/Anilla" in b}
                    m.agarre = len(dedos)>=2 and bool(dedos-{"mano"})
                    m.toca_hoja = any(b=="Objeto/HojasCerradas" for _,b in manos)
                m.en_destino = ("objeto","Destino") in ev
                m.apoyo_objeto = any(a=="objeto" for a,_ in ev)
                m.alcance = np.linalg.norm(links[i,self.manos]-self.bases[i]-m.objeto,axis=1).min()
            if self.tarea=="imitar":
                objetivo = self.objetivos_gesto[i,min(e.indice,2)]
                medidas = [q[i,self.j[ACCIONES.index(n+"_"+s)]] for s in ("izq","der") for n in ("hombro","codo")]
                error = np.abs(np.r_[medidas,xyz[2]-1.17,xyz[0]]-objetivo)
                m.error_gesto = max(error[:4].max(),error[4]*4,error[5]*2)
            m.delta_giro = dr[i,0 if self.nivel==2 else 2]
            if self.tarea=="pirueta":
                yaw = (np.pi,np.pi/2,0.)[self.nivel] * np.sign(e.giro)
                m.error_aterrizaje = (Rotation.from_euler("z",-yaw)*self.rumbo_inicial[i].inv()*rot[i]).magnitude()
            fase = e.indice
            premios[i] = e.evaluar(m)
            cambiado |= e.indice != fase
        if cambiado or self.tarea=="imitar":
            self.actualizar_modelo()
        return self.observar(), premios
