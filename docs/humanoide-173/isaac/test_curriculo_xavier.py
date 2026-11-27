import unittest
import numpy as np
from curriculo_xavier import DT, PLANES, Episodio, Medida
from entrenar_habilidades import configurar


def lectura(xy=(0,0), **cambios):
    base = dict(pelvis=np.array([*xy,1.17]), vertical=1., velocidad=0., pies=np.array([.5,.5]))
    return Medida(**(base|cambios))


def mantener(ep,m,t):
    for _ in range(round(t/DT)):
        ep.evaluar(m)


class Reglas(unittest.TestCase):
    def test_rutas_respetan_puerta(self):
        e=Episodio("caminar",0)
        e.evaluar(lectura((1.,.8)))
        mantener(e,lectura((0,.8)),.5)
        self.assertEqual(e.fase,"puerta")
        e.evaluar(lectura((0,.4)))
        for fase in ("puerta","rodeo","parar"):
            self.assertEqual(e.fase,fase)
            mantener(e,lectura(e.meta()),1.)
        self.assertTrue(e.exito)

    def test_silla_y_suelo_requieren_inicio_fisico(self):
        for nivel in (0,2):
            e=Episodio("levantarse",nivel)
            mantener(e,lectura(),1.)
            self.assertEqual(e.fase,"incorporar")
            e.evaluar(lectura(asiento=nivel==0,vertical=.1 if nivel==2 else 1.))
            mantener(e,lectura(),.6)
            self.assertEqual(e.fase,"paso")

    def test_imitar_exige_cada_pose_y_ventana_temporal(self):
        e=Episodio("imitar",2)
        mantener(e,lectura(error_gesto=.8),3.1)
        self.assertTrue(e.hecho)
        self.assertFalse(e.exito)
        e=Episodio("imitar",2)
        mantener(e,lectura(error_gesto=.05),3.25)
        self.assertTrue(e.exito)

    def test_transporte_y_mision(self):
        for tarea in ("mision","transportar"):
            e=Episodio(tarea,0)
            if tarea=="mision":
                e.evaluar(lectura(asiento=True))
                mantener(e,lectura(),.60)
                mantener(e,lectura(e.meta(),avance_pies=.5),.15)
            elevado=e.origen+[0,0,.12]
            mantener(e,lectura(objeto=elevado,agarre=False),.4)
            self.assertEqual(e.fase,"recoger")
            mantener(e,lectura(objeto=elevado,agarre=True,toca_mano=True),.25)
            punto=e.meta()
            e.evaluar(lectura((.73,punto[1]-.3),objeto=elevado,agarre=True,toca_mano=True))
            mantener(e,lectura(punto,objeto=np.r_[punto,1.],agarre=True,toca_mano=True),.15)
            self.assertEqual(e.fase,"depositar")
            mantener(e,lectura(objeto=e.destino,en_destino=True),.8)
            self.assertFalse(e.hecho)
            e.evaluar(lectura(objeto=e.destino,en_destino=True,toca_mano=True))
            mantener(e,lectura(objeto=e.destino,en_destino=True),.65)
            self.assertTrue(e.exito)

    def test_tijeras_contacto_con_hoja_invalida(self):
        e=Episodio("tijeras",0)
        e.evaluar(lectura(toca_hoja=True))
        self.assertTrue(e.hecho)
        self.assertFalse(e.exito)

    def test_correr_requiere_dos_vuelos_y_velocidad_horizontal(self):
        e=Episodio("correr",1)
        mantener(e,lectura(e.meta()),.15)
        mantener(e,lectura(e.meta()),.15)
        meta=e.meta()
        mantener(e,lectura(meta,velocidad_horizontal=.8),.2)
        self.assertEqual(e.fase,"carrera")
        for lado in range(2):
            e.evaluar(lectura(meta,pies=np.zeros(2)))
            e.evaluar(lectura(meta,velocidad_horizontal=.8,pies=np.eye(2)[lado]))
        self.assertEqual(e.fase,"parar")
        mantener(e,lectura(e.meta()),1.)
        self.assertTrue(e.exito)

    def test_enceste_trayectoria_descendente_y_radio(self):
        for x,exito in ((0,True),(.2,False)):
            e=Episodio("encestar",0)
            mantener(e,lectura(objeto=e.origen+[0,0,.1],agarre=True,toca_mano=True),.25)
            e.evaluar(lectura(objeto=e.destino+[x,-.5,-.2],rapidez_objeto=2.,velocidad_vertical_objeto=1.))
            e.evaluar(lectura(objeto=e.destino+[x,0,.2],rapidez_objeto=2.))
            e.evaluar(lectura(objeto=e.destino+[x,0,-.02],rapidez_objeto=2.))
            mantener(e,lectura(),1.)
            self.assertEqual(e.exito,exito)

    def test_pirueta_exige_un_solo_salto_y_aterrizaje_estable(self):
        e=Episodio("pirueta",1)
        for _ in range(3):
            e.evaluar(lectura())
            e.evaluar(lectura(pies=np.zeros(2),delta_giro=.5))
        self.assertEqual(e.fase,"girar")
        e.evaluar(lectura())
        for _ in range(4):
            e.evaluar(lectura(pies=np.zeros(2),delta_giro=np.pi/8))
        mantener(e,lectura(),1.)
        self.assertTrue(e.exito)

    def test_todos_tienen_comando_y_limite_de_tiempo(self):
        self.assertEqual(configurar([]).modo,"plan")
        for tarea,plan in PLANES.items():
            for entorno in plan.entornos:
                for nivel in range(3):
                    a=configurar(["entrenar","--tarea",tarea,"--entorno",entorno,"--nivel",str(nivel)])
                    e=Episodio(a.tarea,a.nivel)
                    e.ticks=round(plan.segundos/DT)-1
                    e.evaluar(lectura())
                    self.assertTrue(e.hecho)
                    self.assertFalse(e.exito)
