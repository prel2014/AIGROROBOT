import unittest
import numpy as np
from criterio_cosecha import Sosten
from situaciones import Secuencias
from frenado import ventana_parada


class TestFrenadoCosecha(unittest.TestCase):
    def test_brazos_esperan_hasta_parar(self):
        s=Secuencias(np.random.default_rng(17),4,solo_frenado=True)
        for fase in (0,1):
            s.fase.fill(fase)
            self.assertFalse(s.consignas().any())
        s.fase.fill(2)
        self.assertTrue((s.consignas().max(axis=1) > .4).all())

    def test_ventana_de_apoyo(self):
        self.assertTrue(ventana_parada(.05,2.1+np.pi,2.))
        self.assertFalse(ventana_parada(.3,2.1,2.))
        self.assertFalse(ventana_parada(.05,1.9,2.))

    def test_contacto_y_sosten_ininterrumpido(self):
        s=Sosten()
        for _ in range(28):
            self.assertFalse(s.medir(1/60,True,.04,.02,{'pulgar','indice'},True,.99,True))
        self.assertFalse(s.medir(1/60,True,.04,.02,{'indice'},True,.99,True))
        self.assertEqual(s.continuo,0)
        for _ in range(30):
            res=s.medir(1/60,True,.04,.02,{'pulgar','indice'},True,.99,True)
        self.assertTrue(res)
        self.assertFalse(s.medir(1/60,True,.04,.02,{'pulgar','indice'},False,.99,True))
