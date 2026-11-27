"""Contrasta las cuatro secuencias con sensores sintéticos sin iniciar Isaac Sim."""
import unittest
import numpy as np
from situaciones import Secuencias


class PruebaSecuencias(unittest.TestCase):
    def setUp(self):
        self.s = Secuencias(np.random.default_rng(32), 16)
        self.si = np.ones(16, bool)
        self.ceros = np.zeros(16)
        self.pies = np.zeros((16, 2))

    def avanzar(self, activo=None, cerca=None, pasos=2, carga=1., rapidez=0.):
        return self.s.actualizar(.05, self.si if activo is None else activo, self.si if cerca is None else cerca,
                                  np.full(16, pasos), self.pies, self.si, self.ceros + rapidez,
                                  self.s.consignas(), self.ceros + carga)[0]

    def test_secuencias_y_terminacion(self):
        self.assertEqual(np.bincount(self.s.tipo).tolist(), [4]*4)
        activos = self.si.copy()
        for _ in range(25):
            activos &= ~self.avanzar(activo=activos)
        self.assertFalse(activos.any())
        np.testing.assert_array_equal(self.s.fase, np.where(np.isin(self.s.tipo, [1, 3]), 3, 2))

    def test_apoyo_y_primera_parada(self):
        for _ in range(10):
            self.avanzar(pasos=0, carga=0.)
        self.assertFalse(self.s.fase.any())
        for _ in range(10):
            self.avanzar(cerca=~self.si, rapidez=1.)
        self.assertFalse(self.s.fase.any())

    def test_alternancia(self):
        filtro = self.s.tipo == 1
        inicial = self.s.consignas()[filtro]
        for _ in range(7):
            self.avanzar()
        np.testing.assert_array_equal(inicial[:, ::-1], self.s.consignas()[filtro])


if __name__ == "__main__":
    unittest.main()

