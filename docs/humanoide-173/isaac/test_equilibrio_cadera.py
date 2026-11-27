import unittest

import numpy as np

from control_marcha import ESPERA_EN_META_S, postura_de_parada, punto_captura
from entrenar_marcha import Pista


class EquilibrioCadera(unittest.TestCase):
    def test_captura_sigue_la_direccion_de_la_fila(self):
        frente = np.array([[0., 1.], [-1., 0.]])
        apoyo = np.array([[3., 4., .029], [8., 1., .029]])
        centro = apoyo.copy()
        centro[:, :2] -= .042 * frente
        centro[:, 2] = .96
        velocidad = np.column_stack((.2 * frente, np.zeros(2)))
        np.testing.assert_allclose(punto_captura(centro, velocidad, apoyo, frente),
                                   .2 * np.sqrt(.96 / 9.81))

    def test_frenado_conserva_el_escalonamiento_y_responde_al_impulso(self):
        inicial = np.array([[.10, .23, .07, .16, .14, .10]])
        quieto = postura_de_parada(inicial, np.array([.002]), np.zeros(1))
        avance = postura_de_parada(inicial, np.array([.018]), np.array([.10]))
        np.testing.assert_allclose(quieto[0, :3] - quieto[0, 3:], .4 * (inicial[0, :3] - inicial[0, 3:]))
        self.assertTrue(np.all(avance[:, 2::3] < quieto[:, 2::3]))

    def test_llegar_no_cuenta_como_parar_sin_incidentes(self):
        env = Pista.__new__(Pista)
        env.secuencia = None
        env.etapa = "marcha"
        env.active = env.en_meta = np.ones(4, bool)
        env.steps = np.array([3, 3, 3, 3])
        env.segundos_estables = np.array([0., ESPERA_EN_META_S, 2., 2.])
        env.caido = np.array([False, False, True, False])
        env.chocado = np.array([False, False, False, True])
        np.testing.assert_array_equal(env.resultado(), [False, True, False, False])

    def test_paso_depende_de_la_camara_propia_y_confirma_vista_libre(self):
        from types import SimpleNamespace
        env = Pista.__new__(Pista)
        env.etapa, env.secuencia = "cruce", None
        env.politica_paso = SimpleNamespace(umbral=.35, confirmacion_s=.10,
            riesgo=lambda rgb: rgb[:, 0, 0, :2] / 255.)
        env.rgb = np.full((16, 256, 256, 3), 255, np.uint8)
        env.active = np.ones(16, bool)
        env.en_travesia = np.zeros(16, bool)
        env.vista_libre = np.zeros(16)
        env.salida_s = np.full(16, np.nan)
        env.time = 1.
        env.rgb[[2, 9]] = 0
        self.assertFalse(env.decidir_salida().any())
        env.rgb[2] = 255
        np.testing.assert_array_equal(np.flatnonzero(env.decidir_salida()), [9])
        env.rgb[2] = 0
        env.rgb[9] = 255
        np.testing.assert_array_equal(np.flatnonzero(env.decidir_salida()), [9])
        np.testing.assert_array_equal(np.flatnonzero(env.decidir_salida()), [2, 9])

    def test_quedarse_esperando_agota_el_tiempo_y_no_aprueba(self):
        env = Pista.__new__(Pista)
        env.etapa, env.secuencia = "cruce", None
        env.active = np.ones(16, bool)
        env.caido = env.chocado = env.en_meta = np.zeros(16, bool)
        env.steps = env.segundos_estables = np.zeros(16)
        env.time, env.horizonte = 2., 65.
        self.assertFalse(env.terminado())
        env.time = env.horizonte
        self.assertTrue(env.terminado())
        self.assertFalse(env.resultado().any())

    def test_red_visual_da_la_misma_decision_sola_o_acompanada(self):
        import torch
        from paso_visual import RedDePaso
        fondos = torch.randint(256, (2, 3, 128, 128), dtype=torch.uint8)
        red = RedDePaso(fondos).eval()
        pixeles = torch.randint(256, (3, 256, 256, 3), dtype=torch.uint8)
        with torch.inference_mode():
            lote = red(pixeles)
            sola = red(pixeles[1:2])
            reordenada = red(pixeles[[2, 1, 0]])
        torch.testing.assert_close(sola[0], lote[1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reordenada, lote[[2, 1, 0]], rtol=1e-5, atol=1e-6)

    def test_cruces_perpendiculares_independientes_del_numero_de_robot(self):
        from paso_visual import itinerarios_cruce
        orden = np.arange(16)
        a = itinerarios_cruce(np.random.default_rng(77), orden)
        b = itinerarios_cruce(np.random.default_rng(77), orden[::-1])
        for original, invertido in zip(a, b):
            np.testing.assert_array_equal(original, invertido[::-1])
        inicio, rumbo, meta, puesto = a
        self.assertEqual(len(np.unique(puesto)), 16)
        for g in range(4):
            filas = rumbo[np.argsort(puesto)][4*g:4*g+4]
            self.assertEqual(filas[0] @ filas[1], 0.)
            np.testing.assert_array_equal(filas[:2], filas[2:])
        self.assertTrue(np.all(((meta-inicio)*rumbo).sum(1) > 2.7))

    def test_movimiento_lateral_procede_de_dos_imagenes_del_mismo_robot(self):
        from paso_visual import movimiento_lateral, vista_lateral
        rgb = np.zeros((3, 256, 256, 3), np.uint8)
        antes = vista_lateral(rgb)
        rgb[0, 96:156, 174:205] = 255
        rgb[1, 96:156, 40:70] = 255
        senal = movimiento_lateral(antes, vista_lateral(rgb))
        self.assertGreater(senal[0], .9)
        np.testing.assert_array_equal(senal[1:], [0., 0.])


if __name__ == "__main__":
    unittest.main()
