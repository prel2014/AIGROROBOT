import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cargas_plantares import integrar_contactos


class PruebaPlantas(unittest.TestCase):
    def test_canales_anatomicos_y_contactos_fuera_de_la_planta(self):
        poses = np.zeros((2, 2, 7))
        giro = Rotation.from_euler("xyz", [18, -24, 90], degrees=True)
        poses[:, :, 3:] = giro.as_quat()
        poses[:, :, :3] = [[[2, 3, .2], [2.3, 3, .2]], [[0, 0, 1], [.3, 0, 1]]]
        z = giro.apply([0., 0., 1.])
        x = giro.apply([1., 0., 0.])
        muestras = []
        for lado, signo in enumerate((1, -1)):
            origen = poses[0, lado, :3]
            esquinas = [(signo*.05, -.10), (-signo*.05, -.10), (signo*.05, .10), (-signo*.05, .10)]
            for indice, (px, py) in enumerate(esquinas):
                punto = origen + giro.apply([px, py, -.029])
                impulso = (1+indice+4*lado) * z + 2*x
                muestras.append((lado, punto, impulso, z))
            muestras.append((lado, origen+giro.apply([.04, .10, .029]), z, z))  # empeine
            muestras.append((lado, origen+giro.apply([.0595, .10, -.029]), 5*x, x))  # lateral
        acumulado = np.zeros((2, 2, 4))
        integrar_contactos(acumulado, poses, muestras)
        np.testing.assert_allclose(acumulado[0], [[1, 2, 3, 4], [5, 6, 7, 8]])
        np.testing.assert_array_equal(acumulado[1], 0.)
        integrar_contactos(acumulado, poses, muestras)
        np.testing.assert_allclose(acumulado[0].sum(), 72.)


if __name__ == "__main__":
    unittest.main()
