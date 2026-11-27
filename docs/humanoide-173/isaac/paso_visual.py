"""Política local de entrada al cruce. Cada fila es la imagen de una única cabeza."""
import numpy as np
import torch
from torch import nn

FOCAL_CRUCE_MM = 3.1


class RedDePaso(nn.Module):
    def __init__(self, fondos=None):
        super().__init__()
        self.register_buffer("fondos", fondos)
        self.register_buffer("descriptores", self.describir(fondos) if fondos is not None else None)
        canales = (6 if fondos is not None else 3, 20, 32, 48, 64)
        capas = []
        for entrada, salida in zip(canales, canales[1:]):
            capas.extend([nn.Conv2d(entrada, salida, 3, stride=2, padding=1),
                          nn.GroupNorm(4, salida), nn.SiLU()])
        self.rasgos = nn.Sequential(*capas)
        self.decision = nn.Sequential(nn.AdaptiveAvgPool2d((3, 8)), nn.Flatten(),
                                      nn.Linear(64*3*8, 96), nn.SiLU(), nn.Linear(96, 2))

    def forward(self, rgb):
        # El mismo submuestreo se aplica a cada imagen de 256 × 256 durante el entrenamiento.
        return self.desde_miniatura(nn.functional.avg_pool2d(rgb.permute(0, 3, 1, 2).float(), 2))

    def desde_miniatura(self, imagen):
        entrada = imagen / 255.
        if self.fondos is not None:
            clave = self.describir(imagen)
            indice = (clave[:, None] - self.descriptores[None]).square().mean(-1).argmin(1)
            diferencia = ((imagen - self.fondos[indice]) / 64.).clamp(-2., 2.)
            entrada = torch.cat((entrada, diferencia), dim=1)
        return self.decision(self.rasgos(entrada))

    @staticmethod
    def describir(imagen):
        cielo = imagen.float()[:, :, :49]
        return nn.functional.adaptive_avg_pool2d(cielo, (4, 10)).flatten(1) / 255.


class PasoVisual:
    def __init__(self, ruta):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pesos = torch.load(ruta, map_location="cpu", weights_only=True)
        if pesos["contrato"] != "xavier-rgb-paso-v1":
            raise ValueError("La política de paso debe recibir una imagen RGB local de 256 × 256.")
        self.red = RedDePaso(pesos["red"].get("fondos")).to(self.device).eval()
        self.red.load_state_dict(pesos["red"])
        self.umbral = pesos["umbral"]
        self.confirmacion_s = pesos["confirmacion_s"]
        self.umbral_movimiento = pesos.get("umbral_movimiento")

    @torch.inference_mode()
    def riesgo(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (256, 256, 3):
            raise ValueError("Se esperaba un lote de imágenes RGB de 256 × 256.")
        return self.red(torch.as_tensor(rgb, device=self.device)).sigmoid().cpu().numpy()


def vista_lateral(rgb):
    """Luminancia reducida del lateral derecho, a la altura de los otros cuerpos."""
    roi = rgb[:, 88:192, 144:256]
    return roi.reshape(len(roi), 52, 2, 56, 2, 3).mean(axis=(2, 4, 5), dtype=np.float32)


def movimiento_lateral(antes, despues):
    # Dos bordes desplazándose en un bloque de 8 x 8 dejan una respuesta local
    # alta. Dos bloques evitan confundir un solo destello con tráfico que se mueve.
    cambio = np.abs(antes - despues)[:, :48] > 17.
    bloques = cambio.reshape(len(cambio), 6, 8, 7, 8).mean(axis=(2, 4)).reshape(len(cambio), -1)
    return np.partition(bloques, -2, axis=1)[:, -2]


def itinerarios_cruce(rng, numeros):
    """Cuatro intersecciones. Varían los sentidos de aproximación y los robots de cada fila."""
    puestos = rng.permutation(16)[numeros]
    grupo, puesto = puestos // 4, puestos % 4
    centros = np.column_stack((np.where(grupo % 2, 3.2, -3.2), np.where(grupo < 2, -3.2, 3.2)))
    sentido = rng.choice([-1., 1.], (4, 2))
    frente = np.eye(2)[puesto % 2] * sentido[grupo]
    inicio = centros - np.where(puesto < 2, .8, 2.)[:, None] * frente
    meta = centros + np.where(puesto < 2, 1.95, .85)[:, None] * frente
    return inicio, frente, meta, puestos
