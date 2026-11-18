# Tareas agrícolas bimanuales

El flujo actual entrena con fruta suelta. La vegetación aporta contexto y oclusiones. [Comandos de uso](../README.md#teacher-y-captura-agrícolas)

| Escena | Orden abreviada | Fases |
| --- | --- | --- |
| `citrus` | Recoge la naranja más cercana y ponla en la bandeja izquierda. | Agarrar → elevar 15 cm → depositar y estabilizar. |
| `banana` | Recoge el plátano con la derecha, pásalo a la izquierda y deposítalo. | Agarrar → elevar 10 cm → confirmar contacto de ambas manos → soltar con la derecha → depositar. |
| `vineyard` | Levanta 15 cm el racimo más cercano y sostenlo. Evita la espaldera. | Agarrar → elevar → sostener 0,5 s. |
| `berries` | Recoge el arándano más cercano con pulgar e índice sin desplazar los demás. | Confirmar pinza opuesta → elevar 3 cm → sostener 0,5 s. |

Cada orden añade «Si ves obstáculos, esquívalos». El simulador comprueba contactos de yemas, penaliza caídas y poses inválidas y reinicia solo las réplicas terminadas. Hay 11 tipos de obstáculos de mesa o pasillo. Cada réplica intenta añadir o retirar uno cada 1,5 a 4,5 s si la zona está libre de brazos y fruta. La luz cambia durante el episodio.

## Formato de demostraciones

La captura guarda la orden global y la subtarea en español según la fase de cada muestra. La API del VLA acepta el paso activo mediante `subtasks` y lo combina con la orden global antes de CLIP. [Campos y unidades](dataset.md#agricultura-bimanual)

## Pendiente

- Medir el éxito físico del teacher por cultivo y calibrar los umbrales de agarre.
- Conectar las demostraciones al entrenamiento VLA bimanual.
- Modelar el desprendimiento y validar la cosecha de frutos unidos a la planta.
