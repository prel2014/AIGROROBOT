# Ensayos de naranja en Isaac Sim

Cuatro órdenes guiadas en español y dos ensayos de comparación. Misma semilla de simulación (751). Los MP4 exterior y RGB-D de cada fila tienen los mismos fotogramas a 15 fps.

| Ensayo | Máxima elevación | Sostén válido ≥15 cm | Depósito | Grabaciones y datos |
|---|---:|---:|---|---|
| Rover estacionado | 0,0 cm | 0,0 s | No | [Exterior](estacionada/robot_000.mp4) · [Ojos RGB-D](estacionada/robot_000_ojos.mp4) · [Mediciones](estacionada/resultado.json) |
| Surco con roca | 17,0 cm | 0,0 s | No | [Exterior](roca/robot_000.mp4) · [Ojos RGB-D](roca/robot_000_ojos.mp4) · [Mediciones](roca/resultado.json) |
| Naranja izquierda | 0,0 cm | 0,0 s | No | [Exterior](izquierda/robot_000.mp4) · [Ojos RGB-D](izquierda/robot_000_ojos.mp4) · [Mediciones](izquierda/resultado.json) |
| Naranja derecha | 0,0 cm | 0,0 s | No | [Exterior](derecha/robot_000.mp4) · [Ojos RGB-D](derecha/robot_000_ojos.mp4) · [Mediciones](derecha/resultado.json) |
| Surco con roca, cierre detenido por contacto | 0,0 cm | 0,0 s | No | [Exterior](roca_cierre_contacto/robot_000.mp4) · [Ojos RGB-D](roca_cierre_contacto/robot_000_ojos.mp4) · [Mediciones](roca_cierre_contacto/resultado.json) |
| Roca próxima al eje de la ruta | Sin recogida | — | — | [Exterior](roca_frenado/robot_000.mp4) · [Ojos RGB-D](roca_frenado/robot_000_ojos.mp4) · [Mediciones](roca_frenado/resultado.json) |

La altura se mide desde el punto inferior del fruto. El sostén requiere oposición del pulgar con dos dedos, velocidad y deslizamiento bajos durante al menos 0,5 s continuos. El depósito requiere reposo en la canasta después de abrir y retirar la mano.

Con el cierre progresivo, el rover recorrió 3,258 m y pasó a 13,7 cm de la roca. La naranja alcanzó 16,95 cm. Solo 3 de los 66 pasos de sostén cumplieron el límite de deslizamiento de todos los dedos cargados. La repetición con la configuración final y la misma semilla reprodujo la elevación y el margen. [Verificación física sin renderizado](verificacion_sin_render.json)

La selección conserva ambas naranjas en la misma rama. Cada informe identifica el fruto solicitado y registra el desplazamiento y la carga en la vecina. La ruta guarda el margen geométrico del chasis respecto a la roca en `ruta.json`. [Comparación de cuatro ajustes de la palma](calibracion_palma.json) · [Comandos](../../docs/rover-cosechador.md#agarre-y-depósito-en-el-surco-de-naranjos)
