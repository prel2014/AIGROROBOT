# Demostraciones HDF5

## Agricultura bimanual

`agriculture_collect` guarda un HDF5 por episodio completo. `dataset.json` indica el esquema `dexman.agriculture.demo.v1`, la semilla, el orden de juntas y el checkpoint. `summary.json` informa resultados por cultivo. [Comando de captura](../README.md#teacher-y-captura-agrícolas)

| Campo | Forma por paso | Contenido |
| --- | --- | --- |
| `rgb` | `[2,224,224,3]` | RGB `uint8`, izquierdo y derecho |
| `depth_m`, `depth_valid` | `[2,224,224]` | Profundidad en metros y máscara válida |
| `angular_state` | `[120]` | `q[60]` en rad y `dq[60]` en rad/s, izquierda antes de derecha |
| `joint_command` | `[60]` | Ángulos aplicados después de limitar el comando, en rad |
| `phase`, `time_s` | Escalares | Fase y tiempo simulado |
| `lighting`, `obstacles` | `[2]`, `[11,5]` | Intensidad y temperatura de color. Posición, radio y presencia de cada obstáculo |
| `next_contact`, `next_target_pose` | `[2,5]`, `[7]` | Contactos de yemas y pose local de la fruta después del comando |

RGB-D, estado y fase corresponden al instante previo al comando. `reward` y `done` se refieren a la transición ejecutada. La orden, el resultado y la calibración se guardan como atributos por episodio. Los archivos `.partial` identifican capturas interrumpidas.

## Elevación de un brazo

El VLA v5 usa el esquema 1.1. Cada fila registra `rgb` y `rgb_secondary` de 224 × 224, profundidad, `qpos[30]`, `qvel[30]`, `action[26]`, recompensa y final del episodio. `joint_command[30]` es la etiqueta antes del filtro, mientras que `joint_target[30]` registra el objetivo filtrado.

Usa episodios del criterio v4 de 15 cm y 0,5 s de sostén. El cargador admite esquema 1.0 con metadatos completos para reconstruir `joint_command`. Divide por episodio antes de entrenar y calcula la normalización solo con el lote de entrenamiento. [Captura y partición del entorno de un brazo](mvp.md)
