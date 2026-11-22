# Demostraciones HDF5

## Cosecha móvil en los cuatro surcos

El corte del 21 de noviembre de 2026 contiene 11 cosechas verificadas y una recuperación auxiliar. Cubre 9 de las 12 combinaciones previstas. [Datos y vídeos por cultivo](../simulaciones/cosecha_surcos_20261121/README.md)

Desde la raíz del repositorio, este comando ejecuta las demostraciones físicas con Isaac Sim y reúne los cultivos en `datasets/agriculture/cosecha_surcos_v1`.

```powershell
..\isaac-sim-6.2.0\python.bat tools\collect_field_cycles.py
```

Cada episodio registra la orden y subtarea en español, ambos ojos RGB-D, poses de cámaras, estado móvil, 62 consignas articulares y cuatro velocidades de rueda comandadas a 30 Hz. El HDF5 guarda la observación antes de cada acción. Los dos MP4 por robot muestran el resultado de una de cada dos acciones, a 15 Hz. Los vídeos y las mediciones quedan en `captures/dataset_ciclo_v1`. La salida actual del estudiante cubre las 62 articulaciones.

El recolector genera `manifest.json` cuando hay cosechas verificadas de naranja, plátano, uva y arándano en las tres condiciones de ruta, rama e iluminación. Agrupa las repeticiones de cada condición en una sola partición. Exige recorrido, fruto desprendido, sostén de al menos 15 cm durante 0,5 s, traslado con agarre y reposo final de 0,5 s en la canasta. También comprueba la visibilidad binocular durante la aproximación y el cierre.

Cada entrada del manifiesto incluye la orden global y los índices de inicio y fin de cada subtarea. El índice final es exclusivo.

Las grabaciones en `recuperaciones/` conservan los pasos donde el robot recuperó oposición entre dedos. El manifiesto las identifica como ejemplos auxiliares y las excluye de las particiones de cosecha continua.

El perfil lateral de vendimia usa pulgar, índice y medio. La variante `--cluster-grasp con_apoyo` acerca el meñique a la parte inferior del racimo. Las capturas nuevas conservan las calibraciones de la mano junto con sus huellas SHA-256.

En el naranjal, el controlador mide la fuerza de cada dedo. Si el dedo medio pierde apoyo durante el desprendimiento, usa su posición medida respecto a la naranja para corregir la consigna articular.

Para el racimo, `contactos_racimo.npz` guarda las fuerzas normales y los puntos de contacto respecto del centro de masa a 120 Hz. El verificador comprueba que el pulgar y al menos dos dedos rodean el fruto, que sus conos de fricción permiten equilibrar fuerzas y momentos en seis dimensiones y que las cargas medidas permiten sostener sus 200 g. [Cierre de fuerza, Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/12-2-3-force-closure/)

El pedúnculo simulado usa un resorte amortiguado y un umbral de rotura de 4 N para la naranja y 4,5 N para el racimo. Son parámetros del banco. En campo se han medido fuerzas medias de tracción de 35 a 88 N en cítricos. Un prototipo de vendimia sujetó y cortó el raquis. [Ensayos en cítricos](https://www.sciencedirect.com/science/article/abs/pii/S1537511009001858) · [Prototipo para uvas](https://www.jircas.go.jp/sites/default/files/publication/jarq/29-3-171-177_0.pdf)

## Agricultura bimanual

`agriculture_collect` guarda un HDF5 por episodio completo. En el torso fijo, `dataset.json` indica el esquema `dexman.agriculture.demo.v1`, la semilla, el orden de juntas y el checkpoint. `summary.json` informa resultados por cultivo. [Comando de captura](../README.md#teacher-y-captura-agrícolas)

| Campo | Forma por paso | Contenido |
| --- | --- | --- |
| `rgb` | `[2,224,224,3]` | RGB `uint8`, izquierdo y derecho |
| `depth_m`, `depth_valid` | `[2,224,224]` | Profundidad en metros y máscara válida |
| `angular_state` | `[120]` | `q[60]` en rad y `dq[60]` en rad/s, izquierda antes de derecha |
| `joint_command` | `[60]` | Ángulos aplicados después de limitar el comando, en rad |
| `next_held`, `next_clearance_m` | `[2]`, escalar | Pinza física por mano y altura del punto más bajo del fruto tras el comando |
| `phase`, `time_s` | Escalares | Fase y tiempo simulado |
| `lighting`, `obstacles` | `[2]`, `[11,5]` | Intensidad y temperatura de color. Posición, radio y presencia de cada obstáculo |
| `next_contact`, `next_target_pose` | `[2,5]`, `[7]` | Contactos de yemas y pose local de la fruta después del comando |

RGB-D, estado y fase corresponden al instante previo al comando. `reward` y `done` se refieren a la transición ejecutada. El atributo `instruction` guarda la orden global en español. `subtasks_by_phase` contiene el texto de cada subtarea por ID de fase. El entrenador bimanual lee `joint_limits_rad[60,2]` y `joint_command_space=applied_joint_target_rad` de los metadatos. Los archivos `.partial` identifican capturas interrumpidas.

Para comparar órdenes en la misma escena, captura los tres ejercicios con igual semilla y número de réplicas. En `dataset.json`, `demonstrator` distingue PPO de PPO más reglas de fase. El comando `dataset split` mantiene juntas todas las órdenes de un mismo cultivo, semilla, réplica y reinicio.

## Observación móvil

`agriculture_collect --rover` guarda demostraciones `dexman.agriculture.mobile.v3` a 30 Hz. Cada fila contiene ambos RGB-D, `mobile_state[132]`, la posición y orientación de cada cámara (`camera_pos_env_m[2,3]`, `camera_quat_ros_xyzw[2,4]`) y `joint_command[62]`. Los índices 130 y 131 del estado son el giro y la elevación **medidos** antes de actuar. Los índices 60 y 61 del comando son sus **consignas** en radianes para el paso siguiente. La orden global se guarda en español junto con las subtareas del episodio.

`rover_eyes` guarda una observación individual del surco con la consigna `head_target_rad[2]` y la revisión del montaje `rover_asset_sha256`. Las observaciones anteriores `mobile.v2` se pueden leer con su checkpoint de 60 salidas en `infer_vla`. [Comandos del rover](rover-cosechador.md#teacher-y-demostraciones-del-rover)

## Elevación de un brazo

El VLA v5 usa el esquema 1.1. Cada fila registra `rgb` y `rgb_secondary` de 224 × 224, profundidad, `qpos[30]`, `qvel[30]`, `action[26]`, recompensa y final del episodio. `joint_command[30]` es la etiqueta antes del filtro, mientras que `joint_target[30]` registra el objetivo filtrado.

Usa episodios del criterio v4 de 15 cm y 0,5 s de sostén. El cargador admite esquema 1.0 con metadatos completos para reconstruir `joint_command`. Divide por episodio antes de entrenar y calcula la normalización solo con el lote de entrenamiento. [Captura y partición del entorno de un brazo](mvp.md)
