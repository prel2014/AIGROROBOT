# Contrato del dataset VLA, versión 1.1

La política v5 aprende a predecir 30 ángulos absolutos y actualizar su memoria a partir de tokens espaciales CLIP de dos RGB sincronizados, texto y q/dq.
Las nuevas capturas publican esquema 1.1. También pueden leerse episodios 1.0 del teacher
corregido v1 si conservan todos los metadatos necesarios para decodificar sus mandos.
Todos los episodios deben declarar `reward_version=4`,
`success_criterion=opposed_contact_minimum_lift_and_hold_v2`, `minimum_lift_required_m=0.15`,
`success_lift_overshoot_m=0.025` y `hold_seconds=0.5`. Estos campos se conservan en el
checkpoint VLA y se comparan con el entorno. Los datos del criterio anterior se recapturan:
su atributo de éxito no demuestra que alcanzaran los 15 cm mínimos.

Un HDF5 por episodio completo. `.partial` identifica un episodio interrumpido y no se usa
en entrenamiento. Por defecto solo se publican episodios con éxito. Se limita el número de
intentos: si el teacher no alcanza el objetivo, el colector falla en vez de grabar indefinidamente.

| Campo | Forma por paso | Unidad/contenido |
|---|---|---|
| rgb | 224×224×3 | uint8, RGB |
| rgb_secondary | 224×224×3 | uint8, vista opuesta sincronizada |
| depth_secondary_m | 224×224×1 | float, profundidad opuesta en m; 0 inválido |
| depth_m | 224×224×1 | float, distancia al plano de imagen en m; 0 inválido |
| qpos | 30 | rad, orden `JOINT_NAMES` |
| qvel | 30 | rad/s, mismo orden |
| fingertip_normal_force_w_n | 5×3 | Fuerza normal del cubo sobre cada yema, en N y ejes del simulador |
| action | 26 | mando absoluto normalizado [-1,1], antes del filtro |
| joint_command | 30 | ángulos deseados antes del filtro, en rad |
| joint_target | 30 | target tras decodificar y suavizar, rad |
| reward | escalar | recompensa de la transición |
| done | escalar | última transición del episodio |
| timestamp_s | escalar | tiempo simulado desde inicio del episodio |

Los atributos incluyen instrucción, éxito, versión de esquema, orden de juntas/acciones,
semilla, checkpoint y su SHA-256, intrínsecos, posición mundial y cuaternión ROS de la cámara
en XYZW. Las dos cámaras son fijas: se guardan dos calibraciones por episodio. Los contratos incluyen `camera`, `camera_secondary`, escenario, forma, dimensiones y umbral de contacto. Las capturas de una sola cámara no se reutilizan en v5.
Las capturas instrumentadas registran el orden de las yemas en `fingertip_contact_names`:
índice, medio, anular, meñique y pulgar. Las capturas nuevas exigen este campo.

Cada fila representa observación(t) y acción(t). Se captura la imagen ANTES de `env.step`:
Isaac Lab reinicia automáticamente al terminar y la imagen devuelta después podría ser del
siguiente episodio. La recompensa/done pertenecen a la acción de esa fila. No se publica una
observación terminal `t+1`; no se mezcla una imagen tras el reset con una acción terminal.

Los splits de entrenamiento/validación se hacen por episodio y quedan en `split.json`.

Los datos nuevos v1 incluyen `arm_center_q`, `arm_half_range_rad`, `action_center_q` y
`action_encoding=normalized_absolute_position_synergy_centered_v1`. Son necesarios para
interpretar las acciones: 0 corresponde al preagarre calibrado, no al punto medio de los limites.
El entrenamiento VLA rechaza contratos de control mezclados, y la inferencia valida que el
orden, límites y filtro del checkpoint coincidan con el simulador. La captura exige un teacher v1 compatible y
solo publica exitos del objetivo final de 15 cm, no de las etapas iniciales del curriculum.
El esquema 1.1 añade `joint_limits_rad` [30,2], `action_smoothing` y
`target_semantics=desired_joint_position_before_smoothing`. Se comprueba que `joint_command`
coincida con la decodificación de `action`. En esquema 1.0, se reconstruye esa etiqueta usando
sus centros, rango del brazo y manifiesto de juntas. No se usa `joint_target` como etiqueta:
ya incluye el filtro, que volverá a aplicarse al actuar en el simulador.
La migración del esquema 1.0 presupone el filtro 0,05 de la tarea corregida original;
grabaciones con otro filtro requieren metadatos explícitos y una configuración de evaluación coincidente.

Los chunks de 8 pasos × 30 ángulos se rellenan al final repitiendo el último paso y tienen una máscara
para que el padding no contribuya a la pérdida. La normalización q/dq se calcula solo con train.
Los formatos LeRobot/RLDS requieren un exportador adicional; estos archivos no se anuncian
como compatibles directamente con dichos formatos.
