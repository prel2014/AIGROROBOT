# Arquitectura

## Distribución del software

Se usa un patrón de **núcleo funcional y capa de ejecución** (functional core / imperative shell),
con módulos agrupados por responsabilidad. El núcleo calcula comandos/recompensas; los adaptadores
gestionan el simulador, modelos y almacenamiento; los CLI construyen y ejecutan cada aplicación.

| Ubicación | Responsabilidad |
| --- | --- |
| `robot/commands.py` | Conversión de 30 mandos independientes a ángulos acotados |
| `perception/clip.py` | Carga congelada de CLIP, embeddings y validación RGB/texto compartida |
| `agents/recurrent.py` | Actor GRU adaptado a RSL-RL |
| `tasks/manipulation/catalog.py` | Órdenes y desplazamientos objetivo |
| `tasks/manipulation/rewards.py` | Métricas y recompensas como funciones de tensores |
| `tasks/manipulation/config.py`, `environment.py` | Configuración e integración de tareas con Isaac Sim |
| `memory_rl/config.py`, `checkpoint.py` | Hiperparámetros y contrato versionado |
| `memory_rl/observations.py`, `vec_env.py` | Composición de observaciones y adaptación al entrenador |
| `memory_rl/training.py` | Entrenamiento y reanudación del optimizador |
| `memory_rl/inference.py`, `evaluation.py` | Predicción con memoria y evaluación sin optimizador |
| `cli/` | Argumentos, carga de dependencias, rutas y ciclo de vida de la aplicación |

Los cinco lanzadores de captura/VLA/RL visual en `scripts/` delegan en `dexman.cli`.
No se duplica el algoritmo PPO de RSL-RL ni el controlador físico de Isaac Lab.
Las reglas para ampliar el proyecto y migrar imports están en [development.md](development.md).

## Objetivo multitarea con memoria

La ruta `memory_rl` implementa el nuevo contrato RGB + q[30] + texto → q_objetivo[30],
con memoria GRU por réplica y entrenamiento PPO directamente sobre la política visual.
Sus instrucciones cambian la recompensa y el criterio de éxito. El actor no recibe qvel,
pose del objeto, identificador de tarea ni coordenadas de la meta; el crítico sí usa estado
privilegiado durante entrenamiento. Los pesos guardan aprendizaje entre sesiones, mientras
la GRU se reinicia al terminar un episodio. Detalles, límites y comandos en
[memory-rl.md](memory-rl.md).

Las secciones siguientes describen el teacher PPO v1 y el VLA Transformer entrenado por imitación.
La nueva ruta convive con ambas y utiliza checkpoints propios incompatibles con ellas.

## Robot y acciones

El conjunto tiene 30 juntas de revolución: 6 UR10e y 24 Shadow. El teacher PPO tiene 26 acciones:
6 del brazo, 16 articulaciones directas de la mano y 4 sinergias de flexión media/distal.
`src/dexman/robot/joints.py` fija el orden para que no dependa de cómo PhysX enumere las juntas.
El manifiesto USD es la fuente de límites físicos y el entorno falla si falta una junta.

La mano descargada usa `robot0_WRJ1/WRJ0`, `FFJ3..0`, etc. No usa la nomenclatura `rh_*`
de la nueva Shadow multiphysics. El mando final de cada dedo no pulgar mueve J1 y J0 con
la misma fracción de recorrido. Es una aproximación de sinergia posicional de 20 mandos,
no el modelo mecánico de tendones de la mano real; las 24 juntas siguen existiendo en PhysX.
La política visual predice las 30 juntas individualmente, sin aplicar esas cuatro sinergias
en inferencia. Las demostraciones del teacher siguen conteniendo sus movimientos acoplados;
para aprender destreza independiente hacen falta demostraciones que la muestren.

La composicion corregida aplica `PhysicsDriveAPI:angular` a las 24 juntas de la mano. El USD
original carecia de esos motores: escribir ganancias por la API de tensores no bastaba para
controlar las articulaciones. Los cuatro esquemas de tendones originales de rigidez cero se
eliminan en la composicion, porque este modelo ya acopla las falanges por software. Se conservan
sin modificar los modelos originales descargados.

En v1 las acciones son posiciones absolutas normalizadas en [-1,1], con un mapa lineal por
tramos: -1/1 corresponden a los limites y 0 a la postura validada de `configs/grasp_seed.json`.
El brazo dispone de ±0,8 rad alrededor de esa postura, intersectado con los limites fisicos;
la mano conserva los limites originales. Un filtro de 0,05 suaviza targets a 30 Hz; los motores
PD y la compensacion de gravedad operan a 120 Hz. El actor empieza cerca de la accion neutra,
con menos ruido en brazo y muneca. v0 conserva la codificacion antigua para diagnostico, pero
sus checkpoints previos no describen la dinamica corregida y no deben reanudarse en v1.

## Entorno y aprendizaje

Una mesa, un cubo de 5 cm/80 g y un brazo fijo por replica, inicialmente en preagarre. La posicion
XY del cubo varia ±2 cm. El objetivo final es elevarlo 15 cm y sostenerlo durante 15 pasos,
con baja velocidad y proximidad del pulgar y de al menos otros dos dedos. La proximidad es
una heurística de agarre: no demuestra por sí sola fuerzas de contacto ni ausencia de deslizamiento.

Observación PPO: q[30], dq[30], vector palma-objeto[3], posición objeto local[3], vector
objeto-meta[3], objetivo articular filtrado codificado[26] = 95 valores. La recompensa combina acercamiento,
proximidad de dedos, elevación, cercanía a la meta y penalización por cambios de acción.
En v1 se usa progreso de un potencial continuo, coste temporal y por desplazamiento lateral,
penalizacion de cambios de accion, y un bono terminal de 100 por exito. Estar cerca sin progresar
no produce una recompensa positiva de acercamiento en cada paso. El cubo debe tener todos sus
vertices separados al menos 5 mm de la mesa; inclinarlo apoyado no cuenta. La tolerancia XY es
6 cm. El criterio v3 exige elevación medida igual o mayor que la altura de etapa, con
sobreelevación máxima de 2,5 cm; no admite terminar por debajo de la meta. Se requiere baja
velocidad y proximidad de los dedos durante 0,5 s. Se termina por exito, caida, escape o 10 s.

El curriculum comienza a 1,5 cm y avanza por 3, 6, 10 y 15 cm cuando al menos el 60 % de una
ventana de 64 episodios termina con exito. La evaluacion, las capturas de demostraciones y el
modo play exigen la etapa final de 15 cm. Los exitos del entrenamiento inicial no se presentan
como validacion de la tarea final. Aprender un acercamiento amplio desde HOME queda fuera de
este primer curriculum: el reset usa la postura fisicamente verificada de preagarre.

PPO usa estados privilegiados y genera demostraciones. Tanto su actor como su crítico usan un
Transformer de estado con normalización empírica: tokens articulares, geometría y objetivos filtrados.
La cámara del teacher es un sensor de supervisión y grabación; PPO recibe los 95 valores de estado.
Sus checkpoints guardan la arquitectura además de pesos, optimizador y currículo. Los MLP anteriores
se rechazan antes de abrir el simulador; el cambio requiere entrenar una corrida nueva.
El contrato de arquitectura del teacher es versión 3: usa el estado del filtro de posición en
las últimas 26 entradas. Los checkpoints anteriores que observaban acciones crudas no se mezclan
con esta representación. La penalización de cambios de acción conserva su definición original.
La normalización del actor se estima con las réplicas TRAIN del bootstrap y después permanece
fija; el crítico actualiza su normalización durante PPO. Tras imitar, la desviación inicial de
exploración es 0,02 en brazo/muñeca y 0,04 en dedos, con rango permitido 0,01..0,35.
El checkpoint v4 separa los grupos del optimizador: 32 actualizaciones iniciales solo del
crítico, con tasa 1e-4; después el actor comienza con 1e-6. La guardia KL de 0,02 comprueba
64 observaciones reales del rollout y revierte el actor y sus momentos si excede ese límite,
reduciendo su tasa hasta un mínimo de 1e-7. La reanudación conserva estas tasas y contadores.

El estudiante VLA v4 usa CLIP ViT-B/32 congelado: conserva sus 49 parches visuales y CLS, proyectados
a 512 dimensiones, junto con el embedding de texto de 512. Proyecta esas modalidades, q/dq normalizado
y memoria previa al ancho común 256 y concatena los tokens. La secuencia de contexto tiene 53 tokens:
50 visuales, uno de texto, uno de estado y uno de memoria. No aplana las modalidades en un único token.
Usa seis bloques pre-LayerNorm, ocho cabezas y capas internas de 1024 unidades, con dropout 0,1 durante
entrenamiento. Ocho consultas producen ocho vectores de 30 ángulos en orden `JOINT_NAMES`. Una consulta
adicional escribe el embedding de memoria de 256 valores. El contexto no atiende a consultas de salida,
y la consulta de memoria no lee las de acciones. Una Tanh y los límites articulares acotan los objetivos.

El entorno `Dexman-UR10e-Shadow-Lift-VLA-v2` acepta directamente esos 30 ángulos y aplica una
vez el filtro de 0,05. Sus observaciones de estado contienen únicamente q/dq; pose del objeto
y meta se usan para medir éxito y recompensa. La inferencia obtiene RGB de la cámara y texto
de la instrucción. Cada réplica pasa la memoria devuelta al paso siguiente y la reinicia
junto con su chunk al terminar un episodio o cambiar la instrucción.

La clonación supervisa posiciones deseadas antes del filtro: `joint_command` del esquema 1.1,
o la decodificación de `action` en grabaciones 1.0 corregidas con metadatos completos. No utiliza
el `joint_target` ya suavizado como etiqueta. La MSE se divide por el semirrecorrido de cada
junta y excluye padding; q/dq se normalizan con estadísticas solo de train. Los checkpoints
guardan versión, orden, unidades, límites y semántica de los objetivos. El entrenamiento propaga
los gradientes a través de la memoria en secuencias contiguas dentro de cada episodio.
Los pesos se guardan como VLA v4 con configuración completa de arquitectura y contrato de características
CLIP. Las secuencias por defecto tienen 16 observaciones; los pasos de padding no actualizan la memoria.
Los checkpoints v3 no son compatibles, pero sus demostraciones HDF5 corregidas pueden reutilizarse.

Esto implementa un baseline de clonación de comportamiento condicionado por lenguaje.
Una sola tarea y una sola instrucción no entrenan un VLA generalista; harán falta más tareas,
objetos, instrucciones, variación visual y evaluaciones por tarea para demostrar generalización.

## Recursos

| Red | Parámetros entrenables | Pesos FP32 aproximados |
| --- | ---: | ---: |
| Teacher: actor Transformer | 607.796 | 2,43 MB |
| Teacher: crítico Transformer | 604.545 | 2,42 MB |
| Estudiante: Transformer VLA v4 | 5.239.070 | 20,96 MB |

Los checkpoints reanudables añaden los estados del optimizador, estadísticas y metadatos.
Los pesos congelados de CLIP se guardan aparte en `backbone/`. Cada observación del VLA contiene
50×512 valores visuales, 512 de texto, 60 de q/dq y 256 de memoria: 105.712 bytes en FP32 antes
de las proyecciones, sin incluir imágenes originales ni activaciones intermedias.

PPO: 32 réplicas y sin cámaras por defecto. Captura: 1 réplica por defecto, configurable con
`--num_envs` para grabación paralela; una cámara RGB-D 224×224 a 30 Hz por réplica.
VLA: batch 8, backbone congelado. Son puntos de partida para la RTX 3060 de 12 GB
detectada, no una garantía de memoria disponible. Reducir réplicas/batch si es necesario.

Se verificaron los motores, un agarre programado con el criterio completo y PPO con 16 replicas.
Las primeras 80 iteraciones registraron aproximadamente 11 % de exitos en la etapa inicial.
La convergencia de PPO a 15 cm y el entrenamiento VLA siguen pendientes.

## Decisiones de fusión y memoria

La concatenación se hace sobre la secuencia de tokens después de proyectar cada modalidad al mismo
ancho. Es el patrón de fusión que muestran [GR-1](https://arxiv.org/html/2312.13139v1) y
[Octo](https://octo-models.github.io/paper.pdf). Sus encoders difieren de los nuestros: GR-1 usa
MAE para RGB y CLIP para texto; Octo usa parches convolucionales y T5. No se presentan como modelos
idénticos ni como evidencia de rendimiento de esta implementación.

La memoria de lectura/escritura y la propagación de gradientes entre pasos siguen el principio de
[Recurrent Memory Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf),
que se evaluó en secuencias y lenguaje. Su aplicación aquí es una adaptación para control robótico.
Mantener detalle espacial también concuerda con la separación semántica/espacial de
[CLIPort](https://cliport.github.io/), cuya política de acciones es distinta.
Estas referencias justifican decisiones de diseño; la capacidad de levantar 15 cm debe medirse en simulación.

El arranque del teacher usa imitación de trayectorias físicas de calibración antes de PPO para
proporcionar ejemplos de despegue y sostén. La inicialización por comportamiento está documentada
en [Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations](https://homes.cs.washington.edu/~todorov/papers/RajeswaranRSS18.pdf).
Aquí se usa inicialización seguida de PPO, sin implementar el algoritmo DAPG completo.
Los éxitos de calibración se registran aparte y no acreditan éxito de la política aprendida.
La validación pendiente incluye pose HOME, orientación del montaje durante movimiento,
autocolisiones, campo de visión con el brazo posado, estabilidad PD y éxito real del agarre.

El flujo para una GPU con varias réplicas, entrenamiento offline y evaluación VLA está en
[vla-single-gpu.md](vla-single-gpu.md). Las pruebas de red con datos sintéticos y CLIP pequeño
se registran por separado en `vla-validation.json`; no son entrenamientos de la tarea robótica.
