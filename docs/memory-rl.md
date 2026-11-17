# RL visual multitarea con memoria

Contrato externo: imagen RGB + ángulos actuales q[30] + orden textual → 30 ángulos
objetivo, en radianes y en orden `JOINT_NAMES`. Son posiciones deseadas antes del filtro
del controlador; la dinámica determina los ángulos que el robot realmente alcanza.

```text
RGB ── codificador visual CLIP ──────┐
Texto ── codificador textual CLIP ───┼─ GRU(h anterior) ─ cabeza de acciones ─ 30 ángulos
q[30] ── normalización por límites ──┘       │
                                          └─ h siguiente, separado por robot

Simulador + objetivo asociado al texto ─ recompensa ─ PPO ─ actualiza GRU y cabeza
Estado completo del simulador ─ crítico ────────────────┘
```

CLIP permanece congelado. El actor usa exclusivamente RGB, texto y q actual. El crítico
recibe también velocidades, geometría, meta, acciones anteriores, tarea y tiempo restante;
esa información no se transmite al actor. Esta separación sigue el enfoque de
[actor-crítico asimétrico](https://arxiv.org/abs/1710.06542).

La GRU de 256 unidades conserva contexto temporal por réplica. PPO de la versión instalada
de RSL-RL entrena secuencias de 64 pasos con máscaras de episodio y estados recurrentes.
Cada reset limpia solo la réplica correspondiente. La API de inferencia también limpia la
memoria al cambiar la orden. Para políticas recurrentes es necesario gestionar explícitamente
los estados y límites de episodio; véase la
[documentación de PPO recurrente](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
Esa referencia describe el principio; este proyecto utiliza RSL-RL, no SB3.

Hay dos mecanismos distintos de memoria:

- Durante el intento, la GRU resume observaciones previas y puede influir en la siguiente acción.
- Entre intentos y sesiones, las actualizaciones de PPO quedan en los pesos del actor/crítico.
  Se guarda también el optimizador para continuar aprendiendo. Al reanudar se inicia una
  simulación nueva, por lo que no se restaura una memoria temporal asociada a otra escena.

`episodes.jsonl` conserva resultados de éxitos y fracasos por tarea. Es un registro de métricas,
no un archivo de trayectorias ni una memoria recuperable por el actor. PPO aprende de rollouts
recientes; no reutiliza indiscriminadamente episodios antiguos como un algoritmo off-policy.
Una memoria episódica con búsqueda de experiencias sería una extensión diferente.

## Órdenes y recompensas

El catálogo ejecutable está en `src/dexman/tasks/manipulation/catalog.py`; las recompensas
están en `tasks/manipulation/rewards.py`. Inicialmente hay tres objetivos:

| Tarea | Una orden admitida | Condición física |
| --- | --- | --- |
| lift | Levanta el cubo rojo 15 cm y sostenlo. | Meta 15 cm por encima del inicio, estable, separado de la mesa y con proximidad de dedos |
| push_x_positive | Empuja el cubo rojo 10 cm hacia X positivo de la base del robot. | Desplazamiento +10 cm en X, sobre la mesa, estable |
| push_x_negative | Empuja el cubo rojo 10 cm hacia X negativo de la base del robot. | Desplazamiento −10 cm en X, sobre la mesa, estable |

Las direcciones se refieren a la base del robot, no al lado izquierdo/derecho de la imagen.
Los objetivos se calculan respecto a la posición inicial aleatoria del cubo. Durante entrenamiento
se muestrean tarea y frase por episodio; hay variantes en inglés y español. La proximidad de los
dedos es una heurística de agarre, no una medición de fuerzas.

La recompensa combina progreso hacia la meta, aproximación, suavidad y bono de éxito.
Se exige mantener la condición 15 pasos (0,5 s). En empuje, elevar la cara inferior del cubo
más de 1,5 cm invalida el intento, para que levantar y recolocar no cuente como empujar.
El episodio termina por éxito, fallo o al vencer un plazo de 10 s. El plazo es parte de la tarea:
se trata como final finito y no se aplica bootstrap de valor después de él.

El actor produce 30 mandos normalizados independientes que el adaptador convierte a posiciones
absolutas dentro de los límites, centradas en la postura de agarre existente. No utiliza las
cuatro sinergias del teacher de 26 acciones. El entorno aplica una vez el filtro de 0,05.

## Uso

Desde la raíz del proyecto, con las dependencias de Isaac Lab y `configs/vla-requirements.txt`:

```powershell
# Ejecutar este comando inicia realmente entrenamiento; no se ha iniciado al escribir el código.
.\scripts\run.ps1 memory_rl --mode train --num_envs 1 --iterations 3000 --output logs/memory_rl/run01 --headless

# Continuar pesos y optimizador en una carpeta NUEVA.
.\scripts\run.ps1 memory_rl --mode train --checkpoint logs/memory_rl/run01/model_2999.pt --num_envs 1 --iterations 1000 --output logs/memory_rl/run02 --headless

# Evaluación por tareas, sin actualizar pesos; omitir --headless para ver la escena.
.\scripts\run.ps1 memory_rl --mode evaluate --checkpoint logs/memory_rl/run01/model_2999.pt --episodes 60 --output logs/memory_rl/eval01 --headless

# Evaluar una orden concreta y su correspondiente objetivo físico.
.\scripts\run.ps1 memory_rl --mode evaluate --checkpoint logs/memory_rl/run01/model_2999.pt --instruction 'Levanta el cubo rojo 15 cm y sostenlo.' --episodes 20 --output logs/memory_rl/eval_lift --headless
```

Las rutas de checkpoint son ejemplos: requieren un entrenamiento previo. La primera ejecución
descarga CLIP si no está disponible. El directorio guarda `encoder/` con procesador y backbone,
checkpoints con contrato/optimizador, TensorBoard, `contract.json`, `run.json` y `episodes.jsonl`.
Conservar `encoder/` junto a los checkpoints. La evaluación añade `evaluation.json` con
cantidad de episodios y tasa de éxito por tarea; una tarea no observada tiene tasa `null`.
Empezar con una réplica y aumentar solo tras medir VRAM y tiempo de render/inferencia.

API de inferencia independiente del simulador:

```python
from dexman.memory_rl.inference import MemoryPredictor

policy = MemoryPredictor("logs/memory_rl/run01/model_2999.pt", num_envs=1, device="cuda:0")
# rgb: uint8 [H,W,3]; q: [30], en radianes y orden JOINT_NAMES.
angles = policy.predict_batch([rgb], [q], ["Levanta el cubo rojo 15 cm y sostenlo."])[0]
# Repetir con nueva imagen/q conserva memoria. Al empezar un intento nuevo:
policy.reset([True])
```

## Alcance y validación

Es una implementación para entrenar; no se entrega una política robótica ya entrenada.
Las pruebas CPU con datos sintéticos comprueban memoria, aislamiento por réplica, actualización
PPO, checkpoints, límites y semántica de recompensas. No demuestran éxito físico.
Resultado del 12-11-2026: **10 pruebas aprobadas**, CLI/configuración importados y lanzador
verificado con un stub. Registro completo en [memory-rl-validation.json](memory-rl-validation.json).
Falta validar esta ruta en Isaac Sim: render, integración completa, alcance de ambos empujes,
convergencia, tasas de éxito y comportamiento ante oclusiones.

La escena actual conserva un cubo rojo y el inicio en preagarre. No aprende aún a manipular
objetos arbitrarios, abrir cajones, apilar ni ejecutar secuencias libres de instrucciones.
Las órdenes fuera del catálogo se rechazan antes de actuar. Añadir una tarea exige escena/reset,
recompensa, criterio de éxito y datos de entrenamiento coherentes; cambiar solo el texto no basta.
CLIP y las variantes textuales tampoco garantizan comprensión de español ni generalización.

Para medir el efecto de la memoria: comparar el mismo checkpoint con memoria conservada frente
a reiniciada en cada paso, usando las mismas tareas/semillas y oclusiones controladas. Evaluar
por separado instrucciones y objetos no vistos cuando se amplíe el catálogo. Cambiar el catálogo
invalida el contrato de checkpoints actuales y requiere una migración explícita o reentrenar.
