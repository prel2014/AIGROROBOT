# Transformer con memoria, varios robots en una GPU

Este escenario usa réplicas de simulación (`num_envs`) en un único proceso y una única GPU.
Cada réplica tiene UR10e, Shadow Hand, mesa, cubo con posición aleatoria y cámara RGB-D.
Los robots comparten la política; no se entrena una red independiente por robot.

**VLA v4.** Recibe tokens espaciales de CLIP para RGB, embedding de texto, estado articular y memoria.
Devuelve objetivos para las 30 juntas, incluidas las falanges, y memoria actualizada.

El teacher del lanzador usa la tarea corregida **v1** y el directorio PPO
`logs/rsl_rl/ur10e_shadow_lift_v1/`. La mano tiene motores angulares explicitos y comienza en
preagarre. La accion neutra mantiene el brazo en esa postura y cierra hacia una forma de agarre
fisicamente verificada. El curriculum eleva primero 1,5 cm y despues 3, 6, 10 y 15 cm; la
promocion exige 60 % de exitos en una ventana de 64 episodios. Las evaluaciones y demostraciones
usan siempre 15 cm. Los checkpoints antiguos no son compatibles con la nueva codificacion.

Para continuar un checkpoint del nuevo teacher Transformer en una carpeta nueva:

```powershell
.\scripts\vla_scenario.ps1 -Stage teacher -Resume -Robots 16 -Headless -Checkpoint 'RUTA_TRANSFORMER.pt' -RunName continuar_tf
```

`-Iterations` limita las iteraciones de la nueva ejecucion (son adicionales al reanudar).
Se recuperan pesos, optimizador, currículo, contadores y estados aleatorios; los episodios
físicos empiezan de nuevo. Los checkpoints MLP anteriores son incompatibles con la arquitectura
nueva y exigen empezar sin `-Resume` ni `-Checkpoint`. La corrida activa no cambia de modelo.

## Modelo

- CLIP ViT-B/32 congelado produce 50 tokens visuales (49 parches + CLS) de 512 valores y un embedding de texto de 512.
- El Transformer proyecta cada modalidad al ancho 256 y concatena visión, texto, estado y memoria sobre el eje de tokens.
- Usa seis bloques de atención de ocho cabezas; recibe 60 valores articulares (q/dq) normalizados con train y memoria de 256 valores.
- Devuelve 8 × 30 ángulos en radianes y el siguiente embedding de memoria. El entrenamiento por imitación usa secuencias contiguas y máscaras al final del episodio.

`teacher` entrena PPO con actor y crítico Transformer de estado; `train` entrena el VLA por
imitación. `evaluate` usa `Dexman-UR10e-Shadow-Lift-VLA-v2`, que recibe 30 ángulos. Los checkpoints
v4 guardan contrato, arquitectura, pesos, optimizador y época. Para cambiar desde un estudiante
v3, vuelve a entrenar con las demostraciones HDF5 compatibles.

Es un baseline apropiado para probar el flujo con una tarea acotada. CLIP no aporta por sí solo precisión de agarre;
hay que medir éxito en el simulador. Una instrucción y un cubo no permiten demostrar comprensión de lenguaje,
generalización a otras tareas o transferencia al robot físico. Añadir diversidad exige modificar tareas/demostraciones
y evaluar por objeto e instrucción. La MSE de validación no sustituye el porcentaje de agarres exitosos.

## Preparación

Ejecutar desde la raíz del proyecto en PowerShell, con conda/venv desactivado:

```powershell
.\scripts\setup.ps1
& ..\isaac-sim-6.2.0\python.bat -m pip install -r configs/vla-ui-requirements.txt
```

La instalacion es obligatoria antes del primer preview. El lanzador comprueba los paquetes del Python de Isaac Sim
antes de iniciar cada etapa y muestra los comandos de preparacion si faltan dependencias.

El segundo comando incorpora Transformers y matplotlib al Python 3.12 de Isaac Sim. Para una máquina sin ventanas,
puede instalarse `configs/vla-requirements.txt` sin matplotlib. No instalar estas dependencias sobre Python 3.14
para ejecutar la simulación. La primera carga de CLIP requiere descarga; el entrenamiento guarda una copia exacta
del backbone y el procesador junto a los checkpoints para inferencia posterior sin red.

Si PowerShell bloquea los scripts por su política de ejecución, ejecutar el script local en un proceso dedicado,
por ejemplo `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\vla_scenario.ps1 -Stage preview -Robots 1`.
Esto no cambia la política persistente del equipo.

En esta revision de Isaac Lab el argumento nativo para simulacion sin ventana es `--viz none`.
`run.ps1` traduce `--headless` automaticamente para preview, PPO, captura e inferencia en simulacion;
`vla_scenario.ps1 -Headless` sigue siendo la interfaz del escenario. En entrenamiento offline
`--headless` es una opcion propia de `train_vla.py` y se conserva.

Las etapas siguientes se ejecutan de forma secuencial para compartir memoria GPU. Empezar con 1 réplica para revisar
pose, colisiones y campo de visión; luego probar 2 y 4 con cámara. El teacher sin cámara puede admitir más réplicas.
No se garantiza un número de robots: depende de la VRAM libre, PhysX, RTX y batch. Headless sigue renderizando las
cámaras y necesita GPU/RTX. `Robots` controla la simulación; `BatchSize` controla el entrenamiento offline.

## 1. Revisar escena y entrenar el teacher PPO

```powershell
.\scripts\vla_scenario.ps1 -Stage preview -Robots 1
.\scripts\vla_scenario.ps1 -Stage preview -Robots 4 -Headless
.\scripts\vla_scenario.ps1 -Stage teacher -Robots 16 -Headless
```

Para ver el teacher mientras entrena, omitir `-Headless`. Preview aplica acciones cero, no demuestra un agarre.
El teacher PPO debe completar el curriculum antes de generar demostraciones. Se conserva la tarea de
levantar el cubo 15 cm y sostenerlo. Los checkpoints PPO aparecen bajo `logs/rsl_rl/ur10e_shadow_lift_v1/`.

## 2. Capturar demostraciones con varios robots

Sustituir las rutas de ejemplo por archivos existentes:

```powershell
.\scripts\vla_scenario.ps1 -Stage collect -Robots 4 -Headless -Episodes 200 -Checkpoint 'RUTA_PPO.pt' -RunName demos01
# Con ventana: omitir -Headless
```

Cada robot publica HDF5 en su propia carpeta `env_000`, `env_001`, etc., bajo `datasets/demos/demos01/<fecha>/`.
`Episodes` y el presupuesto de intentos se aplican al total del conjunto. Se guardan solo éxitos completos; las
grabaciones que siguen abiertas al alcanzar la cuota quedan como `.partial` y se excluyen del entrenamiento.
La observación se copia antes de actuar para no mezclar imágenes del siguiente episodio tras el autoreset.
Cada cámara conserva su calibración mundial. Las semillas/ID de réplica quedan en los metadatos.
El esquema 1.1 guarda `joint_command` antes del filtro y `joint_target` después. Se entrena con
el primero, para aplicar el filtro una sola vez en inferencia. Las grabaciones 1.0 corregidas
con centros, límites y orden de juntas recuperables se convierten al leerlas, sin modificar archivos.
`-Instruction 'Pick up the red cube and hold it above the table.'` permite registrar texto explícito.

## 3. Entrenar el Transformer

```powershell
# Sin ventanas; solo ejecutar cuando quieras iniciar el entrenamiento
.\scripts\vla_scenario.ps1 -Stage train -Headless -Dataset datasets/demos/demos01 -Epochs 20 -BatchSize 8 -RunName vla01
# Con gráfica de pérdidas: otra corrida, otro nombre
.\scripts\vla_scenario.ps1 -Stage train -Dataset datasets/demos/demos01 -Epochs 20 -BatchSize 8 -RunName vla02
```

Este entrenamiento usa las grabaciones y no abre el simulador. En modo con interfaz muestra train/validation MSE
en matplotlib; para observar movimientos se usa la evaluación de la etapa siguiente. `Robots` no afecta esta etapa.
Se necesitan al menos tres episodios exitosos para separar entrenamiento, validación y prueba por episodio.
`-Horizon 8` configura los pasos futuros; `-Horizon 1` produce un único vector de 30 ángulos por observación.
Los datos de todas las réplicas se descubren recursivamente. El directorio de salida debe ser nuevo.

Salida: `checkpoints/vla/vla01/{best.pt,last.pt,processor/,backbone/,metrics.jsonl,split.json}`.
`last.pt` y `best.pt` son checkpoints de inferencia, no contienen el estado del optimizador para reanudar entrenamiento.
La curva también se exporta como `loss.png` si se usa la gráfica. Para ajustar workers, LR y horizonte usar directamente:

```powershell
.\scripts\run.ps1 train_vla --dataset datasets/demos/demos01 --output checkpoints/vla/vla03 --epochs 20 --batch-size 8 --workers 0 --learning-rate 0.0001 --horizon 8 --headless
```

## 4. Inferencia en simulación

```powershell
# Cuatro robots visibles, acciones por lote
.\scripts\vla_scenario.ps1 -Stage evaluate -Robots 4 -Episodes 20 -Checkpoint checkpoints/vla/vla01/best.pt -RunName eval_visible
# Cuatro robots sin ventana
.\scripts\vla_scenario.ps1 -Stage evaluate -Robots 4 -Headless -Episodes 50 -Checkpoint checkpoints/vla/vla01/best.pt -RunName eval_headless
```

La política predice un chunk para cada robot. Por defecto ejecuta solo su primera acción y vuelve a observar a 30 Hz
(`ExecuteSteps=1`, horizonte deslizante). Con `-ExecuteSteps 4` aplica cuatro acciones de la secuencia
antes de sustituirla. La memoria de cada réplica se actualiza con cada observación y se pone a cero
al terminar el episodio o cambiar la instrucción. Los 30 ángulos se acotan, se suavizan y van directamente
a los motores, sin pasar por el decodificador de 26 mandos del teacher. Las demostraciones del
teacher mantienen sus movimientos acoplados; 30 salidas no generan por sí solas destreza adicional.
No se asume que el cálculo de CLIP mantenga 30 Hz de tiempo real: la frecuencia del control es tiempo simulado.

`logs/vla/<RunName>.json` incluye éxitos, retornos, pasos y success_rate. Comparar checkpoints con la misma semilla
y cantidad de episodios. El CLI directo permite `--seed`, `--instruction` y `--max-steps`;
el lanzador admite `-Instruction` para inferencia y evaluación.
Una instrucción nueva cambia la entrada del modelo; la condición de éxito física sigue siendo levantar el cubo.

## 5. Inferencia offline, sin iniciar simulación

```powershell
.\scripts\vla_scenario.ps1 -Stage infer -Checkpoint checkpoints/vla/vla01/best.pt -Episode 'RUTA_EPISODIO.h5' -RunName pred01
```

Guarda `logs/vla/pred01.npz` con `joint_angles` (`[8,30]` en radianes y orden `JOINT_NAMES`)
y `memory` (`[256]`). Usa `-MemoryIn logs/vla/pred01.npz -Frame 1` y un `-RunName` nuevo para
continuar la misma trayectoria. Sin `-MemoryIn`, la memoria inicial es cero.

## Estado de validación

Para leer las metricas de una corrida activa y guardar un resumen JSON:

```powershell
& ..\isaac-sim-6.2.0\python.bat tools/training_status.py --run logs/rsl_rl/ur10e_shadow_lift/NOMBRE_CORRIDA --output captures/training_status.json
```

Un entrenamiento sin camaras no conserva imagenes de sus estados. Para observar un checkpoint,
`tools/capture_teacher.py` abre una evaluacion separada de hasta cuatro robots, copia el checkpoint
y guarda PNG originales, una lamina temporal `robots.png` y metadatos `capture.json` en un directorio nuevo.
No captura el estado exacto del proceso de entrenamiento. Requiere VRAM libre para otra instancia de Isaac Sim;
el primer render RTX puede tardar varios minutos compilando shaders.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& ..\isaac-sim-6.2.0\python.bat tools/capture_teacher.py --checkpoint 'RUTA_PPO.pt' --output captures/nueva_captura --num_envs 4 --viz none
```

Las pruebas offline están adaptadas a 30 ángulos, entradas obligatorias, congelación de CLIP,
actualización de la cabeza, máscaras, inferencia por lote, independencia de falanges y separación
de episodios paralelos. **Las pruebas CPU de v2 pasan**; véase `refactor-validation.json`.
Los resultados anteriores de `vla-validation.json` son históricos y corresponden a la implementación de 26 mandos.
No equivalen a comprobar PhysX, cámaras RTX, agarre ni memoria de varios robots en Isaac Sim.
Consultar [refactor-validation.json](refactor-validation.json) para la verificación actual de CPU.

Referencias de arquitectura: [CLIP en Transformers](https://huggingface.co/docs/transformers/model_doc/clip).
El modo de múltiples robots usa el `DirectRLEnv` vectorizado y el launcher de la revisión local fijada de Isaac Lab.
