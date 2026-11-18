# AIGROROBOT

UR10e + Shadow Hand en Isaac Sim 6.2.0 e Isaac Lab 3, instalados al mismo nivel que este repo.

La simulación agrícola monta dos UR10e con manos articuladas y dos ojos RGB-D en un torso fijo. El teacher PPO usa 64 réplicas por defecto, 16 por cultivo, con luz y obstáculos dinámicos. Practica agarre, elevación, traspaso y colocación de fruta suelta con contactos medidos. El VLA bimanual produce 8 pasos de 60 ángulos y actualiza una memoria de 256 valores. [USD y capturas](assets/agriculture/README.md) · [Plan agrícola](docs/plan-agricola.md)

## Teacher y captura agrícolas

Desde la raíz del repo en PowerShell:

```powershell
# Validar el teacher y el reinicio indexado sin entrenar
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 64 --validate

# Entrenar desde cero
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 64 --max_iterations 3000 --run-name agro_teacher01

# Adquirir demostraciones desde el último checkpoint de esa corrida
.\scripts\run.ps1 agriculture_collect --headless --num_envs 64 --checkpoint logs/rsl_rl/agriculture_bimanual/agro_teacher01 --episodes 100 --max-attempts 640 --output datasets/agriculture/agro_demos01
```

Para abrir el teacher en una ventana, reemplaza `--headless` por `--viz kit`. Para reanudar, añade `--resume logs/rsl_rl/agriculture_bimanual/agro_teacher01` y elige un nuevo `--run-name`. La captura guarda éxitos por defecto y deja un resumen por cultivo. Añade `--keep-failures` para inspeccionar intentos fallidos. [Formato de captura](docs/plan-agricola.md#formato-de-demostraciones)

## Objetivos del entorno de un brazo

- Levantar el objeto al menos 15 cm y sostenerlo 0,5 s con contactos medidos opuestos. Penalizar caídas, poses inválidas y contacto de yemas/objeto con obstáculos o distractores.
- Teacher PPO con Transformer de estado. Estudiante VLA v5: dos RGB → CLIP congelado → Transformer propio, junto con texto, q/dq y memoria. Salida: 8 × 30 ángulos y memoria de 256 valores.
- Preparar selección, empuje, transporte y colocación en ocho situaciones. Las [24 órdenes y subtareas](docs/robot-orders.md) están catalogadas; su ejecución jerárquica todavía no está conectada al controlador ni entrenada.

## Procedimiento del entorno de un brazo

1. Configura dependencias con `scripts/setup.ps1`; usa el Python de Isaac Sim, sin conda/venv. Versiones en [runtime.lock.json](configs/runtime.lock.json).
2. Revisa los [escenarios](assets/procedural/catalog.json) y las [capturas reales](assets/previews/environments.png). Los USD propios se incluyen en `assets/procedural/`; los assets originales del robot se descargan con setup.
3. Antes de entrenar, valida de nuevo el preagarre para cada geometría. Los checkpoints y datos anteriores no acreditan el nuevo criterio de contacto y las dos vistas: requieren nuevas corridas y demostraciones.
4. Entrena y evalúa el teacher de elevación; exige 90 % de éxito por semilla. Captura demostraciones aprobadas, divide por episodio, entrena y evalúa el VLA. Empuje/colocación requieren sus propios controladores de demostración; no se obtienen cambiando el texto de una elevación.

## Comandos del entorno de un brazo

Desde la raíz del repo, en PowerShell. Los comandos de entrenamiento quedan disponibles para cuando se autorice iniciarlo.

```powershell
# Preparación inicial
.\scripts\setup.ps1
& ..\isaac-sim-6.2.0\python.bat -m pip install -c configs/runtime-constraints.txt -r configs/vla-requirements.txt

# Revisar un entorno y guardar sus dos RGB-D; no entrena
.\scripts\run.ps1 scene_gallery --scenario narrow_passage --output assets/previews --headless
# Regenerar USD y catálogo sin iniciar el simulador
& ..\isaac-sim-6.2.0\python.bat tools/export_scenarios.py

# Entrenar teacher nuevo; omite -Headless para abrir ventana y ambos paneles RGB-D
.\scripts\mvp.ps1 -Stage teacher -Robots 4 -Iterations 3000 -Headless -RunName teacher01
$teacher = 'logs/rsl_rl/ur10e_shadow_lift_v1/teacher01/model_2999.pt'
.\scripts\mvp.ps1 -Stage evaluate-teacher -Teacher $teacher -Robots 4 -Headless -RunName teacher_eval01

# Adquirir datos y dividir episodios
.\scripts\mvp.ps1 -Stage collect -Teacher $teacher -TeacherEvaluation logs/mvp/teacher_eval01/evaluation.json -Episodes 200 -Robots 4 -Headless -RunName demos01
.\scripts\mvp.ps1 -Stage split -Dataset datasets/demos/demos01 -RunName split01

# Entrenar y evaluar estudiante
.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split01.json -Epochs 20 -BatchSize 8 -Headless -RunName vla01
.\scripts\mvp.ps1 -Stage evaluate-student -Student checkpoints/vla/vla01/best.pt -Robots 4 -Headless -RunName vla_eval01

# Ver inferencia en ventana, con un checkpoint NUEVO compatible
.\scripts\run.ps1 play_vla --checkpoint checkpoints/vla/vla01/best.pt --num_envs 1 --episodes 20 --viz kit
```

`-Scenario clear_bench` selecciona un escenario para teacher/evaluación/captura/correcciones; usa el mismo en todo el proceso. Por defecto se usa `red_cube`.
Para continuar teacher: añade `-Resume -Teacher RUTA/model.pt`; para estudiante: `-Resume -Student RUTA/last.pt`. Elige otro `-RunName`; conserva pesos, optimizador y estado de aprendizaje. Headless/ventana no cambia el contrato.

Las dos profundidades se graban en metros para inspección; la red actual recibe los RGB mediante CLIP. El planificador usa geometría del simulador y margen XY, todavía sin planificación de colisiones del brazo completo. Las capturas no demuestran éxito de una política.
