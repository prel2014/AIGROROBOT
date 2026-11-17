# AIGROROBOT

UR10e + Shadow Hand con VLA.

## Objetivos

- Agarrar un cubo de 5 cm en simulación, elevarlo al menos 15 cm desde el preagarre y sostenerlo durante 0,5 s. Se penalizan caídas, órdenes inválidas y estados físicamente inválidos.
- Entrenar un teacher PPO con Transformer de estado y un estudiante VLA con tokens espaciales de CLIP, texto, estado articular y memoria. El VLA devuelve objetivos para las 30 juntas y actualiza la memoria.

## Procedimiento

1. Usa el Python 3.12 de Isaac Sim 6.2.0. `isaac-sim-6.2.0/`, `IsaacLab/` y este repo deben estar al mismo nivel. La revisión de Isaac Lab está fijada en [runtime.lock.json](configs/runtime.lock.json).
2. Entrena el teacher PPO y evalúalo en la meta final. El protocolo exige al menos un 90 % de éxito en cada una de tres semillas, con 200 episodios por semilla.
3. Con el mismo checkpoint aprobado, captura demostraciones HDF5 con RGB-D, contacto de las yemas, instrucción, estado articular y acciones. Divide los datos por episodio en entrenamiento, validación y prueba.
4. Entrena el VLA con el manifiesto de datos y evalúa `best.pt` en simulación con el mismo protocolo.

## Comandos

Ejecuta desde la raíz del repo en PowerShell, con conda y venv desactivados. Usa nombres de corrida nuevos al repetir.

**Preparar el entorno**

```powershell
# Solo para configurar el entorno por primera vez
.\scripts\setup.ps1
& ..\isaac-sim-6.2.0\python.bat -m pip install -c configs/runtime-constraints.txt -r configs/vla-requirements.txt

# Verificar el entorno y probar la escena
.\scripts\run.ps1 doctor
.\scripts\run.ps1 preview --task Dexman-UR10e-Shadow-Lift-RGBD-v1 --num_envs 1 --max_steps 1 --headless
```

**Entrenar y probar el teacher**

```powershell
.\scripts\mvp.ps1 -Stage teacher -Robots 32 -Iterations 3000 `
  -StartLiftCm 15 -ExplorationStdScale 0.5 -Headless -RunName teacher01
$teacher = 'logs/rsl_rl/ur10e_shadow_lift_v1/teacher01/model_2999.pt'
.\scripts\mvp.ps1 -Stage evaluate-teacher -Teacher $teacher -Robots 16 -Headless -RunName teacher_eval01
```

Para entrenar con cámara visible y el panel **Cámara RGB-D del teacher**, omite `-Headless`.
El panel permite elegir una réplica y muestra su RGB, su profundidad y la orden de la tarea.
La arquitectura Transformer nueva requiere empezar sin `-Resume`: los checkpoints MLP anteriores se conservan,
pero son incompatibles. Los checkpoints nuevos sí recuperan pesos, optimizador y currículo.
Las corridas nuevas inicializan el actor con trayectorias físicas de calibración antes de PPO.
`bootstrap.json` informa esa fase por separado; los aciertos aprendidos se miden durante PPO y la evaluación.
El teacher actual observa los objetivos filtrados, fija la normalización del actor tras imitar
y usa exploración reducida. Los checkpoints con otra representación requieren empezar una corrida nueva.
PPO inicia con 32 actualizaciones del crítico y después limita el cambio del actor mediante KL.
El formato actual de checkpoint v4 conserva esta fase y las tasas del actor y del crítico.
El criterio de tarea v3 exige alcanzar la altura mínima; los checkpoints que admitían éxitos por debajo de la meta requieren entrenar de nuevo.
Una corrida nueva puede comenzar directamente en la meta final con `-StartLiftCm 15`.
`-ExplorationStdScale 0.5` reduce el ruido antes de validar el actor inicial; solo se aplica al empezar desde cero.

Para continuar el último teacher guardado, usa un nombre nuevo. `-Iterations` indica actualizaciones adicionales.

```powershell
.\scripts\mvp.ps1 -Stage teacher -Resume -Robots 4 -Iterations 3000 -Headless -RunName teacher_cont01
```

Puedes elegir el origen con `-Teacher 'RUTA/model_400.pt'`. Guarda cada 25 actualizaciones y al finalizar.
Reanudar conserva el estado completo y no repite la inicialización. `-BootstrapUpdates 0` permite comparar PPO sin imitación inicial.

**Adquirir datos**

Ejecuta la captura cuando `logs/mvp/teacher_eval01/evaluation.json` indique `passed: true`.

```powershell
.\scripts\mvp.ps1 -Stage collect -Teacher $teacher `
  -TeacherEvaluation logs/mvp/teacher_eval01/evaluation.json -Episodes 200 -Robots 4 -Headless -RunName demos01
.\scripts\mvp.ps1 -Stage split -Dataset datasets/demos/demos01 -RunName split01
```

**Entrenar y probar el VLA**

```powershell
.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split01.json `
  -Epochs 20 -BatchSize 8 -Headless -RunName vla01
.\scripts\mvp.ps1 -Stage train -Resume -Student checkpoints/vla/vla01/last.pt `
  -Epochs 10 -Headless -RunName vla_cont01
.\scripts\mvp.ps1 -Stage evaluate-student -Student checkpoints/vla/vla01/best.pt `
  -Robots 4 -Headless -RunName vla_eval01
```

El VLA v4 usa 6 bloques de atención, memoria de 256 valores y secuencias de 16 observaciones.
`last.pt` guarda arquitectura, pesos, optimizador y época; reanudar conserva datos y normalización.
Los datos y checkpoints del estudiante deben declarar el criterio mínimo de 15 cm y 0,5 s; los anteriores se rechazan.
Omitir `-Headless` abre la gráfica de pérdida. Los checkpoints VLA v3 requieren volver a entrenar con los HDF5.

Más detalles en [la guía del MVP](docs/mvp.md) y [el formato del dataset](docs/dataset.md).
