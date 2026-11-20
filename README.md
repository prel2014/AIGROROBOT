# AIGROROBOT

Manipulación agrícola de fruta suelta con dos brazos UR10e, manos articuladas y dos cámaras RGB-D. El teacher PPO usa 64 réplicas por defecto. [Tareas y currículo](docs/plan-agricola.md) · [Arquitectura](docs/architecture.md) · [Referencias](docs/referencias-aprendizaje.md) · [Capacidad de VRAM](docs/recursos-vram.md)

El teacher y el estudiante del rover controlan los brazos y orientan ambos ojos RGB-D. Para usar este perfil en el puesto de cosecha, añade `--rover` al teacher y al recolector. [Validar los ojos, entrenar y capturar](docs/rover-cosechador.md#teacher-y-demostraciones-del-rover)

Para inspeccionar el rover entre hileras curvas de naranjos, plátanos, vides y arándanos, abre una de las [cuatro parcelas en Isaac Sim](docs/rover-cosechador.md#parcelas-de-cultivo).

## Preparación

En Windows coloca `isaac-sim-6.2.0`, `IsaacLab` (`release/3.0.0`) y este repositorio en la misma carpeta. Ejecuta desde la raíz del repo en PowerShell, sin conda ni venv. Los scripts usan el Python 3.12 de Isaac Sim. [Versiones fijadas](configs/runtime.lock.json)

```powershell
.\scripts\setup.ps1
& ..\isaac-sim-6.2.0\python.bat tools/download_assets.py
```

## Teacher y captura agrícolas

```powershell
# Ver el entorno y los dos ojos en Isaac Sim
.\scripts\run.ps1 agriculture_preview --scenario citrus --viz kit --live --output logs/agriculture_preview

# Validar el teacher sin entrenar
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 64 --validate

# Entrenar la primera etapa desde un informe físico aprobado
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 64 --grasp-calibration "$env:TEMP\agriculture_grasp.json" --max_iterations 100 --run-name agro_grasp01

# Capturar desde el checkpoint de un teacher de tarea completa
$teacher = 'logs/rsl_rl/agriculture_bimanual/agro_teacher01'
.\scripts\run.ps1 agriculture_collect --headless --num_envs 64 --checkpoint $teacher --episodes 100 --max-attempts 640 --output datasets/agriculture/agro_demos01
```

Para entrenar en ventana, sustituye `--headless` por `--viz kit`. El panel **Ojos RGB-D · Teacher agrícola** permite elegir un robot y ver el RGB y la profundidad de sus dos ojos. Para continuar, añade `--resume $teacher` y usa otro `--run-name`. La captura usa una carpeta de salida nueva y conserva episodios exitosos por defecto. Añade `--keep-failures` para guardar también los fallidos.

Añade `--record-video` para guardar un MP4 por robot e iteración PPO en `videos/epoch_000000/robot_000.mp4`. Cada video muestra RGB y profundidad de ambos ojos a 224 × 224 por vista. Con Scoop puedes indicar el binario mediante `--ffmpeg "$env:USERPROFILE\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"`.

Para aprender el cierre y la elevación inicial desde demostraciones físicas del arándano, añade `--initialize-grasp-from-demos` al entrenamiento nuevo con calibración. La corrida registra el éxito determinista previo a PPO. Para capturar esa habilidad con `agriculture_collect`, pasa el mismo informe mediante `--grasp-calibration`. Los datos se identifican como `calibrated_grasp`.

## Estudiante con órdenes en español

```powershell
.\scripts\run.ps1 train_vla --bimanual --dataset datasets/agriculture/agro_demos01 --output checkpoints/vla_bimanual
.\scripts\run.ps1 play_vla --viz kit --checkpoint checkpoints/vla_bimanual/best.pt --crop berries --instruction "Pinza el arándano más cercano, levántalo 3 cm y sostenlo sin mover los demás." --episodes 5 --output logs/vla_arandano.json
```

El estudiante observa ambos ojos y el estado medido en cada paso, conserva memoria por robot y manda 60 ángulos en el torso fijo o 62 en el rover. El checkpoint selecciona el robot al ejecutar `play_vla`. El entrenamiento alterna la orden sola y la orden con subtarea. La evaluación usa la orden sola y mide el objetivo físico del cultivo. [Formato HDF5](docs/dataset.md) · [USD y capturas](assets/agriculture/README.md) · [Flujo de un brazo](docs/mvp.md)
