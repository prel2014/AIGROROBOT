# AIGROROBOT

Manipulación agrícola de fruta suelta con dos brazos UR10e, manos articuladas y dos cámaras RGB-D. El teacher PPO usa 64 réplicas por defecto, 16 por cultivo, con luz y obstáculos variables. [Objetivos por cultivo](docs/plan-agricola.md)

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

# Entrenar desde cero
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 64 --max_iterations 3000 --run-name agro_teacher01

# Capturar desde el checkpoint más reciente de la corrida
$teacher = 'logs/rsl_rl/agriculture_bimanual/agro_teacher01'
.\scripts\run.ps1 agriculture_collect --headless --num_envs 64 --checkpoint $teacher --episodes 100 --max-attempts 640 --output datasets/agriculture/agro_demos01
```

Para entrenar en ventana, sustituye `--headless` por `--viz kit`. El panel **Ojos RGB-D · Teacher agrícola** permite elegir un robot y ver el RGB y la profundidad de sus dos ojos. Para continuar, añade `--resume $teacher` y usa otro `--run-name`. La captura usa una carpeta de salida nueva y conserva episodios exitosos por defecto. Añade `--keep-failures` para guardar también los fallidos.

El modelo VLA bimanual está implementado. Su cargador de demostraciones y el entrenamiento del estudiante siguen pendientes. [Formato HDF5](docs/dataset.md) · [USD y capturas](assets/agriculture/README.md) · [Flujo de un brazo](docs/mvp.md)
