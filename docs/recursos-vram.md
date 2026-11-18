# Capacidad de simulación en 12 GiB

Medición en una RTX 3060 de 12.288 MiB con Isaac Sim 6.2 y RSL-RL 5.5.1. Se interpretó cada nodo como una réplica del robot. Cada prueba simuló 32 pasos con el actor. La pasada hacia atrás incluyó actor y crítico, un minilote del tamaño de PPO y memoria reservada para Adam. La prueba se limitó a calcular gradientes. El pico físico se muestreó con `nvidia-smi` cada 100 ms e incluye el escritorio.

| Perfil | Réplicas | Pico físico | Margen de VRAM | Minilote |
| --- | ---: | ---: | ---: | ---: |
| Headless con dos RGB-D de 224 × 224 y ocho capturas por rollout | 64 | 9.097 MiB | 3.191 MiB | 256 |
| Headless sin cámaras | 128 | 6.164 MiB | 6.124 MiB | 512 |

**Capacidad comprobada para el teacher** de 128 réplicas sin cámaras y 64 con RGB-D. Son los mayores valores ensayados en sus respectivos perfiles. La medición tiene el alcance de un rollout, no incluye la memoria del estudiante VLA y usa las cuatro escenas base. El número de texturas, las capturas por rollout y otros procesos en la GPU cambian el presupuesto efectivo.

Desde la raíz del repositorio, ejecuta una prueba de capacidad con el perfil que vayas a usar.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& ..\isaac-sim-6.2.0\python.bat tools/agriculture_vram.py --viz none --num_envs 64 --rgbd --output "$env:TEMP\agriculture_vram.json"
```

El JSON informa la reserva máxima de PyTorch y la forma de la carga ensayada. Registra la VRAM física en otra consola durante toda la prueba.

```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 100
```
