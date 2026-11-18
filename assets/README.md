# Assets de simulación

| Ruta | Contenido |
| --- | --- |
| [agriculture/](agriculture/README.md) | Torso, frutas, plantas, obstáculos, texturas y capturas. |
| [procedural/](procedural/README.md) | Ocho escenas de un brazo con USD y catálogo de objetos. |
| `robots/ur10e_shadow/` | Composición USD del UR10e y la Shadow Hand con 30 articulaciones. |
| `vendor/` | Modelos originales descargados para Isaac Sim 6.2. |

Descarga los modelos del robot desde la raíz del repo. El descargador guarda la procedencia y las sumas SHA-256 en `assets/manifest.json`.

```powershell
& ..\isaac-sim-6.2.0\python.bat tools/download_assets.py
```

Conserva los avisos de los autores. [Catálogo NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots_manipulator.html) · [Universal Robot](https://github.com/ros-industrial/universal_robot) · [Shadow Robot](https://github.com/shadow-robot/sr_common)
