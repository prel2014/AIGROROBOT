# Escenarios procedurales

Los ocho USD incluyen objetos, distractores, mesas y obstáculos con colisiones y masas.
Se generan desde `src/dexman/tasks/manipulation/scenarios.py`, que también configura
la simulación. Son geometrías propias de este proyecto, sin dependencias externas ni texturas descargadas.
Los USD son exportaciones portables de los props; el robot y las cámaras se añaden en Isaac Lab.

Regenerar sin simulación ni entrenamiento:

```powershell
& ..\isaac-sim-6.2.0\python.bat tools/export_scenarios.py
```

`catalog.json` incluye dimensiones, masas, objetivos, órdenes y subtareas. Las rutas
usan cajas conocidas del simulador y un margen XY de 8 cm; no validan el volumen
completo del brazo ni constituyen un detector visual. Un destino bloqueado produce
una orden de detenerse. La ejecución jerárquica todavía no está conectada al controlador.

Las capturas reales están en `assets/previews/`; muestran escenas sin política ni entrenamiento.
El robot de NVIDIA sigue requiriendo los assets locales descritos en `assets/README.md`.
