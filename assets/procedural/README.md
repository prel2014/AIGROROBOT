# Escenarios procedurales de un brazo

Ocho USD con mesas, objetos, distractores y obstáculos. Isaac Lab añade el robot y las cámaras. El catálogo incluye dimensiones, masas, objetivos y órdenes. [Catálogo](catalog.json)

Desde la raíz del repo:

```powershell
.\scripts\run.ps1 scene_gallery --scenario narrow_passage --output logs/scene_gallery --headless
& ..\isaac-sim-6.2.0\python.bat tools/export_scenarios.py
```

El primer comando captura el escenario y el segundo regenera los USD. La ejecución jerárquica de las órdenes propuestas sigue pendiente. [Órdenes](../../docs/robot-orders.md)
