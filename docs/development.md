# Desarrollo y organización

El proyecto mantiene el layout Python `src/` y aplica **núcleo funcional y capa de ejecución**.
El objetivo es poder cambiar la percepción, la política o una tarea sin mezclar simulación,
lectura de archivos, CLI y cálculos de recompensas en una misma clase.

## Dependencias

La ruta MVP y su protocolo están en [mvp.md](mvp.md). `agents/teacher.py` adapta la persistencia
de PPO sin modificar RSL-RL; `tasks/lift/curriculum.py` conserva su progresión; `evaluation/`
comparte cuotas y métricas; `data/validation.py` y `manifest.py` verifican episodios y particiones.
`data/dagger.py` separa etiquetas teacher de acciones ejecutadas por el estudiante. Los lanzadores
de `scripts/mvp.ps1` aplican la puerta de aceptación y componen estas piezas.

- `robot/commands.py` y las funciones de recompensas operan sobre tensores; no importan Isaac ni RSL-RL.
- El catálogo textual usa la biblioteca estándar. El paquete `tasks` conserva el registro Gym de
  tareas mediante cadenas de importación; registrar una tarea no construye el simulador.
- `perception` encapsula Transformers; lo reutilizan imitación VLA y RL con memoria.
- `agents/recurrent.py` es el adaptador de la GRU a RSL-RL. La inferencia sigue necesitando las
  clases de modelo de RSL-RL, pero no carga sus algoritmos, runners ni optimizadores.
- `memory_rl/vec_env.py` convierte observaciones/acciones y registra episodios. Usa las interfaces
  del entorno y del codificador; no decide hiperparámetros ni carga checkpoints.
- `training.py`, `inference.py` y `evaluation.py` tienen ciclos de vida separados. Evaluar solo
  reconstruye el actor; reanudar entrenamiento recupera actor, crítico y optimizador.
- `cli` es el punto donde se conectan las dependencias. `scripts/` conserva los nombres de comandos
  existentes y delega en las funciones `main()` del paquete.

Se usan diccionarios de configuración nuevos por llamada porque esa es la interfaz de RSL-RL.
El runner copia la configuración y el contrato para impedir que cambios internos del backend
alteren los metadatos guardados. No se introduce un framework adicional de inyección de dependencias.

## Añadir comportamiento

Para una tarea, modificar `tasks/manipulation/catalog.py`, implementar su métrica/recompensa en
`rewards.py` y adaptar escena/reset cuando haga falta. La dimensión del crítico se calcula con el
número de tareas. El catálogo queda
en los checkpoints: cambiarlo requiere una nueva versión o migración explícita.

Para otra percepción, proporcionar el mismo contrato de `VisualFeatures`: llamada con
`images, instructions, qpos` y retorno de características por réplica; además, `dimension` y
persistencia mediante `save`. El adaptador admite inyectar un codificador en las pruebas.

Para otra memoria, cambiar el actor y su configuración/versionado. No mezclar memoria entre
réplicas: conservar estados entre pasos y limpiar los episodios terminados. La GRU actual conserva
el historial de un episodio; esta reorganización no introduce memoria episódica entre intentos.

## Compatibilidad y limpieza

Se sustituyeron los antiguos `memory_rl/policy.py`, `runtime.py`, `tasks.py` y `environment.py`.
Sus responsabilidades se distribuyen según la tabla de [architecture.md](architecture.md).
El nuevo import de la API pública es:

```python
from dexman.memory_rl.inference import MemoryPredictor
```

Los checkpoints recurrentes anteriores a esta reorganización se leen con el mismo formato y
pesos. El cargador traduce únicamente el nombre antiguo de la clase del actor al nuevo módulo,
sin modificar el archivo. No carga clases arbitrarias indicadas por un checkpoint.
El estudiante usa checkpoints VLA v4 con arquitectura explícita y admite el HDF5 existente
para volver a entrenar. El teacher principal usa Transformers para actor y crítico; los
checkpoints MLP anteriores se rechazan antes de iniciar la simulación.

Los originales de Isaac Sim/Isaac Lab, USD, datos, checkpoints y capturas permanecen fuera de
esta limpieza. Las tareas v0 y herramientas que las usan siguen siendo diagnósticos explícitos;
no son código huérfano y no se usan como entrenamiento nuevo por defecto.

Los informes de validación conservan la fecha y el entorno de cada ejecución histórica.
