# Organización del proyecto

| Componente | Qué editar |
| --- | --- |
| Escenas agrícolas y PPO | `src/dexman/tasks/agriculture/`. `scene.py` monta cada cultivo, `objectives.py` define sus metas y `environment.py` puntúa las fases. |
| Variación por réplica | `src/dexman/tasks/agriculture/randomization.py`. Controla luz, obstáculos y semillas. |
| Teacher bimanual | `src/dexman/agents/agriculture.py`. Define el Transformer, PPO y el contrato de checkpoint. |
| Captura y VLA bimanual | `src/dexman/data/agriculture.py` escribe el HDF5. `src/dexman/vla/bimanual.py` define el modelo del estudiante. |
| Ejecución | `scripts/run.ps1` expone los comandos y `scripts/check_runtime.py` comprueba dependencias. |

Al cambiar sensores, orden de juntas, objetivos o semántica de acciones, revisa el contrato del checkpoint y el formato HDF5. Valida antes de entrenar con `agriculture_teacher --validate`. [Comandos](../README.md) · [Arquitectura](architecture.md) · [Datos](dataset.md)
