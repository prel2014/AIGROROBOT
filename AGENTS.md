# Reglas de arquitectura del repositorio

Aplica **arquitectura hexagonal (puertos y adaptadores)** al código nuevo y a las partes que modifiques. Coloca las reglas de tarea, los límites, las unidades y los formatos en el núcleo. Conecta el simulador, los modelos preentrenados, los archivos y la interfaz de usuario desde los adaptadores. [Diseño y dependencias](docs/architecture.md#patrón-de-software)

- Mantén las reglas de éxito, las fases y los contratos libres de imports de Kit, Omni y PhysX. Convierte las poses y sensores del simulador en tensores del contrato antes de aplicar esas reglas.
- Inyecta los adaptadores en los puntos de entrada de entrenamiento, adquisición, evaluación e inferencia. Usa los contratos compartidos para detectar incompatibilidades de orden articular, unidades, dimensiones y memoria.
- Conserva la medición física de contacto, elevación y sostén en toda etapa de entrenamiento. La trayectoria de calibración puede orientar la exploración. El simulador determina el resultado.
- Registra la semilla, el perfil de simulación y la versión del contrato de cada corrida. Contrasta el criterio físico con una ejecución en Isaac Sim y el contrato de datos con una comprobación sin simulador cuando corresponda.
