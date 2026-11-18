# Migración a manipulación agrícola bimanual

Objetivo inicial: seleccionar, recoger y depositar frutas con un robot de torso fijo, dos brazos y dos ojos RGB-D. La cosecha de frutos todavía unidos a la planta es una segunda etapa. Los escenarios geométricos anteriores sirven como referencia de diagnóstico de física.

Isaac Sim monta 60 juntas por robot, dos ojos RGB-D y cuatro cultivos. El lote de 64 réplicas está conectado al teacher PPO bimanual y a la captura RGB-D desde su checkpoint. La primera tarea se limita a fruta suelta. [Guía de USD y capturas](../assets/agriculture/README.md)

## 1. Biblioteca de frutas y plantas

- [ ] Consolidar variantes de tamaño, madurez y geometría para plátano, naranja, uva y arándano. Registrar autor, licencia, fuente y versión para cada malla y textura. El catálogo del primer lote está en [assets/agriculture](../assets/agriculture/README.md).
- [ ] Normalizar GLB, glTF u OBJ a USD en metros y eje Z arriba. Conservar UV, materiales y partes separadas. En plantas, separar hojas, ramas y pedúnculos. En racimos, separar el objetivo que se cosecha de lo que queda en la planta. [Importación de mallas en Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)
- [ ] Preparar la física por especie. Usar colisiones que sigan el contorno de la fruta, masa e inercia medidas o documentadas y fricción calibrada. Un plátano curvo necesita varios convexos para conservar la concavidad. Una uva suelta, un racimo y un arándano individual requieren propiedades y criterios de contacto distintos.
- [ ] Comparar la malla visual con la colisión en Isaac Sim 6.2. Probar reposo, caída, rodamiento, deslizamiento y agarre instrumentado antes de generar demostraciones. Para frutos unidos, medir y validar el modelo de desprendimiento por fuerza, par o corte. Calibrar indicadores de daño por especie antes de usarlos como métricas.

## 2. Torso, brazos y ojos RGB-D

- [ ] Prototipo de torso fijo, dos módulos UR10e y dos manos articuladas como punto de partida. Parametrizar hombros y confirmar la disponibilidad de mano izquierda o la configuración necesaria para montar el efector. Medir alcance común y comprobar las posturas de entrega, cesta y acceso a ramas.
- [ ] Componer el torso y los dos brazos en USD con marcos definidos, articulaciones y nombres únicos por lado. Exponer 60 juntas si se mantienen los dos módulos actuales de 30. Confirmar límites, acoplamientos, sensores y colisiones entre ambos brazos, manos y torso. [Flujo de ensamblaje en Isaac Sim 6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup_tutorials/tutorial_manipulator_workflow.html)
- [ ] Montar dos RGB-D en la parte alta y frontal del torso, separados lateralmente y con campos de visión solapados. Ajustar separación, inclinación, distancia mínima y resolución usando el fruto más pequeño visible en zonas de agarre y entrega. Verificar que las manos no tapen simultáneamente las dos vistas.
- [ ] Grabar RGB, profundidad métrica, máscara de profundidad válida, intrínsecos, extrínsecos y marca de tiempo de cada ojo. Calibrar ambos al marco del torso. Un sensor de contacto de cada mano permitirá comprobar agarre y transferencias. [Sensores de cámara de Isaac Sim 6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera.html)

## 3. Escenas, órdenes y subtareas del robot

| Escena | Orden de ejemplo | Subtareas con condiciones verificables |
| --- | --- | --- |
| Clasificación de fruta suelta | «Pon la naranja madura en la cesta izquierda y deja el plátano. Si ves obstáculos, esquívalos». | Identificar el objetivo → elegir brazo → acercarse → confirmar agarre → transportar → verificar fruta suelta dentro de la cesta. |
| Entrega entre brazos | «Recoge el plátano con la derecha, pásalo a la izquierda y colócalo en la bandeja». | Agarrar con la derecha → llegar a la zona común → confirmar contacto estable de la izquierda → liberar la derecha → depositar. |
| Naranjo | «Recoge la naranja madura detrás de la rama y deja la verde». | Recuperar la vista → planear alrededor de la rama → alinear → agarrar → separar cuando el modelo del tallo lo permita → verificar la rama y depositar. |
| Vid en espaldera | «Sujeta el racimo con una mano y separa el pedúnculo con la otra. Evita el alambre». | Distinguir racimo y pedúnculo → sostener el peso → posicionar la segunda mano → desprender mediante el efector validado → retirar sin engancharse → depositar. |
| Arbusto de arándanos | «Recoge el arándano azul visible sin tirar de los verdes». | Resolver la oclusión → seleccionar una baya → acercar pulgar e índice → comprobar pinza de baja fuerza → separar sin mover el racimo vecino → depositar. |
| Platanera | «Sostén la mano de plátanos mientras separas el tallo y apóyala en la bandeja». | Identificar tallo y zona de apoyo → estabilizar la carga → separar → repartir el peso → llevar a la bandeja y soltar. |
| Acceso bloqueado | «Busca otra entrada. Si ninguna es segura, detente y deja la fruta donde está». | Detectar obstrucción → probar otra trayectoria o mano → detener la tarea si no hay alternativa válida. |

- [x] Preparar variación de luz, tono de mesa y obstáculos físicos por semilla, réplica y episodio. El catálogo de 11 tipos distingue obstáculos de mesa y de pasillo. Los reinicios mantienen los obstáculos fuera de las posiciones iniciales de fruta, bandejas y el apoyo del robot.
- [x] Ejecutar un lote de 64 réplicas, con 16 por cultivo, reinicio indexado y cambios de luz y obstáculos en tiempo simulado. Los eventos comprueban la separación de brazos y fruta antes de cambiar el obstáculo.
- [x] Conectar la variación al reinicio del PPO bimanual y reiniciar solo las réplicas terminadas. Cada cultivo aporta 16 réplicas por defecto.
- [x] Conectar fases de agarre, elevación, transferencia y colocación para fruta suelta. Los cítricos y las uvas se elevan 15 cm. El plátano requiere traspaso de derecha a izquierda. La pinza de arándano requiere oposición de pulgar e índice.
- [ ] Validar el alcance y los umbrales de contacto por cultivo. Agregar oclusiones, variantes de follaje y desprendimiento para cosecha en planta.
- [ ] Separar fallos por objeto incorrecto, caída, deslizamiento, colisión no permitida, límite articular, desprendimiento incorrecto y destino bloqueado. Medir también fuerza excesiva y daño de fruta o planta cuando se haya calibrado el modelo correspondiente.

## 4. Teacher, dataset y Transformer VLA

- [x] Conectar el teacher PPO con un Transformer de estado de 3 bloques y dos consultas de acción, una por brazo. El actor recibe 320 características y produce 60 comandos normalizados. El crítico predice un valor por réplica. El controlador registra los 60 ángulos aplicados tras limitar el comando.
- [x] Capturar demostraciones con el actor de un checkpoint bimanual, con reinicio por réplica y registro de éxitos, caídas, poses inválidas, desplazamiento de bayas vecinas y tiempos límite.
- [ ] Cambiar el contrato de datos. Guardar RGB-D por ojo, orden global, fase actual, estado articular, comandos sincronizados, eventos de contacto y resultado. Reservar plantas, distribuciones y semillas completas para validar generalización.
- [x] Usar un Transformer propio compartido de seis bloques, ancho 256 y ocho cabezas de atención. Concatena 100 tokens RGB de CLIP, 98 de profundidad, uno de texto, uno por brazo y uno de memoria. Las 16 consultas de acción comparten atención entre lados y pasos.
- [x] Con dos módulos de 30 juntas y torso fijo, el estado angular es `q[60] + dq[60]`. La salida conjunta es `ángulos[8,60]` en radianes más `memoria[256]`. El orden de juntas es izquierda seguida de derecha.
- [ ] Validar resolución, horizonte y latencia para pinza fina. Alimentar la subtarea activa en el texto y reiniciar la memoria únicamente en las réplicas que cambien de episodio.
- [ ] Versionar robot, observaciones y checkpoints. CLIP congelado se puede reutilizar como encoder RGB. El teacher y la cabeza de acciones se entrenarán con datos del nuevo montaje porque cambian la entrada, observación y actuación. La predicción conjunta en bloques sigue el principio empleado en [ACT para manipulación bimanual](https://roboticsproceedings.org/rss19/p016.html).

### Formato de demostraciones

Cada episodio completo se guarda en HDF5 con orden de ojos izquierdo y derecho. Cada muestra contiene RGB `[2,224,224,3]`, profundidad métrica y máscara válida `[2,224,224]`, estado angular `[120]` y comando aplicado `[60]` en radianes. La orden y la calibración se guardan por episodio. La fase, el tiempo simulado, la luz, los obstáculos y las señales de contacto se guardan por muestra. `dataset.json` declara el orden de juntas y el checkpoint usado. `summary.json` cuenta resultados por cultivo.

RGB-D y estado corresponden al instante anterior al comando. El contacto, la pose de la fruta y el resultado de ese paso corresponden al instante posterior. El esquema `dexman.agriculture.demo.v1` requiere el cargador bimanual cuando se conecte el entrenamiento del estudiante.

## 5. Orden de entrega y validación

1. USD agrícolas inspeccionados, torso de dos brazos en escena y grabación de los dos ojos con calibración.
2. Clasificación de naranjas y plátanos sueltos con cada brazo, seguida de una entrega bimanual con registro de contacto.
3. Cosecha de un fruto accesible en una planta. Luego cosecha con oclusión, en espaldera y con frutos pequeños.
4. Evaluación por especie, escena y orden. Medir éxito de tarea y de subtarea, caídas, colisiones, fuerzas, daño modelado y respuesta al bloqueo.
5. Publicar el código y los USD con texturas y atribución. El README documentará entrenamiento, adquisición de datos y evaluación en ventana una vez que esas rutas estén conectadas.

Este documento organiza la migración. Las tareas marcadas `[ ]` requieren implementación o validación y todavía no constituyen un robot bimanual operable.
