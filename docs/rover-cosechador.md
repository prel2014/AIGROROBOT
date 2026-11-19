# Rover cosechador en surcos

En Isaac Sim 6.2 abre el [surco con el robot](../assets/agriculture/robot/rover/surco.usdc) con **File → Open** y pulsa **Play** para ver la postura de marcha. También puedes inspeccionar el [rover ensamblado](../assets/agriculture/robot/rover/rover.usdc) y el [chasis STEP](../assets/agriculture/robot/rover/chasis.step). El STEP usa milímetros y el USD metros, con X hacia delante, Y hacia la izquierda y Z hacia arriba.

## Mecánica y terreno

- Cuatro ruedas motrices de radio 0,23 m, batalla de 1,10 m y vía de 1,02 m. Cada rueda tiene un guardabarros solidario al chasis con 17 mm de holgura vertical en la corona. La cubierta mide 1,48 × 0,89 m y la canasta ocupa el 60 % delantero.
- Los dos UR10e con manos Shadow se apoyan en columnas inclinadas. Cada brida avanza 75 mm y se acerca 26 mm al eje del rover para recoger los codos en marcha. El montaje queda a 1,08 m de altura y abre 12° hacia cada lado. La distancia recta desde cada brida hasta su punto de entrega pasa de 619 a 557 mm.
- Cada mano dispone de una entrada a la canasta con borde bajo de 622 mm. Dos rampas de 15,6° confluyen en un lecho común. En PhysX, el revestimiento tiene contacto rígido con restitución de 0,02. Los parámetros de masa, fricción y rebote son hipótesis de simulación. El daño de la fruta requiere medición independiente.
- El bastidor trasero integra nueve tubos cuadrados de 45 a 80 mm con paredes de 3 o 4 mm. Los largueros y el parachoques unen la parte inferior. Dos travesaños y cuatro diagonales vinculan los hombros y el mástil. Con acero de 7.850 kg/m³, la masa nominal añadida es 39,4 kg; el modelo actualiza masa y centro de masas del chasis. La resistencia de soldaduras y la flexión bajo carga quedan por dimensionar.
- El surco contiene una malla de tierra firme con caballones y ondulaciones y un tramo de 1,2 × 1,5 m con tres capas de tierra móvil. Sus 8.640 partículas granulares PhysX GPU tienen diámetro nominal de 24 mm. Pueden desplazarse por contacto y dejar una rodada. La resolución representa terrones. El fondo rígido limita el hundimiento.
- El firme usa fricción estática/dinámica de 0,75/0,58, el neumático 0,92/0,73 y el material granular PBD 0,55. Calibrar con pruebas de arranque, frenado y giro a igual carga. Medir desplazamiento del suelo después de asentarlo y repetir la pasada sobre la misma huella. La humedad, la compactación permanente y el barro cohesivo requieren calibración específica.
- Para el futuro entrenamiento, sortear por episodio pendiente, profundidad de las rodadas, espesor de tierra suelta y adherencia con semilla registrada. Medir memoria y tiempo por paso con ambas cámaras y LiDAR al aumentar las réplicas. El catálogo de 64 variantes puede repartirse entre lotes según la VRAM disponible.

El prototipo de viñedo BACCHUS usa dos UR10e. Un brazo sostiene el racimo y el otro trabaja en el pedúnculo. Para este rover se adoptó el acceso independiente de cada mano a la recepción frontal. Las rampas se inspiran en el receptor inclinado de Zhang et al. [Diseño de BACCHUS](https://bacchus-project.eu/cases-robot-harvesting/) · [Vendimia en vídeo](https://www.youtube.com/watch?v=_9vffrpZoN0) · [Receptor del cosechador de manzanas](https://doi.org/10.1002/rob.22268)

## Alcance y colisiones de los brazos

El soporte vertical de la cámara interfería con el brazo superior al girar hacia la izquierda. El travesaño nuevo deja libre ese sector. En el rover, PhysX limita el giro del hombro a −45°…+60°, su elevación articular a −60°…0° y el codo a −10°…+85°. Las colisiones entre los eslabones móviles y el chasis están activas.

En el ensayo de traslado y entrega, los brazos recogidos quedaron dentro del ancho de 1,16 m del tren de ruedas, con un margen lateral mínimo de 10 mm. Las palmas descargadas llegaron a la vez a (0,13; ±0,23; 0,72) m respecto a la base con menos de 0,4 mm de error. PhysX registró una separación mínima de 2 mm entre falanges vecinas y cero impulso entre ellas. En un ensayo separado se soltó una naranja de 160 g sobre cada rampa. Ambas permanecieron en la canasta y aparecieron con profundidad válida en los dos ojos a 224 × 224 píxeles.

Desde la raíz del proyecto, ejecuta el barrido del brazo izquierdo o el ensayo simultáneo de las dos palmas sobre sus puntos de entrega.

~~~powershell
..\isaac-sim-6.2.0\python.bat tools\rover_arm_sweep.py --output logs\barrido_izquierdo.json
..\isaac-sim-6.2.0\python.bat tools\rover_basket_reach.py --output logs\alcance_canasta.json
~~~

## Sensores y entradas

| Canal | Origen en Isaac Sim 6.2 | Entrada |
| --- | --- | --- |
| Dos ojos RGB-D | Cámaras RTX a 1,72 m, separadas 0,44 m sobre un travesaño trasero. Focal de 12 mm, orientación de 22° hacia su lado y 43° hacia abajo | RGB por CLIP y profundidad óptica en metros por ojo |
| LiDAR de mástil | `OmniLidar` y `LidarSensor`, perfil giratorio nativo | Retornos con tiempo de captura y máscara de validez |
| IMU del chasis | `IsaacImuSensor` y `IMUSensor` | Aceleración con gravedad en m/s² y giro en rad/s, en ejes del sensor |
| Cuatro encoders | Velocidades medidas por las articulaciones PhysX | rad/s en orden delantera izq., delantera der., trasera izq., trasera der. |

El centro óptico del LiDAR queda en **(−0,62; 0; 1,90) m** respecto al chasis, con origen nominal en el suelo bajo el centro de la cubierta. El mástil trasero deja despejada la abertura de la canasta. Un brazo extendido a esa altura y el follaje pueden ocluir direcciones del barrido; comprobar cobertura por azimut con los brazos recogidos, extendidos y descargando fruta.

[Rover en el surco](../assets/agriculture/robot/rover/surco.png) · [Recepción de dos naranjas](../assets/agriculture/robot/rover/recepcion.png) · [Bastidor trasero](../assets/agriculture/robot/rover/trasera.png) · [Captura de los ojos](../assets/agriculture/robot/rover/ojos.png). Cada columna corresponde a un ojo. Arriba aparece RGB y abajo profundidad con escala de 0 a 4 m. Para guardar el RGB y la profundidad métrica original, usa una carpeta de salida nueva.

```powershell
.\scripts\run.ps1 rover_eyes --headless --output captures\rover_ojos_01
```

El [contrato móvil](../src/dexman/tasks/agriculture/mobile_state.py) combina los 120 valores articulares de brazos y manos, cuatro velocidades de rueda y seis canales de IMU en un estado de **130 valores**. El LiDAR, los dos ojos, la orden en español y la memoria entran por canales separados. Para el piloto móvil se proponen 60 consignas articulares y dos de base (avance y giro). Registrar la velocidad medida junto a cada consigna. Las demostraciones de mesa conservan su contrato original.

## Criterio de cosecha

«Avanza por el surco de naranjos. Si ves obstáculos, esquívalos. Detente junto a una naranja madura, recógela y déjala en la canasta».

1. Medir avance, giro y deriva. Durante evaluación, comparar la velocidad de cada rueda multiplicada por su radio con la velocidad longitudinal del chasis en esa rueda. Excluir casi reposo del cociente de patinaje. En la política, estimar el movimiento mediante LiDAR, cámaras e IMU. La pose exacta de PhysX sirve como referencia de evaluación.
2. Detenerse a alcance de la fruta sin invadir la planta. Comprobar inclinación lateral, pérdida de contacto de rueda y estabilidad con la canasta cargada.
3. El racimo objetivo de cada vid se sitúa en el extremo de un brote hacia el pasillo. Para entrenar la cosecha falta modelar el desprendimiento físico del pedúnculo a una carga calibrada. Confirmar oposición de dedos, separación del tallo, elevación de 15 cm y depósito estable en la canasta con la mano abierta.

El [generador del surco](../tools/design_harvest_rover.py) reutiliza las cotas del CAD. Para regenerar el USD desde la raíz del repositorio, cierra primero el escenario en Kit.

```powershell
..\isaac-sim-6.2.0\python.bat tools\design_harvest_rover.py
```

Fuentes de NVIDIA. [Partículas granulares y contacto con articulaciones](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html) · [Cámaras RTX](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_camera.html) · [LiDAR RTX](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_rtx_lidar.html).
