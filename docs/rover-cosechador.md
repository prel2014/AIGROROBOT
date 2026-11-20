# Rover cosechador en surcos

En Isaac Sim 6.2 abre el [surco con el robot](../assets/agriculture/robot/rover/surco.usdc) con **File → Open** y pulsa **Play** para ver la postura de marcha. También puedes inspeccionar el [rover ensamblado](../assets/agriculture/robot/rover/rover.usdc) y el [chasis STEP](../assets/agriculture/robot/rover/chasis.step). El STEP usa milímetros y el USD metros, con X hacia delante, Y hacia la izquierda y Z hacia arriba.

## Mecánica y terreno

- Cuatro ruedas motrices de radio 0,23 m, batalla de 1,10 m y vía de 1,02 m. Cada neumático tiene 18 pares de tacos en espiga con colisión, llanta cóncava de seis radios y seis pernos. Los guardabarros van unidos al chasis. La cubierta mide 1,48 × 0,89 m y la canasta ocupa el 60 % delantero.
- Los dos UR10e con manos Shadow se apoyan en columnas inclinadas. Cada brida avanza 75 mm y se acerca 26 mm al eje del rover para recoger los codos en marcha. El montaje queda a 1,08 m de altura y abre 12° hacia cada lado. La distancia recta desde cada brida hasta su punto de entrega pasa de 619 a 557 mm.
- Cada mano dispone de una entrada a la canasta con borde bajo de 622 mm. Dos rampas de 15,6° confluyen en un lecho común. En PhysX, el revestimiento tiene contacto rígido con restitución de 0,02. Los parámetros de masa, fricción y rebote son hipótesis de simulación. El daño de la fruta requiere medición independiente.
- El bastidor trasero integra diez tubos cuadrados de 44 a 80 mm con paredes de 3 o 4 mm. Los largueros y el parachoques unen la parte inferior. Los travesaños y las diagonales vinculan los hombros y el mástil. Una rama inclinada independiente sostiene el LiDAR. La masa nominal de los diez tubos es de 41,1 kg con acero de 7.850 kg/m³. La resistencia de soldaduras y la flexión bajo carga quedan por dimensionar.
- El surco contiene una malla de tierra firme con caballones y ondulaciones y un tramo de 1,2 × 1,5 m con tres capas de tierra móvil. Sus 8.640 partículas granulares PhysX GPU tienen diámetro nominal de 24 mm. Pueden desplazarse por contacto y dejar una rodada. La resolución representa terrones. El fondo rígido limita el hundimiento.
- El firme usa fricción estática/dinámica de 0,75/0,58, el neumático 0,92/0,73 y el material granular PBD 0,55. Calibrar con pruebas de arranque, frenado y giro a igual carga. Medir desplazamiento del suelo después de asentarlo y repetir la pasada sobre la misma huella. La humedad, la compactación permanente y el barro cohesivo requieren calibración específica.
- Para el futuro entrenamiento, sortear por episodio pendiente, profundidad de las rodadas, espesor de tierra suelta y adherencia con semilla registrada. Medir memoria y tiempo por paso con ambas cámaras y LiDAR al aumentar las réplicas. El catálogo de 64 variantes puede repartirse entre lotes según la VRAM disponible.

El prototipo de viñedo BACCHUS usa dos UR10e. Un brazo sostiene el racimo y el otro trabaja en el pedúnculo. Para este rover se adoptó el acceso independiente de cada mano a la recepción frontal. Las rampas se inspiran en el receptor inclinado de Zhang et al. [Diseño de BACCHUS](https://bacchus-project.eu/cases-robot-harvesting/) · [Vendimia en vídeo](https://www.youtube.com/watch?v=_9vffrpZoN0) · [Receptor del cosechador de manzanas](https://doi.org/10.1002/rob.22268)

## Alcance y colisiones de los brazos

El soporte vertical de la cámara interfería con el brazo superior al girar hacia la izquierda. El travesaño nuevo deja libre ese sector. En el rover, PhysX limita el giro del hombro a −45°…+60°, su elevación articular a −60°…0° y el codo a −10°…+85°. Las colisiones entre los eslabones móviles y el chasis están activas.

En el ensayo de traslado y entrega, los brazos recogidos quedaron dentro del ancho de 1,16 m del tren de ruedas, con un margen lateral mínimo de 10 mm. Las palmas descargadas llegaron a la vez a (0,13; ±0,23; 0,72) m respecto a la base con menos de 0,4 mm de error mientras se movía el cabezal. En un ensayo separado se soltó una naranja de 160 g sobre cada rampa. Ambas permanecieron en la canasta. Con los brazos recogidos y la cabeza centrada, el ojo izquierdo vio ambas frutas con profundidad válida y el brazo ocultó una al ojo derecho.

Desde la raíz del proyecto, ejecuta el barrido del brazo izquierdo o el ensayo simultáneo de las dos palmas sobre sus puntos de entrega.

~~~powershell
..\isaac-sim-6.2.0\python.bat tools\rover_arm_sweep.py --output logs\barrido_izquierdo.json
..\isaac-sim-6.2.0\python.bat tools\rover_basket_reach.py --output logs\alcance_canasta.json
~~~

## Sensores y entradas

| Canal | Origen en Isaac Sim 6.2 | Entrada |
| --- | --- | --- |
| Dos ojos RGB-D | Cámaras RTX sobre un cabezal común, separadas 0,44 m. En posición cero miran 22° hacia su lado y 43° hacia abajo, a 1,72 m de altura | RGB por CLIP y profundidad óptica en metros por ojo |
| Cabezal articulado | Dos juntas motrices PhysX. Giro de −70° a +70° (positivo a la izquierda) y elevación de −22° a +38° (positivo hacia arriba), relativos a la vista de cosecha | Dos ángulos medidos por los encoders, en radianes |
| LiDAR de mástil | `OmniLidar` y `LidarSensor`, perfil giratorio nativo | Retornos con tiempo de captura y máscara de validez |
| IMU del chasis | `IsaacImuSensor` y `IMUSensor` | Aceleración con gravedad en m/s² y giro en rad/s, en ejes del sensor |
| Cuatro encoders | Velocidades medidas por las articulaciones PhysX | rad/s en orden delantera izq., delantera der., trasera izq., trasera der. |

El LiDAR queda fijo en **(−0,855; 0; 2,02) m** respecto al chasis. En posición neutra, los centros de las cámaras quedan en **(−0,430; ±0,220; 1,72) m**, a 160 mm por delante del eje de inclinación. El origen nominal está en el suelo bajo el centro de la cubierta. El conjunto trasero del LiDAR sobresale 142 mm del parachoques, contando la tapa. En 120 orientaciones de la cabeza, sus piezas móviles dejaron libres las 32 elevaciones del LiDAR en los 720 azimuts comprobados.

[Rover en el surco](../assets/agriculture/robot/rover/surco.png) · [Rueda](../assets/agriculture/robot/rover/llanta.png) · [Cabezal centrado](../assets/agriculture/robot/rover/cabezal_centrado.png) · [Cabezal girado](../assets/agriculture/robot/rover/cabezal.png) · [Bastidor trasero](../assets/agriculture/robot/rover/trasera.png) · [Recepción de dos naranjas](../assets/agriculture/robot/rover/recepcion.png).

La [captura neutra](../assets/agriculture/robot/rover/ojos.png) muestra el RGB arriba y la profundidad de 0 a 4 m abajo. Cada columna corresponde a un ojo. Para inspeccionar ambas naranjas con el ojo derecho, gira la cabeza a +65° e inclínala a −2°. [Vista de la canasta con la cabeza girada](../assets/agriculture/robot/rover/ojos_canasta_girada.png)

Para guardar ambos RGB-D y los ángulos medidos, elige una carpeta de salida nueva.

```powershell
.\scripts\run.ps1 rover_eyes --pan-deg 70 --tilt-deg -17 --output captures\rover_lateral_01
```

Para mover el cabezal en la ventana del simulador, abre el surco, pulsa **Play** y cambia el objetivo angular de `/Surco/Rover/head_pan_joint` o `/Surco/Rover/head_tilt_joint`. El actuador simulado de inclinación admite 15 N·m y 40°/s. En 54 poses de CAD, la menor holgura de la barra de cámaras con el soporte del LiDAR fue de 20 mm.

En una malla de 1° en ambos ejes (8.601 poses), el LiDAR y su soporte trasero proyectaron cero píxeles en los dos ojos de 224 × 224. En los extremos, al mirar al suelo, aparece el poste inferior del cabezal en el borde de la imagen (hasta 0,63 %), además de elementos del bastidor. [Comparación del montaje](../assets/agriculture/robot/rover/comparacion_montaje.png) · [Vista del ojo derecho antes y después](../assets/agriculture/robot/rover/comparacion_ojo_derecho.png)

Los **132 valores** del [estado móvil](../src/dexman/tasks/agriculture/mobile_state.py) son, en orden, 60 posiciones de brazos y manos, 60 velocidades articulares, cuatro velocidades de rueda, tres aceleraciones y tres velocidades angulares de la IMU, giro y elevación del cabezal. El estudiante combina estas medidas con los dos ojos y la orden en español. Predice 60 ángulos de brazos y manos y dos consignas para el cabezal, en radianes. El LiDAR y la IMU están fijos al chasis.

## Teacher y demostraciones del rover

El rover trabaja estacionado frente a la fruta. El borde de la mesa deja libre la muñeca izquierda al reiniciar el robot. El teacher PPO observa el estado físico y la dirección de cada ojo respecto al fruto. El estudiante usa las imágenes y vuelve a observar tras aplicar cada acción. En dos réplicas se grabaron 45 pasos por robot con giros opuestos de 22,4° y un error angular final máximo de 0,63°. [Ver los cuatro ojos y su profundidad](../assets/agriculture/robot/rover/ojos_teacher.png)

```powershell
.\scripts\run.ps1 agriculture_teacher --rover --crop citrus --num_envs 1 --headless --validate
.\scripts\run.ps1 agriculture_teacher --rover --crop citrus --num_envs 1 --headless --run-name rover_citrus01
$teacher = 'logs/rsl_rl/agriculture_bimanual/rover_citrus01'
.\scripts\run.ps1 agriculture_collect --rover --crop citrus --num_envs 1 --headless --checkpoint $teacher --episodes 20 --max-attempts 200 --output datasets/agriculture/rover_citrus01
.\scripts\run.ps1 train_vla --bimanual --dataset datasets/agriculture/rover_citrus01 --output checkpoints/vla_rover01
.\scripts\run.ps1 play_vla --viz kit --crop citrus --checkpoint checkpoints/vla_rover01/best.pt --episodes 1 --output logs/evaluacion_rover01.json
```

La captura requiere un checkpoint del teacher móvil y guarda episodios exitosos por defecto. Para los cuatro cultivos, omite `--crop` en el teacher y el recolector y usa al menos cuatro réplicas. Cambia `--headless` por `--viz kit` para seleccionar un robot en el panel del teacher. Cada réplica del rover con el panel, captura o video activa dos cámaras RTX. [Campos por fotograma](dataset.md#observación-móvil)

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
