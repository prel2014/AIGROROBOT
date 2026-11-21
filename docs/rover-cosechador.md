# Rover cosechador en surcos

En Isaac Sim 6.2 abre el [surco con el robot](../assets/agriculture/robot/rover/surco.usdc) con **File → Open** y pulsa **Play** para ver la postura de marcha. También puedes inspeccionar el [rover ensamblado](../assets/agriculture/robot/rover/rover.usdc) y el [chasis STEP](../assets/agriculture/robot/rover/chasis.step). El STEP usa milímetros y el USD metros, con X hacia delante, Y hacia la izquierda y Z hacia arriba.

## Parcelas de cultivo

Cada USD incluye el rover en la entrada, tres hileras curvas de 24 m, dos calles, cabeceras de 5 m y cerca de postes y alambre. En el selector de cámara elige **General**, **Calle** o **Fruto** bajo `/Parcela/Vistas`. [Galería de Isaac Sim](../assets/agriculture/previews/parcelas/galeria.jpg) · [Capturas individuales](../assets/agriculture/previews/parcelas)

| Abrir en Isaac Sim | Hileras × plantas | Entre hileras / plantas | Detalle del terreno y del cultivo |
| --- | --- | --- | --- |
| [Naranjal](../assets/agriculture/parcelas/naranjal.usdc) | 3 × 8 | 6 / 3 m | Ladera con roca, copa abierta, corteza, hojas nervadas y naranjas con piel porosa. |
| [Platanal](../assets/agriculture/parcelas/platanal.usdc) | 3 × 8 | 4 / 3 m | Rodadas de hasta 8 cm, tierra granular y balde. Racimos verdes, flor, hojas rasgadas y pseudotallos. |
| [Viñedo](../assets/agriculture/parcelas/vinedo.usdc) | 3 × 13 | 3 / 1,8 m | Espaldera sobre pendiente transversal, sarmientos y racimos. Un poste caído invade la calle. |
| [Parcela de arándanos](../assets/agriculture/parcelas/arandanal.usdc) | 3 × 24 | 3 / 1 m | Camellones de 35 cm, tierra granular y rama caída. Bayas con cáliz y cuatro tonos de maduración. |

La referencia geométrica de ida, giro y regreso está en `/Parcela/Navegacion`, atributo `ruta:centroNominal`, en metros locales. Las cabeceras y los obstáculos están disponibles para ensayar navegación. El teacher actual opera en el puesto de cosecha. Quedan por validar el recorrido autónomo y el desprendimiento de fruta en estas parcelas.

Para fotografiar una parcela y comprobar con PhysX el suelo, una rodada, la cerca y el obstáculo, ejecuta desde la raíz del repositorio. Sustituye `citrus` por `banana`, `vineyard` o `berries` para las demás.

```powershell
..\isaac-sim-6.2.0\python.bat tools\render_crop_field.py --crop citrus
```

Las dimensiones y semillas están en [field_layout.py](../src/dexman/tasks/agriculture/field_layout.py). Regenera los cuatro USD con `..\isaac-sim-6.2.0\python.bat tools\build_crop_fields.py`. Referencias de trazado y plantación. [FAO, curvas de nivel](https://www.fao.org/4/u3160e/u3160e07.htm) · [UF/IFAS, cítricos](https://ask.ifas.ufl.edu/publication/HS1302) · [INIA, plátano](https://repositorio.inia.gob.pe/server/api/core/bitstreams/c15c824a-9483-4525-8c8b-eb465bc4930d/content) · [Oregon State, uva de mesa](https://extension.oregonstate.edu/catalog/ec-1639-growing-table-grapes) · [INIA, arándanos](https://biblioteca.inia.cl/server/api/core/bitstreams/6fd5c9f9-0d6d-43eb-ac11-9ae014bd8326/content)

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

En cada brazo, una junta fija une la muñeca UR10e al montaje de la mano Shadow. Un casquillo entre dos bridas cubre los 40 mm entre las mallas. Las tres piezas tienen colisión y pertenecen al cuerpo rígido del montaje de la mano.

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

[Rover en el surco](../assets/agriculture/robot/rover/surco.png) · [Recepción de dos naranjas](../assets/agriculture/robot/rover/recepcion.png).

La [captura neutra](../assets/agriculture/robot/rover/ojos.png) muestra el RGB arriba y la profundidad de 0 a 4 m abajo. Cada columna corresponde a un ojo. Para inspeccionar ambas naranjas con el ojo derecho, gira la cabeza a +65° e inclínala a −2°. [Vista de la canasta con la cabeza girada](../assets/agriculture/robot/rover/ojos_canasta_girada.png)

Para guardar ambos RGB-D y los ángulos medidos, elige una carpeta de salida nueva.

```powershell
.\scripts\run.ps1 rover_eyes --pan-deg 70 --tilt-deg -17 --output captures\rover_lateral_01
```

Para mover el cabezal en la ventana del simulador, abre el surco, pulsa **Play** y cambia el objetivo angular de `/Surco/Rover/head_pan_joint` o `/Surco/Rover/head_tilt_joint`. El actuador simulado de inclinación admite 15 N·m y 40°/s. En 54 poses de CAD, la menor holgura de la barra de cámaras con el soporte del LiDAR fue de 20 mm.

En una malla de 1° en ambos ejes (8.601 poses), el LiDAR y su soporte trasero proyectaron cero píxeles en los dos ojos de 224 × 224. En los extremos, al mirar al suelo, aparece el poste inferior del cabezal en el borde de la imagen (hasta 0,63 %), además de elementos del bastidor.

Los **132 valores** del [estado móvil](../src/dexman/tasks/agriculture/mobile_state.py) son, en orden, 60 posiciones de brazos y manos, 60 velocidades articulares, cuatro velocidades de rueda, tres aceleraciones y tres velocidades angulares de la IMU, giro y elevación del cabezal. El estudiante combina estas medidas con los dos ojos y la orden en español. Predice 60 ángulos de brazos y manos y dos consignas para el cabezal, en radianes. El LiDAR y la IMU están fijos al chasis.

## Teacher y demostraciones del rover

### Recorrido y cosecha en cuatro cultivos

Desde la raíz, ejecuta este ejemplo con una carpeta de salida nueva. Usa `citrus`, `banana`, `vineyard` o `berries`. Cada ensayo recorre unos 3,3 m desde la calle hasta la planta y prueba dos aproximaciones con manos distintas. El controlador recoge el brazo inactivo y planifica el codo y la muñeca con respecto a la línea de visión de ambos ojos.

```powershell
..\isaac-sim-6.2.0\python.bat tools\film_field_cycle.py --viz none --crop berries --output captures\mi_cosecha\arandanal
python tools\field_video_index.py captures\mi_cosecha
```

Abre `index.html` para comparar los dos MP4 sincronizados y saltar al cierre, sostén o depósito. Con `--extended-lift` se ensaya un recorrido de muñeca 7,5 cm mayor y un 12 % más de cierre articular.

`resultado.json` registra el recorrido, el agarre con contacto, el sostén de 15 cm durante 0,5 s y el reposo en la canasta. `visibilidad.json` mide los obstáculos delante del fruto con la profundidad de cada ojo. `rayos.json` registra el primer colisionador identificado entre el ojo y el centro del fruto, incluida la mano. Las fuerzas de rotura del pedúnculo son parámetros del ensayo.

### Agarre y depósito de plátano

Desde la raíz, con FFmpeg disponible y una carpeta de salida nueva, ejecuta el ensayo guiado en Isaac Sim 6.2.

```powershell
..\isaac-sim-6.2.0\python.bat tools\film_field_cycle.py --viz none --crop banana --output captures\ensayo_platano
```

El robot 000 usa la postura derecha calibrada y el 001 ensaya el acceso lateral izquierdo. Para probar la otra trayectoria derecha, añade `--banana-grasps derecha_despejada lateral`. El éxito exige que el punto inferior del plátano suba al menos 15 cm durante 0,5 s, que el pulgar y dos dedos mantengan el agarre durante el traslado y que la fruta repose en la canasta tras la retirada.

Cada rover guarda dos MP4 sincronizados, exterior y RGB-D de ambos ojos. `subpasos.json` registra oposición, deslizamiento, velocidad y altura a 120 Hz. `pixeles.json` cuenta los píxeles visibles del fruto en cada ojo a 15 fps.

| Trayectoria de la mano derecha | Sostén por encima de 15 cm | Vista exterior | Ojos RGB-D | Datos de la simulación |
| --- | ---: | --- | --- | --- |
| Calibrada | 4,40 s | [Abrir MP4](videos/platano/derecha_calibrada/robot_000.mp4) | [Abrir MP4](videos/platano/derecha_calibrada/robot_000_ojos.mp4) | [Ver datos](../simulaciones/platano_cosecha_verificada) |
| Despejada | 4,43 s | [Abrir MP4](videos/platano/derecha_despejada/robot_000.mp4) | [Abrir MP4](videos/platano/derecha_despejada/robot_000_ojos.mp4) | [Ver datos](../simulaciones/platano_surco_dos_ojos) |

Las dos carpetas conservan la orden en español, la semilla y los resultados de ambas réplicas. `mediciones.json` y `contactos.json` registran la ejecución a 30 Hz. `poses_fisicas.npz` guarda 501 muestras por robot a 15 Hz. `verificacion_robot_000.json` relaciona las métricas del robot filmado con las huellas SHA-256 de sus dos MP4.

Para reproducir las poses de la trayectoria calibrada en Isaac Sim desde la raíz del repositorio, usa una carpeta de salida nueva.

```powershell
..\isaac-sim-6.2.0\python.bat tools\render_field_replay.py --viz none --source simulaciones\platano_cosecha_verificada --robot 0 --eye-size 512 --output captures\reproduccion_platano
```

### Agarre y depósito en el surco de naranjos

Para ejecutar las tres situaciones con la misma semilla, desde la raíz del repositorio y con FFmpeg disponible, elige una carpeta nueva.

```powershell
.\scripts\naranjal.ps1 -Salida captures\naranjal_01
```

| Ensayo | Orden y comprobación |
|---|---|
| `estacionada` | «Recoge la naranja exterior por el costado». La base empieza en la estación y se mide su deriva. |
| `roca` | «Avanza por el surco. Si ves obstáculos, esquívalos». El planificador usa la posición de la roca en el mapa y mide el margen del chasis durante el recorrido. El rover se detiene si pierde el margen. |
| `izquierda` y `derecha` | «De las dos naranjas, recoge la de la izquierda» o «recoge la de la derecha». Se conserva la disposición física, se interpreta el lado desde el rover y se miden contacto, desplazamiento y desprendimiento de la vecina. |

Todas continúan con «Elévala 15 cm, sostenla medio segundo y déjala en la canasta». Cada orden guarda un MP4 exterior y otro con el RGB y la profundidad de ambos ojos. `resultado.json` distingue agarre, sostén y depósito. Son trayectorias guiadas por un controlador. Para repetir una orden, añade por ejemplo `-Tarea derecha`. Para medir sin vídeo, añade `-SoloFisica`. [Ensayos registrados](../simulaciones/naranjal)

Con la roca, el cierre progresivo consiguió la mayor elevación. En las tareas estacionadas se conserva la consigna de los motores cuando PhysX mide oposición. La selección izquierda usa el brazo izquierdo y la derecha, el derecho.

También se puede pasar directamente una de las órdenes de selección en español.

```powershell
..\isaac-sim-6.2.0\python.bat tools\film_field_cycle.py --viz none --crop citrus --orden "Recoge la naranja de la izquierda" --output captures\seleccion_izquierda_01
```

El banco anterior permite comparar tres trayectorias de aproximación con el rover estacionado.

```powershell
..\isaac-sim-6.2.0\python.bat tools\rover_orange_trial.py --viz none --output captures\surco_naranjal_01
```

El rover queda estacionado frente al tercer árbol de la segunda hilera. Una rama se extiende 32 cm hacia la calle y sostiene una naranja de 160 g en lugar del fruto decorativo `Orange02`. El pedúnculo aplica una fuerza elástica amortiguada con un umbral de rotura de 4 N elegido para la simulación. Se prueban una entrada frontal derecha, una lateral izquierda y una diagonal derecha.

Cada robot guarda dos MP4 sincronizados a 15 fps. `robot_NNN.mp4` muestra las vistas exterior y superior con la elevación desde la posición inicial del fruto. `robot_NNN_ojos.mp4` muestra el RGB de ambos ojos en la fila superior y su profundidad en la inferior. En la escala de 0,055 a 2,5 m, lo cercano aparece claro.

`resultado.json` separa el agarre, el sostén a 15 cm durante 0,5 s y el depósito con la mano retirada. PhysX mide el contacto del pulgar y dos dedos, el deslizamiento, la altura y el contacto con la canasta. `mediciones.json` conserva esas señales por paso. Las trayectorias son guiadas por cinemática inversa.

### Plan de ensayos para naranja, uva y arándano

El siguiente bloque propone nueve situaciones de cosecha guiada en Isaac Sim. Primero se aísla el agarre con el rover estacionado. Después se incorpora el recorrido por el surco y, por último, la selección de fruto o la coordinación de ambas manos.

| Cultivo | 1. Agarre en estación | 2. Recorrido y obstáculos | 3. Tarea de mayor dificultad |
| --- | --- | --- | --- |
| Naranja | Tomar una naranja exterior desde un costado, oponer el pulgar a dos dedos y desprenderla con tracción progresiva. | Seguir la curva del naranjal, rodear una roca y estacionarse del lado que dé acceso a la rama. | Colocar dos naranjas físicas en la misma escena. Alternar las órdenes «recoge la de la izquierda» y «recoge la de la derecha», medidas desde el rover estacionado. Comprobar cuál llega a la canasta. |
| Racimo de uvas | Entrar por el lado libre de la espaldera, sostener el racimo desde abajo, desprenderlo y frenar el balanceo antes de trasladarlo. | Recorrer el viñedo con pendiente transversal. Ante el poste caído, usar un paso con anchura suficiente o detenerse. Situar la base para mantener el brazo separado del alambre. | Comparar con un agarre bimanual. Una mano sostiene el racimo y la otra estabiliza su parte superior. Medir los contactos de cada mano y comprobar que el racimo queda sujeto tras el desprendimiento. |
| Arándano | Centrar una baya en ambos ojos y tomarla con pulgar e índice. Subir primero 3 cm para calibrar la pinza y después completar la elevación y el depósito. | Mantener las ruedas en la calle, bordear la rama caída y buscar otro ángulo si una hoja oculta la baya. | Colocar bayas vecinas con física. Recoger dos bayas de una en una, con órdenes distintas y cambio de iluminación entre recogidas. Medir el desplazamiento de cada vecina y verificar cada depósito por separado. |

Las naranjas de selección tienen identificadores, colisiones y poses registradas. En el ensayo bimanual de uva se comprobará que la zona superior tenga un colisionador que la mano pueda sujetar. El pedúnculo actual se desprende por carga.

Cada cosecha completa exige que el punto inferior del fruto suba al menos 15 cm desde su altura inicial durante 0,5 s continuos. Se comprobarán el agarre hasta la apertura sobre la canasta, el reposo dentro de ella durante 0,5 s con los dedos retirados y los contactos con obstáculos. Se registrarán la semilla, las órdenes y subtareas en español, el deslizamiento y la visibilidad en cada ojo. Cada robot tendrá dos MP4 sincronizados, exterior y RGB-D. En las variantes con oclusión inicial, se medirá cuánto tarda en recuperar la vista del fruto antes de cerrar la mano.

### Política móvil y adquisición

El teacher PPO de mesa trabaja con el rover estacionado frente a la fruta. Observa el estado físico y la dirección de cada ojo respecto al fruto. El estudiante usa las imágenes y vuelve a observar tras aplicar cada acción.

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
