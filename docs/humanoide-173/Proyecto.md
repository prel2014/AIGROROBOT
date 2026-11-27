# Humanoide bípedo de 1,73 m

Estudio de actuadores y primeras interfaces imprimibles para Bambu Lab A1 con PLA. El CAD de cuerpo completo fija proporciones y centros articulares de partida. Los dos soportes permiten comprobar el encaje en banco. El levantamiento y el transporte de la torre siguen pendientes de dimensionamiento y ensayos.

### Caderas instaladas en el robot completo

El [STEP principal de Xavier](cad/humanoide_xavier.step) incorpora seis actuadores de cadera. Cada salida de flexión se une al asiento original del muslo con sus cuatro M5. El travesaño está fijo a la pelvis. En el [visor de cuerpo completo](http://127.0.0.1:3245/C:/Users/prodriguezl1/Downloads/Hacking/humanoide-173?file=cad/humanoide_xavier.step.py) se pueden combinar giro, apertura y flexión independientemente para cada pierna. La rodilla y el tobillo acompañan al muslo y conservan sus propios controles.

Los conectores de servicio quedan detrás de la pelvis. Desde cada conector, la funda de CAN y alimentación de la rodilla entra por la parte posterior del muslo. El recorrido animado es aproximado y la longitud de su línea central cambia al mover la pierna. La longitud de corte, la reserva de cable para el movimiento y el radio mínimo de curvatura se determinan en banco. El cableado de los seis actuadores de cadera y la validación bajo carga también están pendientes.

Con las piernas extendidas, pies a igual altura y pelvis fija, la distancia entre los bordes internos de las suelas es 95,7 mm en neutro. El primer contacto ocurre al cerrar simultáneamente ambas piernas 3,195° por lado. Este resultado corresponde a esa postura. Isaac reproduce los tres ejes de cada cadera con las mismas convenciones de giro que el CAD. [Cálculo con la geometría de las suelas](cad/verificar_aduccion.py)

En la [vista lateral del cuerpo completo](cad/xavier_completo_paso_lateral_20261127T192827Z.png), la cadera derecha flexiona 35° y la rodilla 20°, con tobillo neutro y pelvis fija. El centro del tobillo avanza 328,8 mm y el punto más bajo de la suela derecha queda 66,8 mm sobre el plano de apoyo izquierdo. [Vista frontal en neutro](cad/xavier_completo_neutro_20261127T192716Z.png)

### Entrenamiento de las caderas en Isaac

Los 16 Xavier llegaron y permanecieron estables en cada una de las tres pruebas de cruce sin coordinador. En los 48 recorridos hubo cero caídas, cero contactos entre robots y cero contactos con el entorno. El último robot completó la parada a los 56,90 s en la semilla 937, a los 56,80 s en la 1201 y a los 51,65 s en la 1417. En la grabación, Xavier 04 espera hasta los 22,20 s para empezar a caminar. [Vídeo de la prueba 937](isaac/cruce_visual_campus_16.mp4) · [Las 16 cámaras a los cuatro segundos](isaac/cruce_visual_campus_16_camaras.jpg) · [Informe por robot](isaac/validacion_cruce_visual.json) · [Pesos de la decisión visual](isaac/paso_visual_campus.pt)

En una prueba complementaria se mantuvo fija la cámara y se colocó un robot del carril transversal alternativamente fuera y dentro de la zona de conflicto. En fotogramas aislados, la CNN indicó paso libre en 46 de 48 vistas despejadas y espera en 34 de 48 vistas ocupadas. Esta medición evalúa la discriminación visual estática. En los recorridos físicos se evalúa la secuencia de imágenes durante toda la espera. [Resultados de las vistas emparejadas](isaac/prueba_pareada_camaras.json)

La entrada al cruce se entrena por imitación con imágenes del campus. Los 16 robots comparten los pesos de la red. Cada cabeza entrega su imagen RGB de 256 × 256 a 20 Hz. La red calcula el riesgo frontal y transversal. Si ha visto tráfico, el robot compara fotogramas de su lateral derecho mientras espera. Empieza a caminar cuando la vista permanece despejada y continúa hasta su meta. La marcha usa la política PPO de la actualización 82, la IMU y los sensores plantares.

Para entrenar una primera red desde la carpeta `Hacking` se necesita el Python de Isaac para renderizar y un Python con PyTorch para ajustar la red. Los dos comandos usan la misma carpeta de imágenes.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_paso_visual.py --recoger --datos .\humanoide-173\isaac\__pycache__\imagenes_cruce --escenas 1000
python .\humanoide-173\isaac\entrenar_paso_visual.py --datos .\humanoide-173\isaac\__pycache__\imagenes_cruce
```

El ajuste con dos robots que pasan consecutivamente usa `--recoger --caravanas --escenas 750 --semilla 6139`. Se pueden registrar esperas y entradas de una política anterior con `--semillas-recorrido`, indicando sus pesos mediante `--modelo`. El ajuste final acepta esas carpetas con `--datos-caravanas` y `--datos-recorridos`. La opción `--fondos` aprende referencias de apariencia a partir de las cámaras de entrenamiento y requiere scikit-learn.

La prueba de cruce ejecuta cuatro intersecciones simultáneas con cuatro Xavier en cada una. Cuenta un recorrido cuando el robot llega a su meta, permanece erguido y estable al menos un segundo y termina sin contactos ni caídas. También registra los que agotan los 65 segundos esperando. El vídeo y las cifras del primer episodio se obtienen durante la misma simulación.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --escenario campus --etapa cruce --validar-marcha 937 1201 1417 --informe .\humanoide-173\isaac\repeticion_cruce_visual.json --video .\humanoide-173\isaac\repeticion_cruce_visual.mp4 --robot-video 4
```

En la prueba histórica con coordinador y los pesos de marcha de la actualización 82, los 16 Xavier completaron la recta y el cruce en cada una de las semillas 937, 1201 y 1417. Los 96 recorridos terminaron con una parada estable, cero caídas y cero contactos con otros robots u obstáculos. En los tres cruces, el último robot completó la maniobra entre 36,10 y 36,55 segundos de simulación. [Informe por robot y semilla](isaac/validacion_marcha_campus.json) · [Grabación del cruce](isaac/cruce_campus_16.mp4) · [Modelo validado](isaac/marcha_campus_validada.pt)

Los 16 Xavier entrenan en el campus con una cámara de 256 × 256 píxeles, una IMU y cuatro celdas de carga por pie y robot. Cada cadera tiene mandos independientes de giro, aducción/abducción y flexión/extensión. Las consignas de marcha se filtran con una constante de 0,20 s y se limita el cambio de consigna de cadera a 45°/s. PPO modula una marcha de referencia. Sus correcciones máximas son 0,55° de giro, 0,32° de apertura y 0,15° en el plano sagital. Los límites mecánicos de las articulaciones se aplican en PhysX. [Control de marcha](isaac/control_marcha.py)

Al llegar, el controlador acorta los últimos pasos hasta que la proyección del centro de masa y su velocidad permiten detenerse sobre los pies. Conserva parte de la diferencia de postura entre ambas piernas y regula los tobillos durante la espera. Una llegada otorga dos puntos y permanecer erguido y por debajo de 0,12 m/s durante un segundo consecutivo otorga otros seis. Una caída o un contacto posterior invalida la aprobación. En el cruce actual, cada Xavier evalúa dos riesgos con su cámara frontal RGB de 256 × 256: alcanzar al de delante y entrar en conflicto con el tráfico transversal. Cuando ve paso libre de manera continua, empieza la marcha y la mantiene hasta alcanzar su propia meta. [Entrenador](isaac/entrenar_marcha.py)

Desde la carpeta `Hacking`, esta orden permite retomar un entrenamiento detenido y añadir 600 actualizaciones. Cada actualización guarda los pesos y el optimizador. La CNN y las cámaras RTX usan la GPU. PhysX usa la CPU para esta ejecución.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --escenario campus --reanudar --iteraciones 600 --pasos 48
```

La evaluación ejecuta la política congelada en recta y en cruce para cada semilla. Comprueba el SHA-256 del checkpoint y guarda las llegadas, caídas, contactos y el tiempo estable por robot. Devuelve un error si algún robot falla.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --escenario campus --validar-marcha 937 1201 1417 --informe .\humanoide-173\isaac\repeticion_marcha_campus.json
```

Para ver a los 16 robots cruzar en Isaac y abrir el mosaico de sus cámaras, reproduce el episodio con los pesos de la evaluación.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --visible --escenario campus --etapa cruce --evaluar 1 --semilla 937
```

## Actuadores comerciales

Precios publicados consultados el 24 de noviembre de 2026. El stock, el envío a Perú y los impuestos se confirman por referencia.

| Modelo | Par nominal o continuo indicado | Par de pico o bloqueo | Tensión y control | Precio de referencia | Uso a evaluar |
| --- | ---: | ---: | --- | --- | --- |
| Waveshare ST3215, SKU 22414 | 0,98 N·m nominal | 2,94 N·m de bloqueo | 12 V, bus serie TTL | US$21,99. [Tienda](https://www.waveshare.com/st3215-servo.htm?sku=22414) · [ficha](https://docs.waveshare.com/ST3215_Servo) | Giro de cuello con apoyo independiente y centro de masa cerca del eje, o ensayos de mecanismos de mano de bajo esfuerzo. |
| ROBOTIS DYNAMIXEL XL330-M288-T | 0,10 N·m estimado por ROBOTIS | 0,52 N·m de bloqueo | 5 V, bus TTL DYNAMIXEL | US$30,94. [Tienda](https://robotis.us/xl/) · [ficha](https://robotis.us/products/dynamixel-xl330-m288-t) | Dedo o pulgar de bajo esfuerzo. Su bus y alimentación requieren su propia interfaz. |
| CubeMars AK45-36 V3.0 KV80 | 8 N·m | 24 N·m de pico | 24 V, CAN, 40 rpm nominales | US$185,90. [Ficha y tienda](https://www.cubemars.com/product/ak45-36-v3-0-kv80-robotic-actuator.html) | Giro de antebrazo, muñeca o ejes secundarios según brazo de palanca y carga. |
| CubeMars AK60-39 V3.0 KV80 | 24 N·m | 72 N·m de pico | 48 V, CAN, 70 rpm nominales | US$448,90. [Ficha y tienda](https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html) | Primera articulación de potencia en banco. Candidato para estudiar codo u otros ejes de mayor demanda con la carga medida. |
| CubeMars AK80-64 KV80 | 48 N·m | 120 N·m de pico | 48 V, 40 rpm nominales | US$989,90 en la tienda con tarjeta de control seleccionada. [Tienda](https://store.cubemars.com/products/ak80-64) · [ficha](https://www.cubemars.com/goods.php?id=1143) | Referencia de mayor par para evaluar rodilla o cadera. Confirmar par, velocidad y disipación durante la marcha. |

Para el CAD del servo utilicé el SKU Waveshare **22414 de 12 V**. Mouser Perú publicó el DFRobot **SER0070** de 12 V a **S/ 96,85** como opción de compra a comparar. Para intercambiarlo, confirma el patrón posterior y la rosca con el vendedor o la pieza en mano. [Mouser Perú](https://www.mouser.pe/en/ProductDetail/DFRobot/SER0070?qs=sGAEpiMZZMsqIr59i2oRchyqFsDZxddudCVCewvAABk%3D) · [DFRobot](https://www.dfrobot.com/product-2962.html)

## Carga y presupuesto

Para estudiar la sensibilidad al peso, supuse 8, 12 y 16 kg, agarre simétrico y 0,30 m de brazo horizontal por lado. La contribución de la torre por lado es `masa × 9,81 × 0,30 / 2`, lo que da **11,8, 17,7 y 23,5 N·m**, respectivamente. El cálculo omite el peso del brazo, la aceleración, las pérdidas y el margen térmico. El AK45-36 supera su par nominal en los tres escenarios con ese brazo de palanca. A 16 kg, el AK60-39 queda a 0,5 N·m de su nominal antes de incorporar esas contribuciones. Las posturas y velocidades reales determinan si una transmisión distinta o un brazo de palanca menor permite usar estos candidatos.

La arquitectura inferior plantea 12 ejes activos como punto de partida. A **S/ 3,50 por US$** como supuesto de presupuesto, seis AK60-39 y seis AK45-36 suman **US$3 808,80 ≈ S/ 13 331** solo en actuadores. Las cantidades ilustran el costo, y cada asignación queda pendiente de dimensionamiento. Faltan electrónica, baterías, herrajes, impresión, envío e impuestos. El alcance de cuerpo completo con torre queda sin una configuración verificada dentro de S/ 12 000 con esta combinación.

## Arquitectura mecánica preliminar

En el [modelo de proporciones](cad/humanoide_conceptual.step.py), +Z apunta arriba y +Y hacia delante. La neutral es erguida, con la planta a Z=0 y la parte superior de la cabeza a Z=1730 mm. Las cotas de centros para tobillo, rodilla, cadera y hombro son 85, 480, 875 y 1400 mm, respectivamente. Las longitudes son decisiones iniciales de diseño y siguen editables.

| Región | Ejes previstos | Comprobación prevista |
| --- | --- | --- |
| Cuello | 3, giro, flexión y ladeo | Girar ±45° como objetivo inicial. Medir interferencia de cables y apoyar el peso de la cabeza en una estructura independiente del servo. |
| Tronco | 3, flexión, inclinación y giro | Alcanzar la torre a distintas alturas y registrar su centro de masa respecto a los pies. |
| Hombros | 3 por lado | Alcanzar ambos puntos de agarre sin interferencia con el tórax. |
| Codos y antebrazos | Flexión de codo y rotación de antebrazo por lado | Simular el paso de brazos extendidos a agarre cercano al torso. |
| Muñecas | 2 por lado | Mantener la pose del agarre con flexión y desviación. |
| Manos | Flexión de dedos y oposición del pulgar | Definir acoples, fuerza y recorrido con el agarre de la torre real. |
| Caderas | 3 por lado | Transferir peso a un pie y girar. |
| Rodillas | 1 por lado | Levantarse y avanzar sin saturar el actuador. Como referencia, los adultos de 20 a 44 años del estudio CDC tuvieron medias de flexión cercanas a 138–142°. |
| Tobillos | 2 por lado | Modificar inclinación y flexión sin perder contacto previsto. |
| Dedos del pie | 1 adicional por lado | Evaluar la flexión del antepié para despegar. |

Los ejes son objetivos de arquitectura. Los límites de control, topes, rangos combinados e interferencias siguen pendientes en el ensamble de cuerpo completo. Los estudios humanos guían los requisitos de movimiento. [CDC, mediciones pasivas](https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index_1704899060.html) · [Estudio de movilidad activa del cuello en tareas cotidianas](https://pubmed.ncbi.nlm.nih.gov/12177547/)

## Revisión de la carrocería

La revisión conserva la altura de 1.730 mm y los centros articulares del modelo de proporciones. Las placas pectorales se encuentran en V sobre el abdomen segmentado. La cabeza aloja una cámara EZVIZ H9c Dual 2K. Las manos tienen cuatro dedos y un pulgar opuesto, y el pie tiene suela, puntera y empeine diferenciados. Los codos y las rodillas incorporan dos orejetas, una lengüeta central y un pasador transversal. En el CAD, la separación lateral entre lengüeta y orejeta es de 0,8 mm por lado. La diferencia nominal entre diámetros de agujero y pasador es de 0,7 mm.

El STEP reimportado con la H9c mide **578 × 268 × 1.730 mm** (ancho, fondo y altura, redondeados al milímetro). La validación geométrica pasó en los **167 sólidos**. En postura neutra se filtraron los pares por sus cajas envolventes y se calcularon las intersecciones de los 237 pares candidatos. Ningún par superó 0,01 mm³ de solape.

En la primera revisión las cubiertas del codo y la rodilla se cruzaban a 60°. Se abrió el pliegue interior de ambas y se rebajaron los laterales junto a las orejetas. En la geometría final se mantuvo fijo el segmento proximal derecho y se giraron la lengüeta y el segmento distal alrededor del pasador, hacia delante en el codo y hacia atrás en la rodilla. A **0°, 15°, 30°, 45°, 60°, 75° y 90°** ninguna pareja de sólidos de esos módulos superó el umbral de solape de 0,01 mm³. Los ángulos intermedios y los movimientos combinados del cuerpo quedan pendientes de verificación.

Las carcasas se modelaron como volúmenes exteriores. Para fabricar el cuerpo faltan el vaciado y la división en piezas para la A1, los alojamientos de actuadores y rodamientos, el cableado y las fijaciones. La masa, el par para caminar o transportar la torre y la resistencia del conjunto siguen pendientes de dimensionamiento.

## Cabeza de visión EZVIZ H9c

Se prevé una cámara completa con su soporte desmontable de fábrica. Promart identifica la H9c Dual 2K de 3 MP con el código de artículo 1000727651. Para el CAD se asumió la revisión **CS-H9c-R100-8H33WKFL**, pendiente de cotejar con la etiqueta de la unidad. El plano acotado muestra **116 mm de ancho con las antenas plegadas, 142,4 mm de fondo hasta el asiento trasero y 145 mm de alto**. La ficha comercial redondea el fondo a 142 mm. La cámara pesa 668 g y requiere alimentación de 12 V CC y 1,5 A. El catálogo step.parts no devolvió un modelo H9c. Se reconstruyeron los volúmenes a partir del plano. Las posiciones de los pivotes, objetivos, pestañas y salida de cable se estimaron a partir de las figuras. [Promart](https://www.promart.pe/camara-wifi-ezviz-h9c-dual-2k-smart-home-doble-lente-3mp-1000727651/p) · [Ficha EZVIZ](referencias/EZVIZ-H9c-Dual-2K-R100.pdf) · [Plano EZVIZ, pág. 1](referencias/EZVIZ-H9c-R100-cotas.pdf)

El bastidor parte del asiento del cuello a Z = 1.532 mm. Dos nervios posteriores sostienen la placa de montaje y una cubierta superior hueca. Una abertura inferior de **30 × 24 mm** permite llevar los ramales de alimentación y Ethernet hacia la espalda. La base del cuello incluye cuatro pasos de Ø4,5 mm sobre un rectángulo de 34 × 28 mm y el cuello tiene cuatro taladros de Ø3,3 × 13 mm para roscar M4 en el prototipo. El bastidor es un único sólido de 139 × 143 × 198 mm. El STL es una malla cerrada de un solo componente.

**Montaje de banco.** Presentar la plantilla física de EZVIZ sobre la cara vertical del bastidor. Confirmar las distancias entre centros, los bordes de la plantilla y la posición del cable respecto al asiento. La placa se entrega sin perforar para esa interfaz. Las coordenadas medidas se pueden incorporar en `OEM_HOLES` antes de regenerar el STL. Fijar el soporte original, guiar ambos ramales y deslizar la cámara hasta que enganchen sus pestañas. Con la unidad en mano, comprobar el acceso para liberarlas, el paso de los conectores, el radio de curvatura y el funcionamiento con la iluminación nocturna. Los tornillos y su agarre efectivo deben dimensionarse para la placa real. [Manual EZVIZ, montaje en pared, págs. 7–8](referencias/EZVIZ-H9c-manual.pdf)

**Verificación óptica y mecánica.** El gran angular superior está orientado 15° hacia abajo y el teleobjetivo inferior parte de la horizontal. Los campos visuales nominales son 108° × 56° y 55° × 28°, respectivamente. En los primeros 400 mm desde cada objetivo, el bastidor queda fuera de ambos campos en posición frontal. El cilindro que reserva el barrido inferior aproximado tiene Ø99 × 94 mm y queda a **10,14 mm** del bastidor. Se comprobó también la ausencia de solape en seis combinaciones de giro e inclinación del modelo. Al girar hacia atrás, el bastidor entra en parte del campo inferior, por ejemplo a 150° de giro y con elevación horizontal. La holgura y la visión útil del equipo físico dependen de la revisión, del cable y de la orientación de las antenas. [Especificaciones de las dos lentes](https://www.ezviz.com/product/H9c-Dual-2K/52810)

Para dimensionar el cuello, la cámara sola aporta unos 6,55 N de peso y 0,066 N·m de momento por cada 10 mm de desplazamiento horizontal de su centro de masa respecto al eje de cabeceo. Faltan la posición de ese centro, el peso del bastidor impreso y el ensayo de retención en movimiento.

## Primeras piezas de banco

| Pieza | Cotas de la interfaz comercial | Decisiones de CAD |
| --- | --- | --- |
| Soporte ST3215 | Carcasa 45,22 × 24,72 mm nominales en el plano. Patrón posterior 24,45 × 20,50 mm. Primera columna a 8,30 mm del eje. Disco trasero Ø19,20 mm. El STEP muestra cuatro barrenos base Ø1,60 mm y un asiento posterior en Y=-25,60 mm en sus coordenadas. | Placa 75 × 44 × 4,2 mm con cuatro pasos Ø2,4 mm, paso Ø20,4 mm para disco y una ventana de 13,7 × 19,6 mm para el resalto y el conector. La rosca M2 y la forma exacta de la cabeza quedan pendientes de confirmación física. |
| Plantilla AK60-39 | Carcasa Ø79 ±0,1 mm y longitud total 67 ±0,2 mm. La tapa de electrónica Ø57 mm sobresale 18,5 mm del asiento en Z = −10,5 mm del STEP. La plantilla toma este asiento y el círculo Ø68 mm. La salida de par está en la otra cara, en Z = +38 mm, con tres M4 en Ø30 mm. | Placa 104 × 104 × 8,5 mm, paso central Ø59 mm y doce agujeros Ø3,4 mm. Para elegir los tornillos hay que distinguir los dos lados del plano, con profundidades M3 de 3 y 6 mm, y comprobar la entrada efectiva del tornillo en el asiento utilizado. |

El [archivo oficial del servo](https://docs.waveshare.com/ST3215_Servo/Resources-And-Documents) enlaza un plano rotulado SCS215. Se usaron las cotas que coinciden con el STEP de Waveshare para el patrón posterior. El modelo alternativo de [step.parts](https://www.step.parts/parts/waveshare_feetech_st3215_servo) se descargó con checksum verificado. El ensamble usa el STEP oficial. Para el motor se usaron el [plano oficial acotado](https://www.cubemars.com/data/cms/202602/ak60-39-v3-0-robotic-actuator-2d-drawing.pdf) y el [STEP oficial](https://www.cubemars.com/data/cms/202602/ak60-39-v3-0-robotic-actuator-3d-drawing.zip).

| Para los prototipos | Cantidad | Referencia |
| --- | ---: | --- |
| ST3215 de 12 V | 1 | Waveshare 22414. |
| Adaptador de bus para ST3215 | 1 | Waveshare Bus Servo Adapter (A), SKU 25514, US$4,99. Para banco, alimentación separada de 12 V y enlace al PC. [Tienda](https://www.waveshare.com/product/modules/bus-servo-adapter-a.htm) · [conexiones](https://docs.waveshare.com/Bus_Servo_Adapter_A/Product-Wiring-Example) |
| AK60-39 V3.0 KV80 | 1 | CubeMars, cotizar con placa de control incluida. |
| Mini tornillos de cabeza plana M2×6 | 4 | Mamut 114769056, S/5,90 por cuatro según el catálogo. Probar rosca, asiento y profundidad en el servo. [Sodimac](https://www.sodimac.com.pe/sodimac-pe/articulo/114768459/Mini-Tornillo-Cabeza-Plana-Phillips-M2.0x6-mm-4-unid./114769056) |
| Pernos M3×12 cabeza botón | 8 para el actuador y hasta 8 para ambos soportes | Mamut 114666309, pack de cuatro. Elegir largo de fijación al banco según el espesor de su superficie. [Sodimac](https://www.sodimac.com.pe/sodimac-pe/articulo/114663450/Perno-Cabeza-Boton-M3x12-con-Tuerca-4-unid./114666309) |
| PLA | 1 soporte de cada tipo | El material ya está disponible. Usar la cara inferior plana como orientación inicial. |

Las placas se montan sobre un banco con hueco o separadores para el cuerpo del actuador, el disco posterior y los conectores, según la orientación elegida. Los rodamientos y resortes del robot quedan pendientes de las cargas y recorridos por eje. Para el primer accionamiento, ensayar retenido en banco, con corriente y velocidad limitadas, acceso a la parada y sin cargar la torre real.

## Archivos y verificación

La animación de 34 segundos usa la geometría del humanoide Xavier. Muestra la colocación de los motores, la bandeja, los conectores de hombro, los tramos flexibles de codo y las conexiones de los dos antebrazos. Las vistas de detalle ocultan las tapas y la piel del tórax.

| Entrega | Archivos |
| --- | --- |
| Humanoide completo en OpenUSD con la guía de montaje en 3D | [Modelo USD](cad/humanoide_xavier.usd) · [exportador](cad/exportar_humanoide_usd.py) |
| Humanoide articulado para Isaac Sim 6.2 y PhysX | [Escena de banco](isaac/banco_xavier.usda) · [robot de base libre](isaac/humanoide_xavier_isaac.usd) · [control por nombre](isaac/controlar.py) |
| Guía práctica de montaje en banco, narrada en español y con 17 capítulos | [Vídeo MP4 con subtítulos](animacion/guia_montaje_en_banco.mp4) · [secuencia editable](animacion/secuencia_guia.py) |
| Ensamblaje del robot completo, plano de cuerpo entero | [Vídeo MP4 de 58 segundos](animacion/ensamblaje_robot_completo.mp4) · [secuencia editable](animacion/montaje_completo.py) |
| Montaje de las placas y el cableado de los brazos | [Vídeo MP4 de 34 segundos](animacion/ensamblaje_humanoide.mp4) · [modelo 3D animado GLB](animacion/ensamblaje_humanoide.glb) · [secuencia editable](animacion/montaje_arnes.py) · [generador de vídeo](animacion/renderizar.py) |
| Humanoide con carrocería revisada | [STEP](cad/humanoide_refinado.step) · [fuente paramétrica](cad/humanoide_refinado.step.py) · [vista](cad/vista_humanoide_refinado.png) · [comparación con el original](cad/comparacion_humanoide.png) |
| Cabeza de visión H9c | [STEP del conjunto](cad/cabeza_h9c.step) · [geometría y parámetros](cad/integracion_h9c.py) · [vista de integración](cad/vista_cabeza_h9c.png) |
| Bastidor de prueba H9c | [STEP](cad/soporte_h9c.step) · [STL para prueba de montaje](cad/soporte_h9c.stl) |
| Humanoide conceptual | [STEP](cad/humanoide_conceptual.step) · [fuente editable](cad/humanoide_conceptual.step.py) · [vista](cad/vista_humanoide_conceptual.png) |
| Montaje ST3215 | [STEP de ensamble](cad/ensamble_st3215.step) · [fuente de ensamble](cad/ensamble_st3215.step.py) · [vista](cad/vista_ensamble_st3215.png) |
| Soporte ST3215 | [STEP](cad/soporte_st3215.step) · [fuente editable](cad/soporte_st3215.step.py) · [3MF](cad/soporte_st3215.3mf) · [STL](cad/soporte_st3215.stl) · [vista](cad/vista_soporte_st3215.png) |
| Montaje AK60-39 | [STEP de ensamble](cad/ensamble_ak60_39.step) · [fuente de ensamble](cad/ensamble_ak60_39.step.py) · [vista](cad/vista_ensamble_ak60_39.png) |
| Plantilla AK60-39 | [STEP](cad/soporte_ak60_39.step) · [fuente editable](cad/soporte_ak60_39.step.py) · [3MF](cad/soporte_ak60_39.3mf) · [STL](cad/soporte_ak60_39.stl) · [vista](cad/vista_soporte_ak60_39.png) |

Verifiqué un sólido cerrado por soporte, dimensiones y número de agujeros en el STEP, mallas STL cerradas y dimensiones de los 3MF en milímetros. El cálculo de intersección de sólidos dio cero interferencia volumétrica detectable entre cada placa y el STEP de su actuador en la pose de montaje, con umbral de 0,0001 mm³. Los dos soportes caben planos en el volumen nominal de la A1. Quedan pendientes el laminado con tu perfil, la prueba de ajuste con tornillos y cables físicos, y las pruebas de carga. [Guía oficial de la A1](https://cdn1.bambulab.com/documentation/quick-start-a75adcb1d5d5e/Quick%20Start%20Guide%20for%20A1.pdf)

### Abrir el USD y recorrer el montaje

El USD abre con el robot ensamblado, en metros y con Z vertical. Para recorrer la guía en Isaac Sim, selecciona `/Humanoide`, cambia la variante `Vista` a `Montaje`, elige la cámara `/Humanoide/CamaraDeMontaje` y reproduce la línea de tiempo a 24 fotogramas por segundo. Los 17 capítulos coinciden con el MP4. El atributo `montaje:capitulo` indica el paso actual y `montaje:comprobacion` describe la revisión de banco correspondiente. Puedes volver a `Ensamblado` para explorar las piezas en su posición de diseño.

La geometría, los materiales y las dos vistas están contenidos en el archivo. El bloque reservado para el futuro convertidor está marcado como geometría de guía. Esta versión sirve para inspección y animación de montaje. La dinámica de marcha y el accionamiento de los motores requieren un modelo físico validado.

### Controlar el humanoide en Isaac Sim 6.2

Abre [la escena de banco](isaac/banco_xavier.usda) y pulsa **Play**. La pelvis está sujeta y las suelas empiezan a 12 cm del suelo. En **Tools → Robotics → Joint Inspector**, selecciona `/World/Humanoide`, busca `codo_der` y cambia **Drives → Target Position** a **30°**. Devuelve la consigna a **0°** para extenderlo. El activo [humanoide_xavier_isaac.usd](isaac/humanoide_xavier_isaac.usd) se puede referenciar en otras escenas con la base libre. Incluye las mallas y materiales. Para trasladar el banco, conserva los dos USD juntos. [Joint Inspector de NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup/joint_inspector.html)

Desde la carpeta `Hacking`, estos comandos ejecutan el banco sin ventana. El primero mueve el codo derecho y cierra la mano izquierda al 50 %. Imprime la consigna y la posición alcanzada después de dos segundos simulados. El segundo enumera los 37 ejes y sus límites. El tercero ensaya el seguimiento y los topes.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\controlar.py --eje codo_der=30 --mano-izq 50
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\controlar.py --listar
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\controlar.py --verificar
```

La escena de banco usa la CPU para la física a 240 Hz y la RTX para dibujar el visor. En `/World/Fisica`, **Enable GPU Dynamics** está desactivado y **Broadphase Type** está en **MBP**. La demostración y el controlador usan un hilo de PhysX para este robot.

La [demostración visible](isaac/mostrar_recorrido.py) empieza con el índice derecho y muestra cada dedo cerrándose y abriéndose por separado. En **Control de dedos**, cada columna corresponde a una mano. Mover un deslizador activa el control manual y conserva las consignas de los demás dedos. **Reproducir secuencia** vuelve al recorrido automático de dedos, brazos y piernas.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\mostrar_recorrido.py
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\controlar.py --dedo indice_der=75 --dedo pulgar_izq=30
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\controlar.py --verificar-dedos
```

El mando `--dedo` acepta un cierre de 0 a 100 % y se puede repetir. La [prueba de independencia](isaac/verificar_dedos.py) aplica la trayectoria de la demostración a un solo dedo cada vez y mide a 30 Hz la posición de las diez almohadillas distales respecto de su propia palma. Comprueba el seguimiento de ambos servos del dedo activo, el retorno a la posición abierta y el desplazamiento de los otros nueve dedos durante la maniobra.

**Resultado del ensayo de dedos en PhysX (CPU).** Los diez dedos completaron el ciclo de 0 a 100 % y de vuelta a 0 %. Entre la posición abierta y la cerrada, las almohadillas de los ocho dedos largos se desplazaron de 66,61 a 66,69 mm. El pulgar derecho se desplazó 33,13 mm y el izquierdo, 33,43 mm. La mayor variación observada en una almohadilla sin mando fue de 0,028 mm. Al reabrir, el mayor error de posición de la almohadilla activa fue de 0,002 mm.

Los mandos corporales son relativos al ensamble erguido. El frente es +Y, la vertical es +Z y el lado derecho corresponde a +X en el CAD. La flexión positiva del codo lleva el antebrazo hacia delante. La rodilla positiva lleva la pantorrilla hacia atrás. La muñeca parte de una inclinación de montaje de 20°. La CLI acepta grados y comprueba cada intervalo. La interfaz de tensores usada por el controlador recibe radianes. [Controles y pivotes del CAD](cad/controles_pose.js) · [Preparación de humanoides en Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup_tutorials/tutorial_humanoid_workflow_prepare_usd.html)

| Eje | Recorrido desde el reposo | Par máx. del *drive* | Velocidad máx. configurada | Origen |
| --- | ---: | ---: | ---: | --- |
| Cuello | −40° a +40° | 0,98 N·m | 90°/s | ST3215 de 12 V, par nominal y velocidad de ensayo |
| Hombro | −20° a +65° | 18 N·m | 45°/s | Actuador virtual |
| Codo | 0° a +90° | 24 N·m | 420°/s | AK60-39, par y velocidad nominales |
| Muñeca | −5° a +25° | 3 N·m | 45°/s | Ajuste de montaje con actuador virtual |
| Cadera, giro | −20° a +20° | 24 N·m | 420°/s | CubeMars AK60-39 V3.0 KV80 |
| Cadera, apertura | −8° a +18° | 24 N·m | 420°/s | CubeMars AK60-39 V3.0 KV80 |
| Cadera, flexión | −15° a +35° | 24 N·m | 420°/s | CubeMars AK60-39 V3.0 KV80 |
| Rodilla | 0° a +80° | 24 N·m | 420°/s | AK60-39, par y velocidad nominales |
| Tobillo | −12° a +12° | 18 N·m | 45°/s | Actuador virtual |
| Dedo largo, motor 1 / motor 2 | 0° a +57° / −57° a 0° | 0,07 N·m | 90°/s | SCS0009, recorrido de la trayectoria CAD y topes de ensayo |
| Pulgar, motor 1 / motor 2 | −33° a 0° / 0° a +33° | 0,07 N·m | 90°/s | SCS0009, recorrido de la trayectoria CAD y topes de ensayo |

Cada mano tiene diez ejes de servo. Por ejemplo, `indice_izq_m1` mueve la primera palanca del índice izquierdo. El cierre porcentual coordina los dos motores de cada dedo y PhysX resuelve las bieletas mediante tres juntas de cierre por dedo. Una combinación arbitraria de los veinte motores puede producir contacto antes de alcanzar la consigna. El cero de simulación corresponde a la pose guardada en el USD. La calibración de los encoders sigue pendiente. [Cinemática y cierre](cad/preparar_controles.py) · [Atribución y modelo original de AmazingHand](referencias/AmazingHand/ATRIBUCION.md) · [Banco de servos](control_mano_can/README.md)

**Alcance físico.** Los topes angulares proceden de los recorridos de diseño. Falta medirlos en el ensamble real. El modelo asigna **24,441 kg** en total, incluyendo seis actuadores de 750 g. La inercia de cada actuador se aproxima como la de un cilindro uniforme fijado al eslabón que sostiene la carcasa. Para el cuerpo se estimaron masas por segmento, centros y tensores de inercia con cajas uniformes. Los componentes móviles de los dedos usan los datos inerciales del MJCF de referencia. El contacto usa cajas en el cuerpo y los pies y envolventes convexas de las cubiertas palmares y de las almohadillas. Doce tramos de cable flexible se conservan como guías en reposo. La deformación de cables, las pérdidas eléctricas, los límites térmicos y el equilibrio en marcha requieren caracterización propia. Los pares de los seis ejes virtuales y las ganancias de todos los reguladores son parámetros editables de simulación. [Generador y parámetros](isaac/preparar.py) · [Comprobación de geometría y topología](isaac/verificar_usd.py) · [CubeMars AK60-39](https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html) · [Waveshare ST3215](https://docs.waveshare.com/ST3215_Servo) · [FEETECH SCS0009](https://ftservo.github.io/ftservo-wiki/products/models/sc-0090-c001/main/)

**Resultado en Isaac Sim 6.2.** A 240 pasos por segundo, los 33 mandos siguieron simultáneamente una pose al 40 % del recorrido elegido desde reposo. Tras dos segundos, el mayor error de posición fue de 1,92° en los hombros y la mayor separación entre los anclajes de los 30 cierres de las manos fue de 0,028 mm. Con órdenes de +65° y −65°, el cuello se detuvo en +40,000° y −40,000°. En una segunda escena con base libre, las suelas partieron 35 mm sobre el suelo y los centros de ambos pies quedaron a 29,0 mm de altura tras 0,5 s, en la cota de apoyo prevista. [Prueba reproducible](isaac/controlar.py)

### Entrenar la marcha de 16 Xavier en una pista compartida

Los 16 humanoides de base libre comparten un suelo de 24 × 24 m. En cada uno de los cuatro cruces llegan dos filas perpendiculares. La posición inicial, los sentidos de llegada y la asignación de robots a las filas varían en cada episodio. Cada palma incluye la masa, la inercia y las superficies de contacto de sus dedos abiertos. Esto reduce el número de cuerpos físicos de 98 a 18 por robot. Una cámara RGB virtual en la cabeza de cada Xavier mira hacia delante y 15° hacia abajo. La IMU virtual calcula la fuerza específica y la velocidad angular en el sistema local de la pelvis. La física avanza a 120 Hz, el control de equilibrio durante la marcha a 60 Hz y la política visual a 20 Hz de tiempo simulado. [Pista para Isaac Sim](isaac/pista_marcha_16.usda)

Al iniciar con `--visible`, la pestaña **Cámaras en vivo · Xavier** muestra las 16 transmisiones RGB de 256 × 256 píxeles, identificadas del 00 al 15. Al pie aparecen la resolución de cada cámara, el contador de imágenes, los FPS por cámara y la relación entre el tiempo simulado y el real. El renderizado usa RTX y PyTorch usa CUDA cuando está disponible. El valor por defecto es CPU para la física. Para usar la GPU también en PhysX, pasa `--fisica gpu`. [Monitor en vivo](isaac/monitor_camaras.py)

Los robots comparten una política PPO. La CNN de cada robot recibe directamente la imagen RGB de 256 × 256 píxeles de su cámara. La política también recibe las lecturas inerciales y articulares, la fase del paso, el mando anterior y la dirección de su meta. PPO ajusta diez consignas articulares y la cadencia. Durante la etapa de marcha el entrenamiento intercala episodios con cruces. Un controlador de equilibrio utiliza la inclinación, la velocidad angular y la posición y velocidad de la pelvis respecto a los apoyos. Sus parámetros se ajustaron con recorridos físicos. Los motores conservan los límites de par, ángulo y velocidad del modelo. Una segunda CNN compartida decide cuándo entrar en el cruce. En ejecución, su única observación es la imagen de la cabeza del propio robot. Los ejemplos para entrenarla por imitación se etiquetan con la geometría y el tráfico de la simulación. [Política visual de paso](isaac/paso_visual.py) · [Entrenamiento visual](isaac/entrenar_paso_visual.py) [Control de equilibrio](isaac/control_marcha.py) · [Entrenador](isaac/entrenar_marcha.py) · [Política](isaac/politica_marcha.py)

El currículo permite comenzar con 12 segundos de equilibrio o entrar directamente en marcha. Durante la marcha se practica un cruce cada tres episodios. Dos episodios consecutivos con 16 de 16 aprobados en la etapa actual permiten avanzar a la siguiente. Una llegada exige que el centro de los pies pase la meta dentro del carril, con el torso erguido y al menos dos alternancias de apoyo. Al llegar, el robot frena y debe conservar el equilibrio hasta el final del episodio. Cualquier caída o choque invalida su aprobación. El informe JSON registra el resultado de cada Xavier y el instante de cada caída o contacto.

Cada suela de contacto de 119 × 245 mm tiene cuatro zonas virtuales de 59,5 × 122,5 mm. En ambos pies el orden es talón interior, talón exterior, antepié interior y antepié exterior. En cada paso físico se asigna el impulso normal de cada contacto a la zona que lo recibió, proyectado sobre la normal de la planta. La suma de seis pasos dividida por 0,05 s da la fuerza media en newtons. El visor muestra el total izquierdo y derecho. Al situar el cursor sobre el número del Xavier se ven sus ocho cargas. El informe de evaluación guarda los máximos por zona. [Plano de las plantas](isaac/plano_cargas_plantares.png) · [Cálculo de las cargas](isaac/cargas_plantares.py)

La red actor-crítico de PyTorch comparte sus pesos entre los 16 robots. La CNN extrae 144 características de la imagen. Se concatenan con 52 valores de estado, incluidos los encoders de los diez ejes de las piernas, sus velocidades, los mandos anteriores y las ocho cargas plantares divididas por el peso del robot. Dos capas densas de 96 y 64 neuronas alimentan el actor gaussiano de once acciones y un crítico escalar. El entrenamiento guarda los pesos en `marcha_cadera_estable.pt`. Las evaluaciones abren la copia congelada `marcha_campus_validada.pt`. La opción `--inicial` transfiere los canales compatibles desde una política de marcha anterior, inicializa a cero los pesos de los canales nuevos y empieza con un optimizador nuevo.

Desde la carpeta `Hacking`, el primer comando continúa el entrenamiento desde el modelo guardado y abre el visor. El segundo evalúa la política con otra semilla durante tres episodios, sin actualizar sus pesos.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --visible --reanudar
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --visible --escenario pista --evaluar 3 --etapa cruce --semilla 101 --informe cruces.json
```

Para empezar con pesos nuevos usa `--modelo` con un archivo de salida nuevo y omite `--reanudar`. Cada actualización guarda los pesos y el estado del optimizador. Con `--verificar` se comprueban las imágenes, la gravedad en reposo, la suma de las cargas frente al peso del modelo y la descarga de ambos pies al elevar un Xavier. También se comprueba la atribución de un contacto controlado antes de reiniciar la pista. En el selector de cámaras del visor, las vistas individuales siguen la ruta `/World/Xavier_00/cabeza/Camara` hasta `/World/Xavier_15/cabeza/Camara`.

### Escenarios de entrenamiento

El entrenador abre el campus por defecto. Para continuar el aprendizaje en un entorno concreto, ejecuta desde la carpeta `Hacking` el siguiente comando y cambia el valor de `--escenario` según la tabla.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --visible --reanudar --escenario campus
```

| Escenario | Espacio y elementos de referencia |
| --- | --- |
| [vivienda](isaac/escenarios/vivienda.usda) | Casa con sala, comedor, cocina y dormitorio alrededor de una galería con lucernario. La cámara ve los cambios entre parqué, alfombra y terrazo, los vanos de las puertas y el mobiliario. |
| [calle](isaac/escenarios/calle.usda) | Calle de barrio inspirada en Trujillo con dos pasos peatonales, veredas, paradero, comercios y vehículos estacionados. Los pasos atraviesan el bordillo a nivel de calzada. |
| [campus](isaac/escenarios/campus.usda) | Campus universitario ficticio inspirado en la UPAO de Trujillo. Una biblioteca y un pabellón de aulas abren hacia un patio con galerías, pérgolas, palmeras, jardineras y bancos. |
| [pista](isaac/pista_marcha_16.usda) | Pista original de 24 × 24 m para reproducir la evaluación de referencia. |

Las tres escenas usan el mismo humanoide y la misma política. El mobiliario principal, las fachadas y los bordillos tienen colisiones estáticas. El episodio registra el robot y el objeto cuando se produce un contacto penalizado. El ejercicio usa metas individuales. La política visual de paso se entrena en el campus. La validez de una política en un escenario se puede medir con `--evaluar` y `--informe`. [Vista de los tres escenarios](isaac/escenarios/vistas.jpg) · [Cámaras de la casa](isaac/escenarios/camaras_vivienda.png) · [Cámaras de la calle](isaac/escenarios/camaras_calle.png) · [Cámaras del campus](isaac/escenarios/camaras_campus.png)

La evaluación del campus usó la política guardada en la iteración 331, en etapa de cruce y con semilla 101. El informe contiene las llegadas, las caídas, los choques y los máximos de carga plantar por robot durante ese episodio. [Resultado del campus](isaac/escenarios/evaluacion_campus.json)

Para regenerar las capas USD se usa `isaac/escenarios_marcha.py` con el Python de Isaac Sim. La cámara `/World/Escenario/Vista` muestra el entorno arquitectónico y `isaac/vistas_escenarios.py` renderiza la lámina en RTX. La madera y el asfalto utilizan juegos fotográficos PBR 2K. La planta de interior y el banco referencian modelos USD con sus propios materiales. Los cojines tienen normales continuas y los vehículos incorporan pasos de rueda, espejos y llantas. [Vista RTX](isaac/escenarios/vistas.jpg) · [Modelos y licencias](isaac/escenarios/texturas/origen.md)

### Entrenamiento con cuatro situaciones

La etapa `situaciones` asigna cuatro Xavier por tarea en cada episodio. Una semilla determina el robot, el lado y la flexión solicitada entre 28° y 48°. Las tareas consisten en flexionar y extender un codo, alternar los brazos, flexionar y extender ambos codos juntos, y caminar entre 0,38 y 0,72 m para detenerse, levantar un antebrazo y bajarlo.

Una postura requiere 0,35 s continuos con error máximo de 8° en ambos codos, velocidad horizontal inferior a 0,16 m/s, pelvis erguida y carga plantar superior al 55 % del peso. La llegada a pie también exige dos alternancias de apoyo. Cada robot termina al completar su secuencia, caer o chocar. El informe conserva tarea, ángulo, lado, fases y motivo del desenlace por robot.

PPO ajusta nueve salidas sobre referencias articulares, seis para las piernas, una para la cadencia y dos para los codos. Recibe la imagen RGB, 47 valores de estado y consignas, y trabaja con las medidas de PhysX. Los dedos permanecen en la configuración de marcha. El ejercicio de brazos mide posturas y todavía queda por entrenar el agarre de un fruto.

Desde `Hacking`, estos comandos muestran la política entrenada y permiten continuar desde los pesos de esta revisión. Cada entorno admite `--escenario vivienda`, `calle` o `campus`.

```powershell
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --etapa situaciones --escenario campus --evaluar 1 --semilla 197 --visible
.\isaac-sim-6.2.0\python.bat .\humanoide-173\isaac\entrenar_marcha.py --etapa situaciones --escenario calle --reanudar --iteraciones 10 --semilla 251
```

El entrenamiento inicial transfirió la política plantar a una red ampliada. Ejecutó 10 actualizaciones en vivienda con semilla 37, 10 en campus con 48 y 16 en calle con 59, con 32 decisiones por robot y actualización. Se calibró el frenado durante este proceso. La política final está en [situaciones.pt](isaac/situaciones.pt). Los informes de evaluación usan los mismos pesos y tres episodios por escenario. [Vivienda](isaac/escenarios/situaciones_vivienda.json) · [Calle](isaac/escenarios/situaciones_calle.json) · [Campus](isaac/escenarios/situaciones_campus.json)

La evaluación obtuvo 107 secuencias completas en 144 intentos individuales, con 37 caídas y cero choques registrados. Hubo 39 éxitos de 48 en vivienda, 34 de 48 en calle y 34 de 48 en campus. La maniobra que requiere más trabajo es frenar y señalar después de caminar. [Demostración de cuatro robots en el campus](isaac/escenarios/situaciones.gif)

| Tarea | Secuencias completas / intentos |
| --- | ---: |
| Flexionar y extender un codo | 36 / 36 |
| Alternar los brazos | 36 / 36 |
| Flexionar y extender los dos codos | 27 / 36 |
| Caminar, parar, señalar y bajar el brazo | 8 / 36 |

## Interior del torso con Jetson AGX Xavier y un banco extraíble

El rediseño conserva los 1.730 mm de altura, la cabeza EZVIZ H9c y las ubicaciones propuestas para los cuatro AK60-39 y el servo del cuello. Los motores se representan mediante sus envolventes exteriores en el STEP de esta variante. Se abrieron dos cavidades en la carrocería de la espalda. El modelo de servicio muestra el torso con sus dos tapas retiradas. [Torso abierto](cad/torso_xavier_abierto.step) · [Humanoide con tapas y motores](cad/humanoide_xavier.step) · [Animación de extracción del computador y del banco](animacion/servicio_interior_xavier.mp4) · [Vista posterior del torso](cad/vista_torso_banco.png) · [Parámetros del diseño](cad/interior_xavier.py)

| Zona | Qué contiene | Cómo se accede |
| --- | --- | --- |
| Tórax, bajo el travesaño de los hombros | Jetson AGX Xavier Developer Kit P2972 en vertical, con la cara de ventilación hacia la espalda. La envolvente del kit mide 105 × 105 × 65 mm. Dos cinchas laterales lo retienen sobre una bandeja con topes. Sobre él se reserva un volumen de 74 × 36 × 17 mm para la regulación DC-DC. | Retirar la tapa ventilada, desconectar los cables y liberar la bandeja. La cara posterior del kit queda a 26 mm de la cara interior de la tapa. Hay 35,5 mm desde cada costado del kit hasta la pared lateral de la cavidad, antes de contar guías y enchufes. [Especificaciones del kit NVIDIA](https://www.nvidia.com/en-in/autonomous-machines/embedded-systems/jetson-agx-xavier/) |
| Abdomen, centrado entre las caderas | Un cartucho con 36 celdas cilíndricas Naylamp SKU 001309. La disposición física es de seis filas por seis columnas, con ejes anteroposteriores, paso de 20 mm y dos separadores aislantes. La masa de las celdas es 1,476 kg y su centro geométrico está a 1.086 mm del suelo. Encima se monta el protector 4S sobre cuatro apoyos de 4 mm, con la PCB KiCad de 72 × 34 × 1,6 mm y una trampilla de servicio. | Desenergizar los equipos, soltar el conector y apartar el mazo. Retirar la tapa del registro y deslizar el cartucho hacia atrás. El cuerpo del casete tiene 2 mm por lado y 4 mm hasta el fondo. El puerto Molex de tres vías se alcanza a través de la tapa dorsal. El lado batería viaja con el casete y el ramal de dos hilos del robot asciende por un conducto propio. [Planos del banco](electronica/bateria_4s9p/Guia_del_banco.md) [Ficha Naylamp de la celda](https://naylampmechatronics.com/baterias/1309-bateria-litio-18650-3v-1600mah.html) |

**Potencia disponible.** Como hipótesis para la electrónica se propone 4S9P, cuatro grupos en serie de nueve celdas en paralelo. A partir de la ficha de Naylamp se obtienen 14,8 V nominales, 16,8 V a plena carga, 14,4 Ah y 213 Wh nominales. El máximo publicado de 0,32 A por celda limita el banco a **2,88 A**, equivalentes a **42,6 W a tensión nominal antes de pérdidas**. Con un rendimiento supuesto del 90 %, quedarían unos 38 W en esa condición. La potencia disponible disminuye al bajar la tensión. Los 36 elementos cuestan S/ 540 al precio consultado de S/ 15 por celda el 24 de noviembre de 2026. El diseño del banco auxiliar fija 1,5 A de servicio y 16,4 V de carga. Se añadieron los esquemas de protector, balanceo y estación externa. Quedan por validar el circuito, la PCB embarcada, los enlaces fusibles y el convertidor de la carga. [Esquemas y cálculos](electronica/bateria_4s9p/Guia_del_banco.md) [Naylamp](https://naylampmechatronics.com/baterias/1309-bateria-litio-18650-3v-1600mah.html)

El ensayo de electrónica debe medir el consumo en el conector del kit con la cámara, periféricos, convertidor y arranque. Los perfiles de potencia del módulo Xavier cubren 10, 15 y 30 W. Un único motor AK60-39 entrega aproximadamente **176 W mecánicos en su punto nominal** (24 N·m a 70 rpm), por encima de la potencia eléctrica total del cartucho. Para que un solo banco impulse el humanoide completo hacen falta celdas, BMS y distribución de potencia dimensionados para el bus de tracción de 48 V, los picos simultáneos y la regeneración. [NVIDIA](https://www.nvidia.com/en-in/autonomous-machines/embedded-systems/jetson-agx-xavier/) · [CubeMars](https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html)

**Comprobaciones de CAD.** La extracción recta del cuerpo del banco y de la bandeja del Jetson hacia la espalda dio 0 mm³ de cruce con la carrocería en postura neutra. También se comprobó el paso vertical entre cavidades y la ausencia de interferencia volumétrica de las dos tapas con sus módulos. El techo del hueco principal del Xavier está 17 mm por debajo del punto más bajo del eje de hombros. La verificación geométrica se hizo con la envolvente publicada del kit y las cotas de las celdas, Ø18 × 65 mm. El STEP oficial P2972 requiere una sesión de NVIDIA. Las rejillas y las interfaces del kit son esquemáticas. Antes de fijar el diseño faltan la orientación y el espacio de los conectores reales, la sujeción bajo vibración, la corriente de balance y el corte real del fusible, la masa completa del cartucho y la temperatura con el torso cerrado. [CAD oficial del kit](https://developer.nvidia.com/embedded/dlc/jetson-xavier-developer-kit-3D-cad-step-model)

**Maquetas de ajuste para la A1.** Para comprobar el recorrido y los retenes con masa inerte se exportaron [la bandeja del Jetson](cad/maquetas_xavier/bandeja_xavier.stl), [el casete](cad/maquetas_xavier/casete_18650.stl), [el separador, dos unidades](cad/maquetas_xavier/separador_36.stl), [la tapa del banco](cad/maquetas_xavier/tapa_banco.stl), [la trampilla del BMS](cad/maquetas_xavier/trampilla_bms.stl) y [la tapa ventilada](cad/maquetas_xavier/tapa_ventilada.stl). La carcasa final de un banco energizado requiere revisar aislamiento, comportamiento térmico y retención mecánica.


## Manos de cinco dedos y unión con la muñeca

Cada mano tiene cuatro dedos largos y un pulgar oponible. Los diez servos SCS0009 se reparten en cinco pares. Dos palancas y sus bieletas accionan las dos falanges de cada dedo. El meñique tiene su propio par y puede cerrar con los otros dedos extendidos. [Movimiento de los dedos](animacion/manos_cinco_dedos.mp4) · [Origen y modificaciones de AmazingHand](referencias/AmazingHand/ATRIBUCION.md)

El asiento del pulgar y la placa que soporta los otros cuatro dedos forman un único sólido. Los dos marcos del pulgar apoyan sobre esta placa y se fijan en tres puntos. El resalto del marco posterior compensa su desfase de 0,6 mm. La cubierta palmar se prolonga alrededor de la raíz del pulgar y deja una abertura para sus bieletas. El meñique tiene un alojamiento propio unido a la placa por dos nervios y una cubierta con espacio para la rótula exterior. [Detalle de la unión](cad/union_pulgar_palma.png) · [Mano derecha](cad/mano_derecha_servo.step) · [Mano izquierda](cad/mano_izquierda_servo.step)

La muñeca utiliza un eje transversal de 8 mm, un tambor de 43 mm y un asiento de 44,6 mm. La holgura axial de diseño es de 0,8 mm por lado. La palma se puede separar del asiento del antebrazo. Para montar el pulgar, presentar sus dos marcos sobre los apoyos de la palma, colocar los tres tornillos M2 × 8 desde el talón y después acoplar el soporte de muñeca y las cubiertas. Los tres tornillos se representan sin rosca. Hay que contrastar la longitud útil y el agarre real de las roscas en una pieza de prueba. [Diez STL para probar el ajuste de ambas manos](cad/maquetas_union_manos.zip)

En el humanoide la muñeca se monta con 20° de inclinación de reposo para separar el pulgar del muslo. El movimiento mostrado corresponde al cierre geométrico de las bieletas. El ajuste impreso, la retención del eje, los cables, el agarre bajo carga y la resistencia de la palma quedan pendientes de ensayo. [Humanoide ensamblado](cad/humanoide_xavier.step) · [Verificación geométrica reproducible](cad/verificar_manos.py)

En ambos lados, la separación nominal entre cada marco del pulgar y su asiento fue inferior a 0,001 mm. Se comprobó contacto sobre una superficie en ambos apoyos. Las almohadillas de los dedos no se intersecaron en las cinco posturas ensayadas. Con los brazos en reposo se midieron 7,69 mm hasta el muslo derecho y 8,09 mm hasta el izquierdo. [Captura CAD derecha](cad/__cadgen__/union_derecha_20261124T165835Z.png) · [Captura CAD izquierda](cad/__cadgen__/union_izquierda_20261124T165835Z.png) · [Captura del conjunto](cad/__cadgen__/humanoide_manos_integradas_20261124T170108Z.png)

El fabricante indica 1 A de bloqueo por servo a 6 V. Si se bloquearan los veinte servos de ambas manos a la vez, exigirían 20 A y 120 W a 6 V. El banco 4S9P descrito arriba ofrece 42,6 W nominales antes de pérdidas. Hace falta medir la corriente en los agarres previstos y dimensionar la alimentación de servos antes de energizar ambas manos. [FEETECH](https://ftservo.github.io/ftservo-wiki/products/models/sc-0090-c001/main/)

## Electrónica de control y recorrido de los cables

Las dos placas de control se desarrollaron para el kit AGX Xavier, los cuatro AK60-39 del ensamble, los veinte SCS0009 de las manos y el ST3215 del cuello. La revisión B del nodo de servos integra el STM32G0B1CBT6 y un conector SWD para programarlo. El plano de cableado incluye conectores, los dos buses CAN y el enlace Ethernet de la H9c. [Diseño y conexiones](electronica/Guia_de_control.md) · [Placas y recorrido del arnés](electronica/placas_y_cableado.pdf) · [Proyectos KiCad](electronica/control_humanoide_esquemas_detallados.zip)

El [banco CAN de la mano](control_mano_can/README.md) contiene el firmware del antebrazo y el programa de la Jetson para leer los diez SCS0009 o comprobar su ida y regreso uno por uno.

La bandeja de la Jetson incorpora dos bastidores laterales para la interfaz CAN y el nodo cervical. Cada antebrazo lleva el STM32 de su mano y un regulador Pololu de 6 V detrás de una tapa ventilada. El mazo llega a la palma por el costado de la muñeca. [Conjunto actualizado](cad/humanoide_xavier.step) · [Detalle del antebrazo abierto](cad/antebrazo_servicio.step) · [Piezas para imprimir](cad/piezas_instalacion_control.zip)

Los ramales de las manos cruzan los hombros y los codos por la cara posterior. La vista de servicio deja a la vista las fichas JST SM y XT30U, los peines de sujeción y las entradas a los antebrazos. [Recorrido de ambos brazos en STEP](cad/cableado_antebrazos.step) · [Vista dorsal sin tapas](cad/__cadgen__/recorrido_brazos_20261124T210040Z.png) · [Vista general](cad/__cadgen__/humanoide_arnes_20261124T210234Z.png)

## Datos para cerrar el dimensionamiento

- Peso medido de la torre con todos sus componentes, dimensiones y caras que se pueden agarrar.
- Altura inicial y final del levantamiento. Transportarla apoyada al torso y levantarla desde el suelo requieren dimensionamientos distintos.
- Volumen y masa de la estructura y de sus piezas móviles. Con estos datos se comprueban margen de par, disipación, flexión, estabilidad y costo por articulación.

## Servicio del cartucho auxiliar 4S9P

El plano mecánico reparte las 36 celdas en cuatro cuadrantes de 3 × 3. El piso superior aísla la placa del protector de las celdas. Sus cuatro taladros Ø2,8 mm forman un rectángulo de 62 × 24 mm para tornillos M2,5. Junto al borde derecho quedan el fusible F1 y el conector. La tapa dorsal deja libre el puerto para poder soltar el ramal del robot antes de extraer el cartucho. Desacoplar sin carga, apartar el mazo, retirar la tapa y sacar el conjunto hacia atrás. La trampilla superior de cuatro M2 se abre con el cartucho extraído. El cargador de 19 V de entrada permanece en la estación externa. [Humanoide](cad/humanoide_xavier.step) · [Cartucho abierto](cad/cartucho_4s9p.step) · [Esquemas eléctricos](electronica/bateria_4s9p/Guia_del_banco.md)

El STEP del humanoide incorpora la PCB de dos caras del protector y sus 41 componentes, exportados desde KiCad. Los conectores JST XH acodados J1 y J2 miran hacia la parte trasera y J3 hacia el costado. Las derivaciones Kelvin de R33 parten de los terminales del shunt. [PCB editable](electronica/bateria_4s9p/protector_cartucho/protector_cartucho.kicad_pcb) · [Detalle de montaje](cad/montaje_bms.step) El modelo separa el fusible y su alojamiento, las dos NTC, el termostato y sus rutas. Las cinco tomas llegan al plano de colectores. Las pletinas 9P, los enlaces fusibles individuales y la unión a cada polo requieren diseño y validación eléctrica antes del montaje con celdas. La posición de los sensores queda sujeta a una medición de temperatura en el cartucho.

Las dos carcasas Molex se modelaron mediante envolventes basadas en los planos del fabricante. Comprobar conector crimpado, enganche, bridas, tolerancias de impresión, contactos del portafusible y alivio de tensión con muestras físicas. El ramal del robot termina en la bahía superior, disponible para el futuro convertidor auxiliar. [Molex 39-01-4031](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/salesdrawingpdf/555/5557/039014031_sd.pdf) · [Molex 39-01-4032](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/salesdrawingpdf/555/5559/039014032_sd.pdf)

El STEP completo se volvió a abrir y se comprobaron siete posiciones del cartucho entre 0 y 140 mm de extracción, con tapa y ramal desacoplados. Se comprobaron además cuatro posiciones de apertura de la trampilla. En esas comprobaciones no se encontraron intersecciones superiores a 0,025 mm³. [Verificación reproducible](cad/verificar_cartucho.py) · [Vista del cartucho](cad/vista_cartucho_4s9p.png) · [Vista del cuerpo completo](cad/vista_humanoide_banco.png)
## Mover el humanoide en el visor 3D

Abre el modelo completo y selecciona **Parameters** en la hoja lateral del STEP. Los 15 controles permiten girar el cuello, ajustar por separado hombro, codo, muñeca, cadera, rodilla y tobillo de cada lado, y cerrar cada mano. **D** indica derecha e **I** izquierda desde el punto de vista del robot. El ajuste de muñeca es relativo a los 20° de reposo del montaje. El cierre de mano recorre de 0 a 100 % e incluye los cinco dedos. **Reset** devuelve todos los mandos a la pose de reposo. [Visor del humanoide](http://127.0.0.1:3245/C:/Users/prodriguezl1/Downloads/Hacking/humanoide-173?file=cad%2Fhumanoide_xavier.step.py) · [Ejemplo de pose](cad/vista_pose_articulada.png)

Los grados describen una pose de inspección con la pelvis fija. La forma de los cables flexibles se aproxima entre sus anclajes. La verificación de colisiones, del equilibrio y de los límites con el prototipo físico queda pendiente. El módulo actúa únicamente en el CAD. [Controles de pose](cad/controles_pose.js) · [Generación de las poses de la mano y rutas](cad/preparar_controles.py)

### Carcasas, tarjetas y mazos de las piernas

Cada muslo y pantorrilla lleva cuatro piezas de pared nominal de 2,8 mm. Las mitades frontal y dorsal se dividen en dos alturas y atornillan a separadores sobre los largueros. La tapa lateral ventilada de la rodilla da acceso al controlador integrado del AK60-39 y a su clavija combinada. En **Parameters**, desactivar **Carcasa exterior de las piernas** descubre los largueros, las dos fundas de servicio y la tarjeta. El mazo naranja lleva potencia de tracción y retorno. El turquesa lleva CAN0. Ambos llegan por el lado del motor solidario al muslo. La torsión admisible en el paso por la cadera y el enfriamiento del controlador requieren ensayos físicos. [Modelo editable de las piernas](cad/tren_inferior.py) · [Integración del driver y los cables](cad/equipo_rodilla.py) · [Carcasas en STL](cad/carcasas_piernas_STL.zip)

### Carcasas y cableado de los brazos

En el [humanoide completo](cad/humanoide_xavier.step), el control **Carcasa exterior de los brazos** abre la vista de servicio en ambos lados. Descubre los dos peines dorsales del húmero, el controlador del motor de codo y el STM32 y el regulador Pololu de 6 V de cada antebrazo. Los paneles del húmero tienen 2,8 mm de pared nominal y cuatro fijaciones M3 cada uno. La tapa abombada del antebrazo conserva los cuatro puntos de fijación y las ranuras frente al regulador. Las ocho cubiertas de ambos brazos están [en formato STL, en milímetros](cad/carcasas_brazos_STL.zip).

Cada codo lleva un AK60-39 con soporte sobre la orejeta del húmero. Una brida con eje y pasador transversal transmite el movimiento a la lengüeta del antebrazo. El driver CubeMars queda bajo la tapa lateral ventilada. Su mazo de tracción entra por dos conectores de servicio en el travesaño del hombro. La funda verde representa los pares de ida y vuelta de CAN0 hasta la derivación del motor. La naranja oscura representa los 48 V con su retorno. Ambas terminan en la clavija combinada del estator. Las otras dos fundas, turquesa y naranja clara, continúan hasta la electrónica de la mano. La alimentación de tracción procede de la distribución externa prevista para el proyecto. Los conectores de servicio de tracción se representan por su envolvente de montaje. [Brazos abiertos y bandeja en STEP](cad/cableado_antebrazos.step)

El control de pose dobla los mazos alrededor de los ejes del hombro y del codo. Para fabricar el conjunto hay que medir la longitud y el radio dinámico admisible de los mazos terminados y comprobar el par transmisible de la brida, la resistencia de los anclajes y la temperatura en carga. [Integración mecánica](cad/equipo_brazos.py) · [Comprobación de ajuste y recorrido](cad/verificar_brazos.py)

Se comprobaron dieciséis combinaciones por brazo, con el hombro a −20°, 0°, 30° y 65° y el codo a 0°, 30°, 60° y 90°. En esas posiciones la separación mínima calculada entre el mazo de la mano y la tapa lateral del motor fue de aproximadamente 2 mm. [Brazo cerrado](cad/brazo_cerrado.png) · [Interior del brazo](cad/brazo_interior.png)

En el brazo izquierdo las dos guías de tracción tienen mordazas desmontables. Los centros de los canales distan 17,0 mm. La guía inferior orienta los mazos hacia salidas a distinta altura. Junto al codo, un manguito bifurcado recibe las dos fundas en alojamientos independientes y conduce los hilos aislados a la clavija P2. El manguito es una pieza de diseño, pendiente de comprobar con el conector real. Las dos bases y las dos tapas de guía se pueden montar con los conectores ya puestos. [Piezas de la guía en STL](cad/guia_traccion_izquierda_STL.zip) · [Detalle del recorrido izquierdo](cad/guia_traccion_izquierda.png)

La separación exterior mínima entre ambas fundas pasó de 0,4 a 6,32 mm en el STEP en reposo. El barrido del visor abarca 86 posiciones de hombro y 91 de codo en pasos de 1°. El recorrido de tracción termina en el estator y acompaña al húmero. La flexión del codo se comprueba contra las piezas móviles del antebrazo. El punto más próximo apareció junto al hombro a −20°, con 6,06 mm entre las fundas. La cota conservadora para todo el recorrido fue de 5,8 mm, contando los refuerzos de protección y el margen numérico del muestreo. En los pasos de las mordazas se comprobó una holgura radial de al menos 0,5 mm. Falta contrastar en banco los diámetros exteriores, la flexibilidad y la retención reales. [Verificación reproducible](cad/verificar_guia_izquierda.py) · [Brazo izquierdo con el codo flexionado](cad/recorrido_traccion_izquierda.png)

## Caderas de tres ejes de Xavier

La variante de encaje incorpora dos motores por cadera. El motor vertical gira el muslo hasta 20° hacia dentro o hacia fuera. El horizontal, con eje anteroposterior, admite 8° de aducción y 18° de abducción. La flexión existente conserva el intervalo de −15° a +35°. Los dos pivotes guardan 224 mm entre centros y están a 875 mm del suelo en postura neutra. Los mandos del visor usan valores negativos para aducir. [Pelvis adaptada](cad/cadera_xavier_instalada.step) · [Fuente paramétrica](cad/cadera_xavier_instalada.step.py) · [Vista del mecanismo con cotas](cad/cadera_xavier_acotada.png)

Se adaptaron la cintura y el escudo pélvicos y los elementos abdominales inferiores para el giro de las horquillas y los apoyos del motor. La cintura incorpora un asiento de 62 × 60 × 14 mm con ocho taladros piloto de Ø5 mm para M6. La variante incluye los dos asientos originales del muslo. La referencia de encaje es el CubeMars AK60-39 V3.0 KV80, con 750 g por unidad y 24 N·m nominales a 48 V. Los cuatro añaden 3,0 kg antes de soportes, tornillos y cableado. El diseño de carga requiere calcular el par durante el apoyo monopodal y comprobar anclaje, tornillería y temperatura en banco. El entrenamiento en Isaac requiere incorporar las nuevas articulaciones y sus parámetros dinámicos al modelo de simulación. [Ficha de CubeMars](https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html) · [Modelo 3D del fabricante](referencias/AK60-39-V3.step) · [Plano acotado](referencias/AK60-39-V3.pdf)

Con pelvis fija, marcha detenida, rodillas extendidas y tobillos neutros, las suelas quedan separadas 95,7 mm. Con 2° de aducción en ambas caderas quedan 35,7 mm y con 3° quedan 5,8 mm. Los vértices de ambos bordes coinciden en las tres coordenadas a 3,195° por lado en las mallas CAD. Este cálculo simétrico depende de la postura de las piernas. [Comparación de posturas](cad/aduccion_piernas_xavier.png) · [Cálculo reproducible](cad/verificar_aduccion.py)

Qosqo Market, de Andes Tricore S.A.C. (RUC 20613934511), publica un servicio de importación a pedido con factura peruana y sin cantidad mínima. Falta una cotización local de cuatro AK60-39 V3.0 KV80 originales con controlador incluido, precio en soles, plazo y garantía. [Servicio de cotización con factura](https://qosqomarket.com/importacion-a-pedido/)


En la reimportación del STEP de la pelvis adaptada se verificaron 274 sólidos válidos. Se compararon los vértices del mecanismo con las superficies de las cinco piezas de carrocería de la variante en posición neutra y en cuatro combinaciones de los extremos de giro, apertura y flexión. En esa muestra ninguna penetración de vértices superó 0,15 mm. Los ocho pilotos del asiento fijo quedaron abiertos en las secciones Z = 936, 942 y 948 mm, con material en los cuatro puntos cardinales a 3,5 mm del centro de cada piloto. [Vista montada con 2° de aducción bilateral](cad/pelvis_instalada_aduccion_20261127T174746Z.png)

### Revisión del soporte trasero de apertura

El [ensamble instalado](cad/cadera_xavier_instalada.step) usa una horquilla superior que sujeta el estator y un bastidor con brida directa para el rotor del AK60-39 V3.0. En reposo, el extremo del mecanismo quedó en Y = −91,0 mm. En el banco anterior estaba en Y = −127,5 mm. Respecto al punto más posterior del escudo pélvico (Y = −85,1 mm), el saliente bajó de 42,4 a 5,9 mm. Al girar la cadera entre −20° y +20° en pasos de 0,1°, la máxima proyección posterior medida del conjunto que gira fue de 10,2 mm respecto al mismo plano. Las medidas excluyen cable y clavija. En esa revisión del soporte trasero, el disco lateral todavía era la brida de prueba de flexión. [Comparación de perfiles](cad/comparacion_cadera_lateral.png) · [Vista oblicua](cad/pelvis_soporte_compacto_20261127T180519Z.png)

Se reimportaron 256 sólidos válidos de la revisión compacta. En nueve combinaciones de apertura (−8°, 0°, +18°) y flexión (−15°, 0°, +35°) se comprobaron en la cadera derecha las interferencias de la carcasa con el bastidor y la salida del muslo, y las del bastidor y la horquilla con la salida y el asiento original del muslo. Los pares examinados dieron cero volumen de intersección a la resolución de 0,001 mm³. Quedan por verificar en banco la rigidez, la carga sobre el apoyo delantero y el actuador, el acceso de montaje y la salida del cable.

### Flexión del muslo con dos CubeMars AK60-39 V3.0 KV80

El STEP instalado lleva seis actuadores de cadera. Las dos unidades nuevas van en el lado medial de los pivotes X (a ±112 mm del centro de la pelvis y a 875 mm de altura). Cada estator se fija al bastidor que abre la pierna y cada rotor acciona directamente el asiento del muslo. La brida del muslo utiliza las tres roscas M4 en círculo de Ø30 mm y los tres alojamientos de pasador Ø4 en círculo de Ø28,5 mm. La placa posterior del estator usa ocho fijaciones M3 en círculo de Ø68 mm. Los rebajes y taladros están modelados. La tornillería, las clavijas y el mazo externo se dimensionan en el montaje de banco. [Ensamble STEP](cad/cadera_xavier_instalada.step) · [Fuente de las interfaces](cad/flexion_xavier.py) · [Plano de CubeMars](referencias/AK60-39-V3.pdf) · [STEP original](referencias/AK60-39-V3.step)

El fabricante da Ø79 ± 0,1 mm y 67 ± 0,2 mm de longitud total. La masa por motor es 750 g. El par nominal es 24 N·m a 48 V y el máximo de corta duración es 72 N·m. Los datos de Isaac para el muslo, la pantorrilla y el pie suman 3,50 kg. Con la pierna en el aire, rodilla extendida, tobillo neutro y 35° de flexión, la suma de los momentos gravitatorios alrededor de X da 7,92 N·m. Con la inercia de cajas del modelo, la inercia aproximada de esos tres segmentos respecto de X es 0,808 kg·m². Las masas pertenecen al modelo y el cálculo omite inercias del reductor, pérdidas, aceleraciones e impactos. Quedan pendientes el ciclo térmico, el esfuerzo sobre el bastidor, la tornillería y las cargas con un pie en apoyo. [Datos del simulador](isaac/preparar.py) · [Ficha del fabricante](https://www.cubemars.com/product/ak60-39-v3-0-kv80-robotic-actuator.html)

En postura neutra, los motores de flexión quedan a 44,6 mm del extremo anterior del escudo y a 45,6 mm del posterior. El borde inferior del motor queda 11,5 mm bajo el escudo. En la vista posterior quedan 10,0 mm entre tapas. El extremo trasero del conjunto sigue siendo el motor de apertura, que sobresale 5,9 mm. Las cotas se refieren al perfil global del escudo y excluyen el arnés enchufado. A partir de los vértices del CAD, dos esferas centradas en los pivotes que encierran los motores nuevos y sus bastidores dejan unos 2,7 mm de separación como estimación conservadora durante el giro. Hace falta comprobar tolerancias, flexión y salida de los cables antes de fabricar. [Perfil acotado](cad/flexion_xavier_perfil_acotado.png) · [Cálculo de las proyecciones](cad/medir_flexion_xavier.py) · [Muslo derecho a 35°](cad/cadera_flexion_accionada_20261127T184918Z.png)

Se reimportaron 352 sólidos válidos. En la cadera derecha se comprobaron las interferencias de las piezas nuevas con la horquilla, el accionamiento de apertura y la salida del muslo en nueve combinaciones de los ángulos −8°, 0°, +18° de apertura y −15°, 0°, +35° de flexión. No se detectaron penetraciones por encima de 0,03 mm³ en los pares probados. Se amplió el hueco interior del escudo para alojar los motores y permitir sus movimientos.

**Compra para la flexión, dos unidades.** Qosqo Market (Andes Tricore S.A.C., RUC 20613934511) publica cotización de importaciones a pedido en Perú con factura, incluso de pocas unidades. Falta confirmación escrita de la revisión original, precio y plazo. Como referencia, la ficha de CubeMars publica US$448,90 por unidad antes de gastos locales. MegaShop también publica el título AK60-39 V3.0 KV80 en Mercado Libre Perú y anuncia factura con IGV. En la ficha declara marca JNYHOBBY, modelo VDQQVQCAB y 50 g, que discrepan del actuador CubeMars de 750 g usado en el CAD. [Cotización local con factura](https://qosqomarket.com/importacion-a-pedido/) · [Solicitud preparada](referencias/cotizacion_flexion_ak60.txt) · [Publicación de MegaShop](https://www.mercadolibre.com.pe/ak6039-v30-kv80-dynamical-modular-brushless-motor-fit-for/up/MPEU4120157675)
