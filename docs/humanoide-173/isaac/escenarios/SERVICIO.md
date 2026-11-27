# Escenarios de servicio para el humanoide

[Ver las ocho escenas renderizadas en Isaac Sim](vistas_servicio.jpg) · [Detalle de los puestos y del andén](detalle_mercado_estacion.jpg) · [Vidrio del invernadero y espejo de la cabina](detalle_vidrio_reflejos.jpg)

En Isaac Sim, abre uno de los archivos USD de esta guía desde **File → Open**. Selecciona la cámara `/World/Escenario/Vista` para ver la distribución. En el mercado y la estación puedes cambiar a `/World/Escenario/VistaDetalle` para examinar el mostrador y el área de espera. Usa **Play** para reproducir los cruces y las puertas. Vuelve al fotograma cero entre intentos.

Cada archivo contiene el entorno completo, en metros y con Z hacia arriba. Para usar el humanoide existente, crea una escena de trabajo, agrega el entorno como subcapa y referencia el humanoide en esa escena. Sitúa la planta de sus pies a la altura de `/World/Escenario/Recorrido/Inicio` y oriéntalo hacia el primer hito. Los puntos `Paso_01`, `Paso_02` y siguientes indican lugares de paso en orden. `Meta` indica dónde termina el recorrido. Los objetos de interacción están bajo `/World/Escenario`.

Los materiales del entorno incluyen mapas PBR de color, normales y rugosidad. El terrazo presenta áridos pulidos y juntas mates. Las chapas de la cabina y las encimeras muestran un cepillado fino. El cartón kraft, la goma estriada, las hojas, las escamas, los cítricos y la corteza del pan tienen patrones propios. Los paños del bus y del invernadero transmiten la luz mediante un material MDL con índice de refracción 1,52. La cubierta satinada difumina el fondo más que los paneles laterales. La cámara `/World/Escenario/VistaMaterial` en el invernadero y el ascensor permite inspeccionar la transmisión y el espejo desde cerca.

Las medidas indicadas en **Evaluación** son criterios propuestos para observar o instrumentar cada ensayo. Los archivos describen el entorno y sus eventos. El control del humanoide y el cálculo de puntuaciones corresponden al ensayo que lo utilice.

### 1. [Mercado de abastos](mercado.usda)

Los [primeros planos de los seis puestos](primeros_planos_mercado.jpg) muestran los poros y gajos de la naranja abierta, los sépalos de los tomates, el ajonjolí del pan, las escamas sobre hielo picado, las marcas de las legumbres y las capas de pétalos. En el árbol USD, las seis cámaras están bajo `/World/Escenario`, desde `DetalleCitricos` hasta `DetalleFlores`.

Seis puestos rodean el pasillo bajo una cubierta de calamina con cerchas metálicas. Frutas exhibe cítricos en jabas y Verduras tiene berenjenas y tomates. Pan muestra panecillos con cortes en la corteza, Pescados tiene una bandeja de hielo, Granos expone sacos abiertos y Flores agrupa los ramos en cubos. La canasta amarilla de 450 g se encuentra al lado izquierdo de **Frutas**, sobre un tablero a 80 cm del suelo. El destino es la superficie libre junto a la balanza de **Caja**.

**Recorrido.** Sal desde el lado de caja, observa el cruce, llega al puesto de frutas y toma la canasta por el asa. Regresa por el pasillo y deposítala en la mesa de caja. En el prim `Carretilla`, la variante USD `momentoCruce` permite repetir el mismo recorrido con dos ritmos. En `temprano` cruza de izquierda a derecha entre los segundos 2 y 10. En `tardio` lo hace entre los segundos 12 y 20.

**Evaluación.** Compara la distancia mínima a la carretilla y el tiempo de espera entre variantes. Al final, comprueba si la base de la canasta queda apoyada por completo sobre la mesa de caja después de soltar el asa.

### 2. [Estación del Metropolitano](estacion.usda)

Compara la [ventanilla desde el andén con el interior del bus](primeros_planos_estacion.jpg). La cámara `DesdeAnden` está a 1,62 m sobre el pavimento y mira a través del cristal lateral izquierdo de la imagen de conjunto. La cámara `DentroDelBus` mira desde el pasillo hacia la cabecera. Hay catorce butacas tapizadas con carcasa y asa, correas con asideros, pulsadores de parada, rejillas de ventilación e iluminación de techo.

El andén y el piso del bus detenido están a 40 cm del suelo. Una línea amarilla marca la espera. El vano del vehículo mide 1,36 m y el espacio horizontal entre el andén y el piso del bus es de 6 cm. El interior cuenta con asientos y barras de apoyo.

Hay dos bancas en los extremos de la zona de espera, una marquesina y un directorio. Desde el centro del andén se distinguen las señales **Embarque 01** y **Salida**. La banda de losetas con botones en relieve corre paralela al borde. Al otro lado del carril se ven los locales comerciales.

**Recorrido.** Acércate a la línea por la derecha, detente ante la banda de losetas y cruza el umbral por el centro de la puerta. Gira a la derecha dentro del bus.

**Evaluación.** Observa si ambos pies permanecen detrás de la línea durante la espera. Registra contactos con el borde del andén y comprueba que ambos pies terminen sobre el revestimiento de goma del bus.

### 3. [Escalinata en ladera](escalinata.usda)

Dos tramos paralelos de 1,80 m de ancho comparten un descanso a 90 cm de altura. El tramo de subida tiene seis peldaños de 15 cm de altura y 34 cm de huella. El de bajada tiene nueve de 10 cm y 36 cm, respectivamente. Una franja ocre señala el borde de cada peldaño.

Las jardineras escalonadas siguen la pendiente a ambos lados. Dos farolas flanquean el descanso y las viviendas dan a la calle superior.

**Recorrido.** Sube por la izquierda, atraviesa el descanso frente al letrero **Mirador**, gira y desciende por la derecha hasta el suelo. Los postes y las barandas exteriores tienen colisión.

**Evaluación.** Anota tramo, peldaño y pie cuando se produzca un tropiezo. Mide por separado la distancia de la punta del pie al borde durante la subida y el apoyo de la planta durante la bajada.

### 4. [Invernadero](invernadero.usda)

Las cámaras `DetalleAlbahaca` y `DetalleMaceta` muestran la [albahaca y el acabado de la maceta de roseta](primeros_planos_invernadero.jpg). La primera planta tiene hojas opuestas unidas al tallo por pecíolos. La segunda concentra sus hojas alrededor del centro del tiesto. El borde redondeado deja a la vista el espesor de la arcilla, las partículas de corteza y la perlita del sustrato.

Dos hileras de siete macetas delimitan un pasillo central de 2,25 m entre tableros. La maceta ocre de 700 g está en la cuarta posición de la hilera izquierda. El tiesto y su follaje se mueven juntos. Las macetas vecinas y sus hojas tienen colisión.

Un depósito alimenta dos líneas de goteo por el borde exterior de los tableros. Las hojas curvas varían de altura entre macetas. Al fondo hay bandejas de germinación con plántulas y sobre las hileras se ven cables de suspensión y luminarias.

**Recorrido.** Entra por el extremo abierto, localiza `MacetaMarcada`, acércate por el pasillo y retírala. Vuelve a la mesa señalada **Semillero** junto a la entrada.

**Evaluación.** Registra si se toma la maceta correcta, cuántas plantas vecinas se tocan y si el tiesto queda en posición vertical sobre el semillero tras soltarlo.

### 5. [Almacén de encomiendas](almacen.usda)

Cuatro estanterías metálicas forman dos filas. En la mesa de recepción hay una encomienda de 44 × 34 × 33 cm y 1,4 kg. En la segunda estantería de la derecha, una marca verde reserva el hueco de entrega entre dos bultos. La superficie de esa balda está a 87 cm.

Las etiquetas **A1**, **A2**, **B1** y **B2** identifican las estanterías. Los paquetes tienen alturas distintas, precintos y etiquetas gráficas de barras. Dos palés cargados enmarcan la persiana del muelle de recepción.

**Recorrido.** Toma `Encomienda` en recepción, gira hacia el pasillo central y avanza hasta la segunda estantería de la fila B. Gira a la derecha y deposita la caja sobre la marca. Las baldas tienen colisión para soportarla.

**Evaluación.** Comprueba que la caja mantenga su altura durante el giro y que se apoye en la balda correcta. Mide el desplazamiento horizontal entre el centro de su base y el centro de la marca verde.

### 6. [Ascensor de edificio](ascensor.usda)

La cabina tiene dos hojas correderas. Los pulsadores `Llamada` y `PisoDos` están a 1,02 m de altura, tienen 12 mm de recorrido y vuelven a su posición por resorte.

El vestíbulo incorpora un directorio de plantas y un indicador de piso. Las jambas de acero y las ranuras del umbral delimitan la entrada. En el interior hay pasamanos, un espejo con marco metálico y un pavimento distinto al del vestíbulo.

**Recorrido.** Acércate al pulsador exterior, pulsa y espera al costado. Las hojas se abren entre los segundos 1 y 3. Entra cuando terminen de abrirse y pulsa el botón del fondo. El cierre transcurre entre los segundos 16 y 18. La apertura sigue ese horario incluso si el pulsador se acciona en otro momento. La prueba ocurre en un solo piso.

**Evaluación.** Mide el desplazamiento de ambos pulsadores y el instante en que se cruza el umbral. Registra cualquier contacto con las hojas.

### 7. [Comedor universitario](comedor.usda)

Cuatro mesas con sillas dejan un corredor central. En el mostrador de servicio hay una bandeja de 46 × 34 cm con borde y un vaso de vidrio de 220 g. Son cuerpos rígidos independientes, por lo que el vaso puede deslizarse o volcarse.

Las mesas tienen tazas, cucharas y servilletas. Al fondo están el menú del día y la línea de servicio con tres cubetas, protector de vidrio y mostrador de retorno.

**Recorrido.** Levanta la bandeja, avanza entre las mesas y gira hacia la mesa del fondo a la derecha. Deposítala desde el lado libre, frente a las sillas.

**Evaluación.** Observa la inclinación máxima de la bandeja durante el giro, el movimiento del vaso respecto a su posición inicial y si termina de pie dentro del borde. El contenido del vaso se representa mediante su masa.

### 8. [Taller de mantenimiento](taller.usda)

El banco tiene una superficie de trabajo a 84 cm. A la izquierda hay una pieza azul de 320 g con un orificio cuadrado pasante de 32 mm. A la derecha hay un pasador de 60 g con vástago de 24 mm de diámetro. El orificio se forma con cuatro sólidos independientes unidos en un mismo cuerpo rígido.

El tornillo de banco ocupa el extremo izquierdo. En el panel perforado cuelgan cuatro llaves y debajo del rótulo se encuentran los cajetines de tornillería. Una lámpara de brazo ilumina el tablero desde la derecha.

**Recorrido.** Sitúate frente al banco, estabiliza la pieza con una mano, toma el pasador con la otra, alinéalo e introdúcelo verticalmente. Para que la cabeza llegue a apoyar en la cara superior, eleva la pieza al menos 4 cm y deja libre la salida inferior del orificio.

**Evaluación.** Registra los intentos de alineación, la profundidad alcanzada por el vástago y el desplazamiento de la pieza durante la inserción. La holgura lateral nominal es de 4 mm por lado al centrar el pasador.

### Regeneración

Abre PowerShell en la carpeta `humanoide-173/isaac` que contiene los generadores e indica la ruta de tu instalación de Isaac Sim. Los dos primeros ejemplos generan los ocho escenarios o una selección. Los dos últimos renderizan todas las vistas con RTX o actualizan solo los diez primeros planos.

```powershell
$pythonIsaac = 'C:\ruta\a\isaac-sim-6.2.0\python.bat'
& $pythonIsaac .\escenarios_servicio.py
& $pythonIsaac .\escenarios_servicio.py mercado ascensor
& $pythonIsaac .\vistas_escenarios.py --servicio
& $pythonIsaac .\vistas_escenarios.py --primeros-planos
```
