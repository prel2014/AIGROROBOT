# Currículo de habilidades de Xavier

Ocho habilidades y una misión combinada disponen de un entrenador PPO manual para Isaac Sim 6.2. La matriz contiene 48 combinaciones de tarea, entorno y nivel. Cada corrida entrena una política compartida para hasta 16 Xavier en estaciones independientes. El cruce con tráfico y su política validada conservan su propio punto de entrada.

| Tarea en la CLI | Entornos | Progresión y medida de éxito |
|---|---|---|
| `caminar` | Vivienda, calle, campus | Atravesar el hueco de la puerta, rodear la silla por el punto indicado y detenerse ante la mesa o el bordillo. Aumenta la distancia. La parada exige un segundo erguido con apoyo bilateral. |
| `levantarse` | Vivienda, campus | Los niveles 0 y 1 parten de un asiento alto de 81 o 79 cm, adaptado a la flexión de la cadera del modelo. El nivel 2 parte tumbado boca arriba. Debe incorporarse, descargar el asiento, avanzar con los pies y parar. |
| `transportar` | Vivienda, campus | Tomar una caja de 250, 370 o 490 g con las dos manos, elevarla más de 7,5 cm, pasar por la puerta con ella y depositarla en la mesa. Se mide el contacto real con el destino antes de soltar y la estabilidad posterior. |
| `imitar` | Vivienda, campus | La cámara observa un demostrador humanoide virtual de frente. Varían el brazo inicial y la amplitud. Las poses incluyen levantar o alternar los brazos, flexionarse y, en el nivel 2, desplazarse lateralmente. Cada pose tiene una ventana temporal y debe mantenerse 0,4 s. |
| `tijeras` | Vivienda, campus | Tomar unas tijeras escolares cerradas por las anillas, levantarlas y soltarlas dentro de la bandeja. Aumenta la variación de orientación inicial. El contacto de la mano con las hojas invalida el intento. |
| `correr` | Campus, calle | Acelerar, correr en recta y frenar. Los niveles 1 y 2 añaden un punto de curva cada vez más desplazado. La carrera exige alternancia de apoyos, dos fases de vuelo observadas y más de 0,60 m/s horizontales al alcanzar la marca. |
| `encestar` | Campus | Recoger una pelota de 22 cm de diámetro de un soporte, elevarla en contacto con la mano y lanzarla. El aro pasa de 1,30 a 1,80 m de altura y se aleja con el nivel. El árbitro interpola el cruce descendente del centro de la pelota y descuenta su radio del hueco libre. Xavier debe acabar estable. |
| `pirueta` | Campus | Media vuelta sobre los pies, cuarto de vuelta en un salto y, como objetivo experimental final, una voltereta de 360° alrededor del eje transversal. En los saltos se integra la rotación durante un solo vuelo. Se comprueban orientación, posición y doble apoyo durante un segundo al aterrizar. |
| `mision` | Vivienda | Levantarse del asiento, dar el primer paso, coger la caja, cruzar la puerta y colocarla en la mesa. Las fases son consecutivas dentro de un único episodio. La carga y la distancia crecen con el nivel. |

## Entradas y control

Cada robot recibe su cámara RGB de 256 × 256, aceleración específica y velocidad angular de la pelvis, vertical local, cuatro cargas por pie, posiciones y velocidades de las articulaciones controladas, acción anterior, fase y dirección local de la meta. La pose objetivo del demostrador sólo interviene en la animación y la puntuación. El entrenador usa la CNN y la actualización PPO del proyecto. Acciona 37 ejes del USD completo, incluidos los de los dedos. Los objetivos articulares incrementales respetan los rangos y límites de velocidad de cada accionamiento. La física usa 120 Hz y el control, 20 Hz.

El agarre, la suelta y el vuelo se miden en física. Los objetos son cuerpos libres. La práctica de imitación emplea un avatar procedural. La carrera, el enceste y la voltereta tienen objetivos de investigación cuya viabilidad dinámica está pendiente de ensayar. Los pesos de marcha y cruce existentes usan un contrato de acciones diferente.

## Preparación sin entrenamiento

Desde la carpeta del proyecto local `Hacking`, en PowerShell:

```powershell
$habilidades = ".\humanoide-173\isaac"
$isaac = ".\isaac-sim-6.2.0\python.bat"
python "$habilidades\entrenar_habilidades.py" plan
python "$habilidades\entrenar_habilidades.py" comprobar
& $isaac "$habilidades\revisar_estaciones.py"
& "$habilidades\curriculo.ps1"
```

En una copia de AIGROROBOT, usa `.\docs\humanoide-173\isaac` como carpeta de habilidades y apunta `$isaac` a la instalación local. El verificador USD compone los escenarios en memoria con OpenUSD. El lanzador por lotes muestra la matriz de corridas por defecto.

## Inicio manual posterior

Ejemplo para empezar con el transporte de la caja:

```powershell
& $isaac "$habilidades\entrenar_habilidades.py" entrenar --tarea transportar --entorno vivienda --nivel 0 --robots 16 --semilla 17 --actualizaciones 300
```

Cada perfil escribe en `runs/habilidades/<tarea>/<entorno>/nivel<N>/semilla<S>/`. Se guarda la política y el optimizador después de cada actualización. El registro JSONL incluye el perfil, las fases alcanzadas, el tiempo y el éxito individual al terminar cada lote de episodios. Para continuar el mismo perfil se usa `--reanudar`. Para transferir pesos a otro nivel o entorno se usa `--inicial RUTA`.

Para evaluar un checkpoint con una semilla nueva y acciones deterministas:

```powershell
& $isaac "$habilidades\entrenar_habilidades.py" evaluar --tarea transportar --entorno vivienda --nivel 0 --semilla 937 --episodios 5 --inicial .\runs\habilidades\transportar\vivienda\nivel0\semilla17\politica.pt
```

El lote completo se inicia únicamente al indicar `-Iniciar`. Recorre los tres niveles por tarea y entorno, transfiriendo los pesos del nivel anterior. El orden del lote es administrativo. Para juzgar dominio, revisar el porcentaje de éxito en semillas reservadas, la fase donde termina cada fallo y el vídeo o la reproducción visible con `--visible`.

```powershell
& "$habilidades\curriculo.ps1" -PythonIsaac $isaac -Iniciar
```

Las habilidades nuevas cuentan con verificación estática del USD y pruebas de sus reglas. La validación física y el entrenamiento quedan para la ejecución que autorice el operador.
