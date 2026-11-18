# Tareas agrícolas bimanuales

El flujo actual trabaja con fruta suelta. La vegetación aporta contexto y oclusiones. [Comandos](../README.md#teacher-y-captura-agrícolas) · [Arquitectura](architecture.md) · [Flujos de referencia](referencias-aprendizaje.md)

| Escena | Orden abreviada | Fases |
| --- | --- | --- |
| `citrus` | Recoge la naranja más cercana y ponla en la bandeja izquierda. | Agarrar → elevar 15 cm → depositar y estabilizar. |
| `banana` | Recoge el plátano con la derecha, pásalo a la izquierda y deposítalo. | Agarrar → elevar 10 cm → confirmar contacto de ambas manos → soltar con la derecha → depositar. |
| `vineyard` | Levanta 15 cm el racimo más cercano y sostenlo. Evita la espaldera. | Agarrar → elevar → sostener 0,5 s. |
| `berries` | Recoge el arándano más cercano con pulgar e índice sin desplazar los demás. | Confirmar pinza opuesta → elevar 3 cm → sostener 0,5 s. |

Cada orden añade «Si ves obstáculos, esquívalos». El simulador comprueba contactos de yemas, penaliza caídas y poses inválidas y reinicia solo las réplicas terminadas. Hay 11 tipos de obstáculos de mesa o pasillo. Cada réplica intenta añadir o retirar uno cada 1,5 a 4,5 s si la zona está libre de brazos y fruta. La luz cambia durante el episodio.

La altura se mide desde el apoyo hasta el punto más bajo de la malla de colisión de la fruta. El agarre exige fuerzas opuestas y limita el deslizamiento durante el levantamiento. `--nominal-scene` desactiva los cambios de luz y obstáculos durante el episodio. `--physics-variation 0.5` activa variación de masa, fricción, ruido y tiempo de respuesta. Al reanudar, el perfil físico queda registrado con cada corrida.

El siguiente comando mide agarre, elevación y liberación en los cuatro cultivos antes de entrenar. Ejecútalo desde la raíz del repositorio.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& ..\isaac-sim-6.2.0\python.bat tools/agriculture_grasp.py --viz none --raised-support --candidates 4 --output "$env:TEMP\agriculture_grasp.json"
```

Para calibrar el espacio de cierre de la mano, añade `--raised-support`. La fruta queda suelta sobre apoyos físicos de 3,5 cm. El ensayo mide la elevación adicional desde su reposo y comprueba la liberación al volver al apoyo.

El currículo recibe ese informe y selecciona un ensayo que haya sostenido y soltado la fruta. En la primera fase usa una sola clase de fruta, dos mandos para cierre y elevación, y conserva el control de los 60 ejes físicos. Se puede empezar desde `closed` para aislar la elevación o desde `pregrasp` para validar el cierre previo.

```powershell
.\scripts\run.ps1 agriculture_teacher --headless --num_envs 4 --grasp-calibration "$env:TEMP\agriculture_grasp.json" --grasp-crop berries --grasp-start pregrasp --validate
```

`--evaluate-episodes 64` mide la política determinista por cultivo sin actualizar pesos. Añade `--resume` con el checkpoint que quieras comparar. Usa la misma calibración y repite con semillas reservadas.

## Formato de demostraciones

La captura guarda la orden global y la subtarea en español según la fase de cada muestra. El bloque textual las combina antes de Multilingual E5. La acción registrada es el objetivo articular aplicado por el controlador. [Campos y unidades](dataset.md#agricultura-bimanual)

## Hallazgos de la revisión

| Área | Hallazgo y decisión |
| --- | --- |
| Exploración y recompensa | El arranque alrededor de la postura del cubo con 60 mandos absolutos dejó entre 95 % y 100 % de muestras fuera de la zona de agarre en la última ventana de la corrida. Se priorizan yemas opuestas y se añade un currículo cerca de una postura físicamente comprobada. |
| Física | En el ensayo previo, el arándano completó 3 de 4 intentos con apoyo elevado. Naranja, plátano y uva perdieron la pinza durante la elevación. Medimos las fuerzas al inicio de la subida y la holgura respecto al apoyo. |
| Medición | El logger promediaba porcentajes de pasos con distinto número de episodios terminados. Ahora recibe un resultado binario por cada episodio terminado. Los picos de altura se registran aparte del éxito. |
| Datos y estudiante | Se usan contratos distintos para una mano y para dos brazos. El estudiante bimanual supervisa 60 objetivos físicos, con máscaras por subtarea y memoria reiniciada al comienzo de cada episodio. |
| Ajustes iniciales | Isaac Sim 6.2 e Isaac Lab 3.0.0 están fijados. Física a 120 Hz, control a 30 Hz. PPO usa 32 pasos por réplica y actualización, cuatro épocas internas, ocho minilotes y tasa inicial de `1e-4`. Se medirán primero cambios de escenario y acción antes de barrer parámetros de PPO. |

El ensayo guiado desde preagarre completó 64 de 64 levantamientos de arándano. La política inicial determinista completó 0 de 64. Los ensayos cartesianos de naranja, plátano y uva siguen sin sostén válido. [Capacidad medida](recursos-vram.md)

## Tareas en orden de ejecución

1. **Escena y contacto.** Comprobar masa e inercia de cada fruta, malla de colisión, fricción, alcance y autochoques. Registrar presión y deslizamiento desde el cierre hasta los primeros milímetros de subida. Repetir sostén y liberación con apoyo y mesa.
2. **Currículo del teacher.** Medir la política inicial con semillas fijas. Empezar en el arándano con el cierre o preagarre calibrado. Entrenar después desde poses más lejanas. Añadir los otros cultivos cuando pasen el ensayo físico y comparar con el teacher de 60 acciones.
3. **Secuencias.** Medir el traspaso y el depósito como habilidades independientes. Encadenarlas con la selección de fruta, la aproximación y la recuperación ante caída o pérdida de contacto.
4. **Variación del escenario.** Introducir pose, dimensiones y masa de fruta por separado. Después añadir fricción, ruido, demora, luz, temperatura de color y obstáculos móviles. Reservar combinaciones para evaluación. [Perfiles de escena](../assets/agriculture/README.md)
5. **Datos.** Capturar episodios completos con marcas de tiempo, dos RGB-D, ángulos, orden, subtarea, límites y resultado. Dividir por episodio y semilla. Medir cobertura por clase, fase y tipo de orden antes de entrenar al estudiante.
6. **Estudiante.** Verificar inferencia, máscaras, normalización, límites y memoria. Variar la orden dentro de la misma escena para comprobar que el modelo depende del texto. Probar ventanas cortas de acciones y reobservación. Comparar memoria frente a historial y medir paráfrasis en español.
7. **Evaluación.** Medir éxito por cultivo y por fase, tiempo, caída, choque y deslizamiento en episodios completos con semillas reservadas. Guardar MP4 individual por robot y época para inspeccionar fallos.
8. **Cosecha.** Incorporar desprendimiento, ramas flexibles y medición de daño si la tarea pasa de fruta suelta a recoger fruta adherida a una planta.
