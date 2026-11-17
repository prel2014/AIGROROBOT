# MVP teacher → estudiante visual

La primera entrega resuelve elevación de un cubo de 5 cm con UR10e + Shadow en simulación.
Parte del preagarre calibrado, exige una elevación medida mínima de 15 cm y sostén durante 0,5 s,
con sobresalto máximo de 2,5 cm. El protocolo de aceptación está en `configs/mvp.json`:
200 episodios **por semilla**, semillas 1001/1002/1003, al menos 90 % de éxito en cada una.
Son metas de aceptación, no resultados alcanzados. El movimiento amplio desde HOME y el
despliegue en hardware no forman parte de este criterio.

## Entorno y datos preservados

Isaac Sim 6.2, Python 3.12 y la revisión de Isaac Lab declarada en `runtime.lock.json`.
`runtime-constraints.txt` fija las versiones principales comprobadas; no es un lock universal
de todas las extensiones binarias de Isaac. Cada ejecución registra las versiones efectivas,
hash de fuentes, assets y checkpoint. Los originales, checkpoints anteriores y `.partial` se conservan.

```powershell
.\scripts\mvp.ps1 -Stage doctor -RunName diagnostico
```

`doctor --stage student --dataset ...` comprueba además los episodios disponibles. No descarga
modelos, no instala paquetes y no arranca el simulador. Las dependencias visuales se preparan
con `configs/vla-requirements.txt`, usando las restricciones del runtime.

## 1. Teacher con currículo reanudable

```powershell
.\scripts\mvp.ps1 -Stage teacher -Robots 32 -Iterations 3000 `
  -StartLiftCm 15 -ExplorationStdScale 0.5 -Headless -RunName teacher01
```

En modo ventana, cada réplica tiene una cámara RGB-D simulada con carcasa y soporte.
El panel **Cámara RGB-D del teacher** muestra la imagen, la profundidad en metros y
la orden fija de la tarea. Actor y crítico PPO usan Transformers de estado, con tres bloques,
ancho 128 y cuatro cabezas. Reciben las observaciones de estado; las imágenes no entran a PPO.

Los checkpoints nuevos guardan actor, crítico, optimizador, nivel/ventana del currículo,
contadores y estados aleatorios. Reanudar crea episodios nuevos; no restaura contactos ni
trayectorias a mitad de episodio. El número de iteraciones solicitado son actualizaciones
adicionales. No se modifican algoritmos ni fuentes de Isaac Lab/RSL-RL.

En corridas nuevas se inicializa el actor por imitación de trayectorias físicas de calibración
para las cinco alturas, antes de PPO. Se seleccionan episodios exitosos y se separan réplicas
para entrenamiento y validación; los periodos inmóviles de espera se excluyen de la imitación.
El informe `bootstrap.json` registra los ejemplos y la pérdida del actor. Los éxitos scriptados
no incrementan los contadores de éxitos de PPO ni habilitan la captura certificada del estudiante.
PPO mantiene la aleatorización del objeto y debe demostrar progreso por sí mismo.
`-StartLiftCm 15` inicia una corrida nueva directamente en la meta final, después de calibrar
las cinco alturas. `-ExplorationStdScale 0.5` reduce la desviación de exploración antes de
validar el actor imitado, respetando el mínimo de 0.01. Esta escala solo se aplica a corridas nuevas con
bootstrap; reanudar conserva la distribución y el currículo guardados.
Este arranque aprovecha la calibración de las cinco alturas y concentra PPO en la meta final.
Entrenar únicamente en la primera altura mejoró esa etapa, pero degradó el agarre a 15 cm en
la comparación física pareada; sus informes están en `logs/launch/teacher_retention_175_*.json`.
Esas comparaciones pequeñas no reemplazan la evaluación de aceptación de tres semillas.
`-BootstrapUpdates 0` desactiva esta inicialización para una comparación explícita; `-Resume`
recupera el estado guardado y no la repite. Los valores por defecto están en `configs/mvp.json`.

```powershell
.\scripts\mvp.ps1 -Stage teacher -Teacher 'RUTA/model_300.pt' -Robots 16 `
  -Resume -Iterations 500 -Headless -RunName teacher02
```

`-Resume` sin `-Teacher` busca el checkpoint guardado más recientemente. El origen puede ser un
archivo o una carpeta de corrida. Usa otro `RunName`; quitar `-Headless` abre la misma tarea con
cámaras y panel. Se guarda cada 25 actualizaciones y al finalizar; una interrupción retoma el
último checkpoint completo. No recupera pasos posteriores que todavía no se habían guardado.

PPO recoge al menos 2048 transiciones por actualización y utiliza cuatro minibatches y cuatro
pasadas. Las primeras 32 actualizaciones entrenan únicamente al crítico. Después, el actor
comienza con tasa `1e-6` y el crítico conserva `1e-4`, con entropía `0.003`. Una guardia mide
la divergencia KL en 64 observaciones del rollout: si supera `0.02`, restaura el actor y sus
momentos del optimizador y reduce su tasa a la mitad, hasta `1e-7`. El crítico conserva su
actualización. Reanudar recupera ambas tasas, los momentos y el progreso de esta fase inicial.
La desviación de
exploración admite `0.01..0.35`; después de la imitación comienza en `0.02` para brazo/muñeca
y `0.04` para dedos. Las estadísticas del actor se fijan con TRAIN durante la inicialización;
el crítico sigue actualizando las suyas durante PPO. El filtro físico de acciones
permanece en `0.05` para conservar el contrato de los checkpoints existentes. Estos ajustes
requieren comprobar mejora en una nueva corrida; el objetivo sigue siendo 15 cm.

Los checkpoints MLP anteriores son incompatibles con el teacher Transformer. Se conservan en
disco, pero no se migran ni se reinician silenciosamente. La primera corrida del Transformer
debe empezar sin `-Resume` ni `-Teacher`, con otro nombre. La tarea y el currículo de alturas
1,5/3/6/10/15 cm permanecen iguales.
El contrato de arquitectura actual del teacher es v3; los anteriores que observaban acciones
crudas o actualizaban continuamente las estadísticas del actor requieren una corrida nueva.
El formato de checkpoint es v4: exige los dos grupos del optimizador y el estado de la guardia;
los checkpoints de entrenamiento anteriores requieren una corrida nueva.
El contrato de tarea v3 descarta los checkpoints del criterio anterior, que podía terminar
por debajo de la altura solicitada. Los contadores de esas corridas no acreditan elevaciones de 15 cm.

## 2. Evaluación final y puerta de aceptación

```powershell
.\scripts\mvp.ps1 -Stage evaluate-teacher -Teacher 'RUTA/model_799.pt' `
  -Robots 16 -Headless -RunName teacher_eval01
```

Siempre se desactiva el currículo y se exige la etapa de 15 cm. Las cuotas de episodios por
réplica evitan sobrecontar robots que fallan rápido. Se guardan motivos de terminación,
altura máxima, tasa por semilla, intervalo Wilson 95 %, latencias p50/p95 y memoria GPU.

Una caída termina el episodio como `dropped` y aplica una penalización terminal de 10 unidades,
además de los términos de recompensa habituales. Una orden fuera de rango o no finita retiene
el último objetivo aplicado y termina como `invalid_action` con recompensa −10.
Un estado medido no finito, una orientación inválida o una junta que exceda sus límites
físicos por más de 0,05 rad termina como `invalid_pose`, también con −10.

La latencia de inferencia incluye preparación y conversión; la simulación se mide por separado.
La memoria máxima de PyTorch no incluye todas las asignaciones del simulador; también se
registra una muestra final de uso del dispositivo, que incluye otros procesos y no es un pico.
No se construye un optimizador para inferencia.

`--require-pass` devuelve error si falta presupuesto o alguna semilla no alcanza el umbral.
Una evaluación fallida conserva los informes. Para una prueba de integración pequeña, usar
el comando de bajo nivel, que informa resultados sin exigir aprobación:

```powershell
.\scripts\run.ps1 evaluate --policy teacher --checkpoint 'RUTA/model_799.pt' `
  --episodes 8 --seeds 1001 --num_envs 4 --output logs/mvp/smoke_eval --headless
```

Una prueba pequeña nunca habilita la captura certificada del lanzador MVP. No usar las semillas
de evaluación para generar datos ni seleccionar hiperparámetros repetidamente; reservar nuevas
semillas de prueba si se desarrolla contra estos resultados.

## 3. Demostraciones y particiones inmutables

```powershell
.\scripts\mvp.ps1 -Stage collect -Teacher 'RUTA/model_799.pt' `
  -TeacherEvaluation logs/mvp/teacher_eval01/evaluation.json -Episodes 200 `
  -Robots 4 -Headless -RunName demos01
.\scripts\mvp.ps1 -Stage split -Dataset datasets/demos/demos01 -RunName split01
```

La puerta comprueba protocolo, éxito y SHA256 del teacher. `collect_demos` sigue disponible
directamente para investigación sin esa puerta; no presenta sus datos como certificados.
Cada captura guarda RGB-D, q/dq, acción teacher, comando antes del filtro, target aplicado,
calibración y procedencia. Solo se publican episodios terminados; no se promueven `.partial`.
La cámara RGB-D fija está en `(0.75, 0.45, 0.95)` m dentro de cada réplica y apunta al área de
agarre con unos 24° hacia abajo. Cinco sensores virtuales miden el contacto de las yemas con el cubo.
Las capturas nuevas guardan el contrato de cámara; no se mezclan vistas
silenciosamente. `play_vla` conserva la cámara antigua para checkpoints sin ese contrato;
el protocolo MVP exige datos y estudiantes de la vista nueva.

`dataset audit` verifica dimensiones, valores finitos, orden temporal, terminación y consistencia
de comandos. `dataset split` necesita al menos tres demostraciones exitosas y crea train/validation/test
por **episodio**, con hashes. Tres son el mínimo estructural, no una cantidad suficiente para aprender.
El conjunto test no entra en estadísticas, entrenamiento ni selección de checkpoint. Los hashes
impiden usar archivos modificados o copiados en varias particiones. Estos splits no prueban
generalización a objetos o colecciones nuevas.

## 4. Entrenar y evaluar el Transformer

```powershell
.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split01.json `
  -Epochs 20 -BatchSize 8 -Headless -RunName student01
.\scripts\mvp.ps1 -Stage train -Resume -Student checkpoints/vla/student01/last.pt `
  -Epochs 10 -Headless -RunName student02
.\scripts\mvp.ps1 -Stage evaluate-student -Student checkpoints/vla/student01/best.pt `
  -Robots 4 -Headless -RunName student_eval01
```

CLIP ViT-B/32 congelado conserva 49 parches visuales más CLS y el embedding de texto.
El Transformer concatena estos tokens, q/dq normalizado y la memoria anterior, tras
proyectarlos al ancho 256. Usa seis bloques y ocho cabezas, con máscara que separa contexto,
consultas de acciones y escritura de memoria. Devuelve ocho objetivos de 30 ángulos en radianes
y el embedding de memoria actualizado. Durante el entrenamiento, los gradientes recorren
secuencias de 16 observaciones contiguas que comienzan con memoria cero. En simulación,
cada réplica reutiliza su memoria en el paso siguiente y la reinicia al terminar el episodio
o cambiar de instrucción. Por defecto se replantea tras cada acción.

`best.pt` se selecciona por pérdida de validación; demostrar destreza requiere rollouts físicos.
La salida conserva 30 juntas independientes, pero el teacher de 26 mandos proporciona cuatro
sinergias: estos datos no acreditan aprendizaje de movimientos independientes de esas falanges.
Los checkpoints VLA v4 guardan también el contrato de tokens y la arquitectura completa.
Para control en lazo cerrado, `ChunkController` conserva la memoria por réplica.

`last.pt` guarda Transformer, AdamW, épocas completas, mejor pérdida, historial y estados
aleatorios, incluido el orden del DataLoader. `-Epochs` añade épocas. `-Resume` recupera el
manifiesto original y verifica sus hashes, conserva normalización, horizonte, longitud de
secuencia, batch y tasa del optimizador. También preserva `best.pt` aunque ninguna época nueva
lo mejore. Sin `-Student`, selecciona el `last.pt` más reciente. Puedes alternar `-Headless`
y la ventana de gráfica de pérdidas; el entrenamiento del estudiante es por imitación offline.
La memoria de una trayectoria física no se guarda como memoria de entrenamiento: cada nueva
secuencia empieza con cero. No se retoma una época interrumpida a mitad.

Los checkpoints VLA v3 requieren una corrida v4 nueva; las demostraciones compatibles se pueden
reutilizar. El warm-start de DAgger admite únicamente estudiantes v4 compatibles y crea un
optimizador nuevo. La reanudación completa necesita los nuevos `last.pt` y sus carpetas locales
`backbone/`, `processor/` y `dataset.json`.

## 5. DAgger y recuperación

```powershell
.\scripts\mvp.ps1 -Stage corrections -Teacher 'RUTA/model_799.pt' `
  -Student checkpoints/vla/student01/best.pt -Robots 4 -Episodes 100 -Headless -RunName corrections01
.\scripts\mvp.ps1 -Stage add-corrections -Manifest datasets/manifests/split01.json `
  -Dataset datasets/corrections/corrections01 -RunName split_dagger01
.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split_dagger01.json `
  -Student checkpoints/vla/student01/best.pt -RunName student_dagger01
```

El estudiante visita estados y el teacher calcula etiquetas sobre esos mismos estados antes
de actuar. Por defecto se ejecuta el teacher con probabilidad 0,2 y el estudiante con 0,8;
`collect_corrections --beta 0` ejecuta únicamente al estudiante. Se registra por separado
la acción ejecutada y la etiqueta teacher. Se conservan fallos y recuperaciones.

Las correcciones supervisan solo **un paso**: concatenar acciones futuras del teacher sobre
una trayectoria del estudiante produciría etiquetas de chunks incorrectas. Las máscaras
excluyen esos futuros. Las correcciones se añaden únicamente a train; validation/test permanecen
idénticos. El warm-start conserva pesos, crea un optimizador nuevo y recalcula estadísticas
solo con train. Es DAgger por rondas; no actualiza pesos mientras se escribe un HDF5.

## Extensión

La ruta multitarea `memory_rl` conserva levantar y empujar en ±X como experimental.
Extender el MVP a esas órdenes
requiere teachers/datos por tarea y evaluación separada; una instrucción fija no demuestra
comprensión de lenguaje libre. Cambiar a ResNet, ajustar CLIP o añadir contactos/tacto requiere
una nueva comparación controlada y un contrato compatible; no se presupone que mejore resultados.
