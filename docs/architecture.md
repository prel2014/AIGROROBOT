# Arquitectura

## Entradas y salidas

El teacher PPO recibe 95 valores privilegiados: q[30], dq[30], objeto respecto a palma[3], posición local[3], objeto-meta[3] y objetivos filtrados[26]. Actor y crítico son Transformers de estado. Devuelve 26 mandos normalizados con cuatro sinergias de dedos.

El estudiante VLA v5 usa CLIP ViT-B/32 congelado sobre dos RGB sincronizados de 224 × 224 y la orden de texto. Cada cámara produce CLS + 49 parches de 512 valores. La fusión concatena 100 tokens visuales, un token de texto, uno de q/dq[60] y uno de memoria[256], proyectados a ancho 256: 103 tokens de contexto. Los índices de posición distinguen ambas vistas.

Seis bloques propios de atención, ocho cabezas y FFN de 1024 procesan el contexto y ocho consultas de acción más una de escritura de memoria. Las salidas son ocho objetivos de 30 ángulos en radianes y un embedding de 256 valores. La memoria se devuelve al siguiente paso; cada réplica la reinicia al terminar o cambiar la orden. La profundidad se conserva para inspección, sin entrar actualmente al Transformer.

CLIP está congelado; el Transformer tiene 5.251.870 parámetros entrenables. Las características de entrada ocupan 208.112 bytes en FP32 por observación, sin imágenes ni activaciones. El entrenamiento usa secuencias contiguas y máscaras; q/dq se normalizan solo con train. Se supervisa `joint_command` antes del filtro de 0,05, aplicado una vez en el simulador.

## Física y éxito

La composición mantiene 30 articulaciones con límites, motores PD y física a 120 Hz; control y cámaras a 30 Hz. El teacher acopla cuatro pares de falanges por software; el VLA predice las 30 articulaciones. Las demostraciones siguen limitadas por la destreza del teacher.

El criterio de recompensa v4 exige pulgar y al menos otros dos dedos con fuerza normal medida >= 0,05 N. Un par debe presentar normales opuestas, puntos en lados opuestos y separación mínima de 1,5 cm. Es una condición necesaria de agarre, no una prueba de cierre de fuerzas ni garantía contra deslizamiento. La proximidad solo aporta señal de acercamiento.

Para éxito final: altura medida >= 15 cm, sobreelevación <= 2,5 cm, error XY < 6 cm, velocidad < 0,15 m/s y separación inferior del objeto > 5 mm durante 0,5 s. El soporte vertical se calcula según cubo, esfera, cilindro o cápsula. Caída, escape, orden inválida, pose inválida o contacto prohibido terminan en fallo. Se miden contactos prohibidos en yemas y objeto; otros enlaces tienen colisión física, sin monitor dedicado de tarea.

Contact offset de 1 mm y rest offset cero reducen agarres visualmente separados. Se usan fricciones moderadas combinadas por mínimo. Los colliders distales originales son cápsulas, una aproximación a las mallas visibles. Estas modificaciones requieren volver a evaluar el agarre dinámico; las capturas estáticas no lo validan.

## Situaciones y subtareas

`tasks/manipulation/scenarios.py` define ocho escenas procedurales, objetos con distinta forma/masa/color, dos distractores y obstáculos cinemáticos inmóviles. Los cuerpos cinemáticos permiten filtrar contactos en GPU; los colliders estáticos por sí solos no permiten ese filtrado. Se reinician también los distractores.

`planner.py` divide transporte en acercar, alinear, agarrar, elevar, pasar por puntos libres, bajar, soltar y verificar. Una máquina de progreso exige condiciones medidas y detiene el plan ante contacto. El planificador usa geometría conocida del teacher e infla cajas en XY 8 cm, incluyendo distractores. No detecta obstáculos visualmente ni comprueba la cinemática/volumen completo del brazo. El catálogo de planes aún no conduce las acciones del rollout.

El flujo principal teacher → HDF5 → VLA conserva la tarea de elevar 15 cm, con selección explícita de escenario y contratos de física/cámara. El catálogo auxiliar de manipulación define nueve metas de elevación, empuje y transporte. Las órdenes de colocación, selección y evasión necesitan demostraciones y evaluaciones específicas; cambiar la frase de una elevación no entrena esas capacidades.

## Versiones y reanudación

Los checkpoints del teacher guardan arquitectura, contrato físico, optimizador, currículo y RNG. Los VLA v5 guardan dos calibraciones de cámara, geometría, criterio de contacto, arquitectura, pesos, estadísticas y optimizador. Los contratos incompatibles se rechazan; las nuevas vistas/criterios requieren nuevas demostraciones y entrenamiento.

La captura de escenarios solo ejecuta ocho pasos manteniendo postura neutra, sin cargar política ni optimizador. Los USD propios y capturas se publican; los modelos originales del robot, pesos, datasets y logs permanecen fuera del repo.
