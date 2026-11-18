# Arquitectura

## Agricultura bimanual

| Modelo | Entradas | Salidas |
| --- | --- | --- |
| Teacher PPO | 320 valores de articulaciones, contactos, fruta, fase y obstáculos. | 60 mandos normalizados, 30 por brazo. El crítico estima un valor por réplica. |
| VLA | Tokens CLIP ViT-B/32 de las dos vistas RGB, profundidad, texto, `q[60] + dq[60]` y memoria de 256 valores. | Ocho pasos de 60 ángulos absolutos en radianes y la siguiente memoria de 256 valores. |

El teacher usa tres bloques Transformer. El VLA usa seis, con ancho 256 y ocho cabezas de atención. El orden de ojos y brazos es izquierda, derecha. La física corre a 120 Hz y el control a 30 Hz. [Tareas y comandos](plan-agricola.md)

## Elevación de un brazo

El teacher Transformer recibe 95 valores de estado y produce 26 mandos con sinergias de dedos. El VLA v5 usa dos RGB, texto, 30 posiciones, 30 velocidades y memoria. Devuelve ocho pasos de 30 ángulos y una memoria de 256 valores. La profundidad se graba para inspección.

El éxito exige levantar el objeto 15 cm y sostenerlo 0,5 s con contactos opuestos. Los checkpoints guardan su contrato de escena y control. Reanudar recupera pesos y optimizador. Las ejecuciones bimanuales requieren su propio checkpoint. [Flujo de elevación](mvp.md)
