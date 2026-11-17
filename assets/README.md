# Modelos y procedencia

`vendor/` contiene una copia de los modelos publicados por NVIDIA para Isaac Sim 6.2:

- `Isaac/Robots/UniversalRobots/ur10e/ur10e.usd`.
- `Isaac/Robots/ShadowRobot/ShadowHand/shadow_hand.usd`.

`manifest.json` registra URL, tamaño y SHA-256 de cada dependencia descargada. Las sumas son
de la copia local; no son firmas publicadas por los fabricantes. Los módulos integrados
`OmniPBR.mdl` / `OmniGlass.mdl` se resuelven mediante Isaac Sim.

`robots/ur10e_shadow/robot.usda` referencia los originales sin modificar sus mallas. Desactiva
el anclaje al mundo de la mano y el terminal vacío del brazo, conserva el único anclaje del
UR10e y añade una junta fija entre la brida y `robot0_hand_mount`. La pose de montaje se
deriva de las transformaciones USD originales. Se asume alineación de ejes y separador de
longitud cero; es una composición para simulación, no el CAD validado de un adaptador real.

`joint_manifest.json` conserva los límites originales, convertidos de grados USD a radianes.
No se ha cambiado la geometría para fingir una mano integrada de fábrica.

La composicion corregida agrega `DriveAPI:angular` a las 24 articulaciones de la mano: el USD
original no contenia esos motores. Los cuatro esquemas de tendones de rigidez cero se retiran
en esta composicion; el control de las falanges medias/distales se acopla por software. La
geometria y los archivos originales de `vendor/` permanecen intactos. `tools/validate_assets.py`
exige los 24 motores para detectar una regresion antes de entrenar.

Catálogo: https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots_manipulator.html

Licencias: el catálogo identifica estos robots como BSD-3 y enlaza sus repositorios de origen:
https://github.com/ros-industrial/universal_robot y https://github.com/shadow-robot/sr_common.
Se deben conservar los avisos de los autores y revisar las condiciones del contenido de NVIDIA
antes de redistribuir los assets. Isaac Lab conserva su LICENSE y avisos en `..\IsaacLab`
respecto a la raíz del proyecto.

## Escenarios propios

[procedural/](procedural/README.md) incluye ocho USD portables con objetos, obstáculos y distractores creados por este proyecto. [previews/](previews/environments.png) contiene capturas reales y ambas vistas RGB-D, sin entrenamiento.
