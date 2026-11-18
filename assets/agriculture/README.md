# Assets y escenas agrícolas

El torso monta dos módulos UR10e con manos articuladas y dos cámaras RGB-D. Los USD usan metros y eje Z hacia arriba. [Galería de escenarios](previews/escenarios.png) · [Vistas RGB-D](previews/ojos_rgbd.png) · [Obstáculos](previews/citrus/obstaculos_campo.png)

## Inspección

Desde la raíz del repositorio. Cambia `citrus` por `banana`, `vineyard` o `berries` para elegir otro cultivo.

```powershell
.\scripts\run.ps1 agriculture_preview --scenario citrus --viz kit --live --output logs/agriculture_preview
.\scripts\run.ps1 agriculture_preview --scenario citrus --randomize --seed 42 --episode 10 --headless --output logs/agriculture_preview
.\scripts\run.ps1 agriculture_batch --headless --num_envs 64 --rgbd --steps 8
```

La vista previa guarda RGB de 640 × 480 por ojo, profundidad en milímetros y calibración en `scene.json`. El lote usa dos ojos de 224 × 224 por réplica. Para comprobar los cambios de luz y obstáculos durante cinco segundos simulados, ejecuta `agriculture_batch` con `--steps 150` y omite `--rgbd`. [Entrenamiento y captura](../../README.md#teacher-y-captura-agrícolas)

## Procedencia y licencias

| Asset local | Fuente y licencia |
| --- | --- |
| [Naranja](models/orange/visual.usda) | [Innerscene](https://www.innerscene.com/tools/library/3d-parts/orange-9fbf79f3), CC0 1.0. GLB convertido a USD. |
| [Naranjo de la escena](props/orange_tree.usdc) | Ramas, hojas y pedúnculos generados para el proyecto. Frutos del modelo de Innerscene. |
| [Naranjo de referencia](models/orange_tree/visual.usda) | [Poly by Google](https://poly.pizza/m/8zS7mfHS0i7), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Conversión y ajuste a 2,4 m. |
| [Plátano](models/bananas/bananas_1k.usdc) | [Poly Haven](https://polyhaven.com/a/bananas), [CC0 1.0](https://polyhaven.com/license). USD con referencias relativas para las texturas. |
| [Platanera](models/banana_plant/visual.usda) | [Innerscene](https://www.innerscene.com/tools/library/3d-parts/banana-plant-with-a-bunch-of-bananas-905b58eb), CC0 1.0. GLB convertido a USD. |
| [Racimo de uvas](models/grapes/visual.usda) | [Innerscene](https://www.innerscene.com/tools/library/3d-parts/bunch-of-grapes-abfcc57f), CC0 1.0. GLB convertido a USD. |
| [Vid en espaldera](models/vineyard/visual.usda) | [3DAssets.dev](https://3dassets.dev/assets/vineyard-and-winery-vine-row-fruiting-f23824de), CC0 1.0. Malla y mapa convertidos a USD. La fuente declara asistencia de IA. |
| [Arándano](models/blueberry/visual.usda) | [Gourmet Blueberry, inciprocal](https://sketchfab.com/3d-models/gourmet-blueberry-62f6c6368c484dd6a85a2a162133d2e0), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). GLB de [Objaverse](https://huggingface.co/datasets/allenai/objaverse/blob/main/glbs/000-154/62f6c6368c484dd6a85a2a162133d2e0.glb) convertido a USD. |
| [Tazón de arándanos](models/blueberries/visual.usda) | [Grab3D](https://grab3d.com/models/food/blueberries/), CC0 1.0. USDZ extraído y orientado a Z arriba. |
| [Arbusto de bayas](models/berry_bush/visual.usda) | [Grab3D](https://grab3d.com/models/nature/berry-bush/), CC0 1.0. Modelo provisional de bayas mixtas, orientado a Z arriba. |

Conserva las texturas y atribuciones junto a los USD. [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
