# Modelos agrícolas para Isaac Sim

Los modelos visuales están en USD, en metros y con Z hacia arriba en los puntos de entrada de la tabla. La vista previa de Isaac Sim combina un torso de base fija, dos módulos UR10e con manos Shadow y dos ojos RGB-D. Expone 30 juntas por lado y dos cámaras de 640 × 480 con 25,6 cm de separación. Cada módulo reutiliza el modelo de mano derecha disponible.

Desde la raíz del repositorio:

```powershell
# Captura de cuatro escenas, sin política
foreach ($escena in @('citrus', 'banana', 'vineyard', 'berries')) {
    .\scripts\run.ps1 agriculture_preview --scenario $escena --headless
}

# Ver en una ventana con las dos cámaras en vivo
.\scripts\run.ps1 agriculture_preview --scenario citrus --viz kit --live
```

Cada carpeta de [capturas](previews) contiene una vista externa, RGB por ojo, profundidad en PNG de 16 bits en milímetros y `scene.json` con intrínsecos, poses de cámara y estado de los brazos. La vista externa sirve para inspección. La fruta suelta cae y descansa sobre la mesa con colisiones y masas iniciales estimadas. La vegetación conserva los frutos unidos como parte de la escena visual. La simulación de desprendimiento, agarre, daño y traspaso requiere la validación del [plan agrícola](../../docs/plan-agricola.md).

La vista previa permite cambiar la intensidad, posición y temperatura de la luz entre 3.900 y 7.800 K, junto con el tono de la mesa. Por episodio pueden aparecer entre cero y tres obstáculos físicos elegidos de 11 tipos. La mesa admite la rama caída, la maceta baja, la manguera, dos formas de roca, las tijeras de poda cerradas, el ladrillo, la pala de mano y el cultivador de tres dientes. El balde y la azada se colocan en los pasillos laterales. Las posiciones reservan espacio para fruta, bandejas, patas y base del robot.

Las variaciones se reproducen con la misma versión de catálogo, semilla, réplica y episodio. `scene.json` registra la configuración elegida, la superficie y la posición física de cada obstáculo de la vista previa.

```powershell
.\scripts\run.ps1 agriculture_preview --scenario citrus --randomize --seed 42 --episode 10 --headless
.\scripts\run.ps1 agriculture_preview --scenario citrus --randomize --seed 42 --episode 17 --headless
```

El lote paralelo inicia 64 réplicas en una escena de PhysX, con 16 de cada cultivo. Durante cada episodio modifica la intensidad y temperatura de la luz a 10 Hz de tiempo simulado. Cada réplica evalúa un evento de aparición o retiro cada 1,5 a 4,5 segundos simulados. El evento se omite si la fruta o los brazos ocupan el área del obstáculo. La secuencia de eventos depende de la semilla y de la trayectoria del robot. El reinicio indexado restaura la fruta y muestrea otra secuencia solo en las réplicas elegidas.

```powershell
# Comprobar la física y los eventos en las 64 réplicas durante 5 s simulados
.\scripts\run.ps1 agriculture_batch --headless --num_envs 64 --steps 150

# Comprobar las 128 cámaras RGB-D de 224 × 224
.\scripts\run.ps1 agriculture_batch --headless --num_envs 64 --rgbd --steps 8
```

El teacher PPO usa este lote con el estado de ambos brazos, la geometría de la escena y los contactos medidos. La adquisición carga un checkpoint bimanual y activa ambos ojos por réplica. [Comandos de entrenamiento y captura](../../README.md#teacher-y-captura-agrícolas)

| Asset | USD local | Fuente, licencia y cambios |
| --- | --- | --- |
| Naranja | [visual.usda](models/orange/visual.usda) | [Orange, Innerscene](https://www.innerscene.com/tools/library/3d-parts/orange-9fbf79f3), CC0 1.0. GLB convertido a USD. |
| Naranjo en la escena | [orange_tree.usdc](props/orange_tree.usdc) | Ramas, 2.464 hojas independientes y pedúnculos generados para el proyecto. Veinte naranjas instancian el modelo de Innerscene CC0. Las ramas tienen colisión estática y las hojas ocluyen los ojos. |
| Naranjo de referencia inicial | [visual.usda](models/orange_tree/visual.usda) | [Orange tree, Poly by Google](https://poly.pizza/m/8zS7mfHS0i7), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Conversión y ajuste de escala a 2,4 m. |
| Plátanos sueltos y racimo | [bananas_1k.usdc](models/bananas/bananas_1k.usdc) | [Bananas, Poly Haven](https://polyhaven.com/a/bananas), [CC0 1.0](https://polyhaven.com/license). USD de origen con tres mapas a 1K y referencias de textura reescritas a rutas relativas. El conjunto incluye cuatro plátanos individuales que se pueden separar del racimo. |
| Platanera con racimo | [visual.usda](models/banana_plant/visual.usda) | [Banana plant with a bunch of bananas, Innerscene](https://www.innerscene.com/tools/library/3d-parts/banana-plant-with-a-bunch-of-bananas-905b58eb), CC0 1.0. GLB convertido a USD con hojas, pseudotallo, pedúnculo y fruta como mallas separadas. |
| Racimo de uvas | [visual.usda](models/grapes/visual.usda) | [Bunch of grapes, Innerscene](https://www.innerscene.com/tools/library/3d-parts/bunch-of-grapes-abfcc57f), CC0 1.0. GLB convertido a USD. |
| Vid en espaldera | [visual.usda](models/vineyard/visual.usda) | [Vine Row Fruiting, 3DAssets.dev](https://3dassets.dev/assets/vineyard-and-winery-vine-row-fruiting-f23824de), CC0 1.0. GLB convertido a USD después de decodificar la geometría cuantizada y el mapa WebP. Geometría de contexto con asistencia de IA declarada por la fuente. |
| Arándano individual escaneado | [visual.usda](models/blueberry/visual.usda) | [Gourmet Blueberry, inciprocal](https://sketchfab.com/3d-models/gourmet-blueberry-62f6c6368c484dd6a85a2a162133d2e0), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). GLB obtenido del [archivo público de Objaverse](https://huggingface.co/datasets/allenai/objaverse/blob/main/glbs/000-154/62f6c6368c484dd6a85a2a162133d2e0.glb) y convertido a USD. |
| Tazón de arándanos | [visual.usda](models/blueberries/visual.usda) | [Blueberries, Grab3D](https://grab3d.com/models/food/blueberries/), CC0 1.0. USDZ extraído a USDC y referenciado desde una composición con Z arriba. Se usa como referencia de agrupación. |
| Arbusto de bayas | [visual.usda](models/berry_bush/visual.usda) | [Berry Bush, Grab3D](https://grab3d.com/models/nature/berry-bush/), CC0 1.0. USDZ extraído a USDC y referenciado desde una composición con Z arriba. Modelo estilizado de bayas mixtas, provisional para estudiar oclusiones. El arbusto de arándano específico sigue pendiente. |

La [dedicación CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) aplica a los modelos indicados. Conserva junto a los USD las texturas y estas atribuciones al publicar el proyecto.

Para convertir nuevas mallas GLB o glTF desde la raíz del repositorio:

```powershell
& ..\isaac-sim-6.2.0\python.bat tools/convert_agricultural_mesh.py path/to/model.glb --output assets/agriculture/models
```

El importador genera un `visual.usda` por fuente, su capa de geometría y sus dependencias locales. Mide el USD resultante y verifica las texturas antes de incorporarlo a una escena. Para primeros planos de cosecha, prioriza follaje y pedúnculos detallados sobre los modelos provisionales de contexto.

Para regenerar el torso, las versiones con colisión para la escena y el naranjo:

```powershell
& ..\isaac-sim-6.2.0\python.bat tools/build_agricultural_props.py
& ..\isaac-sim-6.2.0\python.bat tools/build_orchard_tree.py
& ..\isaac-sim-6.2.0\python.bat tools/build_field_obstacles.py
```
