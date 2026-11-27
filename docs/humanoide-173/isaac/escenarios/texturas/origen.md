# Texturas de los escenarios Xavier

## Vehículos y viviendas de Gazebo

La adaptación de AIGROROBOT convierte las mallas OBJ y COLLADA a USD, conserva sus coordenadas UV y normales, ajusta las unidades a metros y añade materiales y colisiones para Isaac Sim. Los atlas provienen de la misma revisión que las mallas. Copyright 2012 Nathan Koenig. [Licencia CC BY 3.0 del catálogo](https://github.com/osrf/gazebo_models/blob/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/LICENSE) · [Condiciones de atribución](https://creativecommons.org/licenses/by/3.0/)

| Modelo | Autoría según el catálogo | Original |
| --- | --- | --- |
| Prius Hybrid | Ian Chen | [Malla e interior](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/prius_hybrid) |
| Pickup | Nate Koenig | [Camioneta](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/pickup) |
| House 1 y House 2 | Cole Biesemeyer y Nate Koenig | [Casa 1](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/house_1) · [Casa 2](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/house_2) |

## Iluminación y superficies exteriores

La calle comercial utiliza el panorama completo de Kloofendal y el naranjal utiliza el paisaje de Clarens. Los archivos HDR 4K aportan luz y horizonte. Las cuadras, los frutales, los camellones y la estación de riego son geometría USD. [Kloofendal 48d Partly Cloudy](https://polyhaven.com/a/kloofendal_48d_partly_cloudy) · [Clarens Midday](https://polyhaven.com/a/clarens_midday) · [Licencia CC0](https://polyhaven.com/license)

La luz ambiental utiliza un cielo HDR de Greg Zaal y Jarod Guest. El revoque, el césped y la tierra seca emplean mapas de color, normales y rugosidad de 1K. [Cielo Kloofendal](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky) · [Revoque](https://polyhaven.com/a/plastered_wall_02) · [Césped](https://polyhaven.com/a/aerial_grass_rock) · [Tierra seca](https://polyhaven.com/a/brown_mud_dry) · [Licencia CC0](https://polyhaven.com/license)

## Superficies fotográficas de la revisión

El asfalto utiliza [Asphalt 04](https://polyhaven.com/a/asphalt_04) y la madera utiliza [Wood Planks Dirt](https://polyhaven.com/a/wood_planks_dirt). Los seis JPG 2K provienen de Poly Haven bajo [CC0](https://polyhaven.com/license). Cada juego incluye color en sRGB, normales tangentes OpenGL y rugosidad en espacio lineal. El motivo se repite cada 4 m en el asfalto y cada 2 m en la madera.

## Modelos 3D

Los árboles de los exteriores utilizan una versión reducida de Tree Small 02. La adaptación para simulación conserva el tronco y las ramas, selecciona hojas completas con semilla 47 y reutiliza la geometría en las hileras del campo. Los mapas de corte por transparencia, color, rugosidad y normales acompañan al USD local. [Tree Small 02](https://polyhaven.com/a/tree_small_02) · [Licencia CC0](https://polyhaven.com/license)

Los modelos 3D de esta revisión están en `../objetos/`. La planta utiliza [Potted Plant 02, de Rico Cilliers](https://polyhaven.com/a/potted_plant_02) y el banco utiliza [Painted Wooden Bench, de Kirill Sannikov](https://polyhaven.com/a/painted_wooden_bench). Se distribuyen el USD original y sus mapas 1K bajo la misma licencia CC0. El tiesto tiene un cilindro de contacto independiente. El banco utiliza su malla estática como colisionador.

## Atlas y panorama originales

Las seis superficies, los cuatro escaparates y el panorama costero son imágenes originales generadas en modo texto a imagen. El atlas de 1536 × 1024 se dividió en seis muestras de 512 × 512. El generador USD define su repetición en metros, su rugosidad y su metalicidad. El panorama es una imagen LDR de iluminación ambiental. Los escaparates se recortan del segundo atlas y los letreros se dibujan por separado para conservar el texto legible.

## Prompt del atlas

Photorealistic material albedo atlas for a Peruvian architectural robotics simulation. A single perfectly aligned 3-column by 2-row grid filling a landscape canvas with aspect ratio 3:2. Six edge-to-edge square swatches, no gutters or borders. TOP LEFT: satin honey oak timber, longitudinal fine grain and several staggered long floorboards running left to right. TOP MIDDLE: warm ivory polished terrazzo with small grey and ochre mineral chips, no joints. TOP RIGHT: charcoal road asphalt with fine visible aggregate, dry, no painted markings. BOTTOM LEFT: terracotta exposed clay brick, running bond, realistic narrow warm-grey mortar, eight brick courses. BOTTOM MIDDLE: sun-aged light grey brushed concrete, subtle fine grains, absolutely no expansion joints. BOTTOM RIGHT: dusty sage woven upholstery, fine visible warp and weft. Each square is a flat, directly overhead, evenly exposed material scan that can repeat. High frequency detail and natural variation. No lighting gradient, no depth perspective, no scenery, no objects, no text or labels.

## Prompt del panorama

Photorealistic 360 degree equirectangular environment panorama, 2:1 aspect, for an architectural daylight renderer. View from a rooftop in the coastal outskirts of Trujillo, Peru on a bright warm late morning, pale blue sky with thin delicate high clouds, subtle marine haze by the horizon, distant muted tan Andean foothills, very distant low-rise masonry rooftops and a few tree crowns below horizon. Level horizon exactly halfway down the image, the entire upper half is sky. The scene is seen from well above rooftops so there are no tall foreground objects. Muted natural tones, plausible photographic exposure, impeccable continuous horizon across the left and right seams. No people, text, labels or landmarks.

## Prompt de los escaparates

Generate a photorealistic texture atlas for street-level shop window displays in an architectural simulation of a neighborhood in Trujillo, Peru. Single landscape image ideally 2:1. Perfect edge-to-edge 2-column 2-row grid of equally sized horizontal rectangular photographs with absolutely no gutters, borders, captions or labels. Each cell shows what is immediately BEHIND the glass of a different shop window, viewed frontally at street eye level looking in, tightly filled with the shop display. Top left: a small artisanal bakery display with rustic baguettes, rolls and pastries on wood shelves, warm interior light. Top right: a bright clean independent pharmacy with orderly pastel boxes on white shelves, mint-green interior. Bottom left: a small neighborhood stationery shop with colored notebooks and pens, paper and books on shelves. Bottom right: a small neighborhood grocery bodega with a colorful orderly display of fruit, glass bottles, jars and food staples. Photographic detail, realistic everyday objects, gentle indoor shadows, clean frontal view with modest depth. No people, no sign text or brands, no architecture or sidewalk outside the display, no mullions, no opaque glass reflections. Each photo exactly fills its quadrant.
