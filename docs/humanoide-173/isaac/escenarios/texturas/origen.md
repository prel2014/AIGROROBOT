# Texturas de los escenarios Xavier

## Material de paredes y vehículos

La vivienda, los comercios y el campus usan el color, las normales OpenGL y la rugosidad de Plastered Wall 02. El material aplica un tinte distinto según la fachada. [Plastered Wall 02 de Poly Haven](https://polyhaven.com/a/plastered_wall_02) · [Licencia CC0](https://polyhaven.com/license)

Los dos vehículos proceden de la adaptación a USD de AIGROROBOT en la carpeta PROPUESTA. El sedán es Prius Hybrid, de Ian Chen, y la camioneta es Pickup, de Nate Koenig. El catálogo original acredita copyright 2012 Nathan Koenig. Las mallas y sus texturas conservan la licencia CC BY 3.0. La escena añade volúmenes de colisión para Isaac Sim. [Prius Hybrid original](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/prius_hybrid) · [Pickup original](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/pickup) · [Licencia incluida](../objetos/LICENCIA_vehiculos_Gazebo.txt)

## Superficies fotográficas de la revisión

El asfalto utiliza [Asphalt 04](https://polyhaven.com/a/asphalt_04) y la madera utiliza [Wood Planks Dirt](https://polyhaven.com/a/wood_planks_dirt). Los seis JPG 2K provienen de Poly Haven bajo [CC0](https://polyhaven.com/license). Cada juego incluye color en sRGB, normales tangentes OpenGL y rugosidad en espacio lineal. El motivo se repite cada 4 m en el asfalto y cada 2 m en la madera.

## Modelos 3D

Los modelos 3D de esta revisión están en `../objetos/`. La planta utiliza [Potted Plant 02, de Rico Cilliers](https://polyhaven.com/a/potted_plant_02) y el banco utiliza [Painted Wooden Bench, de Kirill Sannikov](https://polyhaven.com/a/painted_wooden_bench). Se distribuyen el USD original y sus mapas 1K bajo la misma licencia CC0. El tiesto tiene un cilindro de contacto independiente. El banco utiliza su malla estática como colisionador.

## Atlas y panorama originales

Las seis superficies, los cuatro escaparates y el panorama costero son imágenes originales generadas en modo texto a imagen. El atlas de 1536 × 1024 se dividió en seis muestras de 512 × 512. El generador USD define su repetición en metros, su rugosidad y su metalicidad. El panorama es una imagen LDR de iluminación ambiental. Los escaparates se recortan del segundo atlas y los letreros se dibujan por separado para conservar el texto legible.

## Prompt del atlas

Photorealistic material albedo atlas for a Peruvian architectural robotics simulation. A single perfectly aligned 3-column by 2-row grid filling a landscape canvas with aspect ratio 3:2. Six edge-to-edge square swatches, no gutters or borders. TOP LEFT: satin honey oak timber, longitudinal fine grain and several staggered long floorboards running left to right. TOP MIDDLE: warm ivory polished terrazzo with small grey and ochre mineral chips, no joints. TOP RIGHT: charcoal road asphalt with fine visible aggregate, dry, no painted markings. BOTTOM LEFT: terracotta exposed clay brick, running bond, realistic narrow warm-grey mortar, eight brick courses. BOTTOM MIDDLE: sun-aged light grey brushed concrete, subtle fine grains, absolutely no expansion joints. BOTTOM RIGHT: dusty sage woven upholstery, fine visible warp and weft. Each square is a flat, directly overhead, evenly exposed material scan that can repeat. High frequency detail and natural variation. No lighting gradient, no depth perspective, no scenery, no objects, no text or labels.

## Prompt del panorama

Photorealistic 360 degree equirectangular environment panorama, 2:1 aspect, for an architectural daylight renderer. View from a rooftop in the coastal outskirts of Trujillo, Peru on a bright warm late morning, pale blue sky with thin delicate high clouds, subtle marine haze by the horizon, distant muted tan Andean foothills, very distant low-rise masonry rooftops and a few tree crowns below horizon. Level horizon exactly halfway down the image, the entire upper half is sky. The scene is seen from well above rooftops so there are no tall foreground objects. Muted natural tones, plausible photographic exposure, impeccable continuous horizon across the left and right seams. No people, text, labels or landmarks.

## Prompt de los escaparates

Generate a photorealistic texture atlas for street-level shop window displays in an architectural simulation of a neighborhood in Trujillo, Peru. Single landscape image ideally 2:1. Perfect edge-to-edge 2-column 2-row grid of equally sized horizontal rectangular photographs with absolutely no gutters, borders, captions or labels. Each cell shows what is immediately BEHIND the glass of a different shop window, viewed frontally at street eye level looking in, tightly filled with the shop display. Top left: a small artisanal bakery display with rustic baguettes, rolls and pastries on wood shelves, warm interior light. Top right: a bright clean independent pharmacy with orderly pastel boxes on white shelves, mint-green interior. Bottom left: a small neighborhood stationery shop with colored notebooks and pens, paper and books on shelves. Bottom right: a small neighborhood grocery bodega with a colorful orderly display of fruit, glass bottles, jars and food staples. Photographic detail, realistic everyday objects, gentle indoor shadows, clean frontal view with modest depth. No people, no sign text or brands, no architecture or sidewalk outside the display, no mullions, no opaque glass reflections. Each photo exactly fills its quadrant.
