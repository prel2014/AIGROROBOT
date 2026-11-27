# Humanoide bípedo Xavier (1,73 m)

Instantánea del proyecto local del 27 de noviembre de 2026. Consulta [Proyecto.md](Proyecto.md) para el diseño y el estado de las pruebas.

Los 16 Xavier completaron las tres pruebas del campus con su cámara individual de 256 × 256. Hubo 48 llegadas estables, cero caídas y cero contactos. [Vídeo](isaac/cruce_visual_campus_16.mp4) · [Informe por robot](isaac/validacion_cruce_visual.json) · [Política visual](isaac/paso_visual_campus.pt)

## Archivos principales

- [USD del robot para Isaac Sim](isaac/humanoide_xavier_isaac.usd).
- [Pista de marcha para 16 robots](isaac/pista_marcha_16.usda).
- [Escenarios de campus, calle y vivienda](isaac/escenarios/), con texturas.
- [Entrenamiento de marcha](isaac/entrenar_marcha.py) y checkpoint `isaac/marcha_cargas.pt`.
- [Fuentes CAD y USD de diseño](cad/).

## Paquetes complementarios

En [paquetes](paquetes/) se incluyen el STEP completo de Xavier, los componentes CAD, electrónica, control CAN, animación y referencias.

Los ZIP grandes están divididos en archivos `.part001`, `.part002`, etc. Descarga todas sus partes junto con `partes.json` y `reunir_partes.py` y ejecuta:

```bash
python paquetes/reunir_partes.py
```

El script reúne cada ZIP y verifica su SHA-256. Después extrae los ZIP sobre esta carpeta conservando sus rutas para reconstruir la estructura del proyecto.

Isaac Sim y sus dependencias se instalan por separado. La fecha de modificación del checkpoint identifica la copia incluida; para resultados de evaluación, consulta `isaac/` y `Proyecto.md`.

Se omitieron cachés CAD, archivos temporales, registros de ejecución y configuraciones locales de KiCad.
