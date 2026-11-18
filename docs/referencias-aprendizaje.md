# Referencias para el aprendizaje robótico

[Biblioteca local de papers y manuales](papers/README.md)

## Flujo de trabajo adoptado

1. **Definir una habilidad y cómo medirla.** Separamos agarrar, elevar, sostener, transferir y depositar. Fijamos unidades, contactos, límites y la condición de término antes de ajustar la red. El flujo de NVIDIA usa el mismo criterio de éxito para recopilar y evaluar. [GR00T E2E con manzana en Isaac Lab](https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow/blob/main/docs/workflows/g1-static-apple.md)
2. **Verificar la escena y la acción con física.** Revisamos colisionadores, límites, fricción, estabilidad y una trayectoria guiada para cada habilidad. IndustReal usa geometría y un currículo de inicialización para tareas de contacto. [IndustReal](https://arxiv.org/abs/2305.17110) · [Factory en Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory)
3. **Entrenar un teacher con estado del simulador.** Empezamos en condiciones físicamente alcanzables y ampliamos la exploración cuando las evaluaciones repetibles muestran progreso. Dexplore usa referencias como guía para la exploración de PPO y después destila una política visual. [Dexplore](https://arxiv.org/abs/2509.09671) · [Código](https://github.com/NVlabs/dexplore)
4. **Comprobar cada habilidad bimanual por separado.** Medimos el cierre, la elevación y la liberación de una mano al transferir a la otra antes de encadenar el depósito. [Bi-DexHands](https://proceedings.neurips.cc/paper_files/paper/2022/hash/217a2a387f52c30755c37b0a73430291-Abstract.html) · [Entornos y código](https://github.com/PKU-MARL/DexterousHands)
5. **Capturar y entrenar al estudiante.** Registramos cada episodio con cámaras, estado, órdenes, acción y resultado. El VLA aprende a emitir bloques cortos de ángulos bimanuales. Se reevaluará con las nuevas imágenes del simulador durante el control. [ACT y ALOHA](https://tonyzhaozh.github.io/aloha/) · [OpenVLA-OFT](https://openvla-oft.github.io/)
6. **Evaluar generalización con episodios completos.** Reservamos semillas, objetos, obstáculos y órdenes. Medimos éxito, caída, deslizamiento, golpes y tiempo por cultivo. En fruta, el daño superficial requiere un criterio específico adicional. [FruitGym](https://arxiv.org/abs/2505.08458) · [DexFruit](https://arxiv.org/abs/2508.07118)

## Aplicación al proyecto

El apoyo de 3,5 cm permite estudiar el cierre sin interferencia de la mesa. Se usará como primer entorno de calibración. La misma habilidad se vuelve a medir con fruta suelta sobre mesa, variaciones de pose y obstáculos. Para seguir órdenes en español, el diseño usa CLIP para las imágenes y Multilingual E5 para el texto. Su desempeño con órdenes agrícolas queda sujeto a evaluación. [Ficha de CLIP](https://github.com/openai/CLIP/blob/main/model-card.md) · [Multilingual E5](https://arxiv.org/abs/2402.05672)

La referencia agrícola de GR00T con G1 registra una manzana suelta. La variante H2 incluye traspaso y depósito, y su documentación señala que la evaluación de la política aprendida está pendiente de validación. Estos proyectos orientan el método y aportan diseños de evaluación para un robot con manos y geometría distintas. [GR00T H2](https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow/blob/main/docs/workflows/h2-pick-handover-place.md)
