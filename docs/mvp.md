# Flujo de elevación de un brazo

El objetivo es levantar el objeto 15 cm y sostenerlo 0,5 s con contactos opuestos medidos. Usa la misma escena en entrenamiento, evaluación y captura. `red_cube` es la predeterminada. [Escenarios](../assets/procedural/README.md) · [Configuración](../configs/mvp.json)

## Entrenar, probar y capturar

Completa la preparación del [README](../README.md) e instala las dependencias del VLA para las fases de estudiante.

```powershell
& ..\isaac-sim-6.2.0\python.bat -m pip install -c configs/runtime-constraints.txt -r configs/vla-requirements.txt

.\scripts\mvp.ps1 -Stage teacher -Robots 4 -Iterations 3000 -Headless -RunName teacher01
$teacher = 'logs/rsl_rl/ur10e_shadow_lift_v1/teacher01/model_2999.pt'
.\scripts\mvp.ps1 -Stage evaluate-teacher -Teacher $teacher -Robots 4 -Headless -RunName teacher_eval01

.\scripts\mvp.ps1 -Stage collect -Teacher $teacher -TeacherEvaluation logs/mvp/teacher_eval01/evaluation.json -Episodes 200 -Robots 4 -Headless -RunName demos01
.\scripts\mvp.ps1 -Stage split -Dataset datasets/demos/demos01 -RunName split01

.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split01.json -Epochs 20 -BatchSize 8 -Headless -RunName vla01
$student = 'checkpoints/vla/vla01/best.pt'
.\scripts\mvp.ps1 -Stage evaluate-student -Student $student -Robots 4 -Headless -RunName vla_eval01
.\scripts\run.ps1 play_vla --checkpoint $student --num_envs 1 --episodes 20 --viz kit
```

Ajusta `$teacher` al checkpoint que exista. La captura del MVP exige una evaluación aprobada del mismo checkpoint. La prueba final usa 200 episodios por semilla y un umbral del 90 % en cada una. [Contrato HDF5](dataset.md#elevación-de-un-brazo)

Para continuar, usa `-Resume -Teacher $teacher` en `-Stage teacher` o `-Resume -Student checkpoints/vla/vla01/last.pt` en `-Stage train`, con un nuevo `-RunName`.

## Correcciones DAgger

```powershell
.\scripts\mvp.ps1 -Stage corrections -Teacher $teacher -Student $student -Robots 4 -Episodes 100 -Headless -RunName corrections01
.\scripts\mvp.ps1 -Stage add-corrections -Manifest datasets/manifests/split01.json -Dataset datasets/corrections/corrections01 -RunName split_dagger01
.\scripts\mvp.ps1 -Stage train -Manifest datasets/manifests/split_dagger01.json -Student $student -Headless -RunName vla_dagger01
```

Las correcciones añaden etiquetas de un paso a la partición de entrenamiento. El arranque con `-Student` en esta última línea conserva pesos e inicia un optimizador nuevo.
