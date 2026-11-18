# VLA de un brazo en una GPU

El VLA v5 usa dos RGB, texto, estado angular y memoria por réplica. El flujo de entrenamiento parte de un teacher de elevación evaluado y HDF5 de la misma escena. [Comandos de entrenamiento y evaluación](mvp.md)

## Inferencia en Isaac Sim

```powershell
$student = 'checkpoints/vla/vla01/best.pt'
.\scripts\run.ps1 play_vla --checkpoint $student --num_envs 4 --episodes 20 --viz kit
```

Usa `--headless` para ejecutar sin ventana. Añade `--scenario` con la escena usada durante la captura si el checkpoint usa una distinta de `red_cube`.

## Inferencia desde un episodio HDF5

```powershell
$episode = 'datasets/demos/demos01/episode_000001.h5'
.\scripts\run.ps1 infer_vla --checkpoint $student --episode $episode --frame 0 --output logs/vla_prediction.npz
```

El archivo `.npz` guarda ángulos y memoria. Puedes pasar su ruta con `--memory-in` al predecir otro paso del mismo episodio. [Formato HDF5](dataset.md#elevación-de-un-brazo)
