# PPO visual con memoria de un brazo

Esta ruta usa CLIP congelado y una GRU de 256 valores por réplica. El actor recibe un RGB, la orden y 30 ángulos actuales. Devuelve 30 ángulos objetivo en radianes. La memoria se limpia al reiniciar el episodio o cambiar la orden. [Órdenes disponibles](../src/dexman/tasks/manipulation/catalog.py)

## Uso

Instala `configs/vla-requirements.txt` con el Python de Isaac Sim. Ejecuta desde la raíz del repo.

```powershell
.\scripts\run.ps1 memory_rl --mode train --num_envs 1 --iterations 3000 --output logs/memory_rl/run01 --headless
$policy = 'logs/memory_rl/run01/model_2999.pt'
.\scripts\run.ps1 memory_rl --mode evaluate --checkpoint $policy --episodes 60 --output logs/memory_rl/eval01 --headless
.\scripts\run.ps1 memory_rl --mode train --checkpoint $policy --num_envs 1 --iterations 1000 --output logs/memory_rl/run02 --headless
```

Ajusta `$policy` a un checkpoint existente y conserva `encoder/` junto a él. Puedes evaluar una orden del catálogo con `--instruction` o usar `--viz kit` en lugar de `--headless` para abrir la escena. `evaluation.json` informa resultados por tarea.

## API sin simulador

```python
from dexman.memory_rl.inference import MemoryPredictor

policy = MemoryPredictor("logs/memory_rl/run01/model_2999.pt")
angles = policy.predict_batch([rgb], [q], ["Levanta el cubo rojo 15 cm y sostenlo."])[0]
policy.reset([True])
```

`rgb` es una imagen `uint8[H,W,3]` y `q` contiene 30 ángulos en radianes ordenados por `JOINT_NAMES`. Llama a `reset` al terminar el episodio.
