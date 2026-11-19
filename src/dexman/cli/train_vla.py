"""Train a recurrent action Transformer on contiguous demonstrations with frozen CLIP."""
import argparse
import functools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from dexman.vla.dataset import Demonstrations, dataset_contract
from dexman.data.manifest import create_manifest, read_manifest, resolve_manifest
from dexman.experiments import (provenance, random_state, resolve_checkpoint, restore_random_state,
                                save_training_checkpoint, seed_everything, sha256, write_json)
from dexman.paths import ROOT
from dexman.perception.bimanual import TEXT_MODEL, VISION_MODEL
from dexman.vla.model import ActionChunkPolicy
from dexman.vla.training import collate, epoch, state_statistics
from dexman.vla.contract import POLICY_VERSION, OUTPUT_SPACE, validate_architecture_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--manifest", help="Use existing, content-verified train/validation/test partitions")
    parser.add_argument("--bimanual", action="store_true", help="Use agricultural RGB-D demonstrations and two arms")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initialize-from", help="Warm-start a DAgger round; optimizer starts fresh")
    source.add_argument("--resume", nargs="?", const="latest",
                        help="Resume a file/directory, or latest last.pt; restores original training settings")
    parser.add_argument("--output")
    parser.add_argument("--model", default=VISION_MODEL)
    parser.add_argument("--text-model", default=TEXT_MODEL, help="Frozen bimanual text encoder")
    parser.add_argument("--epochs", type=int, default=20, help="Additional complete epochs, including on resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, help="Consecutive training steps for memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", action="store_true", help="Offline training without a window (default)")
    display.add_argument("--live-plot", action="store_true", help="Open an interactive loss plot; needs matplotlib")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.bimanual:
        from transformers import AutoTokenizer, CLIPImageProcessor

        from dexman.perception.bimanual import load_frozen_bimanual
        from dexman.vla.bimanual import BimanualActionPolicy
        from dexman.vla.bimanual_contract import OUTPUT_SPACE as output_space, POLICY_VERSION as policy_version
        from dexman.vla.bimanual_contract import architecture_config
        from dexman.vla.bimanual_data import BimanualDemonstrations as loader_type
        from dexman.vla.bimanual_data import collate as collate_batch, dataset_contract as read_contract
        from dexman.vla.bimanual_training import epoch as run_epoch

        if args.horizon != BimanualActionPolicy.horizon:
            parser.error("El VLA agrícola bimanual usa un horizonte fijo de ocho comandos")
        default_dataset, default_output = "datasets/agriculture", "checkpoints/vla_bimanual"
    else:
        policy_version, output_space = POLICY_VERSION, OUTPUT_SPACE
        loader_type, collate_batch, read_contract, run_epoch = Demonstrations, collate, dataset_contract, epoch
        default_dataset, default_output = "datasets/demos", "checkpoints/vla"
    tokenizer = None
    args.dataset = args.dataset or default_dataset
    args.output = args.output or default_output
    args.batch_size = (2 if args.bimanual else 8) if args.batch_size is None else args.batch_size
    args.sequence_length = (8 if args.bimanual else 16) if args.sequence_length is None else args.sequence_length
    if min(args.horizon, args.epochs, args.batch_size) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("horizon, epochs, batch-size and learning-rate must be positive; workers >= 0")
    if args.sequence_length < 2:
        parser.error("sequence-length must be at least two to train memory")
    checkpoint, saved, best_saved = None, None, None
    try:
        if args.resume:
            checkpoint = resolve_checkpoint(args.resume, ROOT / default_output, "last.pt")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved.get("policy_version") != policy_version:
                raise ValueError(f"Checkpoint incompatible. Se requiere versión {policy_version} y la opción de robot correspondiente.")
            if args.bimanual:
                if saved["config"]["architecture"] != architecture_config():
                    raise ValueError("La arquitectura bimanual guardada no coincide con la actual")
                args.text_model = saved["config"]["text_model"]
            else:
                validate_architecture_config(saved["config"]["architecture"], horizon=saved["config"]["horizon"])
            if saved.get("training", {}).get("version") != 1:
                raise ValueError("Checkpoint has no resumable student state; use --initialize-from for a warm-start")
            training = saved["training"]
            if saved["metrics"]["validation_normalized_joint_mse"] == training["best_validation_loss"]:
                best_saved = saved
            else:
                best_saved = torch.load(checkpoint.parent / "best.pt", map_location="cpu", weights_only=True)
                if (best_saved.get("policy_version") != policy_version or
                        best_saved["metrics"]["validation_normalized_joint_mse"] != training["best_validation_loss"] or
                        best_saved["metrics"]["epoch"] > training["completed_epochs"]):
                    raise ValueError("Source best.pt does not match this checkpoint's history; resume from last.pt")
            # Continuation keeps the same normalization, data order and optimizer configuration.
            for key in ("model", "horizon", "sequence_length", "batch_size", "seed", "learning_rate"):
                setattr(args, key, saved["config"][key])
            manifest, partitions = read_manifest(checkpoint.parent / "dataset.json")
            if args.manifest and read_manifest(args.manifest)[0] != manifest:
                raise ValueError("Resume requires the original dataset partitions; use --initialize-from for new data")
        elif args.manifest:
            manifest, partitions = read_manifest(args.manifest)
        else:
            if args.bimanual and not Path(args.dataset).is_dir():
                raise ValueError(f"No existe la carpeta de demostraciones agrícolas {args.dataset}. "
                                 "Genera episodios exitosos con agriculture_collect.")
            manifest = create_manifest(args.dataset, args.seed)
            partitions = resolve_manifest(manifest)
        train_files, val_files = partitions["train"], partitions["validation"]
        args.robot_contract = read_contract(train_files + val_files + partitions["test"])
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    seed_everything(args.seed)
    args.output_space = output_space
    if args.bimanual:
        args.global_instruction_training = saved["config"].get("global_instruction_training", False) if saved else True
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; use a new run name to preserve earlier checkpoints")
    if checkpoint or args.initialize_from:
        from dexman.vla.inference import BimanualVLAPredictor, load_predictor
        previous = load_predictor(checkpoint or args.initialize_from, args.device)
        if isinstance(previous, BimanualVLAPredictor) != args.bimanual:
            raise ValueError("El checkpoint requiere el modo de robot correspondiente")
        if previous.robot_contract != args.robot_contract or previous.model.horizon != args.horizon:
            raise ValueError("Warm-start requires the same robot contract and output horizon")
        processor, model = previous.processor, previous.model
        if args.bimanual:
            tokenizer = previous.tokenizer
            args.model, args.text_model = previous.sources
    elif args.bimanual:
        processor = CLIPImageProcessor.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
        encoders = load_frozen_bimanual(args.model, args.text_model)
        model = BimanualActionPolicy(args.robot_contract["joint_limits_rad"], *encoders).to(args.device)
    else:
        processor = CLIPProcessor.from_pretrained(args.model)
        model = ActionChunkPolicy(args.model, args.horizon,
            joint_limits=args.robot_contract["joint_limits_rad"]).to(args.device)
    args.architecture = model.architecture_config
    parameters = {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    if saved is None and not args.bimanual:
        mean, std = state_statistics(train_files)
        model.state_mean.copy_(mean)
        model.state_std.copy_(std)
    output.mkdir(parents=True, exist_ok=False)
    split = {key: [str(path.resolve()) for path in files] for key, files in partitions.items()}
    # Reserve this run exclusively before writing snapshots and the frozen CLIP assets.
    with (output / "split.json").open("x", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)
        stream.write("\n")
    write_json(output / "dataset.json", manifest)
    write_json(output / "run.json", {"arguments": vars(args), "provenance": provenance(),
        "parameters": parameters,
        "resume": None if checkpoint is None else {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "resume_semantics": "Transformer, optimizer, completed epochs, normalization, dataset partitions and RNG; new sequences start with zero memory",
        "test_partition_used_for_training": False,
        "selection": "lowest validation loss; closed-loop success must be evaluated separately"})
    # Keep the exact frozen weights and processor with this run for offline inference.
    processor.save_pretrained(output / "processor")
    model.backbone.save_pretrained(output / "backbone")
    if args.bimanual:
        tokenizer.save_pretrained(output / "tokenizer")
        model.text_encoder.save_pretrained(output / "text_encoder")
    print(f"VLA {policy_version}: {parameters['trainable']:,} trainable parameters; "
          f"{parameters['frozen']:,} frozen encoder parameters; "
          f"{args.architecture['vision_tokens']} visual tokens; memory unroll {args.sequence_length}", flush=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generators = [torch.Generator().manual_seed(args.seed + offset) for offset in (0, 1)]
    start, best, history = 0, float("inf"), []
    if saved is not None:
        training = saved["training"]
        optimizer.load_state_dict(training["optimizer"])
        start, best = training["completed_epochs"], training["best_validation_loss"]
        history = list(training["history"])
        for generator, state in zip(generators, training["loader_rng"], strict=True):
            generator.set_state(state.cpu())
        restore_random_state(training)
        # A branch must retain the best model even if no additional epoch improves it.
        save_training_checkpoint(output / "best.pt", best_saved)
        save_training_checkpoint(output / "last.pt", saved)
        print(f"Resumed student from {checkpoint}; next epoch {start + 1}; batch {args.batch_size}; "
              f"sequence {args.sequence_length}; learning rate {optimizer.param_groups[0]['lr']}", flush=True)
    collator = functools.partial(collate_batch, processor=processor, **({"tokenizer": tokenizer} if args.bimanual else {}))
    def dataset(files, training):
        if args.bimanual and args.global_instruction_training:
            return loader_type(files, args.horizon, args.sequence_length, prompts="both" if training else "global")
        return loader_type(files, args.horizon, args.sequence_length)

    loaders = [DataLoader(dataset(files, training), batch_size=args.batch_size,
                         shuffle=training, num_workers=args.workers,
                         generator=generators[index],
                         pin_memory=str(args.device).startswith("cuda"),
                         collate_fn=collator)
               for index, (files, training) in enumerate([(train_files, True), (val_files, False)])]
    if args.live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        figure, axis = plt.subplots()
        axis.set(xlabel="Epoch", ylabel="Range-normalized joint-angle MSE", title="Frozen CLIP + memory Transformer")
        train_curve, = axis.plot([], [], label="Train")
        val_curve, = axis.plot([], [], label="Validation")
        axis.legend()
    with (output / "metrics.jsonl").open("w") as metrics:
        for index in range(start, start + args.epochs):
            train_loss = run_epoch(model, loaders[0], args.device, optimizer)
            val_loss = run_epoch(model, loaders[1], args.device)
            record = {"epoch": index + 1, "train_normalized_joint_mse": train_loss,
                      "validation_normalized_joint_mse": val_loss}
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(record, flush=True)
            improved = val_loss < best
            best = min(best, val_loss)
            history.append(record)
            snapshot = {"policy_version": policy_version, "head": model.head.state_dict(),
                        "config": vars(args), "metrics": record,
                        "training": {"version": 1, "optimizer": optimizer.state_dict(),
                                     "completed_epochs": index + 1, "best_validation_loss": best,
                                     "history": history, "loader_rng": [g.get_state() for g in generators],
                                     **random_state()}}
            if not args.bimanual:
                snapshot.update(state_mean=model.state_mean.cpu(), state_std=model.state_std.cpu())
            save_training_checkpoint(output / "last.pt", snapshot)
            if improved:
                save_training_checkpoint(output / "best.pt", snapshot)
            if args.live_plot:
                xs = [value["epoch"] for value in history]
                train_curve.set_data(xs, [value["train_normalized_joint_mse"] for value in history])
                val_curve.set_data(xs, [value["validation_normalized_joint_mse"] for value in history])
                axis.relim()
                axis.autoscale_view()
                figure.canvas.draw_idle()
                plt.pause(0.01)
    if args.live_plot:
        figure.savefig(output / "loss.png")
