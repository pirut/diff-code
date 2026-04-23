from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from code_diffusion.data.dataset import diffusion_collate_fn


def compute_mask_ratio(step: int, *, min_ratio: float, max_ratio: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return max_ratio
    alpha = min(1.0, step / warmup_steps)
    return min_ratio + (max_ratio - min_ratio) * alpha


def run_preflight_batch(model, dataset, config: dict) -> dict[str, float | int]:
    device = next(model.parameters()).device
    model.train()
    model.zero_grad(set_to_none=True)

    mask_ratio = compute_mask_ratio(
        step=0,
        min_ratio=config["mask_ratio_min"],
        max_ratio=config["mask_ratio_max"],
        warmup_steps=config["warmup_steps"],
    )
    dataset.set_mask_ratio(mask_ratio)

    batch_size = min(int(config["batch_size"]), len(dataset))
    batch = diffusion_collate_fn([dataset[index] for index in range(batch_size)])
    batch = {key: value.to(device) for key, value in batch.items()}

    outputs = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        mask_positions=batch["mask"],
        use_cache=False,
    )

    masked_logits = outputs.logits[batch["mask"]]
    masked_labels = batch["labels"][batch["mask"]]
    if masked_labels.numel() == 0:
        raise ValueError("Preflight batch has zero masked tokens.")

    loss = F.cross_entropy(masked_logits, masked_labels)
    loss.backward()

    with torch.no_grad():
        predictions = masked_logits.argmax(dim=-1)
        accuracy = (predictions == masked_labels).float().mean().item()
        grad_sq_sum = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad_sq_sum += float(parameter.grad.detach().float().pow(2).sum().item())

    return {
        "loss": float(loss.item()),
        "masked_accuracy": float(accuracy),
        "batch_size": int(batch["input_ids"].shape[0]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "attention_tokens": int(batch["attention_mask"].sum().item()),
        "masked_tokens": int(batch["mask"].sum().item()),
        "gradient_norm": float(grad_sq_sum ** 0.5),
    }


def train(
    model,
    tokenizer,
    dataset,
    config: dict,
    *,
    benchmark_cases: list[dict[str, object]] | None = None,
    benchmark_options: dict[str, object] | None = None,
    checkpoint_callback=None,
) -> dict[str, float | str]:
    device = next(model.parameters()).device
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    benchmark_controller = _initialize_benchmark_controller(
        dataset=dataset,
        config=config,
        benchmark_cases=benchmark_cases,
    )

    resume_state = _maybe_restore_training_state(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        config=config,
        controller_state=benchmark_controller,
    )
    dataloader = _build_training_dataloader(
        dataset=dataset,
        config=config,
        task_type_weights=benchmark_controller.get("task_type_weights")
        if benchmark_controller.get("task_reweighting_enabled")
        else None,
    )

    iterator = iter(dataloader)
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0
    completed_steps = 0
    start_step = 0

    if resume_state is not None:
        running_loss = float(resume_state.get("running_loss", 0.0))
        running_accuracy = float(resume_state.get("running_accuracy", 0.0))
        completed_steps = int(resume_state.get("completed_steps", 0))
        start_step = int(resume_state.get("step", 0))
        print(
            f"resumed_training step={start_step} "
            f"completed_steps={completed_steps} "
            f"avg_loss={running_loss / max(completed_steps, 1):.4f} "
            f"avg_masked_acc={running_accuracy / max(completed_steps, 1):.4f}"
        )

    for step in range(start_step + 1, config["train_steps"] + 1):
        mask_ratio = compute_mask_ratio(
            step=step - 1,
            min_ratio=config["mask_ratio_min"],
            max_ratio=config["mask_ratio_max"],
            warmup_steps=config["warmup_steps"],
        )
        dataset.set_mask_ratio(mask_ratio)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mask_positions=batch["mask"],
            use_cache=False,
        )

        masked_logits = outputs.logits[batch["mask"]]
        masked_labels = batch["labels"][batch["mask"]]
        if masked_labels.numel() == 0:
            continue

        loss = F.cross_entropy(masked_logits, masked_labels)
        loss.backward()

        grad_clip = config.get("grad_clip")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == masked_labels).float().mean().item()

        completed_steps += 1
        running_loss += float(loss.item())
        running_accuracy += accuracy

        if step % config.get("log_every", 10) == 0 or step == 1:
            avg_loss = running_loss / completed_steps
            avg_accuracy = running_accuracy / completed_steps
            print(
                f"step={step} "
                f"mask_ratio={mask_ratio:.3f} "
                f"loss={loss.item():.4f} "
                f"avg_loss={avg_loss:.4f} "
                f"masked_acc={accuracy:.4f} "
                f"avg_masked_acc={avg_accuracy:.4f}"
            )

        save_every = config.get("save_every")
        if save_every and step % save_every == 0:
            checkpoint_dir = output_dir / f"step-{step}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            benchmark_result = _maybe_benchmark_checkpoint(
                model=model,
                tokenizer=tokenizer,
                config=config,
                checkpoint_dir=checkpoint_dir,
                benchmark_cases=benchmark_cases,
                benchmark_options=benchmark_options,
            )
            controller_update = _apply_benchmark_controller(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                optimizer=optimizer,
                config=config,
                checkpoint_dir=checkpoint_dir,
                benchmark_result=benchmark_result,
                benchmark_options=benchmark_options,
                controller_state=benchmark_controller,
            )
            if controller_update["reset_dataloader"]:
                dataloader = _build_training_dataloader(
                    dataset=dataset,
                    config=config,
                    task_type_weights=benchmark_controller.get("task_type_weights"),
                )
                iterator = iter(dataloader)
            _save_training_state(
                checkpoint_dir=checkpoint_dir,
                optimizer=optimizer,
                step=step,
                completed_steps=completed_steps,
                running_loss=running_loss,
                running_accuracy=running_accuracy,
                benchmark_controller=benchmark_controller,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    checkpoint_dir,
                    {
                        "benchmark": benchmark_result,
                        "controller": controller_update["event"],
                    },
                )

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    final_benchmark_result = _maybe_benchmark_checkpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=final_dir,
        benchmark_cases=benchmark_cases,
        benchmark_options=benchmark_options,
    )
    final_controller_update = _apply_benchmark_controller(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        config=config,
        checkpoint_dir=final_dir,
        benchmark_result=final_benchmark_result,
        benchmark_options=benchmark_options,
        controller_state=benchmark_controller,
    )
    _save_training_state(
        checkpoint_dir=final_dir,
        optimizer=optimizer,
        step=config["train_steps"],
        completed_steps=completed_steps,
        running_loss=running_loss,
        running_accuracy=running_accuracy,
        benchmark_controller=benchmark_controller,
    )
    if checkpoint_callback is not None:
        checkpoint_callback(
            final_dir,
            {
                "benchmark": final_benchmark_result,
                "controller": final_controller_update["event"],
            },
        )

    metrics = {
        "final_loss": running_loss / max(completed_steps, 1),
        "final_masked_accuracy": running_accuracy / max(completed_steps, 1),
    }
    if benchmark_controller.get("enabled"):
        metrics["best_benchmark_score"] = float(benchmark_controller.get("best_score") or 0.0)
        metrics["best_checkpoint"] = str(benchmark_controller.get("best_checkpoint") or "")
    return metrics


def _build_training_dataloader(
    *,
    dataset,
    config: dict,
    task_type_weights: dict[str, float] | None = None,
) -> DataLoader:
    sampler = None
    shuffle = True
    if task_type_weights and hasattr(dataset, "get_weighted_sample_weights"):
        sampler = WeightedRandomSampler(
            weights=dataset.get_weighted_sample_weights(task_type_weights),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get("num_workers", 0),
        collate_fn=diffusion_collate_fn,
    )


def _maybe_restore_training_state(
    *,
    model,
    dataset,
    optimizer,
    config: dict,
    controller_state: dict[str, object],
) -> dict[str, object] | None:
    checkpoint_dir_raw = config.get("resume_checkpoint_dir")
    if not checkpoint_dir_raw:
        return None

    checkpoint_dir = Path(str(checkpoint_dir_raw))
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    if not trainer_state_path.exists():
        return None

    state = torch.load(trainer_state_path, map_location="cpu")
    optimizer_state = state.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    controller_snapshot = state.get("benchmark_controller")
    if isinstance(controller_snapshot, dict):
        _restore_benchmark_controller(
            dataset=dataset,
            controller_state=controller_state,
            snapshot=controller_snapshot,
        )

    return state


def _initialize_benchmark_controller(
    *,
    dataset,
    config: dict,
    benchmark_cases: list[dict[str, object]] | None,
) -> dict[str, object]:
    enabled = bool(benchmark_cases) and bool(config.get("benchmark_controller_enabled", True))
    task_reweighting_enabled = (
        enabled
        and bool(config.get("benchmark_controller_reweight_task_mix", True))
        and hasattr(dataset, "set_task_type_weights")
        and hasattr(dataset, "get_task_type_weights")
    )
    task_type_weights = None
    if task_reweighting_enabled:
        task_type_weights = dataset.set_task_type_weights(config.get("task_type_weights", {}))

    return {
        "enabled": enabled,
        "task_reweighting_enabled": task_reweighting_enabled,
        "task_type_weights": task_type_weights,
        "best_score": None,
        "best_checkpoint": None,
        "plateau_count": 0,
        "lr_decay_count": 0,
        "history": [],
    }


def _restore_benchmark_controller(
    *,
    dataset,
    controller_state: dict[str, object],
    snapshot: dict[str, object],
) -> None:
    for key in (
        "enabled",
        "task_reweighting_enabled",
        "best_score",
        "best_checkpoint",
        "plateau_count",
        "lr_decay_count",
        "history",
    ):
        if key in snapshot:
            controller_state[key] = snapshot[key]

    snapshot_weights = snapshot.get("task_type_weights")
    if controller_state.get("task_reweighting_enabled") and snapshot_weights:
        controller_state["task_type_weights"] = dataset.set_task_type_weights(snapshot_weights)
    else:
        controller_state["task_type_weights"] = snapshot_weights


def _maybe_benchmark_checkpoint(
    *,
    model,
    tokenizer,
    config: dict,
    checkpoint_dir: Path,
    benchmark_cases: list[dict[str, object]] | None,
    benchmark_options: dict[str, object] | None,
) -> dict[str, object] | None:
    if not benchmark_cases:
        return None

    from code_diffusion.evaluation import benchmark_loaded_model, render_benchmark_markdown

    options = dict(benchmark_options or {})
    was_training = model.training
    model.eval()
    try:
        result = benchmark_loaded_model(
            model=model,
            tokenizer=tokenizer,
            config=config,
            cases=benchmark_cases,
            steps=int(options.get("steps") or config["diffusion_steps"]),
            temperature=float(options.get("temperature", 0.0)),
            top_k=int(options["top_k"]) if options.get("top_k") not in {None, 0} else None,
            top_p=float(options["top_p"]) if options.get("top_p") not in {None, 0, 1, 1.0} else None,
            show_samples=int(options.get("show_samples", 1)),
        )
    finally:
        if was_training:
            model.train()

    payload = {
        "cases_file": str(options.get("cases_label", "inline")),
        "checkpoint_count": 1,
        "results": [
            {
                "checkpoint": str(checkpoint_dir),
                "checkpoint_label": checkpoint_dir.name,
                **result,
            }
        ],
    }
    benchmark_dir = Path(config["output_dir"]) / str(options.get("output_subdir", "benchmarks"))
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    json_path = benchmark_dir / f"{checkpoint_dir.name}.json"
    md_path = benchmark_dir / f"{checkpoint_dir.name}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_benchmark_markdown(payload), encoding="utf-8")

    summary = payload["results"][0]
    print(
        f"benchmark checkpoint={summary['checkpoint_label']} "
        f"exact_match_rate={summary['exact_match_rate']:.3f} "
        f"avg_similarity={summary['avg_similarity']:.3f}"
    )
    return summary


def _apply_benchmark_controller(
    *,
    model,
    tokenizer,
    dataset,
    optimizer,
    config: dict,
    checkpoint_dir: Path,
    benchmark_result: dict[str, object] | None,
    benchmark_options: dict[str, object] | None,
    controller_state: dict[str, object],
) -> dict[str, object]:
    if not controller_state.get("enabled") or not benchmark_result:
        return {"reset_dataloader": False, "event": None}

    score = _score_benchmark_summary(benchmark_result, config)
    improvement_threshold = float(config.get("benchmark_controller_improvement_threshold", 1e-4))
    previous_best = controller_state.get("best_score")
    improved = previous_best is None or score > float(previous_best) + improvement_threshold
    if improved:
        controller_state["best_score"] = score
        controller_state["best_checkpoint"] = checkpoint_dir.name
        controller_state["plateau_count"] = 0
        _save_best_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=Path(config["output_dir"]),
            dirname=str(config.get("benchmark_controller_best_dirname", "best")),
        )
        print(
            f"benchmark-controller improved checkpoint={checkpoint_dir.name} "
            f"score={score:.4f}"
        )
    else:
        controller_state["plateau_count"] = int(controller_state.get("plateau_count", 0)) + 1

    lr_event = _maybe_decay_learning_rate(
        optimizer=optimizer,
        config=config,
        controller_state=controller_state,
    )
    if lr_event is not None:
        print(
            f"benchmark-controller lr_decay old_lr={lr_event['old_lr']:.6g} "
            f"new_lr={lr_event['new_lr']:.6g} "
            f"decays={lr_event['decay_count']}"
        )

    weights_event = None
    reset_dataloader = False
    if controller_state.get("task_reweighting_enabled"):
        weights_event = _maybe_reweight_task_mix(
            dataset=dataset,
            config=config,
            controller_state=controller_state,
            benchmark_result=benchmark_result,
        )
        if weights_event is not None:
            reset_dataloader = True
            rendered = " ".join(
                f"{name}={value:.3f}"
                for name, value in sorted(weights_event["task_type_weights"].items())
            )
            print(f"benchmark-controller task_weights {rendered}")

    event = {
        "checkpoint": checkpoint_dir.name,
        "score": score,
        "improved": improved,
        "best_score": controller_state.get("best_score"),
        "best_checkpoint": controller_state.get("best_checkpoint"),
        "plateau_count": controller_state.get("plateau_count"),
        "lr_event": lr_event,
        "weights_event": weights_event,
    }
    controller_state["history"].append(event)
    _write_controller_state(
        output_dir=Path(config["output_dir"]),
        benchmark_options=benchmark_options,
        controller_state=controller_state,
        event=event,
    )
    return {"reset_dataloader": reset_dataloader, "event": event}


def _score_benchmark_summary(summary: dict[str, object], config: dict) -> float:
    weights = config.get("benchmark_controller_score_weights", {}) or {}
    exact_weight = float(weights.get("exact_match", 0.7))
    similarity_weight = float(weights.get("similarity", 0.3))
    denom = max(exact_weight + similarity_weight, 1e-8)
    exact_match = float(summary.get("exact_match_rate", 0.0))
    similarity = float(summary.get("avg_similarity", 0.0))
    return (exact_weight * exact_match + similarity_weight * similarity) / denom


def _save_best_checkpoint(*, model, tokenizer, output_dir: Path, dirname: str) -> None:
    best_dir = output_dir / dirname
    if best_dir.exists():
        shutil.rmtree(best_dir)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)


def _maybe_decay_learning_rate(
    *,
    optimizer,
    config: dict,
    controller_state: dict[str, object],
) -> dict[str, float | int] | None:
    patience = int(config.get("benchmark_controller_plateau_patience", 2))
    if patience <= 0 or int(controller_state.get("plateau_count", 0)) < patience:
        return None

    decay_factor = float(config.get("benchmark_controller_lr_decay_factor", 0.5))
    min_lr = float(config.get("benchmark_controller_min_learning_rate", 1e-5))
    old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    new_lrs = [max(old_lr * decay_factor, min_lr) for old_lr in old_lrs]
    if all(abs(new_lr - old_lr) < 1e-12 for old_lr, new_lr in zip(old_lrs, new_lrs)):
        return None

    for group, new_lr in zip(optimizer.param_groups, new_lrs):
        group["lr"] = new_lr

    controller_state["plateau_count"] = 0
    controller_state["lr_decay_count"] = int(controller_state.get("lr_decay_count", 0)) + 1
    return {
        "old_lr": old_lrs[0],
        "new_lr": new_lrs[0],
        "decay_count": int(controller_state["lr_decay_count"]),
    }


def _maybe_reweight_task_mix(
    *,
    dataset,
    config: dict,
    controller_state: dict[str, object],
    benchmark_result: dict[str, object],
) -> dict[str, object] | None:
    current_weights = dict(controller_state.get("task_type_weights") or {})
    if not current_weights:
        return None

    task_scores: dict[str, float] = {}
    for task_name, metrics in dict(benchmark_result.get("by_task_type", {})).items():
        mapped_buckets = _map_benchmark_task_to_task_buckets(str(task_name))
        if not mapped_buckets:
            continue
        score = _score_benchmark_summary(metrics, config)
        for bucket in mapped_buckets:
            existing = task_scores.get(bucket)
            task_scores[bucket] = score if existing is None else max(existing, score)

    if not task_scores:
        return None

    overall_score = _score_benchmark_summary(benchmark_result, config)
    strength = float(config.get("benchmark_controller_task_mix_strength", 1.0))
    momentum = float(config.get("benchmark_controller_task_mix_momentum", 0.5))
    min_weight = float(config.get("benchmark_controller_min_task_weight", 0.05))

    proposed = dict(current_weights)
    for bucket, bucket_score in task_scores.items():
        if bucket not in proposed:
            continue
        shortfall = max(0.0, overall_score - bucket_score)
        proposed[bucket] *= 1.0 + strength * shortfall

    proposed = _normalize_weights(proposed, min_weight=min_weight)
    blended = {}
    for name, current_weight in current_weights.items():
        target_weight = proposed.get(name, current_weight)
        blended[name] = (1.0 - momentum) * float(current_weight) + momentum * float(target_weight)
    blended = _normalize_weights(blended, min_weight=min_weight)

    if not _weights_changed(current_weights, blended):
        return None

    updated = dataset.set_task_type_weights(blended)
    controller_state["task_type_weights"] = updated
    return {
        "task_type_weights": updated,
        "task_scores": task_scores,
        "overall_score": overall_score,
    }


def _map_benchmark_task_to_task_buckets(task_name: str) -> list[str]:
    aliases = {
        "masked_reconstruction": ["masked_reconstruction"],
        "span_completion": ["fim"],
        "fim": ["fim"],
        "bug_fix": ["bug_fix"],
        "draft_refinement": ["refinement"],
        "refinement": ["refinement"],
    }
    return aliases.get(task_name, [])


def _normalize_weights(weights: dict[str, float], *, min_weight: float = 0.0) -> dict[str, float]:
    if not weights:
        return {}

    cleaned = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(len(cleaned), 1)
        cleaned = {name: uniform for name in cleaned}
    else:
        cleaned = {name: value / total for name, value in cleaned.items()}

    min_weight = max(0.0, float(min_weight))
    if min_weight <= 0:
        return cleaned

    item_count = len(cleaned)
    capped_min = min(min_weight, 1.0 / max(item_count, 1))
    reserved = capped_min * item_count
    if reserved >= 1.0:
        uniform = 1.0 / max(item_count, 1)
        return {name: uniform for name in cleaned}

    remainder = 1.0 - reserved
    return {name: capped_min + remainder * value for name, value in cleaned.items()}


def _weights_changed(
    current_weights: dict[str, float],
    new_weights: dict[str, float],
    *,
    tolerance: float = 1e-4,
) -> bool:
    if current_weights.keys() != new_weights.keys():
        return True
    return any(abs(float(current_weights[name]) - float(new_weights[name])) > tolerance for name in current_weights)


def _write_controller_state(
    *,
    output_dir: Path,
    benchmark_options: dict[str, object] | None,
    controller_state: dict[str, object],
    event: dict[str, object],
) -> None:
    subdir = str((benchmark_options or {}).get("output_subdir", "benchmarks"))
    controller_dir = output_dir / subdir
    controller_dir.mkdir(parents=True, exist_ok=True)

    state_payload = {
        "best_score": controller_state.get("best_score"),
        "best_checkpoint": controller_state.get("best_checkpoint"),
        "plateau_count": controller_state.get("plateau_count"),
        "lr_decay_count": controller_state.get("lr_decay_count"),
        "task_type_weights": controller_state.get("task_type_weights"),
        "history_length": len(controller_state.get("history", [])),
        "last_event": event,
    }
    (controller_dir / "controller_state.json").write_text(
        json.dumps(state_payload, indent=2),
        encoding="utf-8",
    )
    with (controller_dir / "controller_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _save_training_state(
    *,
    checkpoint_dir: Path,
    optimizer,
    step: int,
    completed_steps: int,
    running_loss: float,
    running_accuracy: float,
    benchmark_controller: dict[str, object],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "completed_steps": int(completed_steps),
        "running_loss": float(running_loss),
        "running_accuracy": float(running_accuracy),
        "optimizer_state_dict": optimizer.state_dict(),
        "benchmark_controller": {
            "enabled": benchmark_controller.get("enabled"),
            "task_reweighting_enabled": benchmark_controller.get("task_reweighting_enabled"),
            "task_type_weights": benchmark_controller.get("task_type_weights"),
            "best_score": benchmark_controller.get("best_score"),
            "best_checkpoint": benchmark_controller.get("best_checkpoint"),
            "plateau_count": benchmark_controller.get("plateau_count"),
            "lr_decay_count": benchmark_controller.get("lr_decay_count"),
            "history": benchmark_controller.get("history", []),
        },
    }
    torch.save(payload, checkpoint_dir / "trainer_state.pt")
