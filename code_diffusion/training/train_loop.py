from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

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


def train(model, tokenizer, dataset, config: dict) -> dict[str, float]:
    device = next(model.parameters()).device
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=diffusion_collate_fn,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    iterator = iter(dataloader)
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0
    completed_steps = 0

    for step in range(1, config["train_steps"] + 1):
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

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    return {
        "final_loss": running_loss / max(completed_steps, 1),
        "final_masked_accuracy": running_accuracy / max(completed_steps, 1),
    }
