from __future__ import annotations

import argparse
import random

import torch

from code_diffusion.config import load_config
from code_diffusion.data import CodeDiffusionDataset
from code_diffusion.models import load_diffusion_model
from code_diffusion.training import train


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the code diffusion prototype.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    set_seed(int(config.get("seed", 7)))

    model, tokenizer = load_diffusion_model(config)
    dataset = CodeDiffusionDataset(
        tokenizer=tokenizer,
        data_dir=config["data_dir"],
        seq_length=config["seq_length"],
        extensions=config["extensions"],
        mask_ratio_min=config["mask_ratio_min"],
        mask_ratio_max=config["mask_ratio_max"],
        config=config,
    )
    dataset.export_summary(config["output_dir"])

    metrics = train(model=model, tokenizer=tokenizer, dataset=dataset, config=config)
    print(metrics)


if __name__ == "__main__":
    main()
