from __future__ import annotations

import argparse
from pathlib import Path

import torch

from code_diffusion.config import load_config
from code_diffusion.inference import generate
from code_diffusion.models import load_diffusion_model
from code_diffusion.utils.tokenization import (
    decode_tokens,
    encode_prompt_with_masks,
    resolve_mask_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative denoising inference.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--prompt", required=True, help="Prompt containing one or more [MASK] slots.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override diffusion steps.")
    parser.add_argument(
        "--mask-span-tokens",
        type=int,
        default=None,
        help="How many mask tokens to expand each [MASK] placeholder into.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional remasking confidence threshold.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional fine-tuned checkpoint path. Defaults to output_dir/final if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)

    if args.checkpoint:
        config["model_name"] = args.checkpoint
    else:
        final_dir = Path(config["output_dir"]) / "final"
        if final_dir.exists():
            config["model_name"] = str(final_dir)

    model, tokenizer = load_diffusion_model(config)
    device = next(model.parameters()).device
    mask_token_id = resolve_mask_token_id(tokenizer)

    prompt_ids, prompt_mask = encode_prompt_with_masks(
        args.prompt,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        default_mask_span=args.mask_span_tokens or config["inference_mask_span"],
    )
    prompt_ids = prompt_ids.unsqueeze(0).to(device)
    prompt_mask = prompt_mask.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)

    generated = generate(
        model=model,
        initial_tokens=prompt_ids,
        attention_mask=attention_mask,
        initial_mask=prompt_mask,
        steps=args.steps or config["diffusion_steps"],
        temperature=args.temperature if args.temperature is not None else config["temperature"],
        top_k=args.top_k if args.top_k is not None else config["top_k"],
        top_p=args.top_p if args.top_p is not None else config["top_p"],
        confidence_threshold=(
            args.confidence_threshold
            if args.confidence_threshold is not None
            else config.get("confidence_threshold")
        ),
        remask_fraction=config.get("remask_fraction", 0.5),
        mask_token_id=mask_token_id,
    )

    print(decode_tokens(tokenizer, generated[0].cpu()))


if __name__ == "__main__":
    main()
