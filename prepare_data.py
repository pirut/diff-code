from __future__ import annotations

import argparse
import json

from code_diffusion.config import load_config
from code_diffusion.data.public_corpus import prepare_public_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a trainable public code corpus.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value with key=value. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory for the prepared corpus.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep the existing output directory instead of recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    summary = prepare_public_corpus(
        config=config,
        output_dir=args.output_dir,
        clean_output=not args.keep_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
