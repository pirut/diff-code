from __future__ import annotations

import argparse
from pathlib import Path

import modal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a checkpoint directory from the Modal outputs volume.")
    parser.add_argument("--run-name", required=True, help="Remote run directory inside the Modal outputs volume.")
    parser.add_argument(
        "--remote-subdir",
        default="final",
        help="Subdirectory inside the run to download, for example final or step-25.",
    )
    parser.add_argument(
        "--volume-name",
        default="code-diffusion-outputs",
        help="Modal volume name that stores training outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional local destination. Defaults to ./outputs/modal_downloads/<run-name>/<remote-subdir>.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching files without downloading them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume = modal.Volume.from_name(args.volume_name, create_if_missing=False)

    remote_root = f"{args.run_name.strip('/')}/{args.remote_subdir.strip('/')}".strip("/")
    entries = volume.listdir(remote_root, recursive=True)
    files = [entry for entry in entries if getattr(entry.type, "name", "") != "DIRECTORY"]
    if not files:
        raise FileNotFoundError(f"No files found in Modal volume path: {remote_root}")

    if args.list_only:
        for entry in files:
            print(f"{entry.path}\t{entry.size}")
        return

    output_dir = Path(args.output_dir or f"./outputs/modal_downloads/{args.run_name}/{args.remote_subdir}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for entry in files:
        relative = Path(entry.path).relative_to(remote_root)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            volume.read_file_into_fileobj(entry.path, handle)
        downloaded += 1
        print(destination)

    print(f"downloaded_files={downloaded}")
    print(f"local_checkpoint_dir={output_dir}")


if __name__ == "__main__":
    main()
