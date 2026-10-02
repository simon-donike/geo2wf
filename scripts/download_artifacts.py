#!/usr/bin/env python3
"""Download the pinned public geo2wf release using the Hugging Face CLI."""

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


# Published together on 2 October 2026. The data commit is recorded in the
# model repository's dataset-links.json at MODEL_REVISION.
MODEL_REPO = "simon-donike/geo2wf-models"
MODEL_REVISION = "b4399d426b80698d4cf73a77c9541071a8ab3d42"
DATA_REPO = "simon-donike/geo2wf-data"
DATA_REVISION = "043c7f23e5f0a1fbda7034c342113664fb6aa81e"

DATA_METADATA = [
    "README.md",
    "ATTRIBUTION.md",
    "schema.json",
    "release.json",
    "SHA256SUMS",
]
MODEL_METADATA = [
    "README.md",
    "ATTRIBUTION.md",
    "dataset-links.json",
    "SHA256SUMS",
    "release/registry.json",
    "release/hosting/README.md",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="?",
        default="metadata",
        choices=("metadata", "data", "models", "all"),
        help="metadata (default): release guides/inventories only; data: full dataset; "
        "models: full model release including matching source; all: both releases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="destination parent, relative to the working directory (default: downloads)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ask the Hub for file sizes and cached status without downloading payloads",
    )
    args = parser.parse_args(argv)
    hf = shutil.which("hf")
    if hf is None:
        parser.exit(
            2,
            "Missing Hugging Face CLI. Install with: uv tool install huggingface_hub\n",
        )

    releases = []
    if args.profile in ("metadata", "data", "all"):
        releases.append(("data", DATA_REPO, DATA_REVISION, DATA_METADATA))
    if args.profile in ("metadata", "models", "all"):
        releases.append(("models", MODEL_REPO, MODEL_REVISION, MODEL_METADATA))

    for name, repo, revision, metadata in releases:
        destination = args.output_dir.expanduser().resolve() / name
        command = [hf, "download", repo]
        if args.profile == "metadata":
            command.extend(metadata)
        command.extend(
            [
                "--repo-type",
                "dataset" if name == "data" else "model",
                "--revision",
                revision,
                "--local-dir",
                str(destination),
            ]
        )
        if args.dry_run:
            command.append("--dry-run")
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                "Download failed; rerun the same command to retry. "
                "See the Hugging Face error above.",
                file=sys.stderr,
            )
            return exc.returncode if exc.returncode > 0 else 1

    if not args.dry_run:
        print(f"Files saved under {args.output_dir.expanduser().resolve()}")
        if args.profile == "metadata":
            print("Metadata only: no imagery or checkpoint weights downloaded.")
        print(
            "For catalog loading and paper reproduction, use the matching source "
            "archive in the model release. See docs/data/index.md."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
