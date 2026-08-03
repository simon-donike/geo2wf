"""Run one of the maintained storm-inference workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=("deterministic-residual", "residual-diffusion"),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.workflow == "deterministic-residual":
        from scripts.run_storm_unet_inference import main as inference_main
    else:
        from scripts.run_storm_diffusion_inference import main as inference_main
    inference_main()
