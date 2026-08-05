"""Evaluate a checkpoint with the shared evaluation workflow."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "intensity-forecast":
        sys.argv.pop(1)
        from scripts.evaluate_intensity_forecast import main as forecast_main

        forecast_main()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "intensity-correction":
        sys.argv.pop(1)
        from scripts.evaluate_intensity_correction import main as intensity_main

        intensity_main()
        return
    from scripts.evaluate_checkpoint import main as evaluate_main

    evaluate_main()
