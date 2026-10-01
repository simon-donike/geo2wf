"""Evaluate retained models or reproduce the fixed conference release."""

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
        choices=(
            "conference",
            "intensity-comparison",
            "intensity-correction",
            "intensity-forecast",
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.workflow == "conference":
        from scripts.conference_release import main as evaluate
    elif args.workflow == "intensity-comparison":
        from scripts.evaluate_intensity_models import main as evaluate
    elif args.workflow == "intensity-correction":
        from scripts.evaluate_intensity_correction import main as evaluate
    else:
        from scripts.evaluate_intensity_forecast import main as evaluate
    evaluate()
