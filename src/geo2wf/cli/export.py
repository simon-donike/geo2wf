"""Run one of the maintained dataset export workflows."""

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
        "dataset",
        choices=(
            "geo-sar",
            "geo-pmw",
            "intensity-cache",
            "intensity-forecast-cache",
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.dataset == "geo-sar":
        from scripts.export_geo_sar_geotiffs import main as export_main
    elif args.dataset == "geo-pmw":
        from scripts.export_geo_pmw_geotiffs import main as export_main
    elif args.dataset == "intensity-cache":
        from scripts.export_unet_intensity_cache import main as export_main
    else:
        from scripts.export_intensity_forecast_cache import main as export_main
    export_main()
