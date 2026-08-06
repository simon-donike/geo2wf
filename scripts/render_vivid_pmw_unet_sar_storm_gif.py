#!/usr/bin/env python3
"""Render direct-PMW and local U-Net wind predictions with vivid PMW colors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.render_pmw_unet_sar_storm_gif as direct_renderer


PMW_STRETCH_MIN_K = 220.0
PMW_STRETCH_MAX_K = 283.0
PMW_LOW = np.array([48, 0, 112], dtype=np.float32)
PMW_MID = np.array([232, 0, 128], dtype=np.float32)
PMW_HIGH = np.array([255, 255, 0], dtype=np.float32)


def main() -> None:
    renderer = direct_renderer.renderer
    renderer.TMIN = PMW_STRETCH_MIN_K
    renderer.TMAX = PMW_STRETCH_MAX_K
    renderer.PMW_LOW = PMW_LOW
    renderer.PMW_MID = PMW_MID
    renderer.PMW_HIGH = PMW_HIGH
    direct_renderer.main()


if __name__ == "__main__":
    main()
