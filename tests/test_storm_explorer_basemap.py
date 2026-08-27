from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_uses_only_the_keyless_openstreetmap_basemap() -> None:
    app = (ROOT / "docs/explorer/app.js").read_text(encoding="utf-8")
    tile_urls = re.findall(r'L\.tileLayer\("([^"]+)"', app)

    assert tile_urls == ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"]
    assert "OpenStreetMap contributors" in app


def test_dashboard_cache_busts_the_keyless_basemap_release() -> None:
    dashboard = (ROOT / "docs/explorer/dashboard.html").read_text(encoding="utf-8")

    assert 'href="styles.css?v=20260827-2"' in dashboard
    assert 'src="app.js?v=20260827-2"' in dashboard
