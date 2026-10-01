"""Check local links and asset references in the built MkDocs website."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    root = Path(__file__).resolve().parents[1] / "site"
    missing = []
    count = 0

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if not (
                    (tag in ("a", "link") and key == "href")
                    or (tag in ("img", "script") and key == "src")
                ):
                    continue
                parsed = urlsplit(value or "")
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                relative = unquote(parsed.path)
                target = (
                    root / relative.lstrip("/")
                    if relative.startswith("/")
                    else self.current.parent / relative
                )
                if not target.exists():
                    missing.append((str(self.current.relative_to(root)), value))

    for page in root.rglob("*.html"):
        # MkDocs copies the custom-theme source templates as static files too.
        if "overrides" in page.relative_to(root).parts:
            continue
        parser = Links()
        parser.current = page
        parser.feed(page.read_text())
        count += 1
    if not count:
        raise SystemExit("No built pages found; run mkdocs build first")
    for page, value in missing:
        print(f"{page}: missing {value}")
    print(f"Checked {count} built pages; {len(missing)} missing local targets")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
