"""Check local image references in README and docs markdown files."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = sorted(
    path
    for path in REPO_ROOT.rglob("*.md")
    if ".git" not in path.parts
)

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)


def is_external_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def normalize_markdown_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return target
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split()[0]


def resolve_local_path(markdown_file: Path, target: str) -> Path | None:
    if is_external_url(target):
        return None

    parsed = urlparse(target)
    if parsed.scheme and parsed.scheme not in {"", "file"}:
        return None

    clean_target = unquote(parsed.path or target)
    if not clean_target:
        return None

    path = Path(clean_target)
    if path.is_absolute():
        return REPO_ROOT / path.relative_to("/") if not path.exists() else path
    return markdown_file.parent / path


def iter_image_targets(markdown_file: Path) -> list[str]:
    text = markdown_file.read_text()
    targets = [normalize_markdown_target(match.group(1)) for match in MARKDOWN_IMAGE_RE.finditer(text)]
    targets.extend(match.group(1).strip() for match in HTML_IMAGE_RE.finditer(text))
    return targets


def main() -> int:
    missing: list[str] = []

    for markdown_file in MARKDOWN_FILES:
        for target in iter_image_targets(markdown_file):
            resolved = resolve_local_path(markdown_file, target)
            if resolved is None:
                continue
            if not resolved.exists():
                relative_markdown = markdown_file.relative_to(REPO_ROOT)
                missing.append(f"{relative_markdown}: {target} -> {resolved}")

    if missing:
        print("Missing local markdown image assets:", file=sys.stderr)
        for item in missing:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(f"Checked local image assets in {len(MARKDOWN_FILES)} markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
