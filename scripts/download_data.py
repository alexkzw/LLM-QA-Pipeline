#!/usr/bin/env python3
"""Standalone CLI: download the reference document(s) this project needs.

One job: fetch the reference PDF into ``data/`` so a freshly-cloned repo can run
without committing large binaries to git. This keeps the repository lean while
making setup a single, reproducible command.

The download is:
  * idempotent  - skips the file if it already exists (use --force to refetch),
  * streamed    - written to disk in chunks, so large files don't exhaust memory,
  * verified    - checks the downloaded size, and an optional SHA-256 checksum,
  * atomic      - writes to a temp file and renames on success, so an
                  interrupted download never leaves a half-written PDF behind.

Examples:
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --dest data/
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm_qa.core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataAsset:
    """A file to download, with optional integrity metadata."""

    filename: str
    url: str
    # Optional SHA-256 of the expected file. If set, the download is rejected
    # unless it matches. Leave as None to skip checksum verification.
    sha256: str | None = None
    # Minimum plausible size in bytes; guards against error pages saved as PDFs.
    min_bytes: int = 100_000


# The reference document(s) this project uses. The OECD Economic Outlook is
# published under CC BY 4.0. Update the URL if OECD changes their hosting path;
# the value below is the document's stable publication landing page download.
REQUIRED_ASSETS: list[DataAsset] = [
    DataAsset(
        filename="oecd_outlook_2026.pdf",
        # NOTE: confirm this resolves to the actual PDF before relying on it in
        # CI. OECD publication URLs occasionally change; if it 404s, grab the
        # current link from https://doi.org/10.1787/2d1956f0-en and update here.
        url="https://www.oecd.org/economic-outlook/june-2026/",
        # Known SHA-256 of the reference copy used to build this project:
        #   2abb06218955d9b705b46edcc2cf74190fd8fa3954fda785232dcf908760d61a
        # Left as None (not enforced) because the URL above is a landing page,
        # not a confirmed direct-PDF link;
        sha256=None,
        min_bytes=1_000_000,
    ),
]


def _looks_like_pdf(path: Path) -> bool:
    """Cheap sanity check: real PDFs start with the %PDF- magic bytes."""
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(asset: DataAsset, dest_dir: Path, force: bool) -> bool:
    """Download a single asset. Returns True on success, False on failure."""
    target = dest_dir / asset.filename

    if target.exists() and not force:
        logger.info("%s already present; skipping (use --force to refetch).", target)
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", asset.url, target)

    # Stream to a temp file in the same directory, then atomically rename.
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest_dir, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        request = urllib.request.Request(
            asset.url,
            headers={"User-Agent": "llm-qa-pipeline/1.0 (data downloader)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.length or 0
            downloaded = 0
            with open(tmp_fd, "wb") as out:
                while True:
                    chunk = response.read(1 << 16)  # 64 KB
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r  {downloaded:,}/{total:,} bytes ({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
        if total:
            print()  # newline after the progress line
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error("Download failed for %s: %s", asset.url, exc)
        return False

    # --- Integrity checks before committing the file --------------------
    size = tmp_path.stat().st_size
    if size < asset.min_bytes:
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "Downloaded file is only %d bytes (expected >= %d). "
            "The URL may be returning an error page rather than the PDF.",
            size,
            asset.min_bytes,
        )
        return False

    if not _looks_like_pdf(tmp_path):
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "Downloaded file is not a valid PDF (missing %%PDF- header). "
            "Check the URL: %s",
            asset.url,
        )
        return False

    if asset.sha256:
        actual = _sha256(tmp_path)
        if actual != asset.sha256:
            tmp_path.unlink(missing_ok=True)
            logger.error(
                "Checksum mismatch for %s.\n  expected: %s\n  actual:   %s",
                asset.filename,
                asset.sha256,
                actual,
            )
            return False
        logger.info("Checksum verified for %s.", asset.filename)

    tmp_path.replace(target)  # atomic on the same filesystem
    logger.info("Saved %s (%d bytes).", target, size)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data"),
        help="Destination directory for downloaded files (default: data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", json_output=False)

    results = [_download(asset, args.dest, args.force) for asset in REQUIRED_ASSETS]

    succeeded = sum(results)
    total = len(results)
    if succeeded == total:
        logger.info("All %d asset(s) ready in '%s'.", total, args.dest)
        return 0

    logger.error("%d of %d asset(s) failed to download.", total - succeeded, total)
    logger.error(
        "If the URL has changed, get the current link from "
        "https://doi.org/10.1787/2d1956f0-en and update REQUIRED_ASSETS, "
        "or download the PDF manually into '%s'.",
        args.dest,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())