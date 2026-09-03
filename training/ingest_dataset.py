"""Extract the raw Roboflow YOLOv8 export archive to ``datasets/raw/``.

Idempotent: if ``datasets/raw/`` already looks populated (contains
``data.yaml``), extraction is skipped unless ``--force`` is passed. This is
purely an unpacking step -- it does not touch the train/valid split or fix
the two known defects in the archive (train/valid leakage, a nonexistent
``test/`` directory referenced by ``data.yaml``). Those are handled by
``training/prepare_dataset.py``.

Usage:
    python training/ingest_dataset.py
    python training/ingest_dataset.py --archive "C:\\path\\to\\archive.zip" --force
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "3D printing error.v7i.yolov8.zip"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "raw"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_already_extracted(output_dir: Path) -> bool:
    """Best-effort check for whether the archive has already been extracted
    into ``output_dir`` (looks for the Roboflow ``data.yaml`` manifest)."""
    return (output_dir / "data.yaml").is_file()


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract every entry in the zip archive at ``archive_path`` into
    ``output_dir``, creating it if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_dir)


def summarize(output_dir: Path) -> dict[str, object]:
    """Walk the extracted tree and count images/labels per top-level split
    directory, for the human-readable summary printed after extraction."""
    summary: dict[str, object] = {}
    top_level_files = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    summary["top_level_files"] = top_level_files

    split_counts: dict[str, dict[str, int]] = {}
    for split_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        n_images = 0
        ext_counts: Counter[str] = Counter()
        if images_dir.is_dir():
            for p in images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    n_images += 1
                    ext_counts[p.suffix.lower()] += 1
        n_labels = 0
        if labels_dir.is_dir():
            n_labels = sum(1 for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        split_counts[split_dir.name] = {
            "images": n_images,
            "labels": n_labels,
            **{f"images{ext}": c for ext, c in ext_counts.items()},
        }
    summary["splits"] = split_counts
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Path to the Roboflow YOLOv8 zip export (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory to extract into (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the output directory already looks populated",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive_path: Path = args.archive
    output_dir: Path = args.output

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if is_already_extracted(output_dir) and not args.force:
        print(f"[ingest] '{output_dir}' already contains an extracted dataset (data.yaml found).")
        print("[ingest] Skipping extraction (pass --force to re-extract).")
    else:
        print(f"[ingest] Extracting '{archive_path}' -> '{output_dir}' ...")
        extract_archive(archive_path, output_dir)
        print("[ingest] Extraction complete.")

    summary = summarize(output_dir)
    print()
    print("=" * 60)
    print("INGEST SUMMARY")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print(f"Top-level files: {summary['top_level_files']}")
    for split_name, counts in summary["splits"].items():  # type: ignore[union-attr]
        print(f"  {split_name}: {counts}")
    if not (output_dir / "test").is_dir():
        print()
        print("[ingest] NOTE: no 'test/' directory in the archive (data.yaml references one that")
        print("[ingest]       does not exist -- this is Defect 2). training/prepare_dataset.py")
        print("[ingest]       builds a real, leak-free test split from scratch, so this is expected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
