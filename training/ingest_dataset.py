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


# bool is_already_extracted(Path output_dir)
# Inputs: Path output_dir - directory to check for a prior extraction
# Outputs: bool - True if output_dir already contains a Roboflow data.yaml manifest
# Description: Best-effort check for whether the archive has already been extracted into
#              output_dir, used to make extraction idempotent.
# Side Effects: None (read-only filesystem check)
def is_already_extracted(output_dir: Path) -> bool:
    return (output_dir / "data.yaml").is_file()


# None extract_archive(Path archive_path, Path output_dir)
# Inputs: Path archive_path - path to the Roboflow YOLOv8 zip export
#         Path output_dir - directory to extract the archive into
# Outputs: None
# Description: Extracts every entry in the zip archive at archive_path into output_dir,
#              creating it if needed.
# Side Effects: Creates output_dir (and parents) if missing; extracts all files from the
#               zip archive onto disk under output_dir.
def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_dir)


# dict[str, object] summarize(Path output_dir)
# Inputs: Path output_dir - root of the extracted dataset tree to walk
# Outputs: dict[str, object] - "top_level_files" (sorted file names at the root) and
#          "splits" (per top-level split directory: image count, label count, and a
#          per-extension image breakdown)
# Description: Walks the extracted tree and counts images/labels per top-level split
#              directory, for the human-readable summary printed after extraction.
# Side Effects: None (read-only filesystem traversal)
def summarize(output_dir: Path) -> dict[str, object]:
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


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed --archive, --output, and --force values
# Description: Defines and parses the CLI arguments for this ingest script.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
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


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Parses args, extracts the archive into datasets/raw/ unless
#              it already looks populated (or --force is given), then prints a summary
#              including a note about the expected-missing test/ directory (Defect 2).
# Side Effects: Raises FileNotFoundError if the archive is missing; extracts the zip archive
#               to disk (via extract_archive) unless already extracted and --force not given;
#               prints ingest progress and a summary table to stdout.
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
