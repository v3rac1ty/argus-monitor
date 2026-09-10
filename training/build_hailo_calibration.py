"""Builds a calibration image directory for Hailo's INT8 post-training
quantization step (the Dataflow Compiler's `runner.optimize`, invoked via
`hailomz compile --calib-path ...` -- see docs/hailo-deployment.md), by
sampling N images (default 256), class-balanced and deterministic (seeded),
from `datasets/argus_bin/train`, and writing them resized/center-cropped
with the EXACT SAME geometry `argus.detectors.classifier.preprocess_classify`
(and `argus.detectors.hailo.preprocess_classify_hailo`) use at inference time
-- via the shared `argus.detectors.classifier.resize_and_center_crop` -- so
the images the compiler sees during calibration are geometrically identical
to what the deployed model will actually be fed, not merely "close enough."

Why class balance matters for quantization quality
----------------------------------------------------
Hailo's post-training quantization derives each layer's INT8 scale and
zero-point from the ACTIVATION STATISTICS this calibration set produces when
run through the fp32 graph -- those statistics are the compiler's only proxy
for "the range of values this layer will see in production." An unbalanced
calibration set (e.g. mostly `normal` images, which this dataset has roughly
2.5x more of than `failure` -- 1,102 vs 445 in `datasets/argus_bin/train`)
tunes those ranges to what a healthy print's activations look like, and
`failure`'s activations get clipped or under-resolved wherever they diverge.
That degrades the model's already-measured fp32 precision (0.988, see
runs/eval_bin_test.json) in a way this script cannot detect -- the loss only
shows up later, in the mandatory post-quantization threshold re-derivation
on-device (see docs/hailo-deployment.md). Balancing 50/50 by class is the
cheapest available defense against quietly baking that skew into the HEF.

Why determinism matters
-------------------------
A fixed seed (and sorting every file list before sampling from it) means
that if the compile step is ever re-run -- a different `--hw-arch`, a
Dataflow Compiler bugfix, a model update -- the SAME 256 images produce the
same quantization result. That makes any regression attributable to the
change that was actually made, not to an accidental change in which images
happened to get sampled this time.

Why `train`, never `val`/`test`
----------------------------------
Sampling only from `train` keeps calibration data out of the split that
docs/hailo-deployment.md's mandatory post-quantization threshold
re-derivation (`training/evaluate_classifier.py --split test`) depends on
for an honest number. If a test image had already shaped the model's own
INT8 quantization, precision measured on it afterward would be measuring
memorization of the calibration set, not generalization.

Usage: python training/build_hailo_calibration.py [--data-dir PATH] [--out PATH]
       [--n N] [--input-size N] [--seed N] [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

# Run-from-anywhere bootstrap (matching training/export_tinygrad_onnx.py): this module
# reuses argus.detectors.classifier.resize_and_center_crop, which is only importable once
# the repo root (and src/) are on sys.path -- pytest already does this via pyproject.toml's
# pythonpath/rootdir insertion; `python training/build_hailo_calibration.py` does not.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from argus.detectors.classifier import resize_and_center_crop  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "argus_bin" / "train"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "hailo_calib"
#: Hailo's own documentation suggests 1,024 images for best quantization results but
#: reports usable results from as few as 64-256; 256 is a practical default for a
#: two-class model that keeps compile time reasonable (see docs/hailo-deployment.md).
DEFAULT_N = 256
#: Must match the deployed model's real input size -- models/argus_bin.onnx (and the HEF
#: compiled from it) is exported static 320x320, see config.trident.yaml.
DEFAULT_INPUT_SIZE = 320
DEFAULT_SEED = 1337

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Pure functions -- listing, allocation, and selection. No I/O writes, no
# cv2 dependency, fully unit-testable with a synthetic directory tree.
# --------------------------------------------------------------------------


# dict[str, list[Path]] list_class_images(Path data_dir)
# Inputs: Path data_dir - a classification split directory containing one subdirectory per
#                 class (e.g. datasets/argus_bin/train/{failure,normal})
# Outputs: dict[str, list[Path]] - {class_name: sorted image paths}, keyed by every
#          subdirectory of data_dir, sorted by filename for determinism
# Description: Lists every image file directly inside each class subdirectory of data_dir. The
#              class set is derived from the directory listing itself (not hardcoded), so this
#              works unchanged for the current binary dataset and any future multi-class one.
# Side Effects: Raises FileNotFoundError if data_dir doesn't exist or contains no
#               subdirectories. Read-only filesystem traversal otherwise.
def list_class_images(data_dir: Path) -> dict[str, list[Path]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"no class subdirectories found under {data_dir}")

    images_by_class: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        images_by_class[class_dir.name] = images
    return images_by_class


# dict[str, int] allocate_per_class_counts(Sequence[str] class_names, int n)
# Inputs: Sequence[str] class_names - class names to allocate the budget across
#         int n - total number of calibration images wanted
# Outputs: dict[str, int] - {class_name: count}, counts summing to exactly n
# Description: Splits n as evenly as possible across class_names, sorted first so the
#              allocation is a pure function of (class_names, n) -- any remainder from integer
#              division goes to the earliest classes in sorted order, so re-running with the
#              same inputs always allocates identically.
# Side Effects: Raises ValueError if class_names is empty or n < 0.
def allocate_per_class_counts(class_names: Sequence[str], n: int) -> dict[str, int]:
    if not class_names:
        raise ValueError("class_names must be non-empty")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    ordered = sorted(class_names)
    num_classes = len(ordered)
    base, remainder = divmod(n, num_classes)
    return {name: base + (1 if i < remainder else 0) for i, name in enumerate(ordered)}


# dict[str, list[Path]] select_calibration_images(Mapping[str, Sequence[Path]] images_by_class, Mapping[str, int] counts, int seed)
# Inputs: Mapping[str, Sequence[Path]] images_by_class - available images per class (from
#                 list_class_images; each list assumed pre-sorted for determinism)
#         Mapping[str, int] counts - how many images to pick per class (from
#                 allocate_per_class_counts)
#         int seed - RNG seed; the same seed always picks the same images given the same inputs
# Outputs: dict[str, list[Path]] - {class_name: selected image paths}, len(value) == counts[key]
# Description: Deterministically samples counts[class] images from each class's available pool
#              via a single seeded random.Random processing classes in sorted order, so the
#              exact same calibration set is reproduced for the same (images_by_class, counts,
#              seed) -- see the module docstring for why that determinism matters.
# Side Effects: Raises ValueError if any class in counts has fewer available images than
#               requested (a silently smaller-than-requested or duplicated-image calibration set
#               would defeat the point of --n).
def select_calibration_images(
    images_by_class: Mapping[str, Sequence[Path]], counts: Mapping[str, int], seed: int
) -> dict[str, list[Path]]:
    rng = Random(seed)
    selected: dict[str, list[Path]] = {}
    for class_name in sorted(counts):
        count = counts[class_name]
        available = list(images_by_class.get(class_name, ()))
        if len(available) < count:
            raise ValueError(
                f"class '{class_name}' has only {len(available)} image(s) under the data "
                f"directory but {count} were requested for a balanced calibration set of this "
                "size -- lower --n or add more images for this class"
            )
        selected[class_name] = rng.sample(available, count)
    return selected


# --------------------------------------------------------------------------
# I/O -- reading source images and writing the calibration set (not unit
# tested directly; exercised via a real run against tiny synthetic images
# in tests).
# --------------------------------------------------------------------------


# list[Path] build_calibration_set(Path data_dir, Path out_dir, int n, int input_size, int seed, bool force=False)
# Inputs: Path data_dir - classification split directory to sample from (default:
#                 datasets/argus_bin/train)
#         Path out_dir - destination directory for the calibration images (default:
#                 datasets/hailo_calib)
#         int n - total number of calibration images to write
#         int input_size - square size to resize+center-crop every selected image to, matching
#                 the deployed model's real input size
#         int seed - RNG seed for deterministic, reproducible selection
#         bool force - if True, wipes an existing out_dir before writing; if False (default) and
#                 out_dir already exists, raises rather than silently mixing calibration sets
#                 from different runs
# Outputs: list[Path] - paths of every calibration image written, sorted by class then filename
# Description: Orchestrates the full pipeline: lists available images per class
#              (list_class_images), allocates the per-class budget (allocate_per_class_counts),
#              selects deterministically (select_calibration_images), then for each selected
#              image reads it, applies the exact production resize_and_center_crop geometry at
#              input_size, and writes it as a JPEG named "<class>__<original_stem>.jpg" (flat
#              directory, no subfolders -- matching what hailomz's --calib-path expects, and
#              keeping the source class visible in the filename for traceability without
#              requiring a label the Dataflow Compiler doesn't consume anyway).
# Side Effects: Raises FileExistsError if out_dir already exists and force=False. Deletes
#               out_dir (recursively) if it exists and force=True. Creates out_dir. Reads every
#               selected source image from disk via cv2.imread; raises IOError if any fails to
#               load. Writes one JPEG per selected image to out_dir.
def build_calibration_set(
    data_dir: Path,
    out_dir: Path,
    n: int,
    input_size: int,
    seed: int,
    force: bool = False,
) -> list[Path]:
    import cv2

    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"calibration output directory '{out_dir}' already exists -- pass --force to "
                "recreate it. Recreating it (rather than merging into it) matters because a "
                "stale directory could silently mix images from a previous run at a different "
                "--n/--seed/--input-size into what the compiler is told is one coherent "
                "calibration set."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_class = list_class_images(data_dir)
    counts = allocate_per_class_counts(list(images_by_class), n)
    selected = select_calibration_images(images_by_class, counts, seed)

    written: list[Path] = []
    for class_name in sorted(selected):
        # Sort the selection itself too: rng.sample's output order depends on
        # internal sampling mechanics, not just the seed+pool -- sorting
        # before writing keeps filenames (and therefore this function's
        # return value) a pure function of the *set* selected, not of
        # incidental sample ordering.
        for src in sorted(selected[class_name]):
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"failed to read calibration source image: {src}")
            cropped = resize_and_center_crop(image, input_size)
            dest = out_dir / f"{class_name}__{src.stem}.jpg"
            ok = cv2.imwrite(str(dest), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise IOError(f"cv2.imwrite failed for {dest}")
            written.append(dest)
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed --data-dir, --out, --n, --input-size, --seed, --force
# Description: Defines and parses the CLI for building the Hailo calibration image set.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Classification train-split directory to sample from (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for the calibration image set (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Total calibration images, class-balanced (default: {DEFAULT_N})")
    parser.add_argument(
        "--input-size", type=int, default=DEFAULT_INPUT_SIZE,
        help=f"Square size to resize+center-crop to, matching the deployed model's real input "
        f"size (default: {DEFAULT_INPUT_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed for deterministic selection (default: {DEFAULT_SEED})")
    parser.add_argument("--force", action="store_true", help="Overwrite --out if it already exists")
    return parser.parse_args(argv)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point: builds the calibration set per the parsed arguments and prints a
#              per-class breakdown of what was written.
# Side Effects: See build_calibration_set. Prints progress and a summary to stdout.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    written = build_calibration_set(
        args.data_dir, args.out, args.n, args.input_size, args.seed, force=args.force
    )
    print(
        f"[build_hailo_calibration] wrote {len(written)} calibration image(s) to {args.out} "
        f"(input_size={args.input_size}x{args.input_size}, seed={args.seed})"
    )
    by_class = Counter(p.name.split("__", 1)[0] for p in written)
    for class_name in sorted(by_class):
        print(f"  {class_name}: {by_class[class_name]}")


if __name__ == "__main__":
    main()
