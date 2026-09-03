"""Build a clean, leak-free dataset at ``datasets/argus/`` from the raw
Roboflow export at ``datasets/raw/`` (see ``training/ingest_dataset.py``).

Fixes both known defects in the raw archive:

Defect 1 (train/valid leakage): Roboflow generated 3 augmented variants per
source photo and scattered them across train/valid, so ~47% of the shipped
validation set is material the model already trained on. This script pools
every image from ``raw/train`` and ``raw/valid`` together, discards the
shipped split entirely, recovers each image's *source identity* (the part of
the filename before the Roboflow ``.rf.<hex>`` suffix), and re-splits by
source -- never by file -- so all variants of one source photo land in
exactly one of train/val/test.

Defect 2 (missing ``test/`` dir): the raw ``data.yaml`` points at a
``test/`` directory that was never included in the export. This script
generates a real held-out test split (from sources never touched by
train/val) and writes a fresh, correct ``data.yaml``.

The train split keeps ALL variants of its sources (more training data). The
val and test splits keep exactly ONE variant per source (deterministic:
first in sorted filename order) -- evaluating on all 3 near-duplicate
variants of one photo would triple eval cost, correlate errors between
variants, and make the resulting precision/recall numbers less honest.

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --raw datasets/raw --out datasets/argus --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "argus"

#: Class order the model is trained/evaluated with -- must match
#: raw data.yaml and src/argus/detectors/onnx_yolo.py:DEFAULT_CLASS_NAMES.
CLASS_NAMES: tuple[str, ...] = ("error extrusion", "spaghetti", "stringing", "warping", "zits")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Matches Roboflow's "<source>.rf.<32-char-hex-ish>.<ext>" filenames, e.g.
# "00001_error_dataset_jpeg.rf.549c891f051b38ac8300be431e761f8c.jpg".
# Everything before ".rf." is the source identity; the hex chunk after it is
# just the augmentation-variant id and is discarded.
SOURCE_ID_RE = re.compile(
    r"^(?P<src>.+)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)

DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


# --------------------------------------------------------------------------
# Pure logic (unit-tested in tests/test_prepare_dataset.py without touching
# the real dataset on disk)
# --------------------------------------------------------------------------


def extract_source_id(filename: str) -> tuple[str, bool]:
    """Recover the source-photo identity from an image filename.

    Strips the Roboflow ``.rf.<hex>`` augmentation-variant suffix, e.g.
    ``"00001_x.rf.549c89....jpg"`` -> ``"00001_x"``. If the filename doesn't
    match that pattern, the whole stem (filename minus extension) is used as
    its own source id instead, and ``matched`` is False so callers can count
    how often this fallback fires.

    Returns ``(source_id, matched)``.
    """
    m = SOURCE_ID_RE.match(filename)
    if m:
        return m.group("src"), True
    stem = Path(filename).stem
    return stem, False


def group_by_source(filenames: Iterable[str]) -> tuple[dict[str, list[str]], int]:
    """Group filenames by source identity.

    Returns ``(groups, fallback_count)`` where ``groups`` maps source id ->
    sorted list of filenames belonging to that source (all its augmentation
    variants), and ``fallback_count`` is how many filenames did not match the
    ``.rf.<hex>`` pattern (see ``extract_source_id``).

    Filenames are sorted (both across and within groups) so downstream
    "first file for this source" selection is deterministic regardless of
    filesystem iteration order.
    """
    groups: dict[str, list[str]] = {}
    fallback_count = 0
    for fn in sorted(filenames):
        src, matched = extract_source_id(fn)
        if not matched:
            fallback_count += 1
        groups.setdefault(src, []).append(fn)
    return groups, fallback_count


def split_sources(
    source_ids: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split source ids into (train, val, test) lists.

    Splits by SOURCE, never by file: this is what guarantees no augmentation
    variant of a training-set photo can leak into val/test. Deterministic
    for a given seed: sort the source ids first (so input order never
    matters), then shuffle with a seeded ``random.Random`` before slicing by
    ratio -- so the same seed always reproduces the same split.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios!r} (sum={sum(ratios)})")

    ids = sorted(source_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    # Give the remainder to test so rounding never invents or drops sources.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return train_ids, val_ids, test_ids


def select_files_for_split(
    groups: dict[str, list[str]],
    source_ids: Sequence[str],
    split: str,
) -> dict[str, list[str]]:
    """Choose which files, per source, belong in a given split.

    ``train`` keeps every variant of each of its sources (maximizes training
    data). ``val``/``test`` keep exactly one variant per source -- the first
    in sorted filename order, i.e. ``groups[source][0]`` since
    ``group_by_source`` already sorts each source's file list -- because the
    3 variants are near-duplicates of one photo and evaluating on all of
    them would triple eval cost and correlate errors between variants.

    Returns a dict mapping source id -> list of selected filenames (for that
    source), restricted to ``source_ids``.
    """
    selected: dict[str, list[str]] = {}
    for sid in source_ids:
        files = groups[sid]
        selected[sid] = list(files) if split == "train" else [files[0]]
    return selected


def assert_no_source_overlap(split_source_ids: dict[str, Sequence[str]]) -> None:
    """Raise ``AssertionError`` if any source id appears in more than one
    split. This is the whole point of splitting by source instead of by
    file, so it's checked explicitly rather than just trusted."""
    seen: dict[str, str] = {}
    for split_name, ids in split_source_ids.items():
        for sid in ids:
            if sid in seen:
                raise AssertionError(
                    f"Source-identity leakage detected: source {sid!r} appears in both "
                    f"'{seen[sid]}' and '{split_name}' splits."
                )
            seen[sid] = split_name


def parse_yolo_label_classes(label_text: str) -> list[int]:
    """Parse a YOLO-format label file's contents and return the class id of
    every bounding-box line (first whitespace-separated token per line).
    Blank lines are ignored; an empty/whitespace-only file legitimately means
    "no objects" and yields an empty list."""
    class_ids: list[int] = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(float(line.split()[0])))
    return class_ids


# --------------------------------------------------------------------------
# Label-row normalization: the raw export mixes YOLO *detection* rows
# ("cls cx cy w h", 5 fields) with *segmentation/polygon* rows
# ("cls x1 y1 x2 y2 ... xn yn", odd field count > 5). Ultralytics refuses any
# label file that mixes the two formats and drops the WHOLE IMAGE, not just
# the offending row. normalize_label_text() rewrites every row to plain
# detection format so no image is lost to this.
# --------------------------------------------------------------------------


def clamp01(value: float) -> float:
    """Clamp a normalized coordinate to the valid [0, 1] range."""
    return max(0.0, min(1.0, value))


@dataclass
class LabelNormalizeStats:
    polygon_rows_converted: int = 0
    malformed_rows_skipped: int = 0
    degenerate_rows_dropped: int = 0
    contains_polygon_row: bool = False


def _row_corner_points(fields: list[str]) -> tuple[int, list[float], list[float]]:
    """Parse one label line's whitespace-split ``fields`` into
    ``(cls_id, xs, ys)`` -- the point list used to derive an axis-aligned
    bounding box.

    A detection row (5 fields: ``cls cx cy w h``) is converted to its two
    bounding corners so it can be run through the exact same clamp/bbox math
    as a polygon row. A segment row (odd field count > 5: ``cls x1 y1 x2 y2
    ... xn yn``) uses its polygon points directly.

    Raises ``ValueError`` if any field isn't numeric -- callers treat that as
    a malformed row (skip and count, don't crash).
    """
    cls_id = int(float(fields[0]))
    nums = [float(v) for v in fields[1:]]
    if len(fields) == 5:
        cx, cy, w, h = nums
        xs = [cx - w / 2, cx + w / 2]
        ys = [cy - h / 2, cy + h / 2]
    else:
        xs = nums[0::2]
        ys = nums[1::2]
    return cls_id, xs, ys


def normalize_label_text(label_text: str) -> tuple[str, LabelNormalizeStats]:
    """Rewrite every row of a YOLO label file's contents to plain 5-field
    detection format (``cls cx cy w h``), 6 decimal places.

    - A 5-field row (``cls cx cy w h``) is treated as its two corner points,
      clamped, and re-emitted -- a no-op in meaning for well-formed input.
    - A row with an odd field count > 5 (``cls x1 y1 ... xn yn``) is a
      segment/polygon row: its points are clamped to [0, 1] and collapsed to
      the axis-aligned box ``cx=(min_x+max_x)/2, cy=(min_y+max_y)/2,
      w=max_x-min_x, h=max_y-min_y``.
    - A row with an even field count > 5 has an incomplete trailing
      coordinate pair -- malformed; skipped and counted, not raised.
    - A row with fewer than 5 fields, or any non-numeric field, is likewise
      malformed; skipped and counted.
    - A box that's degenerate after clamping (``w <= 0`` or ``h <= 0``) is
      dropped silently (not counted as malformed -- it was a structurally
      valid row that just clamps away to nothing).
    - Blank lines are ignored (an empty file legitimately means "no
      objects" and stays empty).

    Returns ``(normalized_text, stats)``.
    """
    stats = LabelNormalizeStats()
    out_lines: list[str] = []

    for raw_line in label_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        n = len(fields)
        is_polygon_shaped = n > 5  # odd (valid polygon) or even (malformed) -- both are "not plain detection"

        if n < 5 or (n > 5 and n % 2 == 0):
            stats.malformed_rows_skipped += 1
            if is_polygon_shaped:
                stats.contains_polygon_row = True
            continue

        try:
            cls_id, xs, ys = _row_corner_points(fields)
        except ValueError:
            stats.malformed_rows_skipped += 1
            continue

        if is_polygon_shaped:
            stats.contains_polygon_row = True

        xs = [clamp01(v) for v in xs]
        ys = [clamp01(v) for v in ys]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 or h <= 0:
            stats.degenerate_rows_dropped += 1
            continue

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        if is_polygon_shaped:
            stats.polygon_rows_converted += 1
        out_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    normalized_text = ("\n".join(out_lines) + "\n") if out_lines else ""
    return normalized_text, stats


# --------------------------------------------------------------------------
# Class-subset / single-class filtering (--classes / --single-class)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassFilter:
    """Which original class ids survive, and what they're remapped to.

    ``keep_ids``: original (0..len(CLASS_NAMES)-1) class ids that survive.
    ``id_remap``: original class id -> output class id. Rows whose class id
        isn't a key here are dropped.
    ``names``: output ``names`` list, indexed by output class id.
    ``single_class``: whether all kept classes were collapsed to id 0.
    """

    keep_ids: frozenset[int]
    id_remap: dict[int, int]
    names: list[str]
    single_class: bool


def default_class_filter(class_names: Sequence[str] = CLASS_NAMES) -> ClassFilter:
    """Identity filter: keep every class, no remap, no collapsing. Used when
    ``--classes``/``--single-class`` aren't passed, so downstream code has a
    single code path regardless of whether filtering is active."""
    ids = range(len(class_names))
    return ClassFilter(
        keep_ids=frozenset(ids),
        id_remap={i: i for i in ids},
        names=list(class_names),
        single_class=False,
    )


def resolve_class_filter(
    classes: Sequence[str] | None,
    single_class: bool,
    class_names: Sequence[str] = CLASS_NAMES,
) -> ClassFilter:
    """Build the class-id remap for ``--classes``/``--single-class``.

    ``classes=None`` (or empty) means "keep every class". Requested names are
    matched against ``class_names`` case-insensitively; an unknown name
    raises ``ValueError``. Kept ids are always ordered by their ORIGINAL id
    (not CLI order), so e.g. ``--classes warping,spaghetti`` and
    ``--classes spaghetti,warping`` produce the same remap.

    If ``single_class`` is set, every kept id maps to output id 0 and the
    output ``names`` list is a single entry joining the kept names with
    ``"+"``. Otherwise kept ids are remapped to a contiguous ``0..k-1``
    range in original-id order.
    """
    if classes:
        lower_to_id = {name.lower(): i for i, name in enumerate(class_names)}
        requested_ids: set[int] = set()
        for raw in classes:
            key = raw.strip().lower()
            if not key:
                continue
            if key not in lower_to_id:
                raise ValueError(f"Unknown class {raw!r} in --classes; valid class names: {list(class_names)}")
            requested_ids.add(lower_to_id[key])
        if not requested_ids:
            raise ValueError("--classes was given but no valid class names were found in it")
        keep_ids_ordered = sorted(requested_ids)
    else:
        keep_ids_ordered = list(range(len(class_names)))

    keep_ids = frozenset(keep_ids_ordered)

    if single_class:
        id_remap = {cid: 0 for cid in keep_ids_ordered}
        names = ["+".join(class_names[cid] for cid in keep_ids_ordered)]
    else:
        id_remap = {cid: new_id for new_id, cid in enumerate(keep_ids_ordered)}
        names = [class_names[cid] for cid in keep_ids_ordered]

    return ClassFilter(keep_ids=keep_ids, id_remap=id_remap, names=names, single_class=single_class)


def apply_class_filter(label_text: str, class_filter: ClassFilter) -> str:
    """Drop rows whose class isn't kept by ``class_filter`` and remap
    surviving rows' class id per ``class_filter.id_remap``.

    Assumes ``label_text`` is already normalized to 5-field detection rows
    (see ``normalize_label_text``); non-numeric or otherwise unparseable
    leading tokens are dropped defensively rather than raising.
    """
    out_lines: list[str] = []
    for raw_line in label_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            cid = int(float(fields[0]))
        except (ValueError, IndexError):
            continue
        new_id = class_filter.id_remap.get(cid)
        if new_id is None:
            continue
        out_lines.append(" ".join([str(new_id), *fields[1:]]))
    return ("\n".join(out_lines) + "\n") if out_lines else ""


# --------------------------------------------------------------------------
# I/O layer: scanning the raw export, copying files, writing outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawImage:
    filename: str
    image_path: Path
    label_path: Path  # may or may not exist on disk


def scan_raw_split(raw_root: Path, raw_split_dir: str) -> list[RawImage]:
    """List every image in ``raw_root/raw_split_dir/images`` (e.g. 'train' or
    'valid'), paired with its expected label path (which may not exist)."""
    images_dir = raw_root / raw_split_dir / "images"
    labels_dir = raw_root / raw_split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected raw images directory not found: {images_dir}. "
            "Did you run training/ingest_dataset.py first?"
        )
    out: list[RawImage] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(RawImage(filename=p.name, image_path=p, label_path=labels_dir / (p.stem + ".txt")))
    return out


def pool_raw_images(raw_root: Path) -> tuple[dict[str, RawImage], int]:
    """Pool images from raw/train and raw/valid into a single filename ->
    RawImage map (the shipped split is discarded here per Defect 1).

    A raw split subdirectory (e.g. ``raw_root/valid``) that doesn't exist at
    all is skipped rather than treated as an error -- this keeps the
    function usable on small synthetic fixtures (e.g. in tests) that only
    populate one of the two. If the subdirectory exists but its ``images``
    folder is missing, that's a malformed extraction and still raises (via
    ``scan_raw_split``).

    Returns ``(pooled, duplicate_count)``; if the same filename somehow shows
    up in both raw split dirs, the first occurrence (train, scanned first)
    wins and the duplicate is dropped and counted.
    """
    pooled: dict[str, RawImage] = {}
    duplicates = 0
    for raw_split_dir in ("train", "valid"):
        if not (raw_root / raw_split_dir).is_dir():
            continue
        for img in scan_raw_split(raw_root, raw_split_dir):
            if img.filename in pooled:
                duplicates += 1
                continue
            pooled[img.filename] = img
    return pooled, duplicates


@dataclass
class SplitStats:
    class_names: tuple[str, ...] = CLASS_NAMES
    n_sources: int = 0
    n_images: int = 0
    n_missing_labels: int = 0
    polygon_rows_converted: int = 0
    malformed_rows_skipped: int = 0
    files_containing_polygons: int = 0
    instances_per_class: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instances_per_class:
            self.instances_per_class = {c: 0 for c in self.class_names}
        if not self.images_per_class:
            self.images_per_class = {c: 0 for c in self.class_names}

    def to_dict(self) -> dict[str, object]:
        return {
            "n_sources": self.n_sources,
            "n_images": self.n_images,
            "n_missing_labels": self.n_missing_labels,
            "polygon_rows_converted": self.polygon_rows_converted,
            "malformed_rows_skipped": self.malformed_rows_skipped,
            "files_containing_polygons": self.files_containing_polygons,
            "instances_per_class": self.instances_per_class,
            "images_per_class": self.images_per_class,
        }


def materialize_split(
    split: str,
    selected: dict[str, list[str]],
    pooled: dict[str, RawImage],
    out_dir: Path,
    class_filter: ClassFilter | None = None,
) -> SplitStats:
    """Copy the chosen images (+ matching labels, when present) for one
    split into ``out_dir/<split>/{images,labels}`` and tally stats.

    Every label file is normalized to plain detection rows (see
    ``normalize_label_text``) so polygon/segment rows in the source data
    survive instead of getting the whole image dropped by Ultralytics. If
    ``class_filter`` is given, rows are then filtered/remapped through it
    (see ``apply_class_filter``); images left with zero surviving labels are
    still kept -- they become legitimate background/negative images.
    """
    cf = class_filter if class_filter is not None else default_class_filter()

    images_out = out_dir / split / "images"
    labels_out = out_dir / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(class_names=tuple(cf.names), n_sources=len(selected))

    for sid in sorted(selected):
        for filename in selected[sid]:
            raw_img = pooled[filename]
            stats.n_images += 1

            shutil.copy2(raw_img.image_path, images_out / filename)

            label_dst = labels_out / (Path(filename).stem + ".txt")
            if not raw_img.label_path.is_file():
                print(f"[prepare] WARNING: no label file for image '{filename}' (split={split}); skipping label copy.")
                stats.n_missing_labels += 1
                continue

            raw_text = raw_img.label_path.read_text(encoding="utf-8")
            normalized_text, norm_stats = normalize_label_text(raw_text)
            stats.polygon_rows_converted += norm_stats.polygon_rows_converted
            stats.malformed_rows_skipped += norm_stats.malformed_rows_skipped
            if norm_stats.contains_polygon_row:
                stats.files_containing_polygons += 1

            # Out-of-range sanity check against the FULL original class
            # list (independent of any --classes filtering below).
            for cid in parse_yolo_label_classes(normalized_text):
                if not (0 <= cid < len(CLASS_NAMES)):
                    print(f"[prepare] WARNING: label '{label_dst.name}' has out-of-range class id {cid}; ignoring.")

            final_text = apply_class_filter(normalized_text, cf)
            label_dst.write_text(final_text, encoding="utf-8")

            class_ids = parse_yolo_label_classes(final_text)
            classes_in_image: set[int] = set()
            for cid in class_ids:
                if 0 <= cid < len(cf.names):
                    stats.instances_per_class[cf.names[cid]] += 1
                    classes_in_image.add(cid)
            for cid in classes_in_image:
                stats.images_per_class[cf.names[cid]] += 1

    return stats


def write_data_yaml(out_dir: Path, class_names: Sequence[str] = CLASS_NAMES) -> Path:
    """Write datasets/argus/data.yaml with correct train/val/test paths
    (fixing Defect 2: the raw data.yaml pointed at a test/ dir that didn't
    exist) and the active class name order (the canonical 5 classes, unless
    narrowed by --classes/--single-class)."""
    data_yaml_path = out_dir / "data.yaml"
    names_literal = ", ".join(f"'{c}'" for c in class_names)
    content = (
        f"# Generated by training/prepare_dataset.py -- do not hand-edit.\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_literal}]\n"
    )
    data_yaml_path.write_text(content, encoding="utf-8")
    return data_yaml_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help=f"Raw extracted dataset dir (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"Output dataset dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for the deterministic source-level split (default: 1337)")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild --out if it already exists")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Comma-separated class names to keep (default: keep all "
            f"{len(CLASS_NAMES)}: {', '.join(CLASS_NAMES)}). Other classes' "
            "labels are dropped; images left with no labels are kept as "
            "background images. Surviving class ids are remapped to a "
            "contiguous 0..k-1 range. Does not affect which sources land in "
            "train/val/test for a given --seed. Example: --classes spaghetti,warping"
        ),
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Collapse all kept classes (see --classes) into a single class id 0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir: Path = args.raw
    out_dir: Path = args.out
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    classes_arg: list[str] | None = None
    if args.classes:
        classes_arg = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        class_filter = resolve_class_filter(classes_arg, args.single_class, CLASS_NAMES)
    except ValueError as exc:
        sys.exit(f"[prepare] ERROR: {exc}")
    if classes_arg or args.single_class:
        print(f"[prepare] Class filter active: classes={classes_arg or list(CLASS_NAMES)} "
              f"single_class={args.single_class} -> output names={class_filter.names}")

    if out_dir.exists():
        if args.force:
            print(f"[prepare] --force: removing existing '{out_dir}' ...")
            shutil.rmtree(out_dir)
        else:
            print(f"[prepare] '{out_dir}' already exists. Pass --force to rebuild it. Proceeding to (over)write into it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Pooling raw/train + raw/valid from '{raw_dir}' (shipped split is discarded) ...")
    pooled, duplicate_count = pool_raw_images(raw_dir)
    print(f"[prepare] Pooled {len(pooled)} unique images ({duplicate_count} duplicate filenames dropped).")

    groups, fallback_count = group_by_source(pooled.keys())
    print(f"[prepare] Recovered {len(groups)} unique source identities from {len(pooled)} images.")
    if fallback_count:
        print(f"[prepare] NOTE: {fallback_count} filenames did not match the '.rf.<hex>' pattern; "
              f"each was treated as its own source (fallback).")

    train_ids, val_ids, test_ids = split_sources(sorted(groups.keys()), seed=args.seed, ratios=ratios)
    print(f"[prepare] Source-level split (seed={args.seed}, ratios={ratios}): "
          f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} sources.")

    split_source_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    assert_no_source_overlap(split_source_ids)
    print("[prepare] Verified: zero source-identity overlap between splits.")

    stats_by_split: dict[str, SplitStats] = {}
    for split, ids in split_source_ids.items():
        selected = select_files_for_split(groups, ids, split)
        stats_by_split[split] = materialize_split(split, selected, pooled, out_dir, class_filter)

    data_yaml_path = write_data_yaml(out_dir, class_filter.names)

    # Re-verify overlap using the files actually written to disk, as an
    # end-to-end sanity check (not just trusting the in-memory split lists).
    written_source_ids: dict[str, list[str]] = {}
    for split in SPLIT_NAMES:
        images_dir = out_dir / split / "images"
        filenames = [p.name for p in images_dir.iterdir() if p.is_file()]
        written_groups, _ = group_by_source(filenames)
        written_source_ids[split] = sorted(written_groups.keys())
    assert_no_source_overlap(written_source_ids)

    total_missing_labels = sum(s.n_missing_labels for s in stats_by_split.values())
    total_polygon_converted = sum(s.polygon_rows_converted for s in stats_by_split.values())
    total_malformed_skipped = sum(s.malformed_rows_skipped for s in stats_by_split.values())
    total_files_with_polygons = sum(s.files_containing_polygons for s in stats_by_split.values())

    report = {
        "seed": args.seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "class_names": list(CLASS_NAMES),
        "class_filter": {
            "requested_classes": classes_arg,
            "single_class": args.single_class,
            "kept_original_class_ids": sorted(class_filter.keep_ids),
            "output_class_names": class_filter.names,
        },
        "total_pooled_images": len(pooled),
        "duplicate_filenames_dropped": duplicate_count,
        "total_unique_sources": len(groups),
        "fallback_source_id_count": fallback_count,
        "source_overlap_check": "PASS: zero overlap between train/val/test source identities",
        "data_yaml": str(data_yaml_path),
        "splits": {split: stats_by_split[split].to_dict() for split in SPLIT_NAMES},
        "total_missing_labels": total_missing_labels,
        "polygon_rows_converted": total_polygon_converted,
        "malformed_rows_skipped": total_malformed_skipped,
        "files_containing_polygons": total_files_with_polygons,
    }
    report_path = out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("DATASET PREPARATION -- SPLIT REPORT")
    print("=" * 72)
    print(f"Pooled images (train+valid, shipped split discarded): {len(pooled)}")
    print(f"Unique source photos recovered: {len(groups)}  (fallback/no-match: {fallback_count})")
    print(f"Duplicate filenames dropped: {duplicate_count}")
    print()
    header = f"{'split':<8}{'sources':>10}{'images':>10}{'missing_labels':>16}"
    print(header)
    print("-" * len(header))
    for split in SPLIT_NAMES:
        s = stats_by_split[split]
        print(f"{split:<8}{s.n_sources:>10}{s.n_images:>10}{s.n_missing_labels:>16}")
    print()
    print(f"Source-identity overlap check: {report['source_overlap_check']}")
    print()
    print("Per-class instance counts (bounding boxes) / images-containing-class, per split:")
    col_w = 16
    print(f"{'class':<20}" + "".join(f"{s:>{col_w}}" for s in SPLIT_NAMES))
    for cname in class_filter.names:
        row = f"{cname:<20}"
        for split in SPLIT_NAMES:
            s = stats_by_split[split]
            row += f"{f'{s.instances_per_class[cname]} / {s.images_per_class[cname]}':>{col_w}}"
        print(row)
    print()
    print(f"Total images with no label file (warned + skipped): {total_missing_labels}")
    print()
    print("Polygon/segment-label recovery (Ultralytics otherwise drops the WHOLE image "
          "for a file mixing detection and segment rows):")
    print(f"  Polygon/segment rows converted to bounding boxes: {total_polygon_converted}")
    print(f"  Malformed label rows skipped (not crashed on):    {total_malformed_skipped}")
    print(f"  Label files that contained a polygon/segment row: {total_files_with_polygons}")
    print()
    print(f"data.yaml written to: {data_yaml_path}")
    print(f"Full report written to: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
