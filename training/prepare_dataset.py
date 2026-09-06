"""Builds a clean, leak-free dataset at ``datasets/argus/`` from the raw Roboflow export at ``datasets/raw/`` (see ``training/ingest_dataset.py``), fixing two defects in the raw archive.

Defect 1 (train/valid leakage): Roboflow generated 3 augmented variants per source photo and scattered them across train/valid (~47% overlap). This pools raw/train+valid, discards the shipped split, recovers each image's source identity from its ``.rf.<hex>`` suffix, and re-splits BY SOURCE -- never by file -- so all variants of one photo land in exactly one split. Val/test keep only the first variant per source (all 3 would triple eval cost and correlate errors); train keeps every variant.

Defect 2 (missing ``test/``): the raw ``data.yaml`` points at a nonexistent ``test/`` dir; this generates a real held-out test split and writes a fresh ``data.yaml``.

Usage: python training/prepare_dataset.py [--raw datasets/raw] [--out datasets/argus] [--seed 1337]
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

# Matches Roboflow's "<source>.rf.<hex>.<ext>" filenames; everything before ".rf." is the
# source identity, the hex chunk is just the augmentation-variant id.
SOURCE_ID_RE = re.compile(
    r"^(?P<src>.+)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)

DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


# --------------------------------------------------------------------------
# Pure logic (unit-tested without touching the real dataset on disk)
# --------------------------------------------------------------------------


# tuple[str, bool] extract_source_id(str filename)
# Inputs: str filename - an image filename from the raw Roboflow export
# Outputs: tuple[str, bool] - (source_id, matched): source_id is the recovered source-photo
#          identity; matched is True if the Roboflow ".rf.<hex>" suffix pattern was found,
#          False if the whole filename stem was used as a fallback source id
# Description: Recovers the source-photo identity from an image filename by stripping the
#              Roboflow ".rf.<hex>" augmentation-variant suffix. This is the group/session unit
#              used everywhere downstream: splitting happens BY SOURCE PHOTO, never by
#              individual file, so all augmented variants of one photo land in exactly one
#              split -- the guard against Defect 1 (train/valid leakage).
# Side Effects: None (pure string parsing)
def extract_source_id(filename: str) -> tuple[str, bool]:
    m = SOURCE_ID_RE.match(filename)
    if m:
        return m.group("src"), True
    stem = Path(filename).stem
    return stem, False


# tuple[dict[str, list[str]], int] group_by_source(Iterable[str] filenames)
# Inputs: Iterable[str] filenames - image filenames to group
# Outputs: tuple[dict[str, list[str]], int] - (groups, fallback_count): groups maps source id
#          to a sorted list of filenames belonging to that source (all its augmentation
#          variants); fallback_count is how many filenames didn't match the ".rf.<hex>" pattern
# Description: Groups filenames by source-photo identity via extract_source_id. The grouping
#              unit is the SOURCE PHOTO (session), not the individual file -- this is what lets
#              callers split by source rather than by file. Filenames are sorted both across
#              and within groups so downstream "first file for this source" selection is
#              deterministic regardless of filesystem iteration order.
# Side Effects: None
def group_by_source(filenames: Iterable[str]) -> tuple[dict[str, list[str]], int]:
    groups: dict[str, list[str]] = {}
    fallback_count = 0
    for fn in sorted(filenames):
        src, matched = extract_source_id(fn)
        if not matched:
            fallback_count += 1
        groups.setdefault(src, []).append(fn)
    return groups, fallback_count


# tuple[list[str], list[str], list[str]] split_sources(Sequence[str] source_ids, int seed, tuple[float, float, float] ratios)
# Inputs: Sequence[str] source_ids - source-photo identities to split (not individual files)
#         int seed - RNG seed for deterministic shuffling, e.g. the CLI's default 1337
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15); must sum to 1.0
# Outputs: tuple[list[str], list[str], list[str]] - (train_ids, val_ids, test_ids) source id
#          lists; any rounding remainder is given to test so no source is invented or dropped
# Description: Deterministically splits source ids (not individual files) into train/val/test.
#              Splitting by SOURCE PHOTO/SESSION rather than by file is what guarantees no
#              augmentation variant of a training-set photo can leak into val/test (the guard
#              against Defect 1). Deterministic for a given seed: source ids are sorted first
#              (so input order never matters), then shuffled with a seeded random.Random before
#              slicing by ratio.
# Side Effects: Raises ValueError if ratios don't sum to 1.0 (within 1e-6). Uses a locally-
#               seeded random.Random(seed); does not touch global RNG state.
def split_sources(
    source_ids: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[str], list[str], list[str]]:
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


# dict[str, list[str]] select_files_for_split(dict[str, list[str]] groups, Sequence[str] source_ids, str split)
# Inputs: dict[str, list[str]] groups - source id -> sorted list of that source's filenames
#         (from group_by_source)
#         Sequence[str] source_ids - the sources assigned to this split (from split_sources)
#         str split - "train", "val", or "test"
# Outputs: dict[str, list[str]] - source id -> list of selected filenames for that source,
#          restricted to source_ids
# Description: Chooses which files, per source, belong in a given split. "train" keeps every
#              augmentation variant of each of its sources (maximizes training data); "val" and
#              "test" keep exactly one variant per source -- the first in sorted filename order
#              -- because the near-duplicate variants would triple eval cost and correlate
#              errors between variants if all were kept.
# Side Effects: None
def select_files_for_split(
    groups: dict[str, list[str]],
    source_ids: Sequence[str],
    split: str,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for sid in source_ids:
        files = groups[sid]
        selected[sid] = list(files) if split == "train" else [files[0]]
    return selected


# None assert_no_source_overlap(dict[str, Sequence[str]] split_source_ids)
# Inputs: dict[str, Sequence[str]] split_source_ids - split name -> source ids assigned to it
# Outputs: None
# Description: Verifies no source-photo identity appears in more than one split. This is the
#              explicit check for the whole point of splitting by source instead of by file
#              (the Defect 1 leakage guard), rather than just trusting split_sources.
# Side Effects: Raises AssertionError naming the offending source and both splits it appears in,
#               if any source-identity leakage is detected. No filesystem or RNG activity.
def assert_no_source_overlap(split_source_ids: dict[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in split_source_ids.items():
        for sid in ids:
            if sid in seen:
                raise AssertionError(
                    f"Source-identity leakage detected: source {sid!r} appears in both "
                    f"'{seen[sid]}' and '{split_name}' splits."
                )
            seen[sid] = split_name


# list[int] parse_yolo_label_classes(str label_text)
# Inputs: str label_text - contents of a YOLO-format label file
# Outputs: list[int] - class id of every bounding-box line, in file order
# Description: Parses a YOLO-format label file's contents and returns the class id of every
#              bounding-box line (first whitespace-separated token per line). Blank lines are
#              ignored; an empty/whitespace-only file legitimately means "no objects" and
#              yields an empty list.
# Side Effects: None (pure parsing)
def parse_yolo_label_classes(label_text: str) -> list[int]:
    class_ids: list[int] = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(float(line.split()[0])))
    return class_ids


# --------------------------------------------------------------------------
# Label-row normalization: rewrites every row to plain detection format, since
# Ultralytics drops the WHOLE IMAGE for a label file mixing detection + polygon rows.
# --------------------------------------------------------------------------


# float clamp01(float value)
# Inputs: float value - a normalized coordinate value
# Outputs: float - value clamped to [0, 1]
# Description: Clamps a normalized coordinate to the valid [0, 1] range.
# Side Effects: None
def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class LabelNormalizeStats:
    polygon_rows_converted: int = 0
    malformed_rows_skipped: int = 0
    degenerate_rows_dropped: int = 0
    contains_polygon_row: bool = False


# tuple[int, list[float], list[float]] _row_corner_points(list[str] fields)
# Inputs: list[str] fields - whitespace-split tokens of one YOLO label line (class id followed
#         by either 4 detection fields or an odd-length list of polygon coordinates)
# Outputs: tuple[int, list[float], list[float]] - (cls_id, xs, ys): the class id and the point
#          list used to derive an axis-aligned bounding box
# Description: Parses one label line's fields into (cls_id, xs, ys). A detection row (5 fields:
#              cls cx cy w h) is converted to its two bounding corners so it can be run through
#              the same clamp/bbox math as a polygon row. A segment row (odd field count > 5)
#              uses its polygon points directly.
# Side Effects: Raises ValueError if any field isn't numeric -- callers treat that as a
#               malformed row (skip and count, don't crash). No I/O.
def _row_corner_points(fields: list[str]) -> tuple[int, list[float], list[float]]:
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


# tuple[str, LabelNormalizeStats] normalize_label_text(str label_text)
# Inputs: str label_text - raw contents of a YOLO label file (may mix detection and
#         segmentation/polygon rows)
# Outputs: tuple[str, LabelNormalizeStats] - (normalized_text, stats): normalized_text has
#          every row rewritten to plain 5-field detection format (cls cx cy w h, 6 decimal
#          places); stats tallies polygon_rows_converted, malformed_rows_skipped,
#          degenerate_rows_dropped, and contains_polygon_row
# Description: Rewrites every row of a YOLO label file to plain 5-field detection format. A
#              5-field row is clamped and re-emitted (a no-op in meaning for well-formed
#              input). A segment/polygon row (odd field count > 5) is clamped to [0, 1] and
#              collapsed to its axis-aligned bounding box. Malformed rows (even field count > 5,
#              fewer than 5 fields, or non-numeric fields) are skipped and counted rather than
#              raised. A box that's degenerate after clamping (w <= 0 or h <= 0) is dropped
#              silently. This exists because Ultralytics refuses any label file that mixes
#              detection and segmentation rows and drops the WHOLE IMAGE, not just the
#              offending row -- normalizing every row avoids losing images to that.
# Side Effects: None (pure text transformation; no I/O)
def normalize_label_text(label_text: str) -> tuple[str, LabelNormalizeStats]:
    """A polygon row (odd field count > 5) is clamped and collapsed to its axis-aligned bbox;
    an even field count > 5, <5 fields, or non-numeric fields are malformed (skipped, counted)."""
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
    """Which original class ids survive and what they're remapped to; rows whose class id
    isn't a key in id_remap are dropped."""

    keep_ids: frozenset[int]
    id_remap: dict[int, int]
    names: list[str]
    single_class: bool


# ClassFilter default_class_filter(Sequence[str] class_names)
# Inputs: Sequence[str] class_names - full class-name list, default CLASS_NAMES (the canonical
#         5 classes: error extrusion, spaghetti, stringing, warping, zits)
# Outputs: ClassFilter - identity filter: keeps every class id, maps each to itself, and uses
#          class_names as-is (single_class=False)
# Description: Builds the identity ClassFilter (keep every class, no remap, no collapsing).
#              Used when --classes/--single-class aren't passed, so downstream code has a
#              single code path regardless of whether filtering is active.
# Side Effects: None
def default_class_filter(class_names: Sequence[str] = CLASS_NAMES) -> ClassFilter:
    ids = range(len(class_names))
    return ClassFilter(
        keep_ids=frozenset(ids),
        id_remap={i: i for i in ids},
        names=list(class_names),
        single_class=False,
    )


# ClassFilter resolve_class_filter(Sequence[str] | None classes, bool single_class, Sequence[str] class_names)
# Inputs: Sequence[str] | None classes - requested class names from --classes, or None/empty to
#         keep every class
#         bool single_class - whether to collapse all kept classes into output id 0
#         Sequence[str] class_names - full class-name list, default CLASS_NAMES
# Outputs: ClassFilter - keep_ids, id_remap (original id -> output id), output names list
#          (joined with "+" if single_class), and the single_class flag
# Description: Builds the class-id remap for --classes/--single-class. Requested names are
#              matched against class_names case-insensitively. Kept ids are always ordered by
#              their ORIGINAL id (not CLI order), so "--classes warping,spaghetti" and
#              "--classes spaghetti,warping" produce the same remap. Without single_class, kept
#              ids are remapped to a contiguous 0..k-1 range in original-id order.
# Side Effects: Raises ValueError for an unknown class name in --classes, or if --classes
#               resolves to no valid names. No I/O.
def resolve_class_filter(
    classes: Sequence[str] | None,
    single_class: bool,
    class_names: Sequence[str] = CLASS_NAMES,
) -> ClassFilter:
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


# str apply_class_filter(str label_text, ClassFilter class_filter)
# Inputs: str label_text - normalized label text (5-field detection rows, see
#         normalize_label_text)
#         ClassFilter class_filter - the class keep/remap rules to apply
# Outputs: str - label text with dropped/unkept rows removed and surviving rows' class ids
#          remapped per class_filter.id_remap; "" if no rows survive
# Description: Drops rows whose class isn't kept by class_filter and remaps surviving rows'
#              class id per class_filter.id_remap. Assumes label_text is already normalized to
#              5-field detection rows; non-numeric or otherwise unparseable leading tokens are
#              dropped defensively rather than raising.
# Side Effects: None (pure text transformation; no I/O)
def apply_class_filter(label_text: str, class_filter: ClassFilter) -> str:
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


# list[RawImage] scan_raw_split(Path raw_root, str raw_split_dir)
# Inputs: Path raw_root - root of the extracted raw Roboflow dataset (datasets/raw/)
#         str raw_split_dir - the shipped split subdirectory to scan, "train" or "valid"
# Outputs: list[RawImage] - one RawImage per image file found, paired with its expected
#          (possibly nonexistent) label path
# Description: Lists every image in raw_root/raw_split_dir/images, paired with its expected
#              label path in raw_root/raw_split_dir/labels (which may not exist on disk).
# Side Effects: Raises FileNotFoundError if the expected images/ directory doesn't exist
#               (pointing the user at training/ingest_dataset.py). Read-only filesystem
#               traversal otherwise.
def scan_raw_split(raw_root: Path, raw_split_dir: str) -> list[RawImage]:
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


# tuple[dict[str, RawImage], int] pool_raw_images(Path raw_root)
# Inputs: Path raw_root - root of the extracted raw Roboflow dataset (datasets/raw/)
# Outputs: tuple[dict[str, RawImage], int] - (pooled, duplicate_count): pooled maps filename to
#          RawImage across both shipped splits; duplicate_count is how many filenames appeared
#          in both raw split dirs (first occurrence, train, wins)
# Description: Pools images from raw/train and raw/valid into a single filename -> RawImage
#              map, discarding the shipped train/valid split entirely (Defect 1: Roboflow's
#              shipped split leaked augmented variants of the same source photo across
#              train/valid). A raw split subdirectory that doesn't exist at all is skipped
#              rather than treated as an error, so this stays usable on synthetic test fixtures
#              that only populate one of the two.
# Side Effects: Read-only filesystem traversal via scan_raw_split; raises FileNotFoundError if
#               a split subdirectory exists but its images/ folder is missing.
def pool_raw_images(raw_root: Path) -> tuple[dict[str, RawImage], int]:
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

    # None __post_init__()
    # Inputs: None (operates on self; class_names, instances_per_class, images_per_class)
    # Outputs: None
    # Description: Dataclass post-init hook that fills instances_per_class and
    #              images_per_class with zero counts for every class in class_names, when the
    #              caller didn't supply them explicitly.
    # Side Effects: Mutates self.instances_per_class and self.images_per_class in place.
    def __post_init__(self) -> None:
        if not self.instances_per_class:
            self.instances_per_class = {c: 0 for c in self.class_names}
        if not self.images_per_class:
            self.images_per_class = {c: 0 for c in self.class_names}

    # dict[str, object] to_dict()
    # Inputs: None (operates on self)
    # Outputs: dict[str, object] - JSON-serializable view of every field on this SplitStats
    # Description: Converts this SplitStats instance into a plain dict for JSON serialization
    #              into split_report.json.
    # Side Effects: None
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


# SplitStats materialize_split(str split, dict[str, list[str]] selected, dict[str, RawImage] pooled, Path out_dir, ClassFilter | None class_filter)
# Inputs: str split - split name, "train", "val", or "test"
#         dict[str, list[str]] selected - source id -> filenames to materialize for this split
#         (from select_files_for_split)
#         dict[str, RawImage] pooled - filename -> RawImage lookup (from pool_raw_images)
#         Path out_dir - output dataset root (e.g. datasets/argus)
#         ClassFilter | None class_filter - class keep/remap rules, default None (uses
#         default_class_filter(), i.e. keep everything)
# Outputs: SplitStats - tallies of sources, images, missing labels, polygon conversions,
#          malformed rows, and per-class instance/image counts for this split
# Description: Copies the chosen images (and matching labels, when present) for one split into
#              out_dir/<split>/{images,labels} and tallies stats. Every label file is
#              normalized to plain detection rows (normalize_label_text) so polygon/segment
#              rows survive instead of the whole image being dropped by Ultralytics. If
#              class_filter is given, rows are then filtered/remapped through it; images left
#              with zero surviving labels are still kept as legitimate background/negative
#              images.
# Side Effects: Creates out_dir/<split>/images and out_dir/<split>/labels directories; copies
#               image files (shutil.copy2) into images_out; reads each label file from disk and
#               writes the normalized/filtered label text to labels_out; prints a WARNING to
#               stdout for images with no label file and for labels with out-of-range class ids.
def materialize_split(
    split: str,
    selected: dict[str, list[str]],
    pooled: dict[str, RawImage],
    out_dir: Path,
    class_filter: ClassFilter | None = None,
) -> SplitStats:
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


# Path write_data_yaml(Path out_dir, Sequence[str] class_names)
# Inputs: Path out_dir - output dataset root (e.g. datasets/argus)
#         Sequence[str] class_names - active class name order, default CLASS_NAMES (the
#         canonical 5), or narrowed by --classes/--single-class
# Outputs: Path - path to the written data.yaml file
# Description: Writes out_dir/data.yaml with correct train/val/test paths (fixing Defect 2: the
#              raw data.yaml pointed at a test/ dir that didn't exist) and the active class name
#              order.
# Side Effects: Writes (overwrites) out_dir/data.yaml on disk.
def write_data_yaml(out_dir: Path, class_names: Sequence[str] = CLASS_NAMES) -> Path:
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


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed options (raw, out, seed, train/val/test-ratio, force,
#          classes, single_class). Notable defaults: --seed 1337, ratios DEFAULT_SPLIT_RATIOS
#          (0.70, 0.15, 0.15).
# Description: Defines and parses the CLI for building the leak-free dataset.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
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


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Resolves the class filter, pools raw/train+raw/valid images
#              (discarding the shipped split), groups them by source-photo identity, splits
#              sources deterministically into train/val/test, verifies zero source overlap
#              (both from the in-memory split and by re-scanning what was actually written to
#              disk), materializes each split's images/labels, writes data.yaml and
#              split_report.json, and prints a full summary report.
# Side Effects: Optionally wipes out_dir with shutil.rmtree when --force is passed; creates
#               out_dir and its split subdirectories; copies image files and writes normalized
#               label files to disk (via materialize_split); writes out_dir/data.yaml (via
#               write_data_yaml) and out_dir/split_report.json; calls sys.exit on an invalid
#               --classes value; prints progress and the full split report to stdout.
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
