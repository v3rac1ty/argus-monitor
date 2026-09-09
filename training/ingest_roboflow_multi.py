"""Merges N Roboflow YOLO object-detection exports into ONE unified, leak-free detection dataset (default ``datasets/argus_multi/``).

Every dataset ingested by this project so far shipped with a leakage defect that
inflated metrics, so this script assumes the same of any new export and *proves*
the answer instead of trusting it:

  * Defect A (augmentation leakage, the AtCo export): 10,757 files were only
    1,930 source photos, variants scattered across train/valid. Guard: recover
    the source identity by stripping Roboflow's ``.rf.<hash>`` suffix
    (``prepare_dataset.extract_source_id``) and union every variant into one group.
  * Defect B (timelapse leakage, the FDM export): 1,912 "images" were ~40 real
    print jobs photographed every ~30s. Frame-level splitting put near-identical
    consecutive frames in train and test; measured accuracy fell 97.3% -> ~58%
    once split by session. Guard: cluster near-duplicate frames by perceptual
    hash and by filename timestamp/sequence, then split by CLUSTER.
  * Defect C (cross-dataset re-upload): the same physical print can appear in two
    different Roboflow uploads. Guard: hash every image in every input AND in the
    reference datasets already on disk, and report every cross-dataset collision
    (exact SHA256 of decoded pixels, plus dhash).
  * Defect D (REDUNDANT SOURCE, measured 2026-09): two downloaded exports,
    ``datasets/rf_defects`` (5,869 files) and ``datasets/rf_failure`` (8,853
    files), looked like a large win but stripped down to 418 source photos each
    -- and all 418 were already inside ``datasets/raw``. They contributed ZERO new
    photographs and are not ingested. Guard: ``find_redundant_sources`` compares
    the ``.rf.``-stripped stem sets of every source BEFORE any copying and fails
    unless ``--allow-redundant-sources`` is passed.

Splitting is over GROUPS, never over files: a group is the transitive closure of
(exact pixel duplicate) U (perceptual near-duplicate) U (Roboflow augmentation
siblings) U (filename-derived print session). Zero group overlap between splits
is asserted twice -- once on the in-memory split, once by re-deriving splits from
what was actually written. The default is 60/25/15: group sizes vary so wildly
that a 70/15/15 group split lands near 89/5/6 by IMAGE count, and a ~5%
validation set is too thin to pick confidence thresholds against.

Effective size is reported as THREE separate numbers per source -- raw file
count, unique ``.rf.``-stem count, and post-clustering group count. The gap
between them (13,822 stereovision files are 2,909 photos) is the single most
decision-relevant fact about this data and is never collapsed into one figure.

License gating: ``--exclude-noncommercial`` (ON by default) omits any source
declared CC BY-NC from the merged output, since this repo is GPL-3.0. Both
current sources are permissive, so it is a no-op today; excluded sources are
still scanned and hashed so the duplicate analysis stays complete.

Usage:
  py training/ingest_roboflow_multi.py --atco datasets/raw \\
      --stereovision datasets/rf_stereovision \\
      [--out datasets/argus_multi] [--reference datasets/fdm_raw] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

# Run-from-anywhere bootstrap: this module reuses training/prepare_dataset.py and
# src/argus/types.py, which are only importable once the repo root and src/ are on
# sys.path (pytest already does this via pyproject's pythonpath; `py training/...`
# does not).
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from argus.types import Severity  # noqa: E402
from training.prepare_dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    ClassFilter,
    apply_class_filter,
    assert_no_source_overlap,
    extract_source_id,
    normalize_label_text,
    parse_yolo_label_classes,
    split_sources,
)

DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "argus_multi"
#: Datasets scanned for collision detection only, never emitted (Defect C).
#: Deliberately EMPTY: datasets/raw is now an ingest source (--atco), and
#: datasets/fdm_raw is 5.9 GB belonging to the classification pipeline, so
#: decoding it on every run is not worth it. Pass --reference to opt in.
DEFAULT_REFERENCE_DIRS: tuple[Path, ...] = ()

DEFAULT_SEED = 1337
#: 60/25/15 over GROUPS. Val is deliberately larger than the usual 15%: group
#: sizes vary so much that a 70/15/15 group split measured out at 89/4.9/6.1 by
#: image count, and ~5% of images is too thin to pick confidence thresholds on.
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.60, 0.25, 0.15)
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

#: Hamming distance between two 64-bit dhashes at or below which two images are
#: treated as the same physical scene (near-duplicate / adjacent timelapse frame).
DEFAULT_PHASH_DISTANCE = 6
#: Gap (seconds) between consecutive filename timestamps that starts a new print
#: session -- same threshold build_classification_dataset.py uses for FDM.
DEFAULT_SESSION_GAP_S = 600.0
#: Gap in a filename's trailing sequence number that starts a new session, e.g.
#: frame_0007 -> frame_0100 is a new job, frame_0007 -> frame_0008 is not.
DEFAULT_SEQ_GAP = 3

#: Minimum fraction of a filename-derived session's SOURCE PHOTOS that must have
#: a perceptual near-neighbour inside that same session for the session to be
#: believed and used for grouping.
#:
#: Filename sessions are single-linkage over sequence numbers and timestamps, so
#: they chain without bound: measured on this roster, "0001_null_dataset" ...
#: "0585_null_dataset" became ONE 585-photo session and stereovision's timestamped
#: photos chained 495 of them together, which collapsed 4,644 photos into 1,406
#: groups and left 64% of all images in val with the 'head' class absent from
#: train and test entirely. Corroboration separates the two cases cleanly, because
#: a real capture session is perceptually self-similar and a numbered photo
#: collection is not -- measured on this roster:
#:     timelapse_mp4-0000..0157 (one print)   61.4%  -> believed
#:     stereovision timestamp session ts26    64.4%  -> believed
#:     0001..0585_null_dataset (a collection)  0.7%  -> rejected
#:     imag-10..imag-999 (a collection)        0.0%  -> rejected
#: Rejected sessions contribute no links; their images are still grouped by
#: perceptual hash and Roboflow augmentation identity, so nothing is un-guarded.
DEFAULT_SESSION_CORROBORATION = 0.25
#: Hamming distance used only for that corroboration vote. Deliberately the loose
#: default rather than --phash-distance: the question is "is this the same scene",
#: which is broader than "is this near-identical enough to be one group".
SESSION_CORROBORATION_DISTANCE = 6
#: Sessions with fewer distinct source photos than this are believed without a
#: vote -- too few members to judge, and too small to do real damage if wrong.
SESSION_CORROBORATION_MIN_STEMS = 3

PHASH_BITS = 64
DHASH_SIZE = 8

#: Below this many groups the split can't produce a meaningful val/test.
MIN_GROUPS_FOR_SPLIT = 10
#: Mean frames-per-group above this means the source is timelapse, not independent photos.
TIMELAPSE_FRAMES_PER_GROUP = 3.0
#: A split holding less than this fraction of its requested IMAGE share is flagged: the group
#: split was honest, but uneven group sizes made the image split lopsided.
SPLIT_SHARE_TOLERANCE = 0.6
#: A source whose unique source-photo stems are at least this percent contained in ANOTHER
#: source contributes (almost) no new photographs and is refused without
#: --allow-redundant-sources. rf_defects and rf_failure both sat at 100% against datasets/raw.
REDUNDANT_SOURCE_PCT = 90.0
#: How many collision examples to embed in split_report.json (counts are always exact).
REPORT_SAMPLE_CAP = 50


# --------------------------------------------------------------------------
# Unified label space + class harmonization table
# --------------------------------------------------------------------------

#: Unified class order written to the output data.yaml. Deterministic and
#: explicit -- the ONNX detector's output index order depends on it, so append
#: only, never reorder.
UNIFIED_CLASSES: tuple[str, ...] = (
    "spaghetti",
    "layer_separation",
    "bed_adhesion",
    "blob_of_death",
    "warping",
    "stringing",
    "zits",
    "head",
    "error_extrusion",
)

#: Severity per unified class, consumed by src/argus/types.py's Severity enum.
#: Only CATASTROPHIC classes feed DetectionResult.p_failure.
SEVERITY_BY_CLASS: dict[str, Severity] = {
    "spaghetti": Severity.CATASTROPHIC,
    "layer_separation": Severity.CATASTROPHIC,
    "bed_adhesion": Severity.CATASTROPHIC,
    "blob_of_death": Severity.CATASTROPHIC,
    "warping": Severity.COSMETIC,
    "stringing": Severity.COSMETIC,
    "zits": Severity.COSMETIC,
    "head": Severity.COSMETIC,
    "error_extrusion": Severity.COSMETIC,
}

#: EDIT ME: source class name (normalized -- see normalize_class_name) -> unified
#: class name, or None to DROP that class's label rows entirely. Contributors:
#:   atco         : error extrusion, spaghetti, stringing, warping, zits
#:   stereovision : Bed Adhesion, Blob of Death, Head, Layer Separation, Spaghetti, Warping
#: Every distinct source class is kept, by explicit request -- nothing is merged
#: away or dropped, so the model can name the specific defect it saw.
#: "head" is the PRINTHEAD, not a defect. It is kept deliberately as a
#: non-actionable distractor class so the model learns that the nozzle is not a
#: failure; it is COSMETIC so it can never trigger a pause/cancel.
#: "error_extrusion" (AtCo's "error extrusion") is under-extrusion / extrusion
#: fault: it degrades the part but the print is still recoverable, so COSMETIC.
CLASS_MAP: dict[str, str | None] = {
    "spaghetti": "spaghetti",
    "layer separation": "layer_separation",
    "layer split": "layer_separation",
    "bed adhesion": "bed_adhesion",
    "blob of death": "blob_of_death",
    "warping": "warping",
    "stringing": "stringing",
    "zits": "zits",
    "head": "head",
    "error extrusion": "error_extrusion",
}


@dataclass(frozen=True)
class SourceSpec:
    """Provenance + license metadata for one ingestable Roboflow export."""

    key: str
    origin: str
    license: str
    noncommercial: bool
    expected_classes: tuple[str, ...]


#: The ingestable sources, in deterministic priority order (earlier wins when an
#: exact-duplicate image appears in two of them). expected_classes is
#: DOCUMENTATION ONLY -- the real class list is read from each export's own
#: data.yaml at runtime, so a re-download with different classes is reported
#: rather than silently mis-mapped.
#:
#: NOT ingestable, deliberately: datasets/rf_defects and datasets/rf_failure.
#: Measured 2026-09, both are augmented re-uploads of one 418-photo subset of the
#: AtCo set (418/418 stems already in datasets/raw), so they add zero new
#: photographs. find_redundant_sources refuses them if they are ever passed.
SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="atco",
        origin="atco/3d-printing-error v7 (AtCo '3D printing error')",
        license="MIT",
        noncommercial=False,
        expected_classes=("error extrusion", "spaghetti", "stringing", "warping", "zits"),
    ),
    SourceSpec(
        key="stereovision",
        origin="yawllen-jectr/stereovision-gyibu v8 (StereoVision)",
        license="CC BY 4.0",
        noncommercial=False,
        expected_classes=("Bed Adhesion", "Blob of Death", "Head", "Layer Separation", "Spaghetti", "Warping"),
    ),
)

SOURCE_SPEC_BY_KEY: dict[str, SourceSpec] = {s.key: s for s in SOURCE_SPECS}

SOURCE_CONFOUND_NOTE = (
    "SOURCE-CONFOUND NOTE: images are copied byte-for-byte (no resize/re-encode), so "
    "resolution, framing and JPEG-artifact statistics still identify which dataset each "
    "image came from. Every output image is tagged with its origin (filename prefix and/or "
    "manifest.jsonl) precisely so the per-source recall diagnostic can measure whether the "
    "model learned dataset origin instead of defect features. Treat aggregate mAP as "
    "optimistic until per-source recall is checked."
)


# --------------------------------------------------------------------------
# Class harmonization (pure)
# --------------------------------------------------------------------------


# str normalize_class_name(str name)
# Inputs: str name - a class name exactly as written in a source export's data.yaml
# Outputs: str - lowercased, punctuation-flattened, whitespace-collapsed key for CLASS_MAP
#          lookup, e.g. "Blob_of-Death " -> "blob of death"
# Description: Canonicalizes a source class name so CLASS_MAP needs one entry per concept
#              instead of one per capitalization/separator variant ("Spaghetti"/"spaghetti",
#              "Layer Separation"/"layer_separation" all collapse to the same key).
# Side Effects: None
def normalize_class_name(name: str) -> str:
    flattened = re.sub(r"[_\-/.]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", flattened).strip()


# str | None map_source_class(str name)
# Inputs: str name - a class name from a source export's data.yaml
# Outputs: str | None - the unified class name, or None if the class is explicitly mapped to
#          None (dropped) or is absent from CLASS_MAP entirely (unknown -> dropped)
# Description: Resolves one source class name into the unified label space via CLASS_MAP.
#              Both explicit drops and unknown names return None; callers distinguish them via
#              is_known_class so an unexpected class in a new export is reported loudly rather
#              than silently vanishing.
# Side Effects: None
def map_source_class(name: str) -> str | None:
    return CLASS_MAP.get(normalize_class_name(name))


# bool is_known_class(str name)
# Inputs: str name - a class name from a source export's data.yaml
# Outputs: bool - True if the name has an entry in CLASS_MAP (even one mapped to None)
# Description: Distinguishes a deliberate drop (present in CLASS_MAP with value None) from an
#              unrecognized class (absent from CLASS_MAP). The latter means the export shipped
#              a class this script has never seen and belongs in the report's warnings.
# Side Effects: None
def is_known_class(name: str) -> bool:
    return normalize_class_name(name) in CLASS_MAP


@dataclass(frozen=True)
class ClassRemap:
    """How one source export's class ids map into UNIFIED_CLASSES."""

    id_remap: dict[int, int]  # source class id -> unified class id; absent id => row dropped
    mapped: dict[str, str]  # source class name -> unified class name
    dropped: list[str]  # source class names deliberately mapped to None
    unmapped: list[str]  # source class names missing from CLASS_MAP entirely

    # ClassFilter to_class_filter(self)
    # Inputs: None (operates on self)
    # Outputs: ClassFilter - a prepare_dataset.ClassFilter carrying this remap and the full
    #          UNIFIED_CLASSES name list
    # Description: Adapts this remap into prepare_dataset's ClassFilter so label rows can be
    #              rewritten with that module's already-tested apply_class_filter (rows whose
    #              class id isn't a key in id_remap are dropped, survivors are renumbered)
    #              instead of a second copy of the same logic here.
    # Side Effects: None
    def to_class_filter(self) -> ClassFilter:
        return ClassFilter(
            keep_ids=frozenset(self.id_remap),
            id_remap=dict(self.id_remap),
            names=list(UNIFIED_CLASSES),
            single_class=False,
        )


# ClassRemap build_class_remap(Sequence[str] source_class_names)
# Inputs: Sequence[str] source_class_names - the export's data.yaml "names" list, in its own
#         class-id order (index == source class id)
# Outputs: ClassRemap - id_remap (source id -> UNIFIED_CLASSES index), plus the mapped/dropped/
#          unmapped name breakdowns for the report
# Description: Builds the source->unified class id remap for one export by running every name
#              through map_source_class. A name mapped to None, or absent from CLASS_MAP, gets
#              no id_remap entry, so its label rows are dropped downstream.
# Side Effects: Raises ValueError if CLASS_MAP names a unified class that isn't in
#               UNIFIED_CLASSES (a typo in the mapping table would otherwise silently produce
#               an out-of-range class id in the written labels). No I/O.
def build_class_remap(source_class_names: Sequence[str]) -> ClassRemap:
    unified_index = {name: i for i, name in enumerate(UNIFIED_CLASSES)}
    id_remap: dict[int, int] = {}
    mapped: dict[str, str] = {}
    dropped: list[str] = []
    unmapped: list[str] = []

    for src_id, src_name in enumerate(source_class_names):
        if not is_known_class(src_name):
            unmapped.append(src_name)
            continue
        unified = map_source_class(src_name)
        if unified is None:
            dropped.append(src_name)
            continue
        if unified not in unified_index:
            raise ValueError(
                f"CLASS_MAP maps {src_name!r} to {unified!r}, which is not in UNIFIED_CLASSES "
                f"{list(UNIFIED_CLASSES)}"
            )
        id_remap[src_id] = unified_index[unified]
        mapped[src_name] = unified

    return ClassRemap(id_remap=id_remap, mapped=mapped, dropped=dropped, unmapped=unmapped)


@dataclass
class LabelRowStats:
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped_by_class: int = 0
    polygon_rows_converted: int = 0
    malformed_rows_skipped: int = 0


# tuple[str, LabelRowStats] remap_label_text(str label_text, ClassFilter class_filter)
# Inputs: str label_text - raw contents of one source YOLO label file (may mix detection and
#         polygon/segment rows)
#         ClassFilter class_filter - the source's remap, from ClassRemap.to_class_filter()
# Outputs: tuple[str, LabelRowStats] - (unified_label_text, stats): text with every row as
#          plain 5-field detection format carrying UNIFIED_CLASSES ids; stats tallies rows in/
#          out, rows dropped for an unmapped class, polygon conversions and malformed rows
# Description: Converts one source label file into the unified label space by chaining
#              prepare_dataset.normalize_label_text (polygon/segment rows collapsed to
#              axis-aligned bboxes, which Ultralytics otherwise refuses, dropping the WHOLE
#              image) with prepare_dataset.apply_class_filter (drops unmapped classes, renumbers
#              survivors). Both steps are reused, not reimplemented.
# Side Effects: None (pure text transformation; no I/O)
def remap_label_text(label_text: str, class_filter: ClassFilter) -> tuple[str, LabelRowStats]:
    normalized, norm_stats = normalize_label_text(label_text)
    remapped = apply_class_filter(normalized, class_filter)

    n_normalized = len(parse_yolo_label_classes(normalized))
    n_remapped = len(parse_yolo_label_classes(remapped))
    stats = LabelRowStats(
        rows_in=n_normalized + norm_stats.malformed_rows_skipped,
        rows_out=n_remapped,
        rows_dropped_by_class=n_normalized - n_remapped,
        polygon_rows_converted=norm_stats.polygon_rows_converted,
        malformed_rows_skipped=norm_stats.malformed_rows_skipped,
    )
    return remapped, stats


# --------------------------------------------------------------------------
# Hashing: exact content hash + dhash perceptual hash
# --------------------------------------------------------------------------


# str content_hash(np.ndarray image)
# Inputs: np.ndarray image - a decoded image array (BGR uint8 as returned by cv2.imread)
# Outputs: str - hex SHA256 digest over the array's dtype, shape and raw pixel bytes
# Description: Exact-duplicate fingerprint taken over DECODED PIXELS, not file bytes, so the
#              same photo re-encoded (PNG vs JPEG, re-saved by Roboflow, stripped EXIF) still
#              collides. dtype and shape are folded into the digest so two arrays with
#              identical bytes but different geometry can't collide.
# Side Effects: None (pure; the caller does the decoding)
def content_hash(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(image.dtype).encode("ascii"))
    h.update(str(image.shape).encode("ascii"))
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


# int dhash(np.ndarray image, int size)
# Inputs: np.ndarray image - a decoded image array (BGR or single-channel)
#         int size - dhash grid height/width, default DHASH_SIZE (8) -> a 64-bit hash
# Outputs: int - the perceptual hash as a size*size-bit integer
# Description: Difference hash: grayscale, resize to (size+1) x size, then emit one bit per
#              horizontally-adjacent pixel pair (1 if the right pixel is brighter). Robust to
#              rescaling, re-compression and small brightness shifts, so consecutive timelapse
#              frames of one print land within a few bits of each other (Defect B). Implemented
#              here with cv2/numpy -- both already dependencies -- rather than adding an
#              imagehash/Pillow dependency.
# Side Effects: None (pure array math)
def dhash(image: np.ndarray, size: int = DHASH_SIZE) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = (small[:, 1:] > small[:, :-1]).astype(np.uint8).flatten()
    return int.from_bytes(np.packbits(diff).tobytes(), "big")


# int hamming(int a, int b)
# Inputs: int a - first perceptual hash
#         int b - second perceptual hash
# Outputs: int - number of differing bits between a and b
# Description: Hamming distance between two dhashes; the similarity metric for near-duplicate
#              clustering.
# Side Effects: None
def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# list[tuple[int, int]] _band_slices(int n_bits, int n_bands)
# Inputs: int n_bits - hash width in bits, PHASH_BITS (64)
#         int n_bands - how many disjoint bands to cut the hash into
# Outputs: list[tuple[int, int]] - (bit_offset, mask) per band, covering all n_bits with no gaps
# Description: Partitions a hash into nearly-equal contiguous bit bands for the pigeonhole
#              blocking used by phash_neighbor_pairs: with n_bands = max_distance + 1, any two
#              hashes within max_distance MUST agree exactly on at least one band, so bucketing
#              by band value finds every near-duplicate pair without an O(n^2) sweep.
# Side Effects: None
def _band_slices(n_bits: int, n_bands: int) -> list[tuple[int, int]]:
    n_bands = max(1, min(n_bands, n_bits))
    edges = [round(i * n_bits / n_bands) for i in range(n_bands + 1)]
    out: list[tuple[int, int]] = []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            out.append((lo, (1 << (hi - lo)) - 1))
    return out


# list[tuple[str, str]] phash_neighbor_pairs(Mapping[str, int] hash_by_key, int max_distance)
# Inputs: Mapping[str, int] hash_by_key - image uid -> its dhash
#         int max_distance - inclusive Hamming-distance threshold, default
#         DEFAULT_PHASH_DISTANCE (6); 0 links only bit-identical hashes
# Outputs: list[tuple[str, str]] - uid pairs to union; unioning all of them yields exactly the
#          near-duplicate clusters (the pairs are a spanning set, not the full pair list)
# Description: Finds every pair of images whose dhashes are within max_distance bits, using
#              pigeonhole band blocking (see _band_slices) so a 30k-image dataset doesn't need
#              450M comparisons. Keys sharing an identical hash are first collapsed onto a
#              single representative, which also makes heavily-duplicated timelapse sources
#              cheap. Deterministic: keys and hashes are visited in sorted order.
# Side Effects: None (pure computation; no I/O or RNG)
def phash_neighbor_pairs(hash_by_key: Mapping[str, int], max_distance: int = DEFAULT_PHASH_DISTANCE) -> list[tuple[str, str]]:
    keys_by_hash: dict[int, list[str]] = {}
    for key in sorted(hash_by_key):
        keys_by_hash.setdefault(hash_by_key[key], []).append(key)

    pairs: list[tuple[str, str]] = []
    for h in sorted(keys_by_hash):
        keys = keys_by_hash[h]
        pairs.extend((keys[0], other) for other in keys[1:])

    if max_distance <= 0:
        return pairs

    rep_of_hash = {h: keys[0] for h, keys in keys_by_hash.items()}
    hashes = sorted(rep_of_hash)
    seen: set[tuple[int, int]] = set()
    for offset, mask in _band_slices(PHASH_BITS, max_distance + 1):
        buckets: dict[int, list[int]] = {}
        for h in hashes:
            buckets.setdefault((h >> offset) & mask, []).append(h)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    if hamming(a, b) <= max_distance:
                        pairs.append((rep_of_hash[a], rep_of_hash[b]))
    return pairs


# --------------------------------------------------------------------------
# Filename-derived session recovery (timestamps + sequence numbers)
# --------------------------------------------------------------------------

#: Filename timestamp shapes seen across these exports, tried in order. The first
#: whose digits parse as %Y%m%d%H%M%S wins. Generalizes
#: build_classification_dataset.parse_fdm_timestamp (which only handled FDM's
#: 17-digit "Image_YYYYMMDDHHMMSSmmm") to arbitrary Roboflow-renamed stems.
_TIMESTAMP_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})[-_ tT](\d{2})[-_:.](\d{2})[-_:.](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{8})[-_ tT](\d{6})(?!\d)"),
    re.compile(r"(?<!\d)(\d{14,17})(?!\d)"),
)

#: Splits a stem into (prefix, trailing number, suffix), e.g. "frame_0042" ->
#: ("frame_", 42, ""). The greedy ".*\D" makes it bind to the LAST digit run.
_SEQ_RE = re.compile(r"^(?P<prefix>.*\D)?(?P<num>\d{2,})(?P<suffix>\D*)$")


# datetime | None parse_filename_timestamp(str stem)
# Inputs: str stem - an image filename stem, ideally already stripped of Roboflow's
#         ".rf.<hash>" suffix
# Outputs: datetime | None - the capture timestamp embedded in the stem, or None if no
#          recognized pattern matched or the digits weren't a valid date/time
# Description: Recovers a capture time from a filename so timelapse frames can be segmented into
#              print sessions (Defect B). Handles the FDM-style 14-17 digit run plus the
#              "YYYY-MM-DD_HH-MM-SS" and "YYYYMMDD_HHMMSS" shapes Roboflow tends to preserve.
# Side Effects: None
def parse_filename_timestamp(stem: str) -> datetime | None:
    for pattern in _TIMESTAMP_RES:
        m = pattern.search(stem)
        if not m:
            continue
        digits = "".join(re.findall(r"\d", "".join(m.groups() or (m.group(0),))))
        if len(digits) < 14:
            continue
        try:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            continue
    return None


# tuple[str, int] | None parse_sequence_key(str stem)
# Inputs: str stem - an image filename stem, ideally already stripped of Roboflow's
#         ".rf.<hash>" suffix
# Outputs: tuple[str, int] | None - (family, index) where family is the stem with its trailing
#          number removed, or None if the stem has no 2+ digit trailing number
# Description: Recovers a frame index from a filename ("frame_0042" -> ("frame_|", 42)) so
#              consecutively-numbered captures from one print job can be segmented into
#              sessions when no timestamp is available. Stems whose only digits are a Roboflow
#              hash should be pre-stripped by the caller.
# Side Effects: None
def parse_sequence_key(stem: str) -> tuple[str, int] | None:
    m = _SEQ_RE.match(stem)
    if not m:
        return None
    return f"{m.group('prefix') or ''}|{m.group('suffix') or ''}", int(m.group("num"))


# dict[str, str] filename_session_groups(Sequence[str] stems, float gap_s, int seq_gap)
# Inputs: Sequence[str] stems - source stems (Roboflow ".rf.<hash>" already stripped) for ONE
#         source dataset
#         float gap_s - seconds between consecutive timestamps that start a new session,
#         default DEFAULT_SESSION_GAP_S (600)
#         int seq_gap - jump in frame index that starts a new session, default DEFAULT_SEQ_GAP (3)
# Outputs: dict[str, str] - stem -> session id ("ts<NNNNN>" or "seq:<family>:<NNNNN>"); stems
#          with neither a timestamp nor a sequence number are absent from the mapping
# Description: Segments one source's filenames into print sessions, the filename-derived half
#              of the Defect B guard (the other half is perceptual-hash clustering). Timestamped
#              stems are sorted chronologically and cut wherever the gap exceeds gap_s -- the
#              same rule as build_classification_dataset.fdm_session_groups, generalized to any
#              timestamp shape. Stems with no timestamp but a trailing frame index are grouped
#              per filename family and cut wherever the index jumps by more than seq_gap. A stem
#              with neither signal is left ungrouped so it stays its own singleton group rather
#              than being forced together with unrelated images.
# Side Effects: None (pure computation; no I/O or RNG)
def filename_session_groups(
    stems: Sequence[str],
    gap_s: float = DEFAULT_SESSION_GAP_S,
    seq_gap: int = DEFAULT_SEQ_GAP,
) -> dict[str, str]:
    unique = sorted(set(stems))
    session_of: dict[str, str] = {}

    timestamped = [(ts, s) for s in unique if (ts := parse_filename_timestamp(s)) is not None]
    timestamped.sort(key=lambda pair: (pair[0], pair[1]))
    idx = -1
    prev: datetime | None = None
    for ts, stem in timestamped:
        if prev is None or (ts - prev).total_seconds() > gap_s:
            idx += 1
        session_of[stem] = f"ts{idx:05d}"
        prev = ts

    families: dict[str, list[tuple[int, str]]] = {}
    for stem in unique:
        if stem in session_of:
            continue
        parsed = parse_sequence_key(stem)
        if parsed is None:
            continue
        family, num = parsed
        families.setdefault(family, []).append((num, stem))

    for family in sorted(families):
        entries = sorted(families[family])
        idx = -1
        prev_num: int | None = None
        for num, stem in entries:
            if prev_num is None or num - prev_num > seq_gap:
                idx += 1
            session_of[stem] = f"seq:{family}:{idx:05d}"
            prev_num = num

    return session_of


# --------------------------------------------------------------------------
# Union-find over leakage groups
# --------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set over image uids; one set == one indivisible leakage group."""

    # None __init__(self, Iterable[str] keys)
    # Inputs: Iterable[str] keys - every uid that must exist as its own singleton set up front
    # Outputs: None
    # Description: Seeds the structure with one singleton set per uid, so an image linked to
    #              nothing still forms a valid (size-1) group.
    # Side Effects: Initializes self._parent and self._rank.
    def __init__(self, keys: Iterable[str]) -> None:
        self._parent: dict[str, str] = {k: k for k in keys}
        self._rank: dict[str, int] = {k: 0 for k in self._parent}

    # str find(self, str key)
    # Inputs: str key - a uid known to this structure
    # Outputs: str - the representative uid of key's set
    # Description: Iterative find with path compression (iterative so a 20k-long chain of
    #              near-duplicate timelapse frames can't blow the recursion limit).
    # Side Effects: Rewrites self._parent entries along the path for future lookups.
    def find(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    # None union(self, str a, str b)
    # Inputs: str a - a uid known to this structure
    #         str b - another uid known to this structure
    # Outputs: None
    # Description: Merges the two uids' sets (union by rank), declaring the images
    #              non-separable across splits.
    # Side Effects: Mutates self._parent and self._rank.
    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    # dict[str, list[str]] components(self)
    # Inputs: None (operates on self)
    # Outputs: dict[str, list[str]] - representative uid -> sorted member uids
    # Description: Materializes the current disjoint sets, sorted within each set for
    #              deterministic downstream group naming.
    # Side Effects: Compacts paths via find as a side effect of traversal.
    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in sorted(self._parent):
            out.setdefault(self.find(key), []).append(key)
        return out


# --------------------------------------------------------------------------
# Scanning source exports
# --------------------------------------------------------------------------


@dataclass
class ImageRecord:
    """One image on disk, plus everything needed to group, split and attribute it."""

    source_key: str
    path: Path
    rel_path: str
    is_reference: bool  # True => scanned for collision detection only, never emitted
    label_path: Path | None = None
    source_stem: str = ""  # Roboflow ".rf.<hash>" stripped
    rf_matched: bool = False
    content_hash: str = ""
    phash: int = 0

    # str uid(self)
    # Inputs: None (operates on self)
    # Outputs: str - "<source_key>:<rel_path>", globally unique across all scanned datasets
    # Description: Stable identity used as the union-find key, the manifest key and the report's
    #              collision attribution, so every reported collision names both its dataset and
    #              its path within it.
    # Side Effects: None
    @property
    def uid(self) -> str:
        return f"{self.source_key}:{self.rel_path}"


@dataclass
class SourceScan:
    """Everything discovered about one scanned dataset root."""

    key: str
    root: Path
    is_reference: bool
    records: list[ImageRecord] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    declared_license: str | None = None  # from the export's own data.yaml, if it says
    n_label_files: int = 0
    n_label_rows: int = 0
    n_missing_labels: int = 0
    rf_fallback_count: int = 0


# list[str] read_data_yaml_class_names(Path root)
# Inputs: Path root - root of an extracted Roboflow YOLO export (contains data.yaml)
# Outputs: list[str] - the export's class names in its own class-id order
# Description: Reads the export's own class list, which is the input to build_class_remap.
#              Accepts both the list form ("names: [a, b]") and the dict form
#              ("names: {0: a, 1: b}") Roboflow has used across versions.
# Side Effects: Reads root/data.yaml. Raises FileNotFoundError if it's missing and ValueError if
#               it carries no usable "names" entry -- either case means the classes can't be
#               harmonized, so failing loudly beats guessing.
def read_data_yaml_class_names(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"No data.yaml found at {data_yaml}. Expected an extracted Roboflow YOLO export "
            "(data.yaml + train/valid/test subdirs)."
        )
    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = doc.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    raise ValueError(f"{data_yaml} has no usable 'names' list; cannot harmonize classes.")


# str | None read_data_yaml_license(Path root)
# Inputs: Path root - root of an extracted Roboflow YOLO export
# Outputs: str | None - the license string the export declares under its "roboflow" block, or
#          None if data.yaml is absent, unreadable or silent about it
# Description: Reads the license the export declares about ITSELF, so split_report.json can show
#              it beside the license hardcoded in SOURCE_SPECS and a disagreement is visible
#              rather than assumed away. Never raises: a missing license is a reporting gap, not
#              a reason to abort an ingest.
# Side Effects: Reads root/data.yaml if present.
def read_data_yaml_license(root: Path) -> str | None:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return None
    try:
        doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    block = doc.get("roboflow")
    if isinstance(block, dict) and block.get("license"):
        return str(block["license"])
    return str(doc["license"]) if doc.get("license") else None


# list[Path] _find_image_dirs(Path root)
# Inputs: Path root - root of an extracted dataset
# Outputs: list[Path] - every directory named "images" under root, plus root itself if it
#          directly holds image files; sorted, deduplicated
# Description: Locates the image directories of a YOLO export without assuming a fixed split
#              layout, since Roboflow exports variously ship train/valid, train/valid/test, or a
#              single flat images/ directory. The shipped split is irrelevant here (this script
#              always re-splits from scratch), so all of them are pooled.
# Side Effects: Read-only recursive filesystem traversal.
def _find_image_dirs(root: Path) -> list[Path]:
    dirs = {p for p in root.rglob("images") if p.is_dir()}
    if any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in root.iterdir()):
        dirs.add(root)
    return sorted(dirs)


# SourceScan scan_source(Path root, str source_key, bool is_reference)
# Inputs: Path root - root of an extracted dataset
#         str source_key - provenance key ("atco", "stereovision", or a reference key like
#         "fdm_raw")
#         bool is_reference - True for datasets scanned only for cross-dataset collision
#         detection (never emitted), which also skips the data.yaml/label requirements
# Outputs: SourceScan - the dataset's records (paths + label paths + recovered source stems),
#          class names, and label-count tallies
# Description: Discovers every image in one dataset root, pairs it with its sibling
#              labels/<stem>.txt when present, and recovers each file's Roboflow source identity
#              via prepare_dataset.extract_source_id (the Defect A augmentation-sibling key --
#              run for every source, including any that claims no augmentations, so the claim is
#              verified rather than assumed). Reference datasets are scanned
#              recursively with no layout or label expectations, since datasets/fdm_raw is a
#              tree of class-named folders rather than a YOLO export.
# Side Effects: Raises FileNotFoundError if root doesn't exist or holds no images; reads
#               data.yaml and every label file for non-reference sources. Read-only otherwise.
def scan_source(root: Path, source_key: str, is_reference: bool = False) -> SourceScan:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root for '{source_key}' does not exist: {root}")

    scan = SourceScan(key=source_key, root=root, is_reference=is_reference)

    if is_reference:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        image_entries = [(p, None) for p in paths]
    else:
        scan.class_names = read_data_yaml_class_names(root)
        scan.declared_license = read_data_yaml_license(root)
        image_entries = []
        for images_dir in _find_image_dirs(root):
            labels_dir = images_dir.parent / "labels"
            for p in sorted(images_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    image_entries.append((p, labels_dir / (p.stem + ".txt")))

    if not image_entries:
        raise FileNotFoundError(f"No images found under '{root}' for source '{source_key}'.")

    for path, label_path in image_entries:
        stem, matched = extract_source_id(path.name)
        if not matched:
            scan.rf_fallback_count += 1
        if label_path is not None:
            if label_path.is_file():
                scan.n_label_files += 1
                scan.n_label_rows += len(parse_yolo_label_classes(label_path.read_text(encoding="utf-8")))
            else:
                scan.n_missing_labels += 1
                label_path = None
        scan.records.append(
            ImageRecord(
                source_key=source_key,
                path=path,
                rel_path=path.relative_to(root).as_posix(),
                is_reference=is_reference,
                label_path=label_path,
                source_stem=stem,
                rf_matched=matched,
            )
        )
    return scan


# None hash_records(Sequence[ImageRecord] records, int workers, int progress_every)
# Inputs: Sequence[ImageRecord] records - records whose content_hash/phash are still unset
#         int workers - decode thread-pool size, default 0 (auto: min(8, cpu_count))
#         int progress_every - print a progress line every N images, default 1000 (0 silences)
# Outputs: None
# Description: Decodes every image once and fills in its exact content hash and dhash -- the raw
#              material for both cross-dataset duplicate detection (Defect C) and near-duplicate
#              session clustering (Defect B). Threaded because cv2's decode releases the GIL;
#              an undecodable file is skipped with a warning and left with an empty
#              content_hash so it can never be grouped with anything.
# Side Effects: Reads and decodes every record's image file; mutates each record's content_hash
#               and phash in place; prints progress and per-file decode warnings to stdout.
def hash_records(records: Sequence[ImageRecord], workers: int = 0, progress_every: int = 1000) -> None:
    n_workers = workers if workers > 0 else min(8, (os.cpu_count() or 4))

    # None _hash_one(ImageRecord rec)
    # Inputs: ImageRecord rec - the record to decode and fingerprint
    # Outputs: None
    # Description: Decodes one image and stores its content hash and dhash on the record.
    # Side Effects: Reads rec.path; mutates rec; prints a warning if the file won't decode.
    def _hash_one(rec: ImageRecord) -> None:
        img = cv2.imread(str(rec.path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ingest_multi] WARNING: could not decode '{rec.path}'; excluded from duplicate analysis.")
            return
        rec.content_hash = content_hash(img)
        rec.phash = dhash(img)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, _ in enumerate(pool.map(_hash_one, records), start=1):
            if progress_every and i % progress_every == 0:
                print(f"[ingest_multi]   hashed {i}/{len(records)} images ...")


# --------------------------------------------------------------------------
# Redundant-source pre-flight (Defect D)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StemContainment:
    """One source's source-photo stems, measured as a percentage already present in another's."""

    source: str
    contained_in: str
    n_shared: int
    n_unique: int
    pct: float

    # dict[str, object] to_dict(self)
    # Inputs: None (operates on self)
    # Outputs: dict[str, object] - JSON-serializable view of every field
    # Description: Converts this containment into a plain dict for split_report.json.
    # Side Effects: None
    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "contained_in": self.contained_in,
            "n_shared_stems": self.n_shared,
            "n_unique_stems": self.n_unique,
            "pct_of_source_already_present": round(self.pct, 2),
        }


# dict[str, set[str]] source_stem_sets(Mapping[str, SourceScan] scans)
# Inputs: Mapping[str, SourceScan] scans - source key -> its scan
# Outputs: dict[str, set[str]] - source key -> the set of distinct SOURCE PHOTO stems it holds
#          (Roboflow's ".rf.<hash>" suffix already stripped by scan_source)
# Description: Recovers each dataset's unique-photograph identity set, the unit that decides
#              whether a source contributes anything new. File counts do not: rf_failure's 8,853
#              files were 418 photos, all of them already in datasets/raw.
# Side Effects: None (reads already-scanned records; no I/O)
def source_stem_sets(scans: Mapping[str, SourceScan]) -> dict[str, set[str]]:
    return {key: {r.source_stem for r in scans[key].records} for key in sorted(scans)}


# dict[str, object] build_stem_overlap_matrix(Mapping[str, set[str]] stem_sets, float threshold_pct)
# Inputs: Mapping[str, set[str]] stem_sets - source key -> its unique source-photo stems
#         float threshold_pct - containment percentage at or above which a source is called
#         redundant, default REDUNDANT_SOURCE_PCT (90.0)
# Outputs: dict[str, object] - the report block: unique-stem count per source, the symmetric
#          pairwise shared-stem counts with both directional percentages, and the list of
#          directed containments at or above threshold_pct
# Description: Computes the pairwise source-photo overlap matrix that makes a redundant re-upload
#              visible (Defect D). Percentages are directional on purpose: a small source fully
#              swallowed by a large one is 100% redundant while the large one is barely affected,
#              and only the directional number exposes that asymmetry.
# Side Effects: None (pure computation; no I/O)
def build_stem_overlap_matrix(
    stem_sets: Mapping[str, set[str]], threshold_pct: float = REDUNDANT_SOURCE_PCT
) -> dict[str, object]:
    keys = sorted(stem_sets)
    pairwise: dict[str, dict[str, object]] = {}
    containments: list[StemContainment] = []

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = stem_sets[a] & stem_sets[b]
            pct_a = 100.0 * len(shared) / len(stem_sets[a]) if stem_sets[a] else 0.0
            pct_b = 100.0 * len(shared) / len(stem_sets[b]) if stem_sets[b] else 0.0
            pairwise[f"{a}|{b}"] = {
                "n_shared_stems": len(shared),
                f"pct_of_{a}": round(pct_a, 2),
                f"pct_of_{b}": round(pct_b, 2),
            }
            if pct_a >= threshold_pct:
                containments.append(StemContainment(a, b, len(shared), len(stem_sets[a]), pct_a))
            if pct_b >= threshold_pct:
                containments.append(StemContainment(b, a, len(shared), len(stem_sets[b]), pct_b))

    return {
        "threshold_pct": threshold_pct,
        "unique_source_stems": {k: len(stem_sets[k]) for k in keys},
        "pairwise": pairwise,
        "redundant_containments": [c.to_dict() for c in containments],
    }


# list[StemContainment] find_redundant_sources(Mapping[str, set[str]] stem_sets, Sequence[str] candidates, float threshold_pct)
# Inputs: Mapping[str, set[str]] stem_sets - source key -> its unique source-photo stems, for
#         EVERY scanned dataset (references included, so containment inside data already on disk
#         counts)
#         Sequence[str] candidates - the source keys actually headed for the output; only these
#         can be reported as redundant
#         float threshold_pct - containment percentage at or above which a source is redundant,
#         default REDUNDANT_SOURCE_PCT (90.0)
#         Sequence[str] | None compare_against - the source keys a candidate may be redundant
#         AGAINST, default None (every scanned source). Callers pass the emitted sources plus the
#         reference datasets: being contained in a source that is itself excluded from the output
#         is not redundancy, since that source's photos are not in the merge either.
# Outputs: list[StemContainment] - one entry per (candidate, other-source) pair where the
#          candidate's stems are at or above threshold_pct contained in the other's, worst first
# Description: Names the sources that would add (almost) no new photographs to the merge. This is
#              the exact condition that made rf_failure worthless -- 418 of 418 stems already in
#              datasets/raw, invisible behind an 8,853 file count.
# Side Effects: None (pure computation; no I/O)
def find_redundant_sources(
    stem_sets: Mapping[str, set[str]],
    candidates: Sequence[str],
    threshold_pct: float = REDUNDANT_SOURCE_PCT,
    compare_against: Sequence[str] | None = None,
) -> list[StemContainment]:
    others = sorted(stem_sets if compare_against is None else set(compare_against) & set(stem_sets))
    found: list[StemContainment] = []
    for key in sorted(candidates):
        own = stem_sets.get(key, set())
        if not own:
            continue
        for other in others:
            if other == key:
                continue
            pct = 100.0 * len(own & stem_sets[other]) / len(own)
            if pct >= threshold_pct:
                found.append(StemContainment(key, other, len(own & stem_sets[other]), len(own), pct))
    return sorted(found, key=lambda c: (-c.pct, c.source, c.contained_in))


# None assert_no_redundant_sources(Sequence[StemContainment] redundant, bool allow)
# Inputs: Sequence[StemContainment] redundant - the containments found by find_redundant_sources
#         bool allow - True to downgrade the failure to a pass (the --allow-redundant-sources
#         escape hatch), default False
# Outputs: None
# Description: Turns the redundant-source finding into a hard stop before a single file is
#              copied, so a re-upload can never be merged in under the impression that its file
#              count means new data. The message names both sources and the exact percentage.
# Side Effects: Raises ValueError listing every redundant source unless allow is True. No I/O.
def assert_no_redundant_sources(redundant: Sequence[StemContainment], allow: bool = False) -> None:
    if not redundant or allow:
        return
    lines = "\n".join(
        f"  - '{c.source}': {c.n_shared}/{c.n_unique} of its unique source photos "
        f"({c.pct:.1f}%) are already present in '{c.contained_in}'"
        for c in redundant
    )
    raise ValueError(
        "REDUNDANT SOURCE(S) detected before any files were copied -- these add (almost) no new "
        f"photographs:\n{lines}\n"
        "File counts hide this: rf_failure's 8,853 files were 418 photos, all already in "
        "datasets/raw. Drop the redundant source, or pass --allow-redundant-sources to merge it "
        "anyway (the duplicates are grouped so they still cannot straddle a split)."
    )


# --------------------------------------------------------------------------
# Grouping: exact dupes U near dupes U augmentation siblings U filename sessions
# --------------------------------------------------------------------------


# float session_corroboration(Sequence[ImageRecord] session_records, int distance)
# Inputs: Sequence[ImageRecord] session_records - every already-hashed record whose filename put
#         it in ONE filename-derived session
#         int distance - Hamming distance counting as "same scene", default
#         SESSION_CORROBORATION_DISTANCE (6)
# Outputs: float - fraction (0.0-1.0) of the session's DISTINCT SOURCE PHOTOS that have a
#          perceptual near-neighbour among the session's other source photos; 0.0 for an empty
#          session
# Description: Scores how much a filename-derived session looks like one real capture session.
#              One photo per source stem votes, never one per file: Roboflow augmentation copies
#              of a single photo are near-duplicates of each other by construction and would push
#              any session to ~100%. A continuously photographed print scores high (measured 61%
#              and 64% on this roster); a folder of independent photos that merely happen to be
#              numbered consecutively scores near zero (measured 0.0%-0.7%).
# Side Effects: None (pure computation over already-hashed records; no I/O)
def session_corroboration(
    session_records: Sequence[ImageRecord], distance: int = SESSION_CORROBORATION_DISTANCE
) -> float:
    rep_by_stem: dict[str, ImageRecord] = {}
    for rec in sorted(session_records, key=lambda r: r.uid):
        if rec.content_hash:
            rep_by_stem.setdefault(rec.source_stem, rec)
    if not rep_by_stem:
        return 0.0
    hashes = {rec.uid: rec.phash for rec in rep_by_stem.values()}
    linked: set[str] = set()
    for a, b in phash_neighbor_pairs(hashes, distance):
        linked.add(a)
        linked.add(b)
    return len(linked) / len(hashes)


# bool session_is_believable(Sequence[ImageRecord] session_records, float min_fraction, int distance, int min_stems)
# Inputs: Sequence[ImageRecord] session_records - the records of one filename-derived session
#         float min_fraction - corroboration a session must reach to be used for grouping,
#         default DEFAULT_SESSION_CORROBORATION (0.25); 0.0 believes every session
#         int distance - "same scene" Hamming distance, default SESSION_CORROBORATION_DISTANCE (6)
#         int min_stems - below this many distinct source photos the session is believed without
#         a vote, default SESSION_CORROBORATION_MIN_STEMS (3)
# Outputs: bool - True if this session may contribute grouping links
# Description: Gates one filename-derived session on perceptual corroboration, which is what stops
#              the sequence/timestamp single-linkage from chaining a numbered photo collection
#              into one enormous fake "print job". Small sessions pass unvoted: too few members to
#              judge, and too little damage if wrong.
# Side Effects: None (pure computation; no I/O)
def session_is_believable(
    session_records: Sequence[ImageRecord],
    min_fraction: float = DEFAULT_SESSION_CORROBORATION,
    distance: int = SESSION_CORROBORATION_DISTANCE,
    min_stems: int = SESSION_CORROBORATION_MIN_STEMS,
) -> bool:
    if min_fraction <= 0.0:
        return True
    if len({r.source_stem for r in session_records}) < min_stems:
        return True
    return session_corroboration(session_records, distance) >= min_fraction


@dataclass
class GroupingResult:
    group_of_uid: dict[str, str] = field(default_factory=dict)
    exact_sets: list[list[str]] = field(default_factory=list)
    near_sets: list[list[str]] = field(default_factory=list)
    cross_source_exact: list[list[str]] = field(default_factory=list)
    cross_source_near: list[list[str]] = field(default_factory=list)
    #: "<a>|<b>" (sorted source keys) -> {"exact": n, "near": n} image counts.
    collision_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    filename_sessions_per_source: dict[str, int] = field(default_factory=dict)
    #: Sessions the filename heuristic proposed but perceptual corroboration rejected.
    rejected_sessions_per_source: dict[str, int] = field(default_factory=dict)
    believed_sessions_per_source: dict[str, int] = field(default_factory=dict)


# GroupingResult assign_groups(Sequence[ImageRecord] records, int phash_distance, float session_gap_s, int seq_gap, bool use_filename_sessions, float session_corroboration_min)
# Inputs: Sequence[ImageRecord] records - every scanned record across all datasets, already
#         hashed
#         int phash_distance - near-duplicate Hamming threshold, default DEFAULT_PHASH_DISTANCE (6)
#         float session_gap_s - filename-timestamp session gap in seconds, default
#         DEFAULT_SESSION_GAP_S (600)
#         int seq_gap - filename sequence-number session gap, default DEFAULT_SEQ_GAP (3)
#         bool use_filename_sessions - whether filename timestamps/sequence numbers contribute
#         to grouping, default True
#         float session_corroboration_min - fraction of a filename session's source photos that
#         must be perceptually self-similar for it to be believed, default
#         DEFAULT_SESSION_CORROBORATION (0.25); 0.0 believes every session
# Outputs: GroupingResult - uid -> group id ("g000000"...), the exact/near duplicate sets, the
#          cross-dataset subsets of each, a per-source-pair collision matrix, and the proposed /
#          believed / rejected filename session counts per source
# Description: Computes the indivisible leakage group of every image as the transitive closure
#              of four links: identical decoded pixels (Defect C, spans datasets), dhash within
#              phash_distance (Defect B, catches timelapse neighbours and re-encoded re-uploads),
#              shared Roboflow source identity within one dataset (Defect A), and shared
#              CORROBORATED filename-derived print session within one dataset (Defect B). The
#              corroboration vote is what keeps the fourth link from chaining: sequence numbers
#              and timestamps are single-linkage signals, so an unvoted session heuristic merged
#              585 independently numbered AtCo photos into one "print job". Group ids are assigned
#              by sorted smallest-member uid, so they're deterministic and independent of scan
#              order. Reference datasets participate in grouping so a collision with a dataset
#              already on disk is detected and reported.
# Side Effects: None (pure computation over already-hashed records; no I/O or RNG)
def assign_groups(
    records: Sequence[ImageRecord],
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
    session_gap_s: float = DEFAULT_SESSION_GAP_S,
    seq_gap: int = DEFAULT_SEQ_GAP,
    use_filename_sessions: bool = True,
    session_corroboration_min: float = DEFAULT_SESSION_CORROBORATION,
) -> GroupingResult:
    result = GroupingResult()
    by_uid = {r.uid: r for r in records}
    uf = UnionFind(by_uid)

    # Link 1: identical decoded pixels (crosses dataset boundaries).
    uids_by_content: dict[str, list[str]] = {}
    for uid in sorted(by_uid):
        rec = by_uid[uid]
        if rec.content_hash:
            uids_by_content.setdefault(rec.content_hash, []).append(uid)
    for digest in sorted(uids_by_content):
        uids = uids_by_content[digest]
        if len(uids) < 2:
            continue
        result.exact_sets.append(uids)
        for other in uids[1:]:
            uf.union(uids[0], other)

    # Link 2: perceptually near-identical images (crosses dataset boundaries).
    hash_by_uid = {uid: by_uid[uid].phash for uid in sorted(by_uid) if by_uid[uid].content_hash}
    near_uf = UnionFind(hash_by_uid)
    for a, b in phash_neighbor_pairs(hash_by_uid, phash_distance):
        uf.union(a, b)
        near_uf.union(a, b)
    result.near_sets = [members for members in near_uf.components().values() if len(members) > 1]

    # Link 3: Roboflow augmentation siblings, per dataset (Defect A).
    by_rf: dict[tuple[str, str], list[str]] = {}
    for uid in sorted(by_uid):
        rec = by_uid[uid]
        if rec.rf_matched:
            by_rf.setdefault((rec.source_key, rec.source_stem), []).append(uid)
    for uids in by_rf.values():
        for other in uids[1:]:
            uf.union(uids[0], other)

    # Link 4: filename-derived print sessions, per dataset (Defect B).
    for source_key in sorted({r.source_key for r in records}):
        source_records = [by_uid[uid] for uid in sorted(by_uid) if by_uid[uid].source_key == source_key]
        sessions = filename_session_groups(
            [r.source_stem for r in source_records], session_gap_s, seq_gap
        )
        result.filename_sessions_per_source[source_key] = len(set(sessions.values()))
        result.believed_sessions_per_source[source_key] = 0
        result.rejected_sessions_per_source[source_key] = 0
        if not use_filename_sessions:
            continue
        by_session: dict[str, list[ImageRecord]] = {}
        for rec in source_records:
            session = sessions.get(rec.source_stem)
            if session is not None:
                by_session.setdefault(session, []).append(rec)
        for sid in sorted(by_session):
            members = by_session[sid]
            if not session_is_believable(members, session_corroboration_min):
                result.rejected_sessions_per_source[source_key] += 1
                continue
            result.believed_sessions_per_source[source_key] += 1
            uids = [r.uid for r in members]
            for other in uids[1:]:
                uf.union(uids[0], other)

    components = uf.components()
    ordered = sorted(components.values(), key=lambda members: members[0])
    for idx, members in enumerate(ordered):
        gid = f"g{idx:06d}"
        for uid in members:
            result.group_of_uid[uid] = gid

    # int _tally(list[list[str]] sets, str kind, list[list[str]] sink)
    # Inputs: list[list[str]] sets - duplicate/near-duplicate uid sets
    #         str kind - "exact" or "near", the collision-matrix sub-key
    #         list[list[str]] sink - list that cross-dataset sets are appended to
    # Outputs: int - number of sets that spanned more than one dataset
    # Description: Splits duplicate sets into same-dataset and cross-dataset ones and tallies the
    #              per-source-pair collision matrix that answers "is this export a re-upload of
    #              something already on disk?".
    # Side Effects: Appends to sink and mutates result.collision_matrix.
    def _tally(sets: list[list[str]], kind: str, sink: list[list[str]]) -> int:
        n_cross = 0
        for members in sets:
            sources = sorted({by_uid[uid].source_key for uid in members})
            if len(sources) < 2:
                continue
            n_cross += 1
            sink.append(members)
            for i, a in enumerate(sources):
                for b in sources[i + 1 :]:
                    cell = result.collision_matrix.setdefault(f"{a}|{b}", {"exact": 0, "near": 0})
                    cell[kind] += sum(1 for uid in members if by_uid[uid].source_key in (a, b))
        return n_cross

    _tally(result.exact_sets, "exact", result.cross_source_exact)
    _tally(result.near_sets, "near", result.cross_source_near)
    return result


# --------------------------------------------------------------------------
# Group-aware splitting (reuses prepare_dataset's already-tested split)
# --------------------------------------------------------------------------


# tuple[list[str], list[str], list[str]] split_groups(Sequence[str] group_ids, int seed, tuple[float, float, float] ratios)
# Inputs: Sequence[str] group_ids - leakage group ids (never individual files)
#         int seed - RNG seed for the deterministic shuffle, default DEFAULT_SEED (1337)
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.60, 0.25, 0.15); must sum to 1.0
# Outputs: tuple[list[str], list[str], list[str]] - (train, val, test) group id lists
# Description: Splits leakage GROUPS (not files) by ratios, delegating to
#              prepare_dataset.split_sources -- identical semantics (sort, seeded shuffle, slice,
#              remainder to test), so there is one implementation of the split rule in the repo.
# Side Effects: Raises ValueError if ratios don't sum to 1.0. Uses a locally-seeded
#               random.Random(seed); does not touch global RNG state.
def split_groups(
    group_ids: Sequence[str],
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[str], list[str], list[str]]:
    return split_sources(group_ids, seed=seed, ratios=ratios)


# None assert_no_group_overlap(Mapping[str, Sequence[str]] split_group_ids)
# Inputs: Mapping[str, Sequence[str]] split_group_ids - split name -> group ids assigned to it
# Outputs: None
# Description: Verifies no leakage group appears in two splits -- the single check the whole
#              script exists to guarantee. Delegates to prepare_dataset.assert_no_source_overlap
#              (same invariant, one implementation).
# Side Effects: Raises AssertionError naming the offending group and both splits. No I/O.
def assert_no_group_overlap(split_group_ids: Mapping[str, Sequence[str]]) -> None:
    assert_no_source_overlap({k: list(v) for k, v in split_group_ids.items()})


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


@dataclass
class SplitStats:
    n_groups: int = 0
    n_images: int = 0
    n_missing_labels: int = 0
    n_empty_labels: int = 0
    rows_dropped_by_class: int = 0
    polygon_rows_converted: int = 0
    malformed_rows_skipped: int = 0
    images_per_source: dict[str, int] = field(default_factory=dict)
    instances_per_class: dict[str, int] = field(default_factory=lambda: {c: 0 for c in UNIFIED_CLASSES})
    images_per_class: dict[str, int] = field(default_factory=lambda: {c: 0 for c in UNIFIED_CLASSES})

    # dict[str, object] to_dict(self)
    # Inputs: None (operates on self)
    # Outputs: dict[str, object] - JSON-serializable view of every field
    # Description: Converts this SplitStats into a plain dict for split_report.json, mirroring
    #              prepare_dataset.SplitStats.to_dict's shape.
    # Side Effects: None
    def to_dict(self) -> dict[str, object]:
        return {
            "n_groups": self.n_groups,
            "n_images": self.n_images,
            "n_missing_labels": self.n_missing_labels,
            "n_empty_labels": self.n_empty_labels,
            "rows_dropped_by_class": self.rows_dropped_by_class,
            "polygon_rows_converted": self.polygon_rows_converted,
            "malformed_rows_skipped": self.malformed_rows_skipped,
            "images_per_source": dict(sorted(self.images_per_source.items())),
            "instances_per_class": self.instances_per_class,
            "images_per_class": self.images_per_class,
        }


# str output_filename(ImageRecord record, str source_tag_mode, set[str] used)
# Inputs: ImageRecord record - the image being emitted
#         str source_tag_mode - "filename", "manifest" or "both"; the origin dataset is prefixed
#         onto the name unless the mode is "manifest"
#         set[str] used - output names already claimed in this dataset (mutated)
# Outputs: str - a unique output filename, e.g. "atco__frame_0042.jpg"
# Description: Builds the provenance-preserving output filename. The "<source_key>__" prefix is
#              what keeps every emitted image attributable to its origin dataset for the
#              per-source recall (source-confound) diagnostic without reading the manifest. A
#              collision -- the same name in two shipped split dirs -- gets a numeric suffix
#              rather than silently overwriting.
# Side Effects: Adds the returned name to used.
def output_filename(record: ImageRecord, source_tag_mode: str, used: set[str]) -> str:
    stem = Path(record.rel_path).stem
    if source_tag_mode in ("filename", "both"):
        stem = f"{record.source_key}__{stem}"
    suffix = record.path.suffix.lower()
    candidate = f"{stem}{suffix}"
    n = 1
    while candidate in used:
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    used.add(candidate)
    return candidate


# tuple[SplitStats, list[dict[str, object]]] materialize_split(str split, Sequence[ImageRecord] records, Mapping[str, ClassFilter] class_filters, Mapping[str, str] group_of_uid, Path out_dir, str source_tag_mode)
# Inputs: str split - "train", "val" or "test"
#         Sequence[ImageRecord] records - the images assigned to this split
#         Mapping[str, ClassFilter] class_filters - source key -> that export's class remap
#         Mapping[str, str] group_of_uid - uid -> leakage group id, for the manifest
#         Path out_dir - output dataset root
#         str source_tag_mode - "filename", "manifest" or "both"
#         bool write_output - actually copy images and write labels, default True; False makes
#         this an accounting-only pass (used by --dry-run, which still wants per-class counts)
# Outputs: tuple[SplitStats, list[dict[str, object]]] - (stats, manifest_rows): per-split tallies
#          and one provenance row per emitted image
# Description: Copies one split's images into out_dir/<split>/images and writes each one's label
#              file remapped into the unified label space (remap_label_text). Images are copied
#              byte-for-byte rather than re-encoded, so detection resolution is preserved -- see
#              SOURCE_CONFOUND_NOTE. An image left with zero rows after class harmonization is
#              still emitted, as a legitimate background/negative example. With write_output
#              False, labels are still read and harmonized so the class balance can be inspected
#              before committing to a multi-GB copy, but nothing is written.
# Side Effects: Creates out_dir/<split>/{images,labels}, copies image files (shutil.copy2) and
#               writes label files -- all skipped when write_output is False. Reads every
#               record's source label file. Prints a warning per image with no label file.
def materialize_split(
    split: str,
    records: Sequence[ImageRecord],
    class_filters: Mapping[str, ClassFilter],
    group_of_uid: Mapping[str, str],
    out_dir: Path,
    source_tag_mode: str = "both",
    write_output: bool = True,
) -> tuple[SplitStats, list[dict[str, object]]]:
    images_out = out_dir / split / "images"
    labels_out = out_dir / split / "labels"
    if write_output:
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

    stats = SplitStats(n_groups=len({group_of_uid[r.uid] for r in records}))
    manifest_rows: list[dict[str, object]] = []
    used: set[str] = set()

    for rec in sorted(records, key=lambda r: r.uid):
        name = output_filename(rec, source_tag_mode, used)
        if write_output:
            shutil.copy2(rec.path, images_out / name)
        stats.n_images += 1
        stats.images_per_source[rec.source_key] = stats.images_per_source.get(rec.source_key, 0) + 1

        if rec.label_path is None:
            print(f"[ingest_multi] WARNING: no label file for '{rec.uid}' (split={split}); writing an empty label.")
            stats.n_missing_labels += 1
            label_text = ""
        else:
            label_text, row_stats = remap_label_text(
                rec.label_path.read_text(encoding="utf-8"), class_filters[rec.source_key]
            )
            stats.rows_dropped_by_class += row_stats.rows_dropped_by_class
            stats.polygon_rows_converted += row_stats.polygon_rows_converted
            stats.malformed_rows_skipped += row_stats.malformed_rows_skipped

        if write_output:
            (labels_out / (Path(name).stem + ".txt")).write_text(label_text, encoding="utf-8")

        class_ids = parse_yolo_label_classes(label_text)
        if not class_ids:
            stats.n_empty_labels += 1
        seen_classes: set[int] = set()
        for cid in class_ids:
            if 0 <= cid < len(UNIFIED_CLASSES):
                stats.instances_per_class[UNIFIED_CLASSES[cid]] += 1
                seen_classes.add(cid)
        for cid in seen_classes:
            stats.images_per_class[UNIFIED_CLASSES[cid]] += 1

        manifest_rows.append(
            {
                "output": f"{split}/images/{name}",
                "split": split,
                "source": rec.source_key,
                "source_path": rec.rel_path,
                "group": group_of_uid[rec.uid],
                "content_sha256": rec.content_hash,
                "dhash": f"{rec.phash:016x}",
                "n_boxes": len(class_ids),
            }
        )
    return stats, manifest_rows


# Path write_data_yaml(Path out_dir, Sequence[str] included_sources)
# Inputs: Path out_dir - output dataset root
#         Sequence[str] included_sources - source keys actually emitted, for the provenance
#         header comment
# Outputs: Path - the written data.yaml path
# Description: Writes the merged dataset's data.yaml: train/val/test paths plus the unified class
#              list in UNIFIED_CLASSES order, with the contributing sources and their licenses
#              recorded as header comments so the effective license of a model trained on this
#              directory is visible without opening split_report.json.
# Side Effects: Writes (overwrites) out_dir/data.yaml.
def write_data_yaml(out_dir: Path, included_sources: Sequence[str]) -> Path:
    path = out_dir / "data.yaml"
    names_literal = ", ".join(f"'{c}'" for c in UNIFIED_CLASSES)
    license_lines = "\n".join(
        f"#   {k}: {SOURCE_SPEC_BY_KEY[k].license} ({SOURCE_SPEC_BY_KEY[k].origin})"
        for k in included_sources
        if k in SOURCE_SPEC_BY_KEY
    )
    content = (
        "# Generated by training/ingest_roboflow_multi.py -- do not hand-edit.\n"
        "# Sources merged into this dataset:\n"
        f"{license_lines or '#   (none)'}\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        f"names: [{names_literal}]\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Warnings
# --------------------------------------------------------------------------


# list[str] build_warnings(Mapping[str, SourceScan] scans, GroupingResult grouping, Mapping[str, ClassRemap] remaps, Sequence[str] included, Sequence[str] excluded, Mapping[str, int] groups_per_source)
# Inputs: Mapping[str, SourceScan] scans - source key -> its scan
#         GroupingResult grouping - the computed grouping/collision analysis
#         Mapping[str, ClassRemap] remaps - source key -> its class remap
#         Sequence[str] included - source keys emitted into the output
#         Sequence[str] excluded - source keys scanned but not emitted (license gating)
#         Mapping[str, int] groups_per_source - source key -> distinct leakage groups its
#         images fall into
#         Mapping[str, SplitStats] | None split_stats - per-split tallies, default None (skips
#         the split-imbalance and empty-class checks)
#         tuple[float, float, float] ratios - the requested group-level ratios, default
#         DEFAULT_SPLIT_RATIOS
#         Sequence[StemContainment] redundant - redundant included sources that were merged
#         anyway under --allow-redundant-sources, default ()
# Outputs: list[str] - human-readable warnings, most severe first; never empty (the
#          source-confound note is always present)
# Description: Turns the leakage analysis into the report's explicit warnings list: redundant
#              sources merged despite adding no new photographs, cross-dataset collisions (naming
#              the dataset pair, which is how a re-upload surfaces), timelapse/augmentation
#              suspicion where a source's images collapse into far fewer groups than frames,
#              sources with too few groups to split meaningfully, class names the mapping table
#              doesn't recognize, missing labels, the license gating actually in effect, and the
#              image-level split skew a group-level split produces when group sizes vary a lot
#              (an even 70/15/15 over groups measured out at a lopsided 89/5/6 over images).
# Side Effects: None
def build_warnings(
    scans: Mapping[str, SourceScan],
    grouping: GroupingResult,
    remaps: Mapping[str, ClassRemap],
    included: Sequence[str],
    excluded: Sequence[str],
    groups_per_source: Mapping[str, int],
    split_stats: Mapping[str, SplitStats] | None = None,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    redundant: Sequence[StemContainment] = (),
) -> list[str]:
    warnings: list[str] = []

    for c in redundant:
        warnings.append(
            f"REDUNDANT SOURCE MERGED ANYWAY: '{c.source}' has {c.n_shared}/{c.n_unique} of its "
            f"unique source photos ({c.pct:.1f}%) already present in '{c.contained_in}', so it "
            "adds (almost) no new photographs -- only more augmented copies of images the "
            "dataset already had. It was merged because --allow-redundant-sources was passed."
        )

    if split_stats:
        total_images = sum(s.n_images for s in split_stats.values()) or 1
        shares = {name: split_stats[name].n_images / total_images for name in SPLIT_NAMES}
        wanted = dict(zip(SPLIT_NAMES, ratios))
        skewed = [n for n in ("val", "test") if shares[n] < wanted[n] * SPLIT_SHARE_TOLERANCE]
        if skewed:
            actual = ", ".join(f"{n}={shares[n] * 100:.1f}%" for n in SPLIT_NAMES)
            target = ", ".join(f"{n}={wanted[n] * 100:.0f}%" for n in SPLIT_NAMES)
            warnings.append(
                f"SPLIT IMBALANCE (by images): groups were split {target} as requested, but the "
                f"IMAGE counts landed at {actual} because group sizes vary a lot (a few groups "
                "hold many frames each). This is the correct, leak-free outcome -- splitting "
                "images evenly would put frames of one print job on both sides -- but "
                f"{'/'.join(skewed)} is smaller than it looks like it should be. Raise the val/"
                "test ratios if you need a larger evaluation set."
            )
        empty = [
            c
            for c in UNIFIED_CLASSES
            if not any(split_stats[n].instances_per_class.get(c, 0) for n in SPLIT_NAMES)
        ]
        if empty:
            warnings.append(
                f"EMPTY CLASSES: {empty} have zero boxes in the merged dataset. Training on a "
                "class with no instances teaches the model nothing about it; drop them from "
                "UNIFIED_CLASSES or add a source that covers them."
            )
        unevaluable = [
            c
            for c in UNIFIED_CLASSES
            if any(split_stats[n].instances_per_class.get(c, 0) for n in SPLIT_NAMES)
            and not (split_stats["val"].instances_per_class.get(c, 0) and split_stats["test"].instances_per_class.get(c, 0))
        ]
        if unevaluable:
            warnings.append(
                f"NOT EVALUABLE: {unevaluable} have instances in the dataset but not in both val "
                "AND test, so per-class recall for them cannot be measured on held-out data."
            )

    for pair in sorted(grouping.collision_matrix):
        cell = grouping.collision_matrix[pair]
        a, b = pair.split("|")
        involves_reference = any(k in scans and scans[k].is_reference for k in (a, b))
        remedy = (
            " Pass --drop-reference-duplicates to omit output images that duplicate a reference "
            "dataset."
            if involves_reference
            else ""
        )
        warnings.append(
            f"CROSS-DATASET COLLISION: '{a}' and '{b}' share images -- {cell['exact']} in exact "
            f"pixel-identical sets, {cell['near']} in perceptual near-duplicate sets. They are "
            "grouped together so they cannot straddle a split, but if one of them was used to "
            "train or evaluate an earlier model, metrics computed against the other are "
            f"contaminated.{remedy}"
        )

    for key in sorted(scans):
        scan = scans[key]
        n_images = len(scan.records)
        n_groups = groups_per_source.get(key, 0)
        if n_groups and n_images / n_groups >= TIMELAPSE_FRAMES_PER_GROUP:
            warnings.append(
                f"TIMELAPSE SUSPECTED in '{key}': {n_images} images collapse into only {n_groups} "
                f"independent groups ({n_images / n_groups:.1f} frames per group), i.e. timelapse "
                "frames of one job and/or Roboflow augmentation copies of one photo. Its real "
                "evaluation sample size is the GROUP count, not the file count -- treat any "
                f"per-source metric as being computed over ~{n_groups} independent scenes."
            )
        if 0 < n_groups < MIN_GROUPS_FOR_SPLIT and not scan.is_reference:
            warnings.append(
                f"TOO FEW GROUPS in '{key}': only {n_groups} independent groups; splitting this "
                "source alone would not produce a meaningful val/test."
            )
        if scan.n_missing_labels:
            warnings.append(f"MISSING LABELS in '{key}': {scan.n_missing_labels} images have no label file.")
        if scan.rf_fallback_count and not scan.is_reference:
            warnings.append(
                f"NOTE for '{key}': {scan.rf_fallback_count}/{n_images} filenames carry no "
                "Roboflow '.rf.<hash>' suffix, so augmentation-sibling grouping was a no-op for "
                "them (expected for an export declaring no augmentations; they are still grouped "
                "by perceptual hash and filename session)."
            )

    for key in sorted(remaps):
        remap = remaps[key]
        if remap.unmapped:
            warnings.append(
                f"UNMAPPED CLASSES in '{key}': {remap.unmapped} are absent from CLASS_MAP, so "
                "their label rows were DROPPED. Add them to CLASS_MAP (or map them to None to "
                "drop them deliberately)."
            )
        if remap.dropped:
            warnings.append(f"DROPPED CLASSES in '{key}': {remap.dropped} are mapped to None in CLASS_MAP.")

    # Only sources dropped *because* they are non-commercial belong in a license
    # warning. A run with no non-commercial sources at all (both current sources
    # are permissive) must stay silent rather than imply gating happened.
    gated = [k for k in excluded if k in SOURCE_SPEC_BY_KEY and SOURCE_SPEC_BY_KEY[k].noncommercial]
    nc_included = [k for k in included if k in SOURCE_SPEC_BY_KEY and SOURCE_SPEC_BY_KEY[k].noncommercial]
    if gated:
        details = ", ".join(f"{k} ({SOURCE_SPEC_BY_KEY[k].license})" for k in gated)
        warnings.append(
            f"LICENSE GATING: excluded non-commercial sources: {details}. The merged output is "
            "buildable under the remaining licenses; pass --no-exclude-noncommercial to include "
            "them (the result is then NOT redistributable under this repo's GPL-3.0)."
        )
    if nc_included:
        warnings.append(
            f"LICENSE: non-commercial sources {nc_included} ARE included. Any model trained on "
            "this dataset inherits their CC BY-NC terms and cannot be redistributed under this "
            "repo's GPL-3.0."
        )

    for key in sorted(included):
        spec = SOURCE_SPEC_BY_KEY.get(key)
        declared = scans[key].declared_license if key in scans else None
        if spec and declared and normalize_class_name(declared) != normalize_class_name(spec.license):
            warnings.append(
                f"LICENSE MISMATCH for '{key}': SOURCE_SPECS says {spec.license!r} but the "
                f"export's own data.yaml declares {declared!r}. Trust the export; fix the table."
            )

    warnings.append(SOURCE_CONFOUND_NOTE)
    return warnings


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


# dict[str, object] build_merged_dataset(Mapping[str, SourceScan] scans, Path out_dir, ...)
# Inputs: Mapping[str, SourceScan] scans - source key -> scanned+hashed dataset (input sources
#         and reference datasets alike); keyword-only
#         Path out_dir - output dataset root; keyword-only
#         Sequence[str] | None include - source keys to emit; None means every non-reference,
#         non-noncommercial source (subject to exclude_noncommercial); keyword-only
#         bool exclude_noncommercial - drop CC BY-NC sources from the output, default True;
#         keyword-only
#         int seed / tuple ratios / int phash_distance / float session_gap_s / int seq_gap /
#         bool use_filename_sessions / float session_corroboration_min / bool
#         drop_reference_duplicates / str source_tag /
#         bool allow_redundant_sources / bool dry_run - see parse_args; all keyword-only
# Outputs: dict[str, object] - the full split_report.json content
# Description: The orchestration core, kept free of argparse and of dataset discovery so tests
#              can drive it with tiny synthetic scans. Applies license gating, refuses redundant
#              sources BEFORE any copying (Defect D), groups every scanned image (assign_groups),
#              drops exact duplicates down to one representative, splits the surviving GROUPS by
#              ratios, asserts zero group overlap, materializes the splits, writes data.yaml plus
#              manifest.jsonl, re-verifies zero overlap from the manifest actually written, and
#              assembles the report.
# Side Effects: Creates out_dir and its split subdirectories, copies images and writes label
#               files, data.yaml and manifest.jsonl (all skipped when dry_run is True). Raises
#               SystemExit-free ValueError if a source is redundant or too few groups exist to
#               split, or AssertionError if group overlap is ever detected. Uses locally-seeded
#               RNG only.
def build_merged_dataset(
    *,
    scans: Mapping[str, SourceScan],
    out_dir: Path,
    include: Sequence[str] | None = None,
    exclude_noncommercial: bool = True,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
    session_gap_s: float = DEFAULT_SESSION_GAP_S,
    seq_gap: int = DEFAULT_SEQ_GAP,
    use_filename_sessions: bool = True,
    session_corroboration_min: float = DEFAULT_SESSION_CORROBORATION,
    drop_reference_duplicates: bool = False,
    source_tag: str = "both",
    allow_redundant_sources: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    all_records = [r for key in sorted(scans) for r in scans[key].records]
    by_uid = {r.uid: r for r in all_records}

    candidate_sources = [k for k in sorted(scans) if not scans[k].is_reference]
    if include is None:
        # A key with no SourceSpec has no declared license, so it can't be
        # non-commercial by declaration and is kept.
        included = [
            k
            for k in candidate_sources
            if not (exclude_noncommercial and k in SOURCE_SPEC_BY_KEY and SOURCE_SPEC_BY_KEY[k].noncommercial)
        ]
    else:
        included = [k for k in candidate_sources if k in include]
    excluded = [k for k in candidate_sources if k not in included]

    # Defect D pre-flight: refuse a source that adds no new photographs BEFORE a
    # single image is copied. Runs against every scanned dataset, references
    # included, so containment inside data already on disk counts.
    stem_sets = source_stem_sets(scans)
    stem_overlap = build_stem_overlap_matrix(stem_sets, REDUNDANT_SOURCE_PCT)
    redundant = find_redundant_sources(
        stem_sets,
        included,
        REDUNDANT_SOURCE_PCT,
        compare_against=list(included) + [k for k in scans if scans[k].is_reference],
    )
    stem_overlap["allow_redundant_sources"] = allow_redundant_sources
    stem_overlap["redundant_included_sources"] = [c.to_dict() for c in redundant]
    assert_no_redundant_sources(redundant, allow_redundant_sources)

    grouping = assign_groups(
        all_records, phash_distance, session_gap_s, seq_gap, use_filename_sessions,
        session_corroboration_min,
    )

    remaps = {k: build_class_remap(scans[k].class_names) for k in candidate_sources}
    class_filters = {k: remaps[k].to_class_filter() for k in candidate_sources}

    # Exact duplicates collapse to one representative so the same pixels are not
    # written twice; source priority order decides which copy survives.
    priority = {spec.key: i for i, spec in enumerate(SOURCE_SPECS)}
    emit: list[ImageRecord] = []
    seen_hashes: set[str] = set()
    reference_hashes = {
        r.content_hash for k in scans if scans[k].is_reference for r in scans[k].records if r.content_hash
    }
    reference_groups = {
        grouping.group_of_uid[r.uid] for k in scans if scans[k].is_reference for r in scans[k].records
    }
    n_exact_dropped = 0
    n_reference_dropped = 0
    for rec in sorted(all_records, key=lambda r: (priority.get(r.source_key, 99), r.uid)):
        if rec.source_key not in included:
            continue
        if rec.content_hash and rec.content_hash in seen_hashes:
            n_exact_dropped += 1
            continue
        if drop_reference_duplicates and (
            rec.content_hash in reference_hashes or grouping.group_of_uid[rec.uid] in reference_groups
        ):
            n_reference_dropped += 1
            continue
        if rec.content_hash:
            seen_hashes.add(rec.content_hash)
        emit.append(rec)

    emit_groups = sorted({grouping.group_of_uid[r.uid] for r in emit})
    if len(emit_groups) < MIN_GROUPS_FOR_SPLIT:
        raise ValueError(
            f"Only {len(emit_groups)} independent leakage groups across the included sources "
            f"(minimum {MIN_GROUPS_FOR_SPLIT} to split {ratios}). Everything collapsed into a "
            "handful of groups, which usually means the images really are a few timelapses -- "
            "or that grouping is too aggressive. Inspect split_report.json, then tune "
            "--phash-distance / --seq-gap or pass --no-filename-sessions."
        )

    train_ids, val_ids, test_ids = split_groups(emit_groups, seed, ratios)
    split_group_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    assert_no_group_overlap(split_group_ids)
    for name in ("val", "test"):
        if not split_group_ids[name]:
            raise ValueError(
                f"Split '{name}' received 0 groups from {len(emit_groups)} total groups at "
                f"ratios {ratios}; refusing to write a dataset with an empty evaluation split."
            )

    split_of_group = {gid: name for name, ids in split_group_ids.items() for gid in ids}
    records_by_split: dict[str, list[ImageRecord]] = {s: [] for s in SPLIT_NAMES}
    for rec in emit:
        records_by_split[split_of_group[grouping.group_of_uid[rec.uid]]].append(rec)

    stats_by_split: dict[str, SplitStats] = {}
    manifest_rows: list[dict[str, object]] = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        stats, rows = materialize_split(
            split,
            records_by_split[split],
            class_filters,
            grouping.group_of_uid,
            out_dir,
            source_tag,
            write_output=not dry_run,
        )
        stats_by_split[split] = stats
        manifest_rows.extend(rows)

    if not dry_run:
        write_data_yaml(out_dir, included)
        if source_tag in ("manifest", "both"):
            (out_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
            )
        # Re-verify from what was actually written, not just the in-memory split.
        written: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
        for row in manifest_rows:
            written[str(row["split"])].append(str(row["group"]))
        assert_no_group_overlap({s: sorted(set(v)) for s, v in written.items()})

    groups_per_source = {
        key: len({grouping.group_of_uid[r.uid] for r in scans[key].records}) for key in sorted(scans)
    }
    warnings = build_warnings(
        scans, grouping, remaps, included, excluded, groups_per_source, stats_by_split, ratios,
        redundant,
    )

    # dict[str, object] _source_report(str key)
    # Inputs: str key - a scanned dataset key
    # Outputs: dict[str, object] - that source's report block
    # Description: Assembles one source's per-source report entry: the THREE honest size numbers
    #              (raw files / unique source photos / independent groups), license and inclusion
    #              status, and the class harmonization result. The three numbers are never
    #              collapsed: 13,822 stereovision files are 2,909 photographs, and a file count
    #              read as a data volume is exactly how a redundant re-upload gets merged.
    # Side Effects: None
    def _source_report(key: str) -> dict[str, object]:
        scan = scans[key]
        spec = SOURCE_SPEC_BY_KEY.get(key)
        n_images = len(scan.records)
        n_groups = groups_per_source.get(key, 0)
        n_stems = len(stem_sets.get(key, set()))
        remap = remaps.get(key)
        return {
            "root": str(scan.root),
            "role": "reference" if scan.is_reference else "input",
            "origin": spec.origin if spec else None,
            "license": spec.license if spec else "unknown",
            "license_declared_in_data_yaml": scan.declared_license,
            "noncommercial": bool(spec.noncommercial) if spec else None,
            "included_in_output": key in included,
            "n_files": n_images,
            "n_unique_source_stems": n_stems,
            "n_groups": n_groups,
            "files_per_unique_source_photo": round(n_images / n_stems, 2) if n_stems else None,
            "n_images": n_images,
            "n_label_files": scan.n_label_files,
            "n_label_rows": scan.n_label_rows,
            "n_missing_labels": scan.n_missing_labels,
            "n_unique_content_hashes": len({r.content_hash for r in scan.records if r.content_hash}),
            "rf_suffix_absent": scan.rf_fallback_count,
            "source_class_names": list(scan.class_names),
            "class_map_applied": dict(remap.mapped) if remap else {},
            "class_map_dropped": list(remap.dropped) if remap else [],
            "class_map_unmapped": list(remap.unmapped) if remap else [],
            "raw_frame_count": n_images,
            "estimated_independent_sessions": n_groups,
            "filename_derived_sessions": grouping.filename_sessions_per_source.get(key, 0),
            "filename_sessions_believed": grouping.believed_sessions_per_source.get(key, 0),
            "filename_sessions_rejected_as_uncorroborated": grouping.rejected_sessions_per_source.get(key, 0),
            "frames_per_session": round(n_images / n_groups, 2) if n_groups else None,
        }

    return {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "dry_run": dry_run,
        "grouping_params": {
            "phash_distance": phash_distance,
            "session_gap_s": session_gap_s,
            "seq_gap": seq_gap,
            "filename_sessions_used": use_filename_sessions,
            "session_corroboration_min": session_corroboration_min,
            "drop_reference_duplicates": drop_reference_duplicates,
        },
        "unified_class_names": list(UNIFIED_CLASSES),
        "severity_by_class": {c: SEVERITY_BY_CLASS[c].value for c in UNIFIED_CLASSES},
        "class_map": {k: v for k, v in sorted(CLASS_MAP.items())},
        "license_gating": {
            "exclude_noncommercial": exclude_noncommercial,
            "included_sources": included,
            "excluded_sources": excluded,
            "included_licenses": sorted(
                {SOURCE_SPEC_BY_KEY[k].license for k in included if k in SOURCE_SPEC_BY_KEY}
            ),
        },
        "sources": {key: _source_report(key) for key in sorted(scans)},
        # The three honest numbers, per source and overall. They are NOT
        # interchangeable: files count augmented copies, unique_source_photos
        # counts distinct photographs, groups counts what the split can actually
        # separate -- and only the last one is the real evaluation sample size.
        "dataset_size": {
            "per_source": {
                key: {
                    "role": "reference" if scans[key].is_reference else "input",
                    "included_in_output": key in included,
                    "raw_files": len(scans[key].records),
                    "unique_source_photos": len(stem_sets.get(key, set())),
                    "independent_groups": groups_per_source.get(key, 0),
                }
                for key in sorted(scans)
            },
            "included_total": {
                "raw_files": sum(len(scans[k].records) for k in included),
                "unique_source_photos": len(set().union(*(stem_sets[k] for k in included))),
                "independent_groups": len(emit_groups),
                "emitted_files": len(emit),
            },
        },
        "source_stem_overlap": stem_overlap,
        "duplicates": {
            "exact_duplicate_sets": len(grouping.exact_sets),
            "near_duplicate_sets": len(grouping.near_sets),
            "cross_dataset_exact_sets": len(grouping.cross_source_exact),
            "cross_dataset_near_sets": len(grouping.cross_source_near),
            "collision_matrix": grouping.collision_matrix,
            "images_dropped_as_exact_duplicates": n_exact_dropped,
            "images_dropped_as_reference_duplicates": n_reference_dropped,
            "cross_dataset_exact_examples": [
                sorted(s) for s in grouping.cross_source_exact[:REPORT_SAMPLE_CAP]
            ],
            "cross_dataset_near_examples": [
                sorted(s)[:8] for s in grouping.cross_source_near[:REPORT_SAMPLE_CAP]
            ],
        },
        "sessions": {
            key: {
                "raw_frames": len(scans[key].records),
                "estimated_independent_sessions": groups_per_source.get(key, 0),
                "filename_derived_sessions": grouping.filename_sessions_per_source.get(key, 0),
                "filename_sessions_believed": grouping.believed_sessions_per_source.get(key, 0),
                "filename_sessions_rejected": grouping.rejected_sessions_per_source.get(key, 0),
            }
            for key in sorted(scans)
        },
        "total_groups": len(set(grouping.group_of_uid.values())),
        "emitted_groups": len(emit_groups),
        "emitted_images": len(emit),
        "splits": {s: stats_by_split[s].to_dict() for s in SPLIT_NAMES},
        "group_overlap_check": (
            "PASS: zero leakage-group overlap between train/val/test (checked on the in-memory "
            "split and re-checked against the written manifest)"
            if not dry_run
            else "PASS (in-memory split only; --dry-run wrote nothing)"
        ),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed options. Notable defaults: --seed 1337, ratios
#          (0.60, 0.25, 0.15), --phash-distance 6, --session-gap-s 600, --seq-gap 3,
#          --exclude-noncommercial ON, --allow-redundant-sources OFF, --source-tag both.
# Description: Defines and parses the CLI for merging multiple Roboflow exports. One
#              --<key> flag is generated per entry in SOURCE_SPECS, so adding a source there
#              adds its flag here.
# Side Effects: None (argparse may print usage/help and exit; no filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for spec in SOURCE_SPECS:
        p.add_argument(
            f"--{spec.key}",
            type=Path,
            default=None,
            help=f"Root of the extracted '{spec.origin}' export ({spec.license}).",
        )
    p.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=None,
        help="Dataset root scanned ONLY for cross-dataset duplicate detection, never emitted. "
        "Repeatable. No default: datasets/raw is now an ingest source (--atco), and "
        "datasets/fdm_raw is 5.9 GB from the classification pipeline, so pass it explicitly "
        "if you want that comparison.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"Output dataset dir (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for the group-level split (default: {DEFAULT_SEED})")
    p.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0], help=f"Group-level train share (default: {DEFAULT_SPLIT_RATIOS[0]})")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_RATIOS[1],
        help=f"Group-level val share (default: {DEFAULT_SPLIT_RATIOS[1]}, deliberately above the "
        "usual 0.15: a 70/15/15 GROUP split measured out at 89/4.9/6.1 by image count, and ~5%% "
        "of images is too thin to pick confidence thresholds against).",
    )
    p.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2], help=f"Group-level test share (default: {DEFAULT_SPLIT_RATIOS[2]})")
    p.add_argument(
        "--exclude-noncommercial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Omit sources declared CC BY-NC from the merged output so the result stays "
        "compatible with this repo's GPL-3.0. ON by default; a no-op today since every source "
        "in SOURCE_SPECS is permissive. Pass --no-exclude-noncommercial to include them anyway.",
    )
    p.add_argument(
        "--allow-redundant-sources",
        action="store_true",
        help=f"Merge a source even when {REDUNDANT_SOURCE_PCT:g}%% or more of its unique source "
        "photos are already present in another source. OFF by default: rf_defects and rf_failure "
        "were 100%% contained in datasets/raw, and their 5,869/8,853 file counts made that "
        "invisible. The overlap matrix is reported either way.",
    )
    p.add_argument(
        "--source-tag",
        choices=("filename", "manifest", "both"),
        default="both",
        help="How each output image stays attributable to its origin dataset for the per-source "
        "recall (source-confound) diagnostic: a '<source>__' filename prefix, a manifest.jsonl "
        "sidecar, or both (default: both).",
    )
    p.add_argument("--phash-distance", type=int, default=DEFAULT_PHASH_DISTANCE, help=f"dhash Hamming distance at or below which two images are one group (default: {DEFAULT_PHASH_DISTANCE})")
    p.add_argument("--session-gap-s", type=float, default=DEFAULT_SESSION_GAP_S, help=f"Filename-timestamp gap (s) starting a new print session (default: {DEFAULT_SESSION_GAP_S:g})")
    p.add_argument("--seq-gap", type=int, default=DEFAULT_SEQ_GAP, help=f"Filename sequence-number gap starting a new print session (default: {DEFAULT_SEQ_GAP})")
    p.add_argument(
        "--filename-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let filename timestamps/sequence numbers contribute to grouping (default: on).",
    )
    p.add_argument(
        "--session-corroboration",
        type=float,
        default=DEFAULT_SESSION_CORROBORATION,
        help="Fraction of a filename-derived session's source photos that must be perceptually "
        f"self-similar before that session is believed (default: {DEFAULT_SESSION_CORROBORATION}). "
        "Sequence numbers and timestamps are single-linkage signals, so without this vote a "
        "folder of 585 consecutively numbered independent photos becomes one 'print job'. Pass 0 "
        "to believe every session (the pre-2026-09 behaviour).",
    )
    p.add_argument("--drop-reference-duplicates", action="store_true", help="Omit output images that duplicate an image in a --reference dataset.")
    p.add_argument("--workers", type=int, default=0, help="Image-decode threads for hashing (default: auto)")
    p.add_argument("--force", action="store_true", help="Wipe and rebuild --out if it already exists")
    p.add_argument("--dry-run", action="store_true", help="Scan, hash and analyze, but write only split_report.json")
    return p.parse_args(argv)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Scans and hashes every provided source plus the reference
#              datasets, runs the merge/grouping/split via build_merged_dataset, writes
#              split_report.json, and prints the full report: the three-number effective size
#              (files / unique source photos / independent groups), the source-photo overlap
#              matrix, cross-dataset collisions, per-split per-class counts, and every warning.
# Side Effects: Reads and decodes every image in every provided dataset root; optionally wipes
#               --out with shutil.rmtree when --force is passed; writes the merged dataset,
#               data.yaml, manifest.jsonl and split_report.json (all but the report skipped under
#               --dry-run); calls sys.exit with a clear message if a path is missing, no source
#               was given, a source is redundant, or the dataset can't be split; prints progress
#               and the report to stdout.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    requested = [(spec.key, getattr(args, spec.key)) for spec in SOURCE_SPECS if getattr(args, spec.key) is not None]
    if not requested:
        sys.exit(
            "[ingest_multi] ERROR: no source given. Pass at least one of "
            + ", ".join(f"--{s.key} <dir>" for s in SOURCE_SPECS)
        )

    if args.reference is None:
        reference_dirs = [(d.name, d) for d in DEFAULT_REFERENCE_DIRS if d.exists()]
        for d in DEFAULT_REFERENCE_DIRS:
            if not d.exists():
                print(f"[ingest_multi] NOTE: default reference dataset '{d}' not present; skipping it.")
    else:
        reference_dirs = [(d.name, d) for d in args.reference]

    # A path passed as both an ingest source and a reference would be scanned,
    # hashed and grouped twice under two keys, and would then collide with itself
    # in every duplicate tally.
    requested_paths = {root.resolve() for _, root in requested}
    deduped: list[tuple[str, Path]] = []
    for key, root in reference_dirs:
        if root.exists() and root.resolve() in requested_paths:
            print(f"[ingest_multi] NOTE: reference '{root}' is already an ingest source; not scanning it twice.")
            continue
        deduped.append((key, root))
    reference_dirs = deduped

    scans: dict[str, SourceScan] = {}
    try:
        for key, root in requested:
            print(f"[ingest_multi] Scanning source '{key}' at '{root}' ...")
            scans[key] = scan_source(root, key, is_reference=False)
            print(f"[ingest_multi]   {len(scans[key].records)} images, classes={scans[key].class_names}")
        for key, root in reference_dirs:
            print(f"[ingest_multi] Scanning reference dataset '{key}' at '{root}' (collision detection only) ...")
            scans[key] = scan_source(root, key, is_reference=True)
            print(f"[ingest_multi]   {len(scans[key].records)} images")
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"[ingest_multi] ERROR: {exc}")

    all_records = [r for key in sorted(scans) for r in scans[key].records]
    print(f"[ingest_multi] Hashing {len(all_records)} images (SHA256 of decoded pixels + dhash) ...")
    hash_records(all_records, args.workers)

    if out_dir.exists() and not args.dry_run:
        if args.force:
            print(f"[ingest_multi] --force: removing existing '{out_dir}' ...")
            shutil.rmtree(out_dir)
        else:
            print(f"[ingest_multi] '{out_dir}' already exists. Pass --force to rebuild it. Proceeding to (over)write into it.")

    print("[ingest_multi] Grouping, license-gating, splitting and materializing ...")
    try:
        report = build_merged_dataset(
            scans=scans,
            out_dir=out_dir,
            exclude_noncommercial=args.exclude_noncommercial,
            seed=args.seed,
            ratios=ratios,
            phash_distance=args.phash_distance,
            session_gap_s=args.session_gap_s,
            seq_gap=args.seq_gap,
            use_filename_sessions=args.filename_sessions,
            session_corroboration_min=args.session_corroboration,
            drop_reference_duplicates=args.drop_reference_duplicates,
            source_tag=args.source_tag,
            allow_redundant_sources=args.allow_redundant_sources,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        sys.exit(f"[ingest_multi] ERROR: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("MULTI-SOURCE ROBOFLOW INGEST -- SPLIT REPORT" + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 78)
    print("EFFECTIVE SIZE -- three separate numbers; a file count is NOT a photo count.")
    header = (
        f"{'source':<16}{'role':<11}{'files':>9}{'photos':>9}{'groups':>9}{'f/photo':>9}  "
        f"{'license':<12}included"
    )
    print(header)
    print("-" * len(header))
    for key, block in report["sources"].items():  # type: ignore[union-attr]
        fpp = block["files_per_unique_source_photo"]
        print(
            f"{key:<16}{block['role']:<11}{block['n_files']:>9}{block['n_unique_source_stems']:>9}"
            f"{block['n_groups']:>9}{(f'{fpp:.2f}' if fpp else '-'):>9}  "
            f"{str(block['license']):<12}{block['included_in_output']}"
        )
    totals = report["dataset_size"]["included_total"]  # type: ignore[index]
    print("-" * len(header))
    print(
        f"{'INCLUDED TOTAL':<16}{'':<11}{totals['raw_files']:>9}{totals['unique_source_photos']:>9}"
        f"{totals['independent_groups']:>9}"
    )
    print(f"  -> {totals['emitted_files']} files actually emitted from "
          f"{totals['unique_source_photos']} unique photographs in "
          f"{totals['independent_groups']} independent groups.")
    print()
    overlap = report["source_stem_overlap"]  # type: ignore[index]
    print(f"SOURCE-PHOTO OVERLAP MATRIX (redundancy threshold {overlap['threshold_pct']:g}%):")
    if overlap["pairwise"]:
        for pair, cell in sorted(overlap["pairwise"].items()):
            extras = ", ".join(f"{k}={v}%" for k, v in sorted(cell.items()) if k.startswith("pct_of_"))
            print(f"  {pair}: {cell['n_shared_stems']} shared source photos ({extras})")
    else:
        print("  (only one source scanned; nothing to compare)")
    if overlap["redundant_included_sources"]:
        for c in overlap["redundant_included_sources"]:
            print(f"  REDUNDANT: '{c['source']}' is {c['pct_of_source_already_present']}% inside "
                  f"'{c['contained_in']}' (merged anyway: --allow-redundant-sources)")
    print()
    dupes = report["duplicates"]  # type: ignore[index]
    print(f"Exact-duplicate sets: {dupes['exact_duplicate_sets']} "
          f"(cross-dataset: {dupes['cross_dataset_exact_sets']})")
    print(f"Near-duplicate sets:  {dupes['near_duplicate_sets']} "
          f"(cross-dataset: {dupes['cross_dataset_near_sets']})")
    for pair, cell in sorted(dupes["collision_matrix"].items()):
        print(f"  collision {pair}: exact={cell['exact']} images, near={cell['near']} images")
    print(f"Images dropped as exact duplicates: {dupes['images_dropped_as_exact_duplicates']}")
    print(f"Images dropped as reference duplicates: {dupes['images_dropped_as_reference_duplicates']}")
    print()
    split_header = f"{'split':<8}{'groups':>9}{'images':>9}{'no_label':>10}{'empty_label':>13}"
    print(split_header)
    print("-" * len(split_header))
    for split in SPLIT_NAMES:
        s = report["splits"][split]  # type: ignore[index]
        print(f"{split:<8}{s['n_groups']:>9}{s['n_images']:>9}{s['n_missing_labels']:>10}{s['n_empty_labels']:>13}")
    print()
    print("Per-class instances / images-containing-class, per split:")
    print(f"{'class':<18}{'severity':<15}" + "".join(f"{s:>18}" for s in SPLIT_NAMES))
    for cname in UNIFIED_CLASSES:
        row = f"{cname:<18}{SEVERITY_BY_CLASS[cname].value:<15}"
        for split in SPLIT_NAMES:
            s = report["splits"][split]  # type: ignore[index]
            cell = f"{s['instances_per_class'][cname]} / {s['images_per_class'][cname]}"
            row += f"{cell:>18}"
        print(row)
    print()
    print(f"Group-overlap check: {report['group_overlap_check']}")
    print()
    print("!" * 78)
    print("WARNINGS")
    print("!" * 78)
    for w in report["warnings"]:  # type: ignore[union-attr]
        print(f"  - {w}")
    print()
    print(f"Full report written to: {report_path}")
    if not args.dry_run:
        print(f"data.yaml written to:   {out_dir / 'data.yaml'}")
        if args.source_tag in ("manifest", "both"):
            print(f"Provenance manifest:    {out_dir / 'manifest.jsonl'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
