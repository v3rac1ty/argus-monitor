"""Build a balanced, leak-free, 6-class image-classification dataset at
``datasets/argus_cls/{train,val,test}/<class_name>/*.jpg`` (Ultralytics
classification layout) from three heterogeneous sources.

    class            | source(s)
    -----------------+---------------------------------------------------
    normal           | Hugging Face ``Masamsa/3d-print-failure-detection``
                      | (the ONLY source of healthy-print images)
    spaghetti        | FDM ``Off_platform`` (all) + a sample of argus_v2's
                      | spaghetti-labeled images (deliberately mixed)
    cracking         | FDM ``Cracking`` (all)
    layer_shifting   | FDM ``Layer_shifting`` (all)
    stringing        | FDM ``Stringing`` (all)
    warping          | FDM ``Warping`` (all)

Leakage hazards this script guards against
--------------------------------------------------------------------------
1. **FDM augmentation variants.** 81 FDM files end ``_aug.jpg`` and 80 end
   ``_original.jpg``, sharing a source stem with a plain ``.jpg`` (161
   files, 1,751 unique source stems total). ``fdm_group_key`` recovers the
   source stem so every variant of one source photo is assigned to exactly
   one split (never split across train/val/test) -- see ``split_groups`` /
   ``assign_splits_for_class``.
2. **argus_v2 Roboflow augmentation variants.** argus_v2 is a YOLO
   detection dataset whose ``train`` split (unlike ``val``/``test``) keeps
   every augmented ``.rf.<hex>`` variant of a source photo (see
   ``training/prepare_dataset.py``'s identical concern for the detection
   dataset). Pooling train+val+test spaghetti-labeled images and splitting
   naively by file would scatter near-duplicate crops/rotations of the same
   source print across this dataset's train/val/test. ``argus_group_key``
   recovers the source identity the same way so those variants are grouped
   too.
3. **FDM sequential time-lapse frames (the big one).** The FDM images are
   NOT independent photos -- each class directory is a sequence of frames
   captured during a handful of real print jobs, encoding a capture
   timestamp in the filename (``Image_YYYYMMDDHHMMSSmmm[.jpg|_aug.jpg|
   _original.jpg]``). Consecutive frames of one print are ~30s apart
   (median) and near-identical, so grouping by source stem alone (hazard
   #1) still let adjacent frames of the SAME print job land in different
   splits -- a model then memorizes specific print jobs (background,
   part geometry, lighting) instead of learning defect appearance, which
   measured as 97.3% top-1 after a 3-epoch smoke train on the old,
   frame-level split. ``parse_fdm_timestamp`` recovers each frame's
   timestamp and ``fdm_session_groups`` segments each class's frames into
   PRINT SESSIONS (a new session starts whenever the gap to the previous
   frame exceeds ``--session-gap-s``, default 600s); the session id -- not
   the source stem -- is the group unit ``assign_splits_for_class`` splits
   on, so every frame of one print job (and all its ``_aug``/``_original``
   variants, which share their source frame's timestamp) lands in exactly
   one split. A frame whose timestamp can't be parsed becomes its own
   singleton session.

   Splitting by session instead of by frame collapses several classes down
   to a handful of independent samples (measured: Cracking 8, Layer_
   shifting 5, Off_platform 3, Stringing 9, Warping 15 -- ~40 sessions
   total across the FDM dataset). See "Evaluation validity" below.

Source-confound mitigation (the main risk of this dataset's design)
--------------------------------------------------------------------------
``normal`` comes ONLY from Hugging Face and most defect classes come ONLY
from the FDM dataset, so a classifier could learn "which dataset is this
image from" (resolution, JPEG compression artifacts, framing) instead of
"is this print failing" -- and still score great on this dataset's own
val/test splits while being worthless in deployment. Two things are done to
reduce (not eliminate) that giveaway:

  1. Every output image is resized so its short side is 512px then
     center-cropped to 512x512 (``resize_and_center_crop``) -- this exactly
     matches ``argus.detectors.classifier.preprocess_classify``'s
     training-time preprocessing contract, and erases the raw-resolution
     gap between FDM's 2048x3072 originals and Hugging Face's much smaller
     ones.
  2. Every output image is re-encoded at an identical JPEG quality
     (``JPEG_QUALITY``, see ``save_normalized_jpeg``) so JPEG artifact
     statistics don't identify the source either.

See ``CONFOUND_WARNING`` below for the full text (also printed and written
into ``split_report.json``). Reported accuracy on this dataset should be
treated as optimistic until real healthy-print frames from the user's own
printer are folded into ``normal``.

Evaluation validity (session splitting's consequence)
--------------------------------------------------------------------------
Because splitting is now by print session, a class's real evaluation
sample size is its SESSION count, not its (much larger) frame count. Some
FDM-only classes only have a handful of sessions:

  - Fewer than ``MIN_SESSIONS_FOR_EVAL`` (3) sessions total: there aren't
    enough independent print jobs to hold out both a val and a test
    session without leaving 0 in train, so ALL of that class's images go
    to train and it is marked not evaluable -- it has no val/test
    accuracy at all.
  - >= 3 sessions: >=1 session each is guaranteed in val and test (see
    ``assign_splits_for_class`` / ``_ensure_min_one_per_split``), but a
    class with only a few sessions total, or only 1 in val/test, still
    has a val/test accuracy computed over a handful of print jobs -- see
    ``EVAL_VALIDITY_WARNING_HEADER`` and the per-class ``evaluable`` /
    ``n_sessions`` / ``sessions_per_split`` fields in ``split_report.json``
    and the printed summary. Treat any such class's val/test number as
    statistically weak, not a real estimate of accuracy.

Usage:
    python training/build_classification_dataset.py
    python training/build_classification_dataset.py --seed 1337 --per-class-cap 550 --force
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FDM_DIR = REPO_ROOT / "datasets" / "fdm_raw" / "FDM-3D-Printing-Defect-Dataset" / "data"
DEFAULT_ARGUS_DIR = REPO_ROOT / "datasets" / "argus_v2"
DEFAULT_HF_CACHE_DIR = REPO_ROOT / "datasets" / "hf_normal_raw"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "argus_cls"

#: datasets-server parquet index for the Hugging Face "normal" source --
#: anonymous, no API key needed. Returns {config: {split: [parquet_url]}}.
HF_DATASET_ID = "Masamsa/3d-print-failure-detection"
HF_PARQUET_INDEX_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}/parquet"
HF_CONFIG = "default"
#: In this dataset's ``label`` column, 0 == normal / healthy print, 1 == failure.
HF_NORMAL_LABEL = 0

#: Class order -- must match the commented-out classification block in
#: config.example.yaml (``detector.class_names``) and, once trained,
#: src/argus/detectors/classifier.py's ONNX model output index order.
CLASS_NAMES: tuple[str, ...] = ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")

#: FDM raw dataset subdirectory -> our class name, for the 4 classes that
#: are FDM-only.
FDM_DEFECT_CLASS_DIRS: dict[str, str] = {
    "Cracking": "cracking",
    "Layer_shifting": "layer_shifting",
    "Stringing": "stringing",
    "Warping": "warping",
}
#: FDM subdirectory that (together with a argus_v2 sample) feeds "spaghetti".
FDM_SPAGHETTI_DIR = "Off_platform"

#: argus_v2 YOLO label class id for "spaghetti" (see data.yaml: index 1 of
#: ['error extrusion', 'spaghetti', 'stringing', 'warping', 'zits']).
ARGUS_SPAGHETTI_CLASS_ID = 1
ARGUS_SPLIT_DIRS: tuple[str, ...] = ("train", "val", "test")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

TARGET_PER_CLASS = 500
DEFAULT_PER_CLASS_CAP = 550
DEFAULT_SEED = 1337
OUTPUT_SIZE = 512
JPEG_QUALITY = 90
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

#: FDM frames are sequential time-lapse captures of a handful of real print
#: jobs (median ~30s apart). A new print SESSION starts whenever the gap
#: between consecutive frames (within a class) exceeds this many seconds --
#: see ``fdm_session_groups``. Overridable via ``--session-gap-s``.
DEFAULT_SESSION_GAP_S = 600

#: A class needs at least this many print sessions to hold out both a val
#: and a test session without leaving 0 sessions in train -- see
#: ``assign_splits_for_class``. Below this, ALL of a class's images go to
#: train and it is marked not evaluable (no val/test accuracy at all).
MIN_SESSIONS_FOR_EVAL = 3

#: Below these session counts a class technically *is* evaluable (see
#: MIN_SESSIONS_FOR_EVAL above) but its val/test accuracy is computed over
#: so few print jobs that it shouldn't be trusted as a per-class number --
#: see ``evaluation_validity_lines``.
MIN_TOTAL_SESSIONS_FOR_CONFIDENT_EVAL = 5
MIN_VAL_TEST_SESSIONS_FOR_CONFIDENT_EVAL = 2

CONFOUND_WARNING = (
    "SOURCE-CONFOUND WARNING: 'normal' images come EXCLUSIVELY from the Hugging Face "
    f"'{HF_DATASET_ID}' dataset, while the 4 FDM-only defect classes (and most of "
    "'spaghetti') come EXCLUSIVELY from the FDM-3D-Printing-Defect-Dataset (a fixed-camera "
    "Ender 3 nozzle close-up rig). A classifier trained on this data can learn to recognize "
    "WHICH DATASET an image came from (residual resolution/framing/lighting cues) rather than "
    "whether the print is actually failing. Every output image is resized+center-cropped to "
    f"{OUTPUT_SIZE}x{OUTPUT_SIZE} and re-encoded at a fixed JPEG quality ({JPEG_QUALITY}) to "
    "reduce this, but that does NOT eliminate it. Treat reported accuracy on this dataset's "
    "val/test splits as optimistic until real healthy-print frames from the user's own "
    "Voron Trident camera are added to 'normal'."
)

EVAL_VALIDITY_WARNING_HEADER = (
    "EVALUATION VALIDITY WARNING: FDM source images are sequential time-lapse frames of a "
    "handful of real print jobs (median ~30s apart), not independent photos, so this dataset "
    "splits by PRINT SESSION (a run of frames with no gap > --session-gap-s) instead of by "
    "frame -- a class's real evaluation sample size is its SESSION count, not its much larger "
    "frame count. A class below is flagged NOT EVALUATED if it has fewer than "
    f"{MIN_SESSIONS_FOR_EVAL} sessions total (everything went to train, no val/test split "
    f"exists for it), or WEAK if it has fewer than {MIN_TOTAL_SESSIONS_FOR_CONFIDENT_EVAL} "
    f"sessions total or fewer than {MIN_VAL_TEST_SESSIONS_FOR_CONFIDENT_EVAL} sessions in val "
    "or test. A WEAK class's val/test accuracy is computed over a handful of print jobs -- a "
    "couple of sessions is not a statistically meaningful basis for trusting a per-class number."
)


# --------------------------------------------------------------------------
# Group-key recovery (leak-free splitting -- see module docstring)
# --------------------------------------------------------------------------

_FDM_VARIANT_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)_(?:aug|original)$")


def fdm_group_key(stem: str) -> str:
    """Recover an FDM source-photo stem by stripping a trailing ``_aug`` or
    ``_original`` augmentation-variant suffix (e.g. ``'Image_123_aug'`` ->
    ``'Image_123'``). A stem with neither suffix is already a source stem
    and is returned unchanged."""
    m = _FDM_VARIANT_SUFFIX_RE.match(stem)
    return m.group("stem") if m else stem


#: Matches Roboflow's "<source>.rf.<hex>" stem shape (same pattern
#: training/prepare_dataset.py uses for the detection dataset's identical
#: leakage hazard; reimplemented locally to keep this script self-contained).
_ROBOFLOW_SOURCE_RE = re.compile(r"^(?P<src>.+)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def argus_group_key(stem: str) -> str:
    """Recover an argus_v2 source-photo identity by stripping the
    ``.rf.<hex>`` Roboflow augmentation-variant suffix (e.g.
    ``'00001_x.rf.549c89ab'`` -> ``'00001_x'``). A stem that doesn't match
    is treated as its own source (returned unchanged)."""
    m = _ROBOFLOW_SOURCE_RE.match(stem)
    return m.group("src") if m else stem


# --------------------------------------------------------------------------
# FDM print-session recovery (leakage hazard #3 -- see module docstring).
# The FDM dataset is sequential time-lapse frames of a handful of real print
# jobs, not independent photos; these functions recover which frames belong
# to the same print job so splitting can group on that instead of on frame
# identity.
# --------------------------------------------------------------------------

#: Matches the 17-digit capture timestamp FDM embeds at the start of every
#: stem (``Image_YYYYMMDDHHMMSSmmm``): the first 14 digits are
#: ``%Y%m%d%H%M%S``, the trailing 3 are milliseconds (unused -- session
#: segmentation only needs second-level ordering). The lookahead requires
#: the digit run to be exactly 17 digits long (terminated by an underscore,
#: e.g. before ``_aug``/``_original``, or end of string) so a longer digit
#: run can never be mistaken for a valid timestamp by partial match. Since
#: the timestamp precedes any ``_aug``/``_original`` suffix, this parses an
#: augmentation variant's stem exactly the same as its source frame's.
_FDM_TIMESTAMP_RE = re.compile(r"^Image_(\d{17})(?=$|_)")


def parse_fdm_timestamp(stem: str) -> datetime | None:
    """Parse the capture timestamp embedded in an FDM stem (source frame or
    ``_aug``/``_original`` variant -- both parse identically, see
    ``_FDM_TIMESTAMP_RE``). Returns ``None`` if ``stem`` doesn't start with
    the expected 17-digit pattern, or the first 14 digits don't form a
    valid ``%Y%m%d%H%M%S`` date/time."""
    m = _FDM_TIMESTAMP_RE.match(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1)[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def fdm_session_groups(stems: Sequence[str], gap_s: float = DEFAULT_SESSION_GAP_S) -> dict[str, str]:
    """Assign every FDM stem in ``stems`` to a print-session group id, so
    that all frames of one print job -- and every ``_aug``/``_original``
    augmentation variant of each of those frames -- land in the same
    dataset split (see module docstring's leakage hazard #3).

    Algorithm:
      1. Recover each stem's *frame id* (``fdm_group_key``, i.e. the stem
         with a trailing ``_aug``/``_original`` suffix stripped) and parse
         that frame id's capture timestamp (``parse_fdm_timestamp``) --
         computed once per distinct frame id, not once per stem, since
         variants share their source frame's timestamp exactly.
      2. Frames with a parseable timestamp are sorted chronologically
         (ties broken by frame id, for determinism) and walked in order; a
         new session starts whenever the gap to the previous frame's
         timestamp exceeds ``gap_s`` seconds. Every stem belonging to a
         frame inherits that frame's session id.
      3. A stem whose frame id has no parseable timestamp gets its own
         singleton group, keyed off the frame id so its variants (if any)
         still land together -- never merged into a real chronological
         session, and counted/reported separately by callers.

    Returns ``{stem: group_id}`` for every stem in ``stems``.
    """
    frame_of_stem: dict[str, str] = {stem: fdm_group_key(stem) for stem in stems}
    frame_timestamp: dict[str, datetime | None] = {
        frame: parse_fdm_timestamp(frame) for frame in set(frame_of_stem.values())
    }

    parsed_frames = sorted(
        (frame for frame, ts in frame_timestamp.items() if ts is not None),
        key=lambda frame: (frame_timestamp[frame], frame),
    )

    session_of_frame: dict[str, str] = {}
    session_idx = -1
    prev_ts: datetime | None = None
    for frame in parsed_frames:
        ts = frame_timestamp[frame]
        assert ts is not None  # narrows for type checkers; guaranteed by the filter above
        if prev_ts is None or (ts - prev_ts).total_seconds() > gap_s:
            session_idx += 1
        session_of_frame[frame] = f"session{session_idx:05d}"
        prev_ts = ts

    return {
        stem: (f"unparsed:{frame}" if frame_timestamp[frame] is None else session_of_frame[frame])
        for stem, frame in frame_of_stem.items()
    }


# --------------------------------------------------------------------------
# Candidate model: one selectable image, with a deferred loader so nothing
# is decoded into memory until it's actually chosen and written.
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    class_name: str
    source: str  # "fdm" | "argus_v2" | "hf"
    group_key: str  # never split across train/val/test -- see module docstring
    output_stem: str  # unique within this candidate's class; used as the output filename
    loader: Callable[[], np.ndarray]  # decodes and returns a BGR image array on demand
    # "unique frames" reporting unit -- differs from group_key only for FDM
    # sources, where group_key is a print-SESSION id (many frames) but
    # frame_key is the older per-source-stem id, so the report can show the
    # size of the frame->session correction (see module docstring's
    # leakage hazard #3). Defaults to group_key (frames == sessions) for
    # every other source, where no such correction applies.
    frame_key: str = ""

    def __post_init__(self) -> None:
        if not self.frame_key:
            self.frame_key = self.group_key


def _file_loader(path: Path) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        return img

    return _load


def _bytes_loader(data: bytes) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img

    return _load


# --------------------------------------------------------------------------
# Source A -- FDM dataset (local disk, already extracted)
# --------------------------------------------------------------------------


def scan_fdm_class_dir(fdm_root: Path, class_dir_name: str) -> list[Path]:
    """List every image directly inside ``fdm_root/class_dir_name``, sorted
    for determinism."""
    d = fdm_root / class_dir_name
    if not d.is_dir():
        raise FileNotFoundError(f"FDM class directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def fdm_candidates(
    fdm_root: Path,
    class_dir_name: str,
    class_name: str,
    session_gap_s: float = DEFAULT_SESSION_GAP_S,
) -> list[Candidate]:
    """Build one Candidate per image in an FDM class subdirectory, grouped
    for SPLITTING by print session (see ``fdm_session_groups`` -- leakage
    hazard #3) and namespaced by ``class_dir_name`` so two different FDM
    subdirectories can never accidentally collide onto the same group
    (verified empirically to never happen in the real data, but namespaced
    defensively regardless). Each candidate's ``frame_key`` separately
    records the older per-source-stem grouping (``fdm_group_key`` -- hazard
    #1) purely for "unique frames vs unique sessions" reporting; it plays
    no role in the actual split."""
    paths = scan_fdm_class_dir(fdm_root, class_dir_name)
    session_by_stem = fdm_session_groups([p.stem for p in paths], session_gap_s)
    out: list[Candidate] = []
    for p in paths:
        stem = p.stem
        frame = fdm_group_key(stem)
        session = session_by_stem[stem]
        out.append(
            Candidate(
                class_name=class_name,
                source="fdm",
                group_key=f"fdm:{class_dir_name}:{session}",
                frame_key=f"fdm:{class_dir_name}:{frame}",
                output_stem=f"fdm_{class_dir_name.lower()}_{stem}",
                loader=_file_loader(p),
            )
        )
    return out


def gather_fdm_candidates(
    fdm_root: Path, session_gap_s: float = DEFAULT_SESSION_GAP_S
) -> tuple[dict[str, list[Candidate]], list[Candidate]]:
    """Scan the whole FDM raw tree once. Returns ``(defect_by_class,
    off_platform)`` where ``defect_by_class`` maps our class name -> all
    candidates for the 4 FDM-only defect classes, and ``off_platform`` is
    the ``Off_platform`` subdirectory's candidates (feeds "spaghetti")."""
    defect_by_class = {
        class_name: fdm_candidates(fdm_root, dir_name, class_name, session_gap_s)
        for dir_name, class_name in FDM_DEFECT_CLASS_DIRS.items()
    }
    off_platform = fdm_candidates(fdm_root, FDM_SPAGHETTI_DIR, "spaghetti", session_gap_s)
    return defect_by_class, off_platform


# --------------------------------------------------------------------------
# Source C -- spaghetti from the existing argus_v2 detection dataset
# --------------------------------------------------------------------------


def has_class_id(label_text: str, class_id: int) -> bool:
    """True if any row of a YOLO label file's contents has ``class_id`` as
    its class (first whitespace-separated token). Blank lines are ignored;
    an empty/whitespace-only file (legitimately "no objects") returns
    False, and a malformed leading token is treated as "not a match" rather
    than raising."""
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cid = int(float(line.split()[0]))
        except ValueError:
            continue
        if cid == class_id:
            return True
    return False


def scan_argus_spaghetti_images(
    argus_root: Path, split_dirs: Sequence[str] = ARGUS_SPLIT_DIRS
) -> list[Path]:
    """Pool every image across ``argus_root/{split}/images`` (for each of
    ``split_dirs``) whose paired label file contains at least one
    spaghetti (class ``ARGUS_SPAGHETTI_CLASS_ID``) row. argus_v2's own
    train/val/test assignment is irrelevant here -- this dataset re-splits
    everything from scratch via ``argus_group_key`` (see module docstring).
    A split directory that doesn't exist is skipped (keeps this usable on
    small synthetic fixtures that only populate one split)."""
    out: list[Path] = []
    for split in split_dirs:
        images_dir = argus_root / split / "images"
        labels_dir = argus_root / split / "labels"
        if not images_dir.is_dir():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_SUFFIXES):
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.is_file():
                continue
            text = label_path.read_text(encoding="utf-8")
            if has_class_id(text, ARGUS_SPAGHETTI_CLASS_ID):
                out.append(img_path)
    return out


def argus_spaghetti_candidates(argus_root: Path) -> list[Candidate]:
    """Build one Candidate per spaghetti-labeled argus_v2 image, grouped by
    source stem (see ``argus_group_key``) so Roboflow augmentation variants
    of one source photo are never split across train/val/test."""
    out: list[Candidate] = []
    for p in scan_argus_spaghetti_images(argus_root):
        stem = p.stem
        group = argus_group_key(stem)
        out.append(
            Candidate(
                class_name="spaghetti",
                source="argus_v2",
                group_key=f"argus_v2:{group}",
                output_stem=f"argus_v2_{stem}",
                loader=_file_loader(p),
            )
        )
    return out


# --------------------------------------------------------------------------
# Source B -- "normal" from Hugging Face (network I/O; not covered by
# tests -- see tests/test_build_classification_dataset.py)
# --------------------------------------------------------------------------


def fetch_hf_parquet_urls(
    dataset_id: str = HF_DATASET_ID, config: str = HF_CONFIG, timeout: float = 30.0
) -> dict[str, str]:
    """Query the HF datasets-server parquet index and return {split_name:
    parquet_url} for ``config``. Anonymous, no API key required."""
    resp = requests.get(HF_PARQUET_INDEX_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    split_urls = data[config]
    return {split: urls[0] for split, urls in split_urls.items() if urls}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20, timeout: float = 60.0) -> Path:
    """Stream-download ``url`` to ``dest``, idempotently: if ``dest``
    already exists and is non-empty, the download is skipped entirely (so
    re-running the pipeline doesn't re-fetch ~580MB of parquet every time).
    Downloads to a ``.part`` sibling first and atomically renames on
    success, so a crash mid-download can't leave a truncated file that
    looks "already cached" on the next run."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest


def download_hf_normal_parquets(cache_dir: Path) -> dict[str, Path]:
    """Fetch the parquet-index URLs and download every split's parquet file
    into ``cache_dir`` (idempotent -- see ``download_file``). Returns
    {split_name: local_path}."""
    urls = fetch_hf_parquet_urls()
    paths: dict[str, Path] = {}
    for split, url in urls.items():
        dest = cache_dir / f"{split}.parquet"
        paths[split] = download_file(url, dest)
    return paths


def load_normal_rows_from_parquet(parquet_path: Path) -> list[bytes]:
    """Read a local parquet file (schema: ``image`` struct{bytes,path},
    ``label`` int) and return the raw encoded-image bytes of every row with
    ``label == HF_NORMAL_LABEL``. Uses polars' native parquet reader (no
    pyarrow dependency needed)."""
    df = pl.read_parquet(parquet_path, columns=["image", "label"])
    normal = df.filter(pl.col("label") == HF_NORMAL_LABEL)
    return [row["image"]["bytes"] for row in normal.iter_rows(named=True)]


def hf_normal_candidates(image_bytes_by_split: Mapping[str, Sequence[bytes]]) -> list[Candidate]:
    """Build one Candidate per HF normal-labeled row. Each row is an
    independent photo (no known augmentation-duplication issue for this
    source, unlike FDM/argus_v2), so each gets its own singleton group,
    namespaced by split + row index for a stable, unique ``output_stem``."""
    out: list[Candidate] = []
    for split in sorted(image_bytes_by_split):
        for idx, data in enumerate(image_bytes_by_split[split]):
            out.append(
                Candidate(
                    class_name="normal",
                    source="hf",
                    group_key=f"hf:{split}:{idx:06d}",
                    output_stem=f"hf_normal_{split}_{idx:06d}",
                    loader=_bytes_loader(data),
                )
            )
    return out


# --------------------------------------------------------------------------
# Deterministic sampling / capping / group-aware splitting
# --------------------------------------------------------------------------


def deterministic_sample(keys: Sequence[str], k: int, seed: int) -> list[str]:
    """Deterministically choose up to ``k`` of ``keys`` for a given seed:
    sort first (so input order never matters), shuffle with a fresh
    ``random.Random(seed)``, then take the first ``k``. If ``k >=
    len(keys)`` every key is returned (still sorted+shuffled)."""
    ordered = sorted(set(keys))
    random.Random(seed).shuffle(ordered)
    if k >= len(ordered):
        return ordered
    return ordered[:k]


def sample_candidates(candidates: Sequence[Candidate], k: int, seed: int) -> list[Candidate]:
    """``deterministic_sample`` lifted to work on ``Candidate`` objects,
    keyed by their (class-unique) ``output_stem``. Raises ``ValueError`` if
    two candidates share an ``output_stem`` -- that would make the sample
    ambiguous/non-deterministic."""
    by_key: dict[str, Candidate] = {}
    for c in candidates:
        if c.output_stem in by_key:
            raise ValueError(f"duplicate output_stem {c.output_stem!r} among candidates")
        by_key[c.output_stem] = c
    chosen_keys = deterministic_sample(list(by_key.keys()), k, seed)
    return [by_key[key] for key in chosen_keys]


def split_groups(
    group_ids: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split group ids into (train, val, test) lists.
    Mirrors ``training/prepare_dataset.py``'s ``split_sources``: sort first
    (input order never matters), shuffle with a seeded ``random.Random``,
    slice by ratio -- the remainder after rounding train/val goes to test
    so no group is invented or dropped."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios!r} (sum={sum(ratios)})")

    ids = sorted(group_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = min(round(n * ratios[0]), n)
    n_val = min(round(n * ratios[1]), n - n_train)
    n_test = n - n_train - n_val

    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val : n_train + n_val + n_test]


def assert_no_group_overlap(split_group_ids: Mapping[str, Sequence[str]]) -> None:
    """Raise ``AssertionError`` if any group id appears in more than one
    split -- the whole point of splitting by group instead of by file, so
    it's checked explicitly rather than just trusted."""
    seen: dict[str, str] = {}
    for split_name, ids in split_group_ids.items():
        for gid in ids:
            if gid in seen:
                raise AssertionError(
                    f"Group-identity leakage detected: group {gid!r} appears in both "
                    f"'{seen[gid]}' and '{split_name}' splits."
                )
            seen[gid] = split_name


def _ensure_min_one_per_split(
    train_ids: list[str], val_ids: list[str], test_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """After a ratio-based ``split_groups`` call, guarantee ``val_ids`` and
    ``test_ids`` each end up with >=1 group (a strict 70/15/15 split of a
    small group count can round val or test down to 0 -- e.g. 3 groups ->
    2/0/1, 5 groups -> 4/1/0 -- even though holding out one group for each
    is the whole point once a class clears ``MIN_SESSIONS_FOR_EVAL``).
    Moves one group at a time out of ``train_ids`` (kept the preferred
    donor so long as it stays non-empty) into whichever of val/test is
    still empty; falls back to borrowing from the other of val/test if
    train_ids can't spare one, so train never drops below 1 group. Pure
    function of its (already seed-shuffled) inputs, so this stays
    deterministic."""
    train_ids, val_ids, test_ids = list(train_ids), list(val_ids), list(test_ids)
    for target in (val_ids, test_ids):
        if target:
            continue
        if len(train_ids) > 1:
            target.append(train_ids.pop())
        elif len(val_ids) > 1:
            target.append(val_ids.pop())
        elif len(test_ids) > 1:
            target.append(test_ids.pop())
    return train_ids, val_ids, test_ids


def assign_splits_for_class(
    candidates: Sequence[Candidate],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[Candidate]]:
    """Group one class's kept candidates by ``group_key`` (a print SESSION
    id for FDM sources -- see leakage hazard #3 -- or an independent-photo
    id otherwise), split the GROUPS 70/15/15 (deterministic for ``seed``),
    verify zero overlap, then expand back to per-split candidate lists
    (every candidate in a group lands in that group's split).

    Two special cases around small group counts (see module docstring's
    "Evaluation validity" section):
      - Fewer than ``MIN_SESSIONS_FOR_EVAL`` groups: there aren't enough
        independent samples to hold out both a val and a test group
        without leaving 0 in train, so EVERY candidate goes to train and
        val/test are both empty -- this is how callers detect "not
        evaluable" (``bool(result["val"])`` is False).
      - ``MIN_SESSIONS_FOR_EVAL`` or more groups: the ratio split runs
        normally, then ``_ensure_min_one_per_split`` guarantees >=1 group
        landed in val and >=1 in test.
    """
    groups: dict[str, list[Candidate]] = {}
    for c in candidates:
        groups.setdefault(c.group_key, []).append(c)

    group_ids = list(groups.keys())
    if len(group_ids) < MIN_SESSIONS_FOR_EVAL:
        return {"train": list(candidates), "val": [], "test": []}

    train_ids, val_ids, test_ids = split_groups(group_ids, seed, ratios)
    train_ids, val_ids, test_ids = _ensure_min_one_per_split(train_ids, val_ids, test_ids)
    assert_no_group_overlap({"train": train_ids, "val": val_ids, "test": test_ids})

    result: dict[str, list[Candidate]] = {"train": [], "val": [], "test": []}
    for split_name, group_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for gid in group_ids:
            result[split_name].extend(groups[gid])
    return result


# --------------------------------------------------------------------------
# Per-class selection recipes (balancing -- see module docstring's table)
# --------------------------------------------------------------------------


def select_normal(hf_pool: list[Candidate], target: int, cap: int, seed: int) -> list[Candidate]:
    """Sample ``min(target, cap, len(hf_pool))`` images from the HF normal
    pool."""
    k = min(target, cap, len(hf_pool))
    return sample_candidates(hf_pool, k, seed)


def select_spaghetti(
    off_platform: list[Candidate],
    argus_pool: list[Candidate],
    target: int,
    cap: int,
    seed: int,
) -> list[Candidate]:
    """ALL of FDM's ``Off_platform`` images are always included, topped up
    with a deterministic sample from the argus_v2 spaghetti pool to reach
    ``min(target, cap)`` total. If ``Off_platform`` alone somehow exceeds
    the cap (only possible with a very small ``--per-class-cap``), the
    combined set is trimmed back down to ``cap`` rather than silently
    ignoring it."""
    budget = min(target, cap)
    kept_off_platform = list(off_platform)
    remaining = max(0, budget - len(kept_off_platform))
    kept_argus = sample_candidates(argus_pool, min(remaining, len(argus_pool)), seed)
    combined = kept_off_platform + kept_argus
    if len(combined) > cap:
        combined = sample_candidates(combined, cap, seed)
    return combined


def select_fdm_defect_class(pool: list[Candidate], cap: int, seed: int) -> list[Candidate]:
    """Use every image in ``pool`` (an FDM-only defect class), unless it
    exceeds ``cap``, in which case sample down to ``cap``."""
    k = min(len(pool), cap)
    return sample_candidates(pool, k, seed)


# --------------------------------------------------------------------------
# Image normalization -- also the source-confound mitigation (see module
# docstring). MUST match argus.detectors.classifier.preprocess_classify's
# resize/crop math exactly (that function's docstring says so explicitly).
# --------------------------------------------------------------------------


def resize_and_center_crop(image: np.ndarray, size: int = OUTPUT_SIZE) -> np.ndarray:
    """Resize ``image`` so its short side is exactly ``size`` px (preserving
    aspect ratio), then center-crop to ``size x size``.

    Mirrors ``argus.detectors.classifier.preprocess_classify``'s resize/crop
    steps (short-side resize + center crop, same defensive
    ``max(size, round(...))`` clamp against float round-off landing a hair
    under ``size``) so training-time and inference-time preprocessing never
    diverge. Also erases the raw-resolution gap between the ~2048x3072 FDM
    originals and the much smaller Hugging Face ones -- one of the two
    source-confound mitigations (see module docstring).
    """
    orig_h, orig_w = image.shape[:2]
    short_side = min(orig_h, orig_w)
    scale = size / short_side

    resized_w = max(size, int(round(orig_w * scale)))
    resized_h = max(size, int(round(orig_h * scale)))

    if (resized_w, resized_h) != (orig_w, orig_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    top = (resized_h - size) // 2
    left = (resized_w - size) // 2
    return resized[top : top + size, left : left + size]


def save_normalized_jpeg(image: np.ndarray, dest: Path, quality: int = JPEG_QUALITY) -> None:
    """Resize+center-crop ``image`` to ``OUTPUT_SIZE`` and write it as a
    JPEG at a fixed quality. Both steps exist purely to erase
    source-identifying cues (raw resolution, JPEG compression-artifact
    statistics) that would otherwise let a classifier cheat by learning
    "which dataset is this from" instead of "is this print failing" (see
    module docstring's CONFOUND_WARNING)."""
    processed = resize_and_center_crop(image, OUTPUT_SIZE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), processed, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError(f"cv2.imwrite failed for {dest}")


# --------------------------------------------------------------------------
# Materialization + report
# --------------------------------------------------------------------------


@dataclass
class ClassReport:
    per_split_counts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_NAMES})
    per_source_counts: dict[str, int] = field(default_factory=dict)
    # "unique frames" (old split unit, Candidate.frame_key) vs "unique
    # sessions" (new split unit, Candidate.group_key) -- see module
    # docstring's leakage hazard #3. Equal for every non-FDM source.
    n_frames: int = 0
    n_sessions: int = 0
    sessions_per_split: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLIT_NAMES})
    # False iff this class had fewer than MIN_SESSIONS_FOR_EVAL sessions,
    # in which case every candidate went to train and val/test are empty
    # -- see assign_splits_for_class.
    evaluable: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": {**self.per_split_counts, "total": sum(self.per_split_counts.values())},
            "source_breakdown": self.per_source_counts,
            "n_frames": self.n_frames,
            "n_sessions": self.n_sessions,
            "sessions_per_split": dict(self.sessions_per_split),
            "evaluable": self.evaluable,
        }


def materialize_class(
    class_name: str,
    split_assignment: Mapping[str, list[Candidate]],
    out_dir: Path,
) -> ClassReport:
    """Write every candidate in ``split_assignment`` to
    ``out_dir/<split>/<class_name>/<output_stem>.jpg`` (normalized via
    ``save_normalized_jpeg``) and tally a ``ClassReport``. ``evaluable`` is
    derived straight from ``split_assignment`` (val and test are both
    non-empty iff the class cleared ``MIN_SESSIONS_FOR_EVAL`` -- see
    ``assign_splits_for_class``), so there's a single source of truth for
    what "not evaluable" means."""
    report = ClassReport(evaluable=bool(split_assignment.get("val")) and bool(split_assignment.get("test")))
    seen_groups: set[str] = set()
    seen_frames: set[str] = set()
    for split_name in SPLIT_NAMES:
        candidates = split_assignment.get(split_name, [])
        report.per_split_counts[split_name] = len(candidates)
        split_groups_seen: set[str] = set()
        class_dir = out_dir / split_name / class_name
        for c in candidates:
            seen_groups.add(c.group_key)
            seen_frames.add(c.frame_key)
            split_groups_seen.add(c.group_key)
            report.per_source_counts[c.source] = report.per_source_counts.get(c.source, 0) + 1
            image = c.loader()
            save_normalized_jpeg(image, class_dir / f"{c.output_stem}.jpg")
        report.sessions_per_split[split_name] = len(split_groups_seen)
    report.n_sessions = len(seen_groups)
    report.n_frames = len(seen_frames)
    return report


def evaluation_validity_lines(class_reports: Mapping[str, ClassReport]) -> list[str]:
    """Build one warning line per class whose val/test accuracy shouldn't
    be trusted (see ``EVAL_VALIDITY_WARNING_HEADER``): NOT EVALUATED for a
    class below ``MIN_SESSIONS_FOR_EVAL`` sessions (no val/test split
    exists at all), or WEAK for a class that has a val/test split but with
    too few sessions total or in val/test to be a meaningful per-class
    estimate. Classes not flagged by either rule are omitted entirely."""
    lines: list[str] = []
    for class_name in CLASS_NAMES:
        r = class_reports[class_name]
        if not r.evaluable:
            lines.append(
                f"  - {class_name}: NOT EVALUATED -- only {r.n_sessions} print session(s) total "
                f"(< {MIN_SESSIONS_FOR_EVAL} required to hold out a val AND a test session "
                "without leaving 0 in train). ALL of its images went to train; it has no "
                "val/test accuracy."
            )
            continue
        reasons: list[str] = []
        if r.n_sessions < MIN_TOTAL_SESSIONS_FOR_CONFIDENT_EVAL:
            reasons.append(f"only {r.n_sessions} sessions total")
        if r.sessions_per_split["val"] < MIN_VAL_TEST_SESSIONS_FOR_CONFIDENT_EVAL:
            reasons.append(f"only {r.sessions_per_split['val']} val session(s)")
        if r.sessions_per_split["test"] < MIN_VAL_TEST_SESSIONS_FOR_CONFIDENT_EVAL:
            reasons.append(f"only {r.sessions_per_split['test']} test session(s)")
        if reasons:
            lines.append(f"  - {class_name}: WEAK -- {', '.join(reasons)}.")
    return lines


def build_dataset(
    *,
    out_dir: Path,
    fdm_defect_by_class: Mapping[str, list[Candidate]],
    fdm_off_platform: list[Candidate],
    argus_spaghetti_pool: list[Candidate],
    hf_normal_pool: list[Candidate],
    seed: int = DEFAULT_SEED,
    per_class_cap: int = DEFAULT_PER_CLASS_CAP,
    target_per_class: int = TARGET_PER_CLASS,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, object]:
    """Pure(ish) orchestration core: given already-gathered candidate pools
    (no network/dataset I/O of its own beyond writing ``out_dir``), select
    + group-aware-split + materialize all 6 classes, and return the full
    report dict. Kept separate from ``main()`` so tests can call this
    directly with small synthetic pools -- no network, no real dataset, no
    monkeypatching required.
    """
    selections: dict[str, list[Candidate]] = {
        "normal": select_normal(hf_normal_pool, target_per_class, per_class_cap, seed),
        "spaghetti": select_spaghetti(fdm_off_platform, argus_spaghetti_pool, target_per_class, per_class_cap, seed),
    }
    for class_name, pool in fdm_defect_by_class.items():
        selections[class_name] = select_fdm_defect_class(pool, per_class_cap, seed)

    class_reports: dict[str, ClassReport] = {}
    overall_split_group_ids: dict[str, set[str]] = {s: set() for s in SPLIT_NAMES}
    for class_name in CLASS_NAMES:
        split_assignment = assign_splits_for_class(selections[class_name], seed, ratios)
        class_reports[class_name] = materialize_class(class_name, split_assignment, out_dir)
        for split_name in SPLIT_NAMES:
            # Namespace by class so groups from different classes (which
            # share no source images) can never be mistaken for overlap. A
            # group contributes multiple candidates within ONE split (all
            # its variants), so dedupe to one entry per group per split --
            # assert_no_group_overlap's job is to catch a group id showing
            # up in *two different* splits, not repeats within one.
            overall_split_group_ids[split_name].update(f"{class_name}::{c.group_key}" for c in split_assignment[split_name])

    # Final end-to-end re-verification across every class combined, not just
    # trusting each class's own per-class check.
    assert_no_group_overlap({s: sorted(ids) for s, ids in overall_split_group_ids.items()})

    total_frames = sum(r.n_frames for r in class_reports.values())
    total_sessions = sum(r.n_sessions for r in class_reports.values())
    eval_lines = evaluation_validity_lines(class_reports)
    eval_warning_text = EVAL_VALIDITY_WARNING_HEADER + "\n" + (
        "\n".join(eval_lines)
        if eval_lines
        else "  (no classes flagged -- every class meets the minimum session thresholds)"
    )
    report: dict[str, object] = {
        "seed": seed,
        "per_class_cap": per_class_cap,
        "target_per_class": target_per_class,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "output_size": OUTPUT_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "class_names": list(CLASS_NAMES),
        "classes": {c: class_reports[c].to_dict() for c in CLASS_NAMES},
        "total_frames": total_frames,
        "total_sessions": total_sessions,
        "total_groups": total_sessions,  # kept as an alias -- "group" == "session" post leakage-hazard-#3 fix
        "group_overlap_check": "PASS: zero group-identity overlap between train/val/test splits (checked per class and across all classes combined)",
        "confound_warning": CONFOUND_WARNING,
        "evaluation_validity_warning": eval_warning_text,
    }
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fdm-dir", type=Path, default=DEFAULT_FDM_DIR, help=f"FDM raw dataset dir (default: {DEFAULT_FDM_DIR})")
    parser.add_argument("--argus-dir", type=Path, default=DEFAULT_ARGUS_DIR, help=f"argus_v2 detection dataset dir (default: {DEFAULT_ARGUS_DIR})")
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR, help=f"Cache dir for downloaded HF parquet files (default: {DEFAULT_HF_CACHE_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"Output dataset dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for deterministic sampling/splitting (default: {DEFAULT_SEED})")
    parser.add_argument("--per-class-cap", type=int, default=DEFAULT_PER_CLASS_CAP, help=f"Upper bound on images kept per class (default: {DEFAULT_PER_CLASS_CAP})")
    parser.add_argument(
        "--session-gap-s",
        type=float,
        default=DEFAULT_SESSION_GAP_S,
        help="Gap (seconds) between consecutive FDM frame timestamps that starts a new print "
        f"session -- see leakage hazard #3 in the module docstring (default: {DEFAULT_SESSION_GAP_S})",
    )
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild --out if it already exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out

    print(f"[build_cls] {CONFOUND_WARNING}")
    print()

    print(f"[build_cls] Scanning FDM dataset at '{args.fdm_dir}' (session gap: {args.session_gap_s:g}s) ...")
    fdm_defect_by_class, fdm_off_platform = gather_fdm_candidates(args.fdm_dir, args.session_gap_s)
    for class_name, cands in fdm_defect_by_class.items():
        n_sessions = len({c.group_key for c in cands})
        print(f"  {class_name}: {len(cands)} images, {n_sessions} print sessions")
    n_off_platform_sessions = len({c.group_key for c in fdm_off_platform})
    print(f"  spaghetti (Off_platform): {len(fdm_off_platform)} images, {n_off_platform_sessions} print sessions")

    print(f"[build_cls] Scanning argus_v2 at '{args.argus_dir}' for spaghetti-labeled images ...")
    argus_pool = argus_spaghetti_candidates(args.argus_dir)
    print(f"  argus_v2 spaghetti pool: {len(argus_pool)} images")

    print(f"[build_cls] Fetching Hugging Face '{HF_DATASET_ID}' parquet files into '{args.hf_cache_dir}' ...")
    hf_paths = download_hf_normal_parquets(args.hf_cache_dir)
    image_bytes_by_split: dict[str, list[bytes]] = {}
    for split, path in sorted(hf_paths.items()):
        rows = load_normal_rows_from_parquet(path)
        image_bytes_by_split[split] = rows
        print(f"  {split}: {len(rows)} normal-labeled rows")
    hf_pool = hf_normal_candidates(image_bytes_by_split)
    print(f"  total HF normal pool: {len(hf_pool)} images")

    if out_dir.exists():
        if args.force:
            print(f"[build_cls] --force: removing existing '{out_dir}' ...")
            shutil.rmtree(out_dir)
        else:
            print(f"[build_cls] '{out_dir}' already exists. Pass --force to rebuild it. Proceeding to (over)write into it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_cls] Selecting, group-aware splitting, normalizing, and writing images ...")
    report = build_dataset(
        out_dir=out_dir,
        fdm_defect_by_class=fdm_defect_by_class,
        fdm_off_platform=fdm_off_platform,
        argus_spaghetti_pool=argus_pool,
        hf_normal_pool=hf_pool,
        seed=args.seed,
        per_class_cap=args.per_class_cap,
    )

    report_path = out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("CLASSIFICATION DATASET BUILD -- SPLIT REPORT")
    print("=" * 78)
    print(f"Seed: {args.seed}   Per-class cap: {args.per_class_cap}   Ratios: {DEFAULT_SPLIT_RATIOS}")
    print(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}   JPEG quality: {JPEG_QUALITY}   Session gap: {args.session_gap_s:g}s")
    print()
    header = f"{'class':<18}{'train':>8}{'val':>8}{'test':>8}{'total':>8}   source breakdown"
    print(header)
    print("-" * len(header))
    for class_name in CLASS_NAMES:
        c = report["classes"][class_name]  # type: ignore[index]
        counts = c["counts"]
        sources = ", ".join(f"{k}={v}" for k, v in sorted(c["source_breakdown"].items()))
        print(
            f"{class_name:<18}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}"
            f"{counts['total']:>8}   {sources}"
        )
    print()
    print("Frames (old split unit) vs sessions (new split unit) -- see leakage hazard #3:")
    header2 = f"{'class':<18}{'n_frames':>10}{'n_sessions':>12}{'train':>8}{'val':>8}{'test':>8}   evaluable"
    print(header2)
    print("-" * len(header2))
    for class_name in CLASS_NAMES:
        c = report["classes"][class_name]  # type: ignore[index]
        sps = c["sessions_per_split"]
        print(
            f"{class_name:<18}{c['n_frames']:>10}{c['n_sessions']:>12}{sps['train']:>8}{sps['val']:>8}{sps['test']:>8}"
            f"   {c['evaluable']}"
        )
    print()
    print(f"Total frames across all classes: {report['total_frames']}   Total sessions across all classes: {report['total_sessions']}")
    print(f"Group-overlap check: {report['group_overlap_check']}")
    print()
    print(report["confound_warning"])
    print()
    print("!" * 78)
    print(report["evaluation_validity_warning"])
    print("!" * 78)
    print()
    print(f"Full report written to: {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
