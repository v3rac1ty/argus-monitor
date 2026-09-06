"""Builds a leak-free, 6-class classification dataset at ``datasets/argus_cls/{train,val,test}/<class_name>/*.jpg`` from FDM, argus_v2, and Hugging Face sources (see ``CLASS_NAMES``).

Splits are GROUP-aware: FDM ``_aug``/``_original`` variants and Roboflow ``.rf.<hash>`` renders of one source image never straddle a split, and FDM frames are further grouped into print SESSIONS -- splitting by frame instead let a 3-epoch smoke train hit 97.3% top-1 via leakage (see ``fdm_session_groups``).

``normal`` is Hugging Face only while most defect classes are FDM only, so accuracy can reflect dataset origin rather than defect presence (``CONFOUND_WARNING``).

Usage: python training/build_classification_dataset.py [--seed N] [--per-class-cap N] [--force]
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

#: Gap (seconds) between consecutive FDM frame timestamps that starts a new print session.
DEFAULT_SESSION_GAP_S = 600

#: Minimum print sessions to hold out both a val and test session; below this, all of a
#: class's images go to train and it's marked not evaluable.
MIN_SESSIONS_FOR_EVAL = 3

#: Below these session-count thresholds, val/test accuracy is too noisy to trust per-class.
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


# str fdm_group_key(str stem)
# Inputs: str stem - an FDM image filename stem (may carry a trailing "_aug" or "_original"
#         augmentation-variant suffix)
# Outputs: str - the recovered source-photo stem, e.g. "Image_123_aug" -> "Image_123"
# Description: Recovers an FDM source-photo stem by stripping a trailing "_aug"/"_original"
#              augmentation-variant suffix. A stem with neither suffix is already a source stem
#              and is returned unchanged. This is leakage hazard #1's grouping key (see module
#              docstring); it is superseded as the actual SPLIT unit by the print-SESSION id
#              from fdm_session_groups (hazard #3), but still used for "unique frames" reporting.
# Side Effects: None
def fdm_group_key(stem: str) -> str:
    m = _FDM_VARIANT_SUFFIX_RE.match(stem)
    return m.group("stem") if m else stem


#: Matches Roboflow's "<source>.rf.<hex>" stem shape (same pattern
#: training/prepare_dataset.py uses for the detection dataset's identical
#: leakage hazard; reimplemented locally to keep this script self-contained).
_ROBOFLOW_SOURCE_RE = re.compile(r"^(?P<src>.+)\.rf\.[0-9a-f]+$", re.IGNORECASE)


# str argus_group_key(str stem)
# Inputs: str stem - an argus_v2 image filename stem (may carry a Roboflow ".rf.<hex>"
#         augmentation-variant suffix)
# Outputs: str - the recovered source-photo identity, e.g. "00001_x.rf.549c89ab" -> "00001_x"
# Description: Recovers an argus_v2 source-photo identity by stripping the ".rf.<hex>" Roboflow
#              augmentation-variant suffix. A stem that doesn't match is treated as its own
#              source (returned unchanged). This is the GROUP unit for splitting argus_v2
#              spaghetti candidates (leakage hazard #2 in the module docstring) -- the split
#              happens by source photo, not by file, so augmentation variants of one source
#              never land in different splits.
# Side Effects: None
def argus_group_key(stem: str) -> str:
    m = _ROBOFLOW_SOURCE_RE.match(stem)
    return m.group("src") if m else stem


# --------------------------------------------------------------------------
# FDM print-session recovery -- groups sequential time-lapse frames by print job.
# --------------------------------------------------------------------------

#: Matches the 17-digit capture timestamp FDM embeds at the start of every
#: Matches the 17-digit capture timestamp FDM embeds at the stem's start (``Image_YYYYMMDDHHMMSSmmm``,
#: first 14 digits = %Y%m%d%H%M%S); lookahead requires exactly 17 digits so a longer run can't match partially.
_FDM_TIMESTAMP_RE = re.compile(r"^Image_(\d{17})(?=$|_)")


# datetime | None parse_fdm_timestamp(str stem)
# Inputs: str stem - an FDM image filename stem (source frame or "_aug"/"_original" variant)
# Outputs: datetime | None - the parsed capture timestamp, or None if stem doesn't start with
#          the expected 17-digit pattern or the first 14 digits aren't a valid
#          %Y%m%d%H%M%S date/time
# Description: Parses the capture timestamp embedded in an FDM stem. A variant's timestamp
#              parses identically to its source frame's (the timestamp precedes any
#              _aug/_original suffix). This is the foundation of leakage hazard #3's fix:
#              recovering real capture time lets fdm_session_groups segment frames into print
#              sessions instead of treating them as independent photos.
# Side Effects: None
def parse_fdm_timestamp(stem: str) -> datetime | None:
    m = _FDM_TIMESTAMP_RE.match(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1)[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


# dict[str, str] fdm_session_groups(Sequence[str] stems, float gap_s)
# Inputs: Sequence[str] stems - FDM image filename stems (source frames and/or
#         _aug/_original variants) for one FDM class directory
#         float gap_s - seconds between consecutive frame timestamps that starts a new print
#         session, default DEFAULT_SESSION_GAP_S (600s)
# Outputs: dict[str, str] - stem -> group_id ("sessionNNNNN" for chronologically-grouped
#          frames, or "unparsed:<frame_id>" for a frame whose timestamp couldn't be parsed)
#          for every stem in stems
# Description: Assigns every FDM stem to a print-session group id, so all frames of one print
#              job -- and every _aug/_original variant of each frame -- land in the same
#              dataset split (the guard against leakage hazard #3, the FDM sequential
#              time-lapse confound: consecutive frames ~30s apart are near-identical, so a
#              model split at the frame level memorizes specific print jobs instead of learning
#              defect appearance). Frames with a parseable timestamp are sorted chronologically
#              (ties broken by frame id) and a new session starts whenever the gap to the
#              previous frame exceeds gap_s seconds; a frame with no parseable timestamp gets
#              its own singleton group, keyed off the frame id.
# Side Effects: None (pure computation; no I/O or RNG)
def fdm_session_groups(stems: Sequence[str], gap_s: float = DEFAULT_SESSION_GAP_S) -> dict[str, str]:
    """Sorts frames chronologically by parsed timestamp and starts a new
    session whenever the gap exceeds ``gap_s``; unparseable stems get their
    own singleton group keyed by frame id."""
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
# Candidate model: a selectable image with a deferred loader.
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    class_name: str
    source: str  # "fdm" | "argus_v2" | "hf"
    group_key: str  # never split across train/val/test -- see module docstring
    output_stem: str  # unique within this candidate's class; used as the output filename
    loader: Callable[[], np.ndarray]  # decodes and returns a BGR image array on demand
    # "unique frames" reporting unit: for FDM, group_key is the session id (many frames) while
    # frame_key is the per-source-stem id; equal to group_key for every other source.
    frame_key: str = ""

    # None __post_init__()
    # Inputs: None (operates on self; group_key, frame_key)
    # Outputs: None
    # Description: Dataclass post-init hook that defaults frame_key to group_key when the
    #              caller didn't set frame_key explicitly (i.e. frames == sessions, true for
    #              every non-FDM source).
    # Side Effects: Mutates self.frame_key in place.
    def __post_init__(self) -> None:
        if not self.frame_key:
            self.frame_key = self.group_key


# Callable[[], np.ndarray] _file_loader(Path path)
# Inputs: Path path - path to an image file on disk
# Outputs: Callable[[], np.ndarray] - a zero-argument loader that decodes and returns the image
#          as a BGR array when called
# Description: Builds a deferred loader for a local image file, so nothing is decoded into
#              memory until a Candidate is actually chosen and materialized.
# Side Effects: None at call time; the returned closure reads the file from disk and decodes it
#               via cv2.imread when invoked.
def _file_loader(path: Path) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        return img

    return _load


# Callable[[], np.ndarray] _bytes_loader(bytes data)
# Inputs: bytes data - raw encoded image bytes (e.g. from a Hugging Face parquet row)
# Outputs: Callable[[], np.ndarray] - a zero-argument loader that decodes and returns the image
#          as a BGR array when called
# Description: Builds a deferred loader for an in-memory encoded image, so nothing is decoded
#              until a Candidate is actually chosen and materialized.
# Side Effects: None at call time; the returned closure decodes the bytes via cv2.imdecode when
#               invoked. No I/O.
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


# list[Path] scan_fdm_class_dir(Path fdm_root, str class_dir_name)
# Inputs: Path fdm_root - root of the extracted FDM raw dataset
#         str class_dir_name - FDM subdirectory name, e.g. "Cracking" or "Off_platform"
# Outputs: list[Path] - image file paths directly inside fdm_root/class_dir_name, sorted for
#          determinism
# Description: Lists every image directly inside an FDM class subdirectory.
# Side Effects: Raises FileNotFoundError if the class directory doesn't exist. Read-only
#               filesystem listing otherwise.
def scan_fdm_class_dir(fdm_root: Path, class_dir_name: str) -> list[Path]:
    d = fdm_root / class_dir_name
    if not d.is_dir():
        raise FileNotFoundError(f"FDM class directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# list[Candidate] fdm_candidates(Path fdm_root, str class_dir_name, str class_name, float session_gap_s)
# Inputs: Path fdm_root - root of the extracted FDM raw dataset
#         str class_dir_name - FDM subdirectory to scan, e.g. "Cracking" or "Off_platform"
#         str class_name - the output classification class name this directory feeds
#         float session_gap_s - print-session gap threshold (seconds), default
#         DEFAULT_SESSION_GAP_S (600s)
# Outputs: list[Candidate] - one Candidate per image, with group_key set to a print-session id
#          (namespaced by class_dir_name) and frame_key set to the older per-source-stem id
# Description: Builds one Candidate per image in an FDM class subdirectory, grouped for
#              SPLITTING by print session (fdm_session_groups, leakage hazard #3) and namespaced
#              by class_dir_name so two FDM subdirectories can never collide onto the same
#              group. frame_key separately records the per-source-stem grouping (hazard #1)
#              purely for "unique frames vs unique sessions" reporting; it plays no role in the
#              actual split.
# Side Effects: Read-only filesystem listing via scan_fdm_class_dir; each Candidate's loader
#               defers image decoding until called.
def fdm_candidates(
    fdm_root: Path,
    class_dir_name: str,
    class_name: str,
    session_gap_s: float = DEFAULT_SESSION_GAP_S,
) -> list[Candidate]:
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


# tuple[dict[str, list[Candidate]], list[Candidate]] gather_fdm_candidates(Path fdm_root, float session_gap_s)
# Inputs: Path fdm_root - root of the extracted FDM raw dataset
#         float session_gap_s - print-session gap threshold (seconds), default
#         DEFAULT_SESSION_GAP_S (600s)
# Outputs: tuple[dict[str, list[Candidate]], list[Candidate]] - (defect_by_class,
#          off_platform): defect_by_class maps class name to all candidates for the 4
#          FDM-only defect classes; off_platform is the Off_platform subdirectory's candidates
#          (feeds "spaghetti")
# Description: Scans the whole FDM raw tree once, building candidates for every FDM-sourced
#              class in one pass.
# Side Effects: Read-only filesystem listing via fdm_candidates/scan_fdm_class_dir for each FDM
#               class subdirectory.
def gather_fdm_candidates(
    fdm_root: Path, session_gap_s: float = DEFAULT_SESSION_GAP_S
) -> tuple[dict[str, list[Candidate]], list[Candidate]]:
    defect_by_class = {
        class_name: fdm_candidates(fdm_root, dir_name, class_name, session_gap_s)
        for dir_name, class_name in FDM_DEFECT_CLASS_DIRS.items()
    }
    off_platform = fdm_candidates(fdm_root, FDM_SPAGHETTI_DIR, "spaghetti", session_gap_s)
    return defect_by_class, off_platform


# --------------------------------------------------------------------------
# Source C -- spaghetti from the existing argus_v2 detection dataset
# --------------------------------------------------------------------------


# bool has_class_id(str label_text, int class_id)
# Inputs: str label_text - contents of a YOLO-format label file
#         int class_id - the class id to look for
# Outputs: bool - True if any row has class_id as its class (first whitespace-separated token)
# Description: Checks whether a YOLO label file's contents contain at least one row of the
#              given class_id. Blank lines are ignored; an empty/whitespace-only file
#              (legitimately "no objects") returns False, and a malformed leading token is
#              treated as "not a match" rather than raising.
# Side Effects: None
def has_class_id(label_text: str, class_id: int) -> bool:
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


# list[Path] scan_argus_spaghetti_images(Path argus_root, Sequence[str] split_dirs)
# Inputs: Path argus_root - root of the argus_v2 YOLO detection dataset
#         Sequence[str] split_dirs - argus_v2 split subdirectories to scan, default
#         ARGUS_SPLIT_DIRS ("train", "val", "test")
# Outputs: list[Path] - image paths whose paired label file contains at least one spaghetti
#          (ARGUS_SPAGHETTI_CLASS_ID) row
# Description: Pools every image across argus_root/{split}/images (for each of split_dirs)
#              whose paired label file contains at least one spaghetti-labeled row. argus_v2's
#              own train/val/test assignment is irrelevant here -- this dataset re-splits
#              everything from scratch via argus_group_key (leakage hazard #2). A split
#              directory that doesn't exist is skipped, keeping this usable on small synthetic
#              fixtures that only populate one split.
# Side Effects: Read-only filesystem traversal; reads each candidate image's paired label file
#               from disk.
def scan_argus_spaghetti_images(
    argus_root: Path, split_dirs: Sequence[str] = ARGUS_SPLIT_DIRS
) -> list[Path]:
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


# list[Candidate] argus_spaghetti_candidates(Path argus_root)
# Inputs: Path argus_root - root of the argus_v2 YOLO detection dataset
# Outputs: list[Candidate] - one Candidate per spaghetti-labeled argus_v2 image, class_name
#          "spaghetti", source "argus_v2", group_key namespaced by the recovered source stem
# Description: Builds one Candidate per spaghetti-labeled argus_v2 image, grouped by source
#              stem (argus_group_key) so Roboflow augmentation variants of one source photo are
#              never split across train/val/test (leakage hazard #2).
# Side Effects: Read-only filesystem traversal via scan_argus_spaghetti_images (reads each
#               image's paired label file); each Candidate's loader defers image decoding until
#               called.
def argus_spaghetti_candidates(argus_root: Path) -> list[Candidate]:
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
# Source B -- "normal" from Hugging Face (network I/O; not unit-tested).
# --------------------------------------------------------------------------


# dict[str, str] fetch_hf_parquet_urls(str dataset_id, str config, float timeout)
# Inputs: str dataset_id - Hugging Face dataset id, default HF_DATASET_ID
#         ("Masamsa/3d-print-failure-detection")
#         str config - dataset config name, default HF_CONFIG ("default")
#         float timeout - HTTP request timeout in seconds, default 30.0
# Outputs: dict[str, str] - {split_name: parquet_url} for config
# Description: Queries the Hugging Face datasets-server parquet index for the given dataset/
#              config and returns the first parquet URL per split. Anonymous, no API key
#              required.
# Side Effects: Makes one HTTP GET request to HF_PARQUET_INDEX_URL (huggingface.co); raises via
#               requests if the request fails (raise_for_status).
def fetch_hf_parquet_urls(
    dataset_id: str = HF_DATASET_ID, config: str = HF_CONFIG, timeout: float = 30.0
) -> dict[str, str]:
    resp = requests.get(HF_PARQUET_INDEX_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    split_urls = data[config]
    return {split: urls[0] for split, urls in split_urls.items() if urls}


# Path download_file(str url, Path dest, int chunk_size, float timeout)
# Inputs: str url - URL to download
#         Path dest - local destination file path
#         int chunk_size - streaming download chunk size in bytes, default 1 << 20 (1 MiB)
#         float timeout - HTTP request timeout in seconds, default 60.0
# Outputs: Path - dest (whether freshly downloaded or already cached)
# Description: Stream-downloads url to dest, idempotently: if dest already exists and is
#              non-empty, the download is skipped entirely (so re-running the pipeline doesn't
#              re-fetch ~580MB of parquet every time). Downloads to a .part sibling first and
#              atomically renames on success, so a crash mid-download can't leave a truncated
#              file that looks "already cached" on the next run.
# Side Effects: Makes an HTTP GET request (streaming) to url; creates dest's parent directory;
#               writes a .part file to disk and renames it to dest on success. Skips all of the
#               above if dest already exists and is non-empty.
def download_file(url: str, dest: Path, chunk_size: int = 1 << 20, timeout: float = 60.0) -> Path:
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


# dict[str, Path] download_hf_normal_parquets(Path cache_dir)
# Inputs: Path cache_dir - local directory to cache downloaded parquet files in
# Outputs: dict[str, Path] - {split_name: local_path} for every HF dataset split
# Description: Fetches the parquet-index URLs and downloads every split's parquet file into
#              cache_dir (idempotent -- see download_file).
# Side Effects: Makes an HTTP GET request to the HF parquet index (fetch_hf_parquet_urls), then
#               one streaming HTTP download per split (download_file), writing parquet files to
#               disk under cache_dir unless already cached.
def download_hf_normal_parquets(cache_dir: Path) -> dict[str, Path]:
    urls = fetch_hf_parquet_urls()
    paths: dict[str, Path] = {}
    for split, url in urls.items():
        dest = cache_dir / f"{split}.parquet"
        paths[split] = download_file(url, dest)
    return paths


# list[bytes] load_normal_rows_from_parquet(Path parquet_path)
# Inputs: Path parquet_path - path to a local HF dataset parquet file (schema: "image"
#         struct{bytes,path}, "label" int)
# Outputs: list[bytes] - raw encoded-image bytes of every row with label == HF_NORMAL_LABEL
# Description: Reads a local parquet file and returns the raw encoded-image bytes of every
#              normal-labeled (healthy print) row. Uses polars' native parquet reader (no
#              pyarrow dependency needed).
# Side Effects: Reads parquet_path from disk.
def load_normal_rows_from_parquet(parquet_path: Path) -> list[bytes]:
    df = pl.read_parquet(parquet_path, columns=["image", "label"])
    normal = df.filter(pl.col("label") == HF_NORMAL_LABEL)
    return [row["image"]["bytes"] for row in normal.iter_rows(named=True)]


# list[Candidate] hf_normal_candidates(Mapping[str, Sequence[bytes]] image_bytes_by_split)
# Inputs: Mapping[str, Sequence[bytes]] image_bytes_by_split - split name -> list of raw
#         encoded-image bytes for that split's normal-labeled rows
# Outputs: list[Candidate] - one Candidate per HF normal-labeled row, class_name "normal",
#          source "hf", each with its own singleton group_key
# Description: Builds one Candidate per HF normal-labeled row. Each row is an independent photo
#              (no known augmentation-duplication issue for this source, unlike FDM/argus_v2),
#              so each gets its own singleton group, namespaced by split + row index for a
#              stable, unique output_stem.
# Side Effects: Each Candidate's loader defers image decoding (from the already-in-memory
#               bytes) until called; no I/O at call time.
def hf_normal_candidates(image_bytes_by_split: Mapping[str, Sequence[bytes]]) -> list[Candidate]:
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


# list[str] deterministic_sample(Sequence[str] keys, int k, int seed)
# Inputs: Sequence[str] keys - candidate keys to sample from (deduplicated)
#         int k - how many keys to choose
#         int seed - RNG seed for deterministic shuffling
# Outputs: list[str] - up to k keys, deterministically chosen for the given seed; all of keys
#          (sorted+shuffled) if k >= len(keys)
# Description: Deterministically chooses up to k of keys for a given seed: sorts first (so
#              input order never matters), shuffles with a fresh random.Random(seed), then
#              takes the first k.
# Side Effects: Uses a locally-created random.Random(seed); does not touch global RNG state.
def deterministic_sample(keys: Sequence[str], k: int, seed: int) -> list[str]:
    ordered = sorted(set(keys))
    random.Random(seed).shuffle(ordered)
    if k >= len(ordered):
        return ordered
    return ordered[:k]


# list[Candidate] sample_candidates(Sequence[Candidate] candidates, int k, int seed)
# Inputs: Sequence[Candidate] candidates - candidates to sample from
#         int k - how many candidates to choose
#         int seed - RNG seed for deterministic shuffling
# Outputs: list[Candidate] - up to k candidates, deterministically chosen for the given seed
# Description: deterministic_sample lifted to work on Candidate objects, keyed by their
#              (class-unique) output_stem.
# Side Effects: Uses a locally-created random.Random(seed) (via deterministic_sample); does not
#               touch global RNG state. Raises ValueError if two candidates share an
#               output_stem, which would make the sample ambiguous/non-deterministic.
def sample_candidates(candidates: Sequence[Candidate], k: int, seed: int) -> list[Candidate]:
    by_key: dict[str, Candidate] = {}
    for c in candidates:
        if c.output_stem in by_key:
            raise ValueError(f"duplicate output_stem {c.output_stem!r} among candidates")
        by_key[c.output_stem] = c
    chosen_keys = deterministic_sample(list(by_key.keys()), k, seed)
    return [by_key[key] for key in chosen_keys]


# tuple[list[str], list[str], list[str]] split_groups(Sequence[str] group_ids, int seed, tuple[float, float, float] ratios)
# Inputs: Sequence[str] group_ids - group identifiers (print sessions or independent-photo
#         ids), not individual files or images
#         int seed - RNG seed for deterministic shuffling, e.g. the CLI's default DEFAULT_SEED
#         (1337)
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15); must sum to 1.0
# Outputs: tuple[list[str], list[str], list[str]] - (train_ids, val_ids, test_ids) group id
#          lists; any rounding remainder is given to test so no group is invented or dropped
# Description: Deterministically splits group ids (not individual files) into train/val/test.
#              Mirrors training/prepare_dataset.py's split_sources: sorts first (input order
#              never matters), shuffles with a seeded random.Random, then slices by ratio. The
#              GROUP is the split unit -- a print session or an independent source photo -- so
#              this is what prevents near-duplicate frames/variants from leaking across splits.
# Side Effects: Raises ValueError if ratios don't sum to 1.0 (within 1e-6). Uses a locally-
#               seeded random.Random(seed); does not touch global RNG state.
def split_groups(
    group_ids: Sequence[str],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[list[str], list[str], list[str]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios!r} (sum={sum(ratios)})")

    ids = sorted(group_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_train = min(round(n * ratios[0]), n)
    n_val = min(round(n * ratios[1]), n - n_train)
    n_test = n - n_train - n_val

    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val : n_train + n_val + n_test]


# None assert_no_group_overlap(Mapping[str, Sequence[str]] split_group_ids)
# Inputs: Mapping[str, Sequence[str]] split_group_ids - split name -> group ids assigned to it
# Outputs: None
# Description: Verifies no group identity (print session or independent-photo id) appears in
#              more than one split. This is the explicit check for the whole point of splitting
#              by group instead of by file (the leakage-hazard guard), rather than just trusting
#              split_groups.
# Side Effects: Raises AssertionError naming the offending group and both splits it appears in,
#               if any group-identity leakage is detected. No filesystem or RNG activity.
def assert_no_group_overlap(split_group_ids: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in split_group_ids.items():
        for gid in ids:
            if gid in seen:
                raise AssertionError(
                    f"Group-identity leakage detected: group {gid!r} appears in both "
                    f"'{seen[gid]}' and '{split_name}' splits."
                )
            seen[gid] = split_name


# tuple[list[str], list[str], list[str]] _ensure_min_one_per_split(list[str] train_ids, list[str] val_ids, list[str] test_ids)
# Inputs: list[str] train_ids - group ids assigned to train by a ratio-based split
#         list[str] val_ids - group ids assigned to val by a ratio-based split
#         list[str] test_ids - group ids assigned to test by a ratio-based split
# Outputs: tuple[list[str], list[str], list[str]] - (train_ids, val_ids, test_ids) adjusted so
#          val_ids and test_ids each have at least 1 group (when possible)
# Description: After a ratio-based split_groups call, guarantees val_ids and test_ids each end
#              up with >=1 group (a strict 70/15/15 split of a small group count can round val
#              or test down to 0, e.g. 3 groups -> 2/0/1). Moves one group at a time out of
#              train_ids (preferred donor so long as it stays non-empty) into whichever of
#              val/test is still empty; falls back to borrowing from the other of val/test if
#              train_ids can't spare one, so train never drops below 1 group. Pure function of
#              its (already seed-shuffled) inputs, so this stays deterministic.
# Side Effects: None
def _ensure_min_one_per_split(
    train_ids: list[str], val_ids: list[str], test_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
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


# dict[str, list[Candidate]] assign_splits_for_class(Sequence[Candidate] candidates, int seed, tuple[float, float, float] ratios)
# Inputs: Sequence[Candidate] candidates - one class's kept candidates (post per-class
#         selection)
#         int seed - RNG seed for deterministic shuffling
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15)
# Outputs: dict[str, list[Candidate]] - {"train": [...], "val": [...], "test": [...]}
#          candidates per split; val and test are both empty if the class had fewer than
#          MIN_SESSIONS_FOR_EVAL groups (not evaluable)
# Description: Groups one class's kept candidates by group_key (a print SESSION id for FDM
#              sources, or an independent-photo id otherwise -- see module docstring), splits
#              the GROUPS 70/15/15 (deterministic for seed), verifies zero overlap, then expands
#              back to per-split candidate lists (every candidate in a group lands in that
#              group's split). Fewer than MIN_SESSIONS_FOR_EVAL groups: every candidate goes to
#              train and val/test are both empty (how callers detect "not evaluable"). Otherwise
#              the ratio split runs normally, then _ensure_min_one_per_split guarantees >=1
#              group landed in val and >=1 in test.
# Side Effects: Uses a locally-seeded random.Random(seed) (via split_groups); does not touch
#               global RNG state. Raises AssertionError (via assert_no_group_overlap) if group
#               overlap is somehow detected. No filesystem I/O.
def assign_splits_for_class(
    candidates: Sequence[Candidate],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[Candidate]]:
    """Fewer than ``MIN_SESSIONS_FOR_EVAL`` groups: everything goes to train
    and val/test are empty (not evaluable)."""
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


# list[Candidate] select_normal(list[Candidate] hf_pool, int target, int cap, int seed)
# Inputs: list[Candidate] hf_pool - all HF normal-labeled candidates
#         int target - target number of images for this class, e.g. TARGET_PER_CLASS (500)
#         int cap - upper bound on images kept for this class
#         int seed - RNG seed for deterministic sampling
# Outputs: list[Candidate] - min(target, cap, len(hf_pool)) sampled candidates
# Description: Samples min(target, cap, len(hf_pool)) images from the HF normal pool for the
#              "normal" class.
# Side Effects: Uses a locally-seeded random.Random(seed) (via sample_candidates); does not
#               touch global RNG state.
def select_normal(hf_pool: list[Candidate], target: int, cap: int, seed: int) -> list[Candidate]:
    k = min(target, cap, len(hf_pool))
    return sample_candidates(hf_pool, k, seed)


# list[Candidate] select_spaghetti(list[Candidate] off_platform, list[Candidate] argus_pool, int target, int cap, int seed)
# Inputs: list[Candidate] off_platform - all FDM Off_platform candidates (always fully kept)
#         list[Candidate] argus_pool - all argus_v2 spaghetti-labeled candidates
#         int target - target number of images for this class, e.g. TARGET_PER_CLASS (500)
#         int cap - upper bound on images kept for this class
#         int seed - RNG seed for deterministic sampling
# Outputs: list[Candidate] - combined off_platform + sampled argus_pool candidates, capped at cap
# Description: Always includes ALL of FDM's Off_platform images, topped up with a deterministic
#              sample from the argus_v2 spaghetti pool to reach min(target, cap) total. If
#              Off_platform alone somehow exceeds the cap (only possible with a very small
#              --per-class-cap), the combined set is trimmed back down to cap rather than
#              silently ignoring it.
# Side Effects: Uses a locally-seeded random.Random(seed) (via sample_candidates); does not
#               touch global RNG state.
def select_spaghetti(
    off_platform: list[Candidate],
    argus_pool: list[Candidate],
    target: int,
    cap: int,
    seed: int,
) -> list[Candidate]:
    budget = min(target, cap)
    kept_off_platform = list(off_platform)
    remaining = max(0, budget - len(kept_off_platform))
    kept_argus = sample_candidates(argus_pool, min(remaining, len(argus_pool)), seed)
    combined = kept_off_platform + kept_argus
    if len(combined) > cap:
        combined = sample_candidates(combined, cap, seed)
    return combined


# list[Candidate] select_fdm_defect_class(list[Candidate] pool, int cap, int seed)
# Inputs: list[Candidate] pool - all candidates for one FDM-only defect class
#         int cap - upper bound on images kept for this class
#         int seed - RNG seed for deterministic sampling
# Outputs: list[Candidate] - min(len(pool), cap) candidates
# Description: Uses every image in pool (an FDM-only defect class), unless it exceeds cap, in
#              which case samples down to cap.
# Side Effects: Uses a locally-seeded random.Random(seed) (via sample_candidates); does not
#               touch global RNG state.
def select_fdm_defect_class(pool: list[Candidate], cap: int, seed: int) -> list[Candidate]:
    k = min(len(pool), cap)
    return sample_candidates(pool, k, seed)


# --------------------------------------------------------------------------
# Image normalization -- must match argus.detectors.classifier.preprocess_classify exactly.
# --------------------------------------------------------------------------


# np.ndarray resize_and_center_crop(np.ndarray image, int size)
# Inputs: np.ndarray image - source image (BGR array) to normalize
#         int size - target output size (both dimensions), default OUTPUT_SIZE (512)
# Outputs: np.ndarray - image resized so its short side is size px (aspect-preserving) and
#          center-cropped to size x size
# Description: Resizes image so its short side is exactly size px (preserving aspect ratio),
#              then center-crops to size x size. Mirrors
#              argus.detectors.classifier.preprocess_classify's resize/crop steps exactly (same
#              defensive max(size, round(...)) clamp against float round-off) so training-time
#              and inference-time preprocessing never diverge. Also erases the raw-resolution
#              gap between the ~2048x3072 FDM originals and the much smaller Hugging Face
#              ones -- one of the two source-confound mitigations (see module docstring).
# Side Effects: None (pure array transformation via cv2.resize; no I/O)
def resize_and_center_crop(image: np.ndarray, size: int = OUTPUT_SIZE) -> np.ndarray:
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


# None save_normalized_jpeg(np.ndarray image, Path dest, int quality)
# Inputs: np.ndarray image - source image (BGR array) to normalize and write
#         Path dest - destination file path for the output JPEG
#         int quality - JPEG encoding quality, default JPEG_QUALITY (90)
# Outputs: None
# Description: Resizes+center-crops image to OUTPUT_SIZE (via resize_and_center_crop) and
#              writes it as a JPEG at a fixed quality. Both steps exist purely to erase
#              source-identifying cues (raw resolution, JPEG compression-artifact statistics)
#              that would otherwise let a classifier cheat by learning "which dataset is this
#              from" instead of "is this print failing" (see module docstring's
#              CONFOUND_WARNING).
# Side Effects: Creates dest's parent directory; writes (overwrites) a JPEG file to disk at
#               dest. Raises IOError if cv2.imwrite fails.
def save_normalized_jpeg(image: np.ndarray, dest: Path, quality: int = JPEG_QUALITY) -> None:
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

    # dict[str, object] to_dict()
    # Inputs: None (operates on self)
    # Outputs: dict[str, object] - JSON-serializable view of this ClassReport: counts (per
    #          split plus total), source_breakdown, n_frames, n_sessions, sessions_per_split,
    #          evaluable
    # Description: Converts this ClassReport instance into a plain dict for JSON serialization
    #              into split_report.json.
    # Side Effects: None
    def to_dict(self) -> dict[str, object]:
        return {
            "counts": {**self.per_split_counts, "total": sum(self.per_split_counts.values())},
            "source_breakdown": self.per_source_counts,
            "n_frames": self.n_frames,
            "n_sessions": self.n_sessions,
            "sessions_per_split": dict(self.sessions_per_split),
            "evaluable": self.evaluable,
        }


# ClassReport materialize_class(str class_name, Mapping[str, list[Candidate]] split_assignment, Path out_dir)
# Inputs: str class_name - output classification class name
#         Mapping[str, list[Candidate]] split_assignment - {"train"/"val"/"test": [...]}
#         candidates for this class, from assign_splits_for_class
#         Path out_dir - output dataset root (e.g. datasets/argus_cls)
# Outputs: ClassReport - tallies of per-split counts, per-source counts, unique frames/sessions,
#          sessions per split, and whether this class is evaluable
# Description: Writes every candidate in split_assignment to
#              out_dir/<split>/<class_name>/<output_stem>.jpg (normalized via
#              save_normalized_jpeg) and tallies a ClassReport. evaluable is derived straight
#              from split_assignment (val and test are both non-empty iff the class cleared
#              MIN_SESSIONS_FOR_EVAL), so there's a single source of truth for what "not
#              evaluable" means.
# Side Effects: Creates out_dir/<split>/<class_name> directories; calls each candidate's loader
#               (decoding an image from disk or memory) and writes a normalized JPEG to disk
#               for every candidate.
def materialize_class(
    class_name: str,
    split_assignment: Mapping[str, list[Candidate]],
    out_dir: Path,
) -> ClassReport:
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


# list[str] evaluation_validity_lines(Mapping[str, ClassReport] class_reports)
# Inputs: Mapping[str, ClassReport] class_reports - class name -> ClassReport from
#         materialize_class, for every class in CLASS_NAMES
# Outputs: list[str] - one warning line per flagged class; classes not flagged by either rule
#          are omitted entirely
# Description: Builds one warning line per class whose val/test accuracy shouldn't be trusted:
#              "NOT EVALUATED" for a class below MIN_SESSIONS_FOR_EVAL sessions (no val/test
#              split exists at all), or "WEAK" for a class that has a val/test split but with
#              too few sessions total or in val/test to be a meaningful per-class estimate.
# Side Effects: None
def evaluation_validity_lines(class_reports: Mapping[str, ClassReport]) -> list[str]:
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


# dict[str, object] build_dataset(Path out_dir, Mapping[str, list[Candidate]] fdm_defect_by_class, list[Candidate] fdm_off_platform, list[Candidate] argus_spaghetti_pool, list[Candidate] hf_normal_pool, int seed, int per_class_cap, int target_per_class, tuple[float, float, float] ratios)
# Inputs: Path out_dir - output dataset root (e.g. datasets/argus_cls); keyword-only
#         Mapping[str, list[Candidate]] fdm_defect_by_class - class name -> candidates, for the
#         4 FDM-only defect classes; keyword-only
#         list[Candidate] fdm_off_platform - FDM Off_platform candidates (feeds "spaghetti");
#         keyword-only
#         list[Candidate] argus_spaghetti_pool - argus_v2 spaghetti-labeled candidates;
#         keyword-only
#         list[Candidate] hf_normal_pool - HF normal-labeled candidates; keyword-only
#         int seed - RNG seed for deterministic sampling/splitting, default DEFAULT_SEED (1337);
#         keyword-only
#         int per_class_cap - upper bound on images kept per class, default
#         DEFAULT_PER_CLASS_CAP (550); keyword-only
#         int target_per_class - target image count per class, default TARGET_PER_CLASS (500);
#         keyword-only
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15); keyword-only
# Outputs: dict[str, object] - the full split report: seed/cap/ratio/output settings, per-class
#          ClassReport dicts, total frames/sessions, the group-overlap check result, the
#          confound warning, and the evaluation-validity warning
# Description: Pure(ish) orchestration core: given already-gathered candidate pools (no
#              network/dataset I/O of its own beyond writing out_dir), selects each class via
#              its per-class recipe, group-aware-splits via assign_splits_for_class, materializes
#              all 6 classes, re-verifies zero group overlap across all classes combined (not
#              just per-class), and assembles the full report dict. Kept separate from main() so
#              tests can call this directly with small synthetic pools -- no network, no real
#              dataset, no monkeypatching required.
# Side Effects: Creates out_dir and its per-split/per-class subdirectories; writes normalized
#               JPEG files to disk for every selected candidate (via materialize_class); uses
#               locally-seeded RNG throughout (does not touch global RNG state); raises
#               AssertionError if group overlap is somehow detected across classes.
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
    """Kept separate from ``main()`` so tests can call this directly with
    small synthetic pools -- no network, no real dataset, no monkeypatching."""
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


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed options (fdm_dir, argus_dir, hf_cache_dir, out, seed,
#          per_class_cap, session_gap_s, force). Notable defaults: --seed DEFAULT_SEED (1337),
#          --per-class-cap DEFAULT_PER_CLASS_CAP (550), --session-gap-s DEFAULT_SESSION_GAP_S
#          (600s, the print-session leakage-hazard-#3 threshold).
# Description: Defines and parses the CLI for building the classification dataset.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
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


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Prints the confound warning, scans the FDM dataset and reports
#              per-class session counts, scans argus_v2 for spaghetti-labeled images, downloads
#              and reads the Hugging Face normal-image parquets, then selects/group-aware-splits/
#              normalizes/writes all 6 classes via build_dataset and prints the full split
#              report (including the evaluation-validity warning).
# Side Effects: Reads the FDM and argus_v2 datasets from local disk; makes HTTP requests to
#               huggingface.co and downloads parquet files to --hf-cache-dir (unless already
#               cached, see download_hf_normal_parquets); optionally wipes --out with
#               shutil.rmtree when --force is passed; creates --out and its class/split
#               subdirectories; writes normalized JPEG images to disk for every selected
#               candidate; writes --out/split_report.json; prints extensive progress and a full
#               summary report to stdout.
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
