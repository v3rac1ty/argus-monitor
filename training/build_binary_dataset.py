"""Builds a leak-free, BINARY (normal vs failure) classification dataset at
``datasets/argus_bin/{train,val,test}/{failure,normal}/*.jpg`` from ONE Hugging
Face source: ``Masamsa/3d-print-failure-detection``.

Why binary, and why one source: ``training/build_classification_dataset.py``'s
6-class dataset mixes an FDM dataset (which has ZERO normal images) with
Hugging Face normals, so ``source`` and ``label`` are perfectly correlated and
a model can score well by recognizing which dataset an image came from (see
that module's ``CONFOUND_WARNING``). This script sidesteps that confound
entirely by drawing BOTH labels from a single source dataset that carries
both classes itself (label 0 == normal, label 1 == failure).

This source's own rows are almost certainly timelapse frames of a small
number of print jobs, so its own train/validation/test boundary is not
trustworthy for a leak-free split. This script therefore:

  1. Pools every row across all three of the source's own splits.
  2. Drops exact-duplicate images (identical decoded pixels).
  3. Clusters near-duplicate images (perceptual hash, Hamming distance) into
     connected components -- each component is one independent SCENE.
  4. Reports, as a headline number, how many of those scenes straddle the
     source's own original train/test boundary -- this is the direct
     evidence for whether that boundary leaked.
  5. Re-splits by SCENE (never by image), stratified by label, targeting
     70/15/15.
  6. Excludes any scene that (rarely) contains both labels -- a single scene
     cannot be both a normal and a failed print, so a mixed scene is a
     data-quality signal, not a class to guess at.

Because the true independent sample size is the scene count, not the raw
image count, the written ``split_report.json`` includes an explicit
``evaluation_validity_warning`` that is blunt about how few scenes back any
val/test metric when the scene count is small.

Usage: python training/build_binary_dataset.py [--out PATH] [--cache-dir PATH] [--seed N] [--phash-distance N] [--per-class-cap N] [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import polars as pl
import requests

# Run-from-anywhere bootstrap: this module reuses training/build_classification_dataset.py
# and training/ingest_roboflow_multi.py, which are only importable once the repo root is on
# sys.path (pytest already does this via pyproject's pythonpath; `python training/...` does not).
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.build_classification_dataset import (  # noqa: E402
    JPEG_QUALITY,
    OUTPUT_SIZE,
    download_file,
    save_normalized_jpeg,
)
from training.ingest_roboflow_multi import (  # noqa: E402
    DEFAULT_PHASH_DISTANCE,
    content_hash,
    dhash,
    phash_neighbor_pairs,
)

DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "argus_bin"
DEFAULT_CACHE_DIR = REPO_ROOT / "datasets" / "hf_binary_raw"

#: datasets-server parquet index -- anonymous, no API key needed. Returns
#: {config: {split: [parquet_url]}}.
HF_DATASET_ID = "Masamsa/3d-print-failure-detection"
HF_PARQUET_INDEX_URL_TEMPLATE = "https://huggingface.co/api/datasets/{dataset_id}/parquet"
HF_CONFIG = "default"

#: In this dataset's ``label`` column, 0 == normal / healthy print, 1 == failure.
HF_LABEL_NORMAL = 0
HF_LABEL_FAILURE = 1

#: Output class directory names, SORTED alphabetically -- index 0 is "failure",
#: index 1 is "normal". Every downstream consumer (training script, ONNX
#: export, inference) must derive class order from sorted folder names, never
#: hardcode a different order.
CLASS_NAMES: tuple[str, ...] = tuple(sorted(("failure", "normal")))

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

DEFAULT_SEED = 1337
DEFAULT_SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

#: Below this many independent scenes (near-duplicate clusters) in the WHOLE
#: deduplicated pool, or below MIN_CLUSTERS_PER_SPLIT_CLASS scenes for one
#: class in one split, val/test metrics rest on too few independent
#: observations to be read as a generalization estimate -- see
#: build_evaluation_validity_warning.
MIN_INDEPENDENT_SCENES_FOR_TRUST = 40
MIN_CLUSTERS_PER_SPLIT_CLASS = 5


# --------------------------------------------------------------------------
# Record model
# --------------------------------------------------------------------------


@dataclass
class HfRecord:
    """One row pooled from one of the source's own splits, plus everything
    needed to dedupe, cluster, split and materialize it."""

    uid: str  # unique across the whole pool, e.g. "train:000123"
    orig_split: str  # the HF split this row came from ("train"/"validation"/"test")
    label: int  # raw HF label: HF_LABEL_NORMAL (0) or HF_LABEL_FAILURE (1)
    content_hash: str  # exact-duplicate fingerprint (see training.ingest_roboflow_multi.content_hash)
    phash: int  # near-duplicate fingerprint (see training.ingest_roboflow_multi.dhash)
    loader: Callable[[], np.ndarray]  # decodes and returns a BGR image array on demand
    output_stem: str  # unique output filename stem (without extension)


# str class_name_for_label(int label)
# Inputs: int label - a raw HF ``label`` column value
# Outputs: str - "normal" for HF_LABEL_NORMAL (0), "failure" for HF_LABEL_FAILURE (1)
# Description: Maps this dataset's raw binary label to our output class name. This is the
#              single place that encodes the "0 == normal, 1 == failure" mapping documented in
#              the module docstring, so nothing else needs to know the raw label values.
# Side Effects: Raises ValueError for any label other than 0 or 1.
def class_name_for_label(label: int) -> str:
    if label == HF_LABEL_NORMAL:
        return "normal"
    if label == HF_LABEL_FAILURE:
        return "failure"
    raise ValueError(f"unrecognized HF label {label!r}; expected {HF_LABEL_NORMAL} (normal) or {HF_LABEL_FAILURE} (failure)")


# Callable[[], np.ndarray] _bytes_loader(bytes data)
# Inputs: bytes data - raw encoded image bytes (from a Hugging Face parquet row)
# Outputs: Callable[[], np.ndarray] - a zero-argument loader that decodes and returns the image
#          as a BGR array when called
# Description: Builds a deferred loader for an in-memory encoded image, so nothing is decoded
#              until explicitly requested (either to compute this row's hashes once, or later to
#              materialize its output JPEG).
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
# Download + parquet loading (network I/O; not unit-tested)
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
# Side Effects: Makes one HTTP GET request to huggingface.co; raises via requests if the request
#               fails (raise_for_status).
def fetch_hf_parquet_urls(dataset_id: str = HF_DATASET_ID, config: str = HF_CONFIG, timeout: float = 30.0) -> dict[str, str]:
    url = HF_PARQUET_INDEX_URL_TEMPLATE.format(dataset_id=dataset_id)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    split_urls = data[config]
    return {split: urls[0] for split, urls in split_urls.items() if urls}


# dict[str, Path] download_hf_split_parquets(Path cache_dir)
# Inputs: Path cache_dir - local directory to cache downloaded parquet files in
# Outputs: dict[str, Path] - {split_name: local_path} for every source split (expected: "train",
#          "validation", "test")
# Description: Fetches the parquet-index URLs and downloads every split's parquet file into
#              cache_dir, reusing training.build_classification_dataset.download_file so the
#              ~582MB download is idempotent (skipped on re-run once cached).
# Side Effects: Makes an HTTP GET request to the HF parquet index (fetch_hf_parquet_urls), then
#               one streaming HTTP download per split (download_file), writing parquet files to
#               disk under cache_dir unless already cached.
def download_hf_split_parquets(cache_dir: Path) -> dict[str, Path]:
    urls = fetch_hf_parquet_urls()
    paths: dict[str, Path] = {}
    for split, url in urls.items():
        dest = cache_dir / f"{split}.parquet"
        paths[split] = download_file(url, dest)
    return paths


# list[tuple[int, bytes]] load_rows_from_parquet(Path parquet_path)
# Inputs: Path parquet_path - path to a local HF dataset parquet file (schema: "image"
#         struct{bytes,path}, "label" int)
# Outputs: list[tuple[int, bytes]] - (label, raw_encoded_image_bytes) for every row, in file
#          order -- UNFILTERED (unlike the 6-class builder's normal-only reader, this dataset's
#          whole point is that it carries both labels)
# Description: Reads a local parquet file and returns every row's raw label and encoded-image
#              bytes. Uses polars' native parquet reader (no pyarrow dependency needed).
# Side Effects: Reads parquet_path from disk.
def load_rows_from_parquet(parquet_path: Path) -> list[tuple[int, bytes]]:
    df = pl.read_parquet(parquet_path, columns=["image", "label"])
    return [(int(row["label"]), row["image"]["bytes"]) for row in df.iter_rows(named=True)]


# list[HfRecord] gather_hf_records(Mapping[str, Path] parquet_paths)
# Inputs: Mapping[str, Path] parquet_paths - split_name -> local parquet path, from
#         download_hf_split_parquets
# Outputs: list[HfRecord] - one HfRecord per row across every split, with content_hash/phash
#          already computed
# Description: Reads every split's parquet file and builds one HfRecord per row. Unlike the
#              6-class builder (which never needs to decode an HF image until final write), this
#              pipeline's whole point is deduplication/clustering, which requires DECODED PIXELS
#              -- so every image is decoded once here to fingerprint it. The loader stored on the
#              record re-decodes from the same original bytes at materialize time; this trades a
#              second decode pass for not holding every decoded array in memory at once.
# Side Effects: Reads and decodes every image once (via content_hash/dhash on the decoded array).
#               No filesystem writes.
def gather_hf_records(parquet_paths: Mapping[str, Path]) -> list[HfRecord]:
    records: list[HfRecord] = []
    for split in sorted(parquet_paths):
        rows = load_rows_from_parquet(parquet_paths[split])
        for idx, (label, data) in enumerate(rows):
            loader = _bytes_loader(data)
            image = loader()
            records.append(
                HfRecord(
                    uid=f"{split}:{idx:06d}",
                    orig_split=split,
                    label=label,
                    content_hash=content_hash(image),
                    phash=dhash(image),
                    loader=loader,
                    output_stem=f"hf_{class_name_for_label(label)}_{split}_{idx:06d}",
                )
            )
    return records


# --------------------------------------------------------------------------
# Exact-duplicate removal + near-duplicate clustering (pure; no I/O)
# --------------------------------------------------------------------------


# tuple[list[HfRecord], int] dedupe_exact_duplicates(Sequence[HfRecord] records)
# Inputs: Sequence[HfRecord] records - pooled records, any order
# Outputs: tuple[list[HfRecord], int] - (kept, n_dropped): kept has exactly one record per
#          distinct content_hash (the lexicographically-first uid among ties, for determinism);
#          n_dropped counts every record removed as an exact duplicate
# Description: Removes exact-pixel-duplicate rows (identical content_hash), keeping the first by
#              uid. Input order never matters -- records are sorted by uid before scanning.
# Side Effects: None (pure; no I/O or RNG)
def dedupe_exact_duplicates(records: Sequence[HfRecord]) -> tuple[list[HfRecord], int]:
    seen: set[str] = set()
    kept: list[HfRecord] = []
    dropped = 0
    for r in sorted(records, key=lambda rec: rec.uid):
        if r.content_hash in seen:
            dropped += 1
            continue
        seen.add(r.content_hash)
        kept.append(r)
    return kept, dropped


# str _uf_find(dict[str, str] parent, str key)
# Inputs: dict[str, str] parent - union-find parent map (key -> its current parent, self if root)
#         str key - a uid known to parent
# Outputs: str - the representative (root) uid of key's set
# Description: Iterative find with path compression (iterative so a long chain of near-duplicate
#              timelapse frames can't blow the recursion limit).
# Side Effects: Rewrites parent entries along the path for future lookups.
def _uf_find(parent: dict[str, str], key: str) -> str:
    root = key
    while parent[root] != root:
        root = parent[root]
    while parent[key] != root:
        parent[key], key = root, parent[key]
    return root


# None _uf_union(dict[str, str] parent, str a, str b)
# Inputs: dict[str, str] parent - union-find parent map, mutated in place
#         str a - a uid known to parent
#         str b - another uid known to parent
# Outputs: None
# Description: Merges a's and b's sets. Always attaches the LEXICOGRAPHICALLY GREATER of the two
#              current roots under the smaller one (rather than union-by-rank), which guarantees
#              -- regardless of the order pairs are unioned in -- that a connected component's
#              final root is always its globally minimum uid. That makes cluster identity fully
#              deterministic without a separate normalization pass.
# Side Effects: Mutates parent.
def _uf_union(parent: dict[str, str], a: str, b: str) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra == rb:
        return
    if ra > rb:
        ra, rb = rb, ra
    parent[rb] = ra


# dict[str, str] cluster_by_phash(Sequence[HfRecord] records, int phash_distance)
# Inputs: Sequence[HfRecord] records - deduplicated records to cluster
#         int phash_distance - inclusive Hamming-distance threshold, default
#         DEFAULT_PHASH_DISTANCE (imported from training.ingest_roboflow_multi, currently 6)
# Outputs: dict[str, str] - uid -> cluster_id for every record in records; cluster_id is the
#          minimum uid within that connected component (see _uf_union)
# Description: Clusters near-duplicate images into connected components ("independent scenes")
#              via phash_neighbor_pairs (reused from training.ingest_roboflow_multi) plus a local
#              union-find. Two images end up in the same cluster iff there's a chain of pairwise
#              Hamming distances each <= phash_distance linking them -- not necessarily that
#              every pair in the cluster is directly close.
# Side Effects: None (pure computation; no I/O or RNG)
def cluster_by_phash(records: Sequence[HfRecord], phash_distance: int = DEFAULT_PHASH_DISTANCE) -> dict[str, str]:
    parent: dict[str, str] = {r.uid: r.uid for r in records}
    hash_by_uid = {r.uid: r.phash for r in records}
    for a, b in phash_neighbor_pairs(hash_by_uid, max_distance=phash_distance):
        _uf_union(parent, a, b)
    return {uid: _uf_find(parent, uid) for uid in parent}


# --------------------------------------------------------------------------
# Leakage audit: does a cluster straddle HF's own split, or mix labels?
# --------------------------------------------------------------------------


# tuple[int, dict[str, set[str]]] straddling_cluster_count(Sequence[HfRecord] records, Mapping[str, str] cluster_of)
# Inputs: Sequence[HfRecord] records - the records that were clustered (must all have entries in
#         cluster_of)
#         Mapping[str, str] cluster_of - uid -> cluster_id, from cluster_by_phash
# Outputs: tuple[int, dict[str, set[str]]] - (straddle_count, orig_splits_by_cluster):
#          straddle_count is how many clusters contain records from more than one of HF's own
#          original splits; orig_splits_by_cluster maps cluster_id -> the set of orig_split
#          values seen in it
# Description: This is the headline leakage-audit number: it directly measures whether HF's own
#              train/validation/test boundary keeps near-duplicate scenes together (0 straddling
#              clusters) or splits them apart (many straddling clusters), which is exactly the
#              failure mode this whole re-split exists to fix.
# Side Effects: None
def straddling_cluster_count(records: Sequence[HfRecord], cluster_of: Mapping[str, str]) -> tuple[int, dict[str, set[str]]]:
    splits_by_cluster: dict[str, set[str]] = {}
    for r in records:
        splits_by_cluster.setdefault(cluster_of[r.uid], set()).add(r.orig_split)
    straddled = sum(1 for splits in splits_by_cluster.values() if len(splits) > 1)
    return straddled, splits_by_cluster


# dict[str, set[int]] find_mixed_label_clusters(Sequence[HfRecord] records, Mapping[str, str] cluster_of)
# Inputs: Sequence[HfRecord] records - the records that were clustered
#         Mapping[str, str] cluster_of - uid -> cluster_id, from cluster_by_phash
# Outputs: dict[str, set[int]] - cluster_id -> the set of distinct labels seen in it, for every
#          cluster containing MORE THAN ONE label; clusters with a single label are omitted
# Description: Flags every cluster whose member images disagree on label. A single physical
#              scene cannot be both a normal and a failed print, so a mixed cluster is a
#              data-quality signal (near-duplicate frames straddling the moment of failure, or a
#              mislabeled row) -- see the module docstring's step 6. Callers exclude these
#              clusters entirely rather than guessing a label for them.
# Side Effects: None
def find_mixed_label_clusters(records: Sequence[HfRecord], cluster_of: Mapping[str, str]) -> dict[str, set[int]]:
    labels_by_cluster: dict[str, set[int]] = {}
    for r in records:
        labels_by_cluster.setdefault(cluster_of[r.uid], set()).add(r.label)
    return {cid: labels for cid, labels in labels_by_cluster.items() if len(labels) > 1}


# dict[str, float] summarize_cluster_sizes(Sequence[int] sizes)
# Inputs: Sequence[int] sizes - one entry per cluster: how many images it contains
# Outputs: dict[str, float] - {"count", "min", "max", "mean", "median"} over sizes; all zero if
#          sizes is empty
# Description: Summarizes the cluster-size distribution for the split report, so a reader can
#              tell at a glance whether "independent scenes" are mostly singletons or dominated
#              by a few huge timelapse bursts.
# Side Effects: None
def summarize_cluster_sizes(sizes: Sequence[int]) -> dict[str, float]:
    if not sizes:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    ordered = sorted(sizes)
    n = len(ordered)
    mean = sum(ordered) / n
    median = ordered[n // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"count": n, "min": ordered[0], "max": ordered[-1], "mean": round(mean, 3), "median": median}


# --------------------------------------------------------------------------
# Cluster-level split assignment (pure; no I/O)
# --------------------------------------------------------------------------


# dict[str, str] assign_clusters_to_splits(Mapping[int, Sequence[tuple[str, int]]] clusters_by_label, int seed, tuple[float, float, float] ratios)
# Inputs: Mapping[int, Sequence[tuple[str, int]]] clusters_by_label - label -> list of
#         (cluster_id, cluster_size) for that label's clusters
#         int seed - RNG seed for deterministic tie-break shuffling
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15); must sum to 1.0
# Outputs: dict[str, str] - cluster_id -> split_name ("train"/"val"/"test") for every cluster in
#          clusters_by_label
# Description: Splits clusters (never images) into train/val/test, independently PER LABEL so
#              every split ends up with both classes (stratification) -- clusters_by_label
#              already partitions by label, so this just runs the same greedy packer on each
#              partition. Within one label, clusters are processed in a seed-shuffled order and
#              each is assigned to whichever split currently has the largest deficit versus its
#              ideal share of the running image total (a standard deterministic proportional
#              allocator) -- clusters are indivisible and vary in size, so this gets close to the
#              target ratios by IMAGE count rather than guaranteeing them exactly.
# Side Effects: Raises ValueError if ratios don't sum to 1.0 (within 1e-6). Uses a locally-seeded
#               random.Random per label; does not touch global RNG state.
def assign_clusters_to_splits(
    clusters_by_label: Mapping[int, Sequence[tuple[str, int]]],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, str]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios!r} (sum={sum(ratios)})")

    ratio_by_split = dict(zip(SPLIT_NAMES, ratios))
    assignment: dict[str, str] = {}
    for label in sorted(clusters_by_label):
        items = sorted(clusters_by_label[label])
        random.Random(seed + label).shuffle(items)

        assigned = {s: 0 for s in SPLIT_NAMES}
        running_total = 0
        for cluster_id, size in items:
            running_total += size
            best_split = max(SPLIT_NAMES, key=lambda s: ratio_by_split[s] * running_total - assigned[s])
            assignment[cluster_id] = best_split
            assigned[best_split] += size
    return assignment


# set[str] cap_clusters_per_class(Mapping[str, int] cluster_label, Mapping[str, int] cluster_size, int cap, int seed)
# Inputs: Mapping[str, int] cluster_label - cluster_id -> its (single) label
#         Mapping[str, int] cluster_size - cluster_id -> how many images it contains
#         int cap - upper bound on images kept per class
#         int seed - RNG seed for deterministic tie-break shuffling
# Outputs: set[str] - the cluster_ids to KEEP; a class's kept clusters never total more than cap
#          images (whole clusters only -- one may be left out entirely rather than split)
# Description: Deterministically caps each class's total image count at cap, without ever
#              breaking a cluster apart: clusters are seed-shuffled (per label, independent RNG
#              stream from assign_clusters_to_splits) and greedily added while they still fit
#              under cap; a cluster that would push the running total over cap is left out
#              entirely rather than truncated.
# Side Effects: Uses a locally-seeded random.Random per label; does not touch global RNG state.
def cap_clusters_per_class(
    cluster_label: Mapping[str, int],
    cluster_size: Mapping[str, int],
    cap: int,
    seed: int,
) -> set[str]:
    by_label: dict[int, list[str]] = {}
    for cid, label in cluster_label.items():
        by_label.setdefault(label, []).append(cid)

    kept: set[str] = set()
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        random.Random(seed + 1000 + label).shuffle(ids)
        total = 0
        for cid in ids:
            size = cluster_size[cid]
            if total + size > cap:
                continue
            kept.add(cid)
            total += size
    return kept


# None assert_no_cluster_overlap(Mapping[str, Sequence[str]] split_cluster_ids)
# Inputs: Mapping[str, Sequence[str]] split_cluster_ids - split name -> cluster ids assigned to it
# Outputs: None
# Description: Verifies no cluster identity (independent scene) appears in more than one split.
#              This is the explicit check for the whole point of splitting by cluster instead of
#              by image, rather than just trusting assign_clusters_to_splits.
# Side Effects: Raises AssertionError naming the offending cluster and both splits it appears in,
#               if any cluster-identity leakage is detected. No filesystem or RNG activity.
def assert_no_cluster_overlap(split_cluster_ids: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for split_name, ids in split_cluster_ids.items():
        for cid in ids:
            if cid in seen:
                raise AssertionError(
                    f"Cluster-identity leakage detected: cluster {cid!r} appears in both "
                    f"'{seen[cid]}' and '{split_name}' splits."
                )
            seen[cid] = split_name


# --------------------------------------------------------------------------
# Orchestration: pure planning (no I/O), separate from materialization
# --------------------------------------------------------------------------


@dataclass
class BinarySplitPlan:
    """The full leak-free split plan, plus every statistic needed for the
    audit report -- computed entirely from record metadata, never from
    decoded pixels (see plan_binary_split)."""

    records_by_split: dict[str, list[HfRecord]]
    total_rows_pooled: int
    exact_duplicates_dropped: int
    total_clusters: int  # independent scenes: ALL clusters post-dedup, pre label-mix exclusion, pre cap
    cluster_sizes: list[int]  # sizes of every one of those clusters
    straddling_clusters: int
    mixed_label_cluster_count: int
    mixed_label_images_excluded: int
    capped_out_cluster_count: int
    clusters_per_split_by_class: dict[str, dict[str, int]]  # only over clusters actually kept+split


# BinarySplitPlan plan_binary_split(Sequence[HfRecord] records, int seed, int phash_distance, tuple[float, float, float] ratios, int | None per_class_cap)
# Inputs: Sequence[HfRecord] records - every pooled row (all HF splits combined), each already
#         carrying its content_hash/phash
#         int seed - RNG seed for deterministic sampling/splitting, default DEFAULT_SEED (1337)
#         int phash_distance - near-duplicate Hamming threshold, default DEFAULT_PHASH_DISTANCE
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15)
#         int | None per_class_cap - upper bound on images kept per class, or None for no cap
# Outputs: BinarySplitPlan - the full leak-free plan and every audit statistic
# Description: The pure core of this whole script: dedupes exact duplicates, clusters
#              near-duplicates into independent scenes, measures how many scenes straddle HF's
#              own original split (BEFORE any exclusion, since that's a property of the source
#              data, not of our choices), excludes any scene that mixes labels, optionally caps
#              each class's total image count (whole clusters only), then splits the remaining
#              clusters 70/15/15 stratified by label. NEVER calls a record's loader, so it is
#              callable purely in-memory with no network or disk access -- this is what makes the
#              whole splitting algorithm unit-testable.
# Side Effects: Uses locally-seeded RNG throughout (via cluster_by_phash's determinism and
#               assign_clusters_to_splits/cap_clusters_per_class); does not touch global RNG
#               state. Raises AssertionError if cluster overlap is somehow detected. No I/O.
def plan_binary_split(
    records: Sequence[HfRecord],
    seed: int = DEFAULT_SEED,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    per_class_cap: int | None = None,
) -> BinarySplitPlan:
    total_rows_pooled = len(records)
    deduped, dropped = dedupe_exact_duplicates(records)
    cluster_of = cluster_by_phash(deduped, phash_distance)

    # Independent-scene bookkeeping over EVERY post-dedup cluster, before any
    # label-purity exclusion or capping -- the headline "how many independent
    # scenes exist at all" number (module docstring step 4 depends on this
    # being computed pre-exclusion).
    cluster_size_all = Counter(cluster_of.values())
    total_clusters = len(cluster_size_all)
    cluster_sizes = sorted(cluster_size_all.values())

    straddle_count, _ = straddling_cluster_count(deduped, cluster_of)

    mixed = find_mixed_label_clusters(deduped, cluster_of)
    mixed_images_excluded = sum(cluster_size_all[cid] for cid in mixed)

    clean = [r for r in deduped if cluster_of[r.uid] not in mixed]

    cluster_label: dict[str, int] = {cluster_of[r.uid]: r.label for r in clean}
    cluster_size_clean = Counter(cluster_of[r.uid] for r in clean)

    kept_cluster_ids = set(cluster_label)
    capped_out_count = 0
    if per_class_cap is not None:
        kept_cluster_ids = cap_clusters_per_class(cluster_label, cluster_size_clean, per_class_cap, seed)
        capped_out_count = len(cluster_label) - len(kept_cluster_ids)

    clusters_by_label: dict[int, list[tuple[str, int]]] = {}
    for cid in kept_cluster_ids:
        clusters_by_label.setdefault(cluster_label[cid], []).append((cid, cluster_size_clean[cid]))

    split_of_cluster = assign_clusters_to_splits(clusters_by_label, seed, ratios)
    assert_no_cluster_overlap({s: [cid for cid, sp in split_of_cluster.items() if sp == s] for s in SPLIT_NAMES})

    records_by_split: dict[str, list[HfRecord]] = {s: [] for s in SPLIT_NAMES}
    for r in clean:
        cid = cluster_of[r.uid]
        if cid not in kept_cluster_ids:
            continue
        records_by_split[split_of_cluster[cid]].append(r)

    clusters_per_split_by_class: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLIT_NAMES}
    for cid, split_name in split_of_cluster.items():
        cname = class_name_for_label(cluster_label[cid])
        clusters_per_split_by_class[split_name][cname] += 1

    return BinarySplitPlan(
        records_by_split=records_by_split,
        total_rows_pooled=total_rows_pooled,
        exact_duplicates_dropped=dropped,
        total_clusters=total_clusters,
        cluster_sizes=cluster_sizes,
        straddling_clusters=straddle_count,
        mixed_label_cluster_count=len(mixed),
        mixed_label_images_excluded=mixed_images_excluded,
        capped_out_cluster_count=capped_out_count,
        clusters_per_split_by_class=clusters_per_split_by_class,
    )


# --------------------------------------------------------------------------
# Materialization + report
# --------------------------------------------------------------------------


# dict[str, dict[str, int]] materialize_split(Mapping[str, list[HfRecord]] records_by_split, Path out_dir)
# Inputs: Mapping[str, list[HfRecord]] records_by_split - split name -> records assigned to it,
#         from a BinarySplitPlan
#         Path out_dir - output dataset root (e.g. datasets/argus_bin)
# Outputs: dict[str, dict[str, int]] - split -> class_name -> number of images actually written
# Description: Writes every record to out_dir/<split>/<class_name>/<output_stem>.jpg, normalized
#              via save_normalized_jpeg (reused from training.build_classification_dataset, so
#              training-time preprocessing is byte-for-byte identical to
#              argus.detectors.classifier.preprocess_classify).
# Side Effects: Creates out_dir/<split>/<class_name> directories; calls each record's loader
#               (decoding its image from the original bytes) and writes a normalized JPEG to disk
#               for every record.
def materialize_split(records_by_split: Mapping[str, list[HfRecord]], out_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASS_NAMES} for s in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        for r in records_by_split.get(split_name, []):
            cname = class_name_for_label(r.label)
            dest = out_dir / split_name / cname / f"{r.output_stem}.jpg"
            image = r.loader()
            save_normalized_jpeg(image, dest, quality=JPEG_QUALITY)
            counts[split_name][cname] += 1
    return counts


# str build_evaluation_validity_warning(int total_clusters, Mapping[str, Mapping[str, int]] clusters_per_split_by_class, Sequence[str] class_names, int min_scenes, int min_per_split_class)
# Inputs: int total_clusters - total independent scenes in the whole deduplicated pool
#         Mapping[str, Mapping[str, int]] clusters_per_split_by_class - split -> class_name ->
#         cluster count, from a BinarySplitPlan
#         Sequence[str] class_names - output class names, default CLASS_NAMES
#         int min_scenes - minimum total scenes considered trustworthy, default
#         MIN_INDEPENDENT_SCENES_FOR_TRUST (40)
#         int min_per_split_class - minimum scenes per class per split considered trustworthy,
#         default MIN_CLUSTERS_PER_SPLIT_CLASS (5)
# Outputs: str - a plain-language warning; loudly worded if either threshold is breached,
#          otherwise a milder note that still names the scene count
# Description: The "critical honesty gate": val/test accuracy on this dataset is only as
#              meaningful as the number of INDEPENDENT SCENES behind it, not the (much larger)
#              raw image count, since many images are near-duplicate frames of the same scene.
#              This makes that distinction explicit and impossible to miss when the scene count
#              is thin.
# Side Effects: None
def build_evaluation_validity_warning(
    total_clusters: int,
    clusters_per_split_by_class: Mapping[str, Mapping[str, int]],
    class_names: Sequence[str] = CLASS_NAMES,
    min_scenes: int = MIN_INDEPENDENT_SCENES_FOR_TRUST,
    min_per_split_class: int = MIN_CLUSTERS_PER_SPLIT_CLASS,
) -> str:
    reasons: list[str] = []
    if total_clusters < min_scenes:
        reasons.append(
            f"only {total_clusters} independent scene(s) (near-duplicate clusters) exist in the "
            f"ENTIRE deduplicated pool, below the {min_scenes} minimum"
        )
    for split in SPLIT_NAMES:
        for cname in class_names:
            n = clusters_per_split_by_class.get(split, {}).get(cname, 0)
            if n < min_per_split_class:
                reasons.append(f"split '{split}' has only {n} '{cname}' cluster(s) (< {min_per_split_class})")

    if reasons:
        return (
            "EVALUATION VALIDITY WARNING: " + "; ".join(reasons) + ". "
            "Reported val/test accuracy on this dataset is computed over a HANDFUL of independent "
            "print scenes, not the much larger raw-image count (many images are near-duplicate "
            "frames of the same scene) -- treat any accuracy/precision/recall number here as "
            "anecdotal, NOT as an estimate of how the model will generalize to new prints."
        )
    return (
        f"Evaluation validity: {total_clusters} independent scenes total, clearing the minimum "
        f"bar of {min_scenes}+ total and {min_per_split_class}+ per class per split -- still a "
        "modest sample, so treat held-out metrics as a useful signal rather than a tight "
        "generalization guarantee."
    )


# dict[str, object] build_dataset(Path out_dir, Sequence[HfRecord] records, int seed, int phash_distance, tuple[float, float, float] ratios, int | None per_class_cap)
# Inputs: Path out_dir - output dataset root (e.g. datasets/argus_bin); keyword-only
#         Sequence[HfRecord] records - every pooled row (all HF splits combined); keyword-only
#         int seed - RNG seed for deterministic sampling/splitting, default DEFAULT_SEED (1337);
#         keyword-only
#         int phash_distance - near-duplicate Hamming threshold, default DEFAULT_PHASH_DISTANCE;
#         keyword-only
#         tuple[float, float, float] ratios - (train, val, test) fractions, default
#         DEFAULT_SPLIT_RATIOS (0.70, 0.15, 0.15); keyword-only
#         int | None per_class_cap - upper bound on images kept per class, or None for no cap;
#         keyword-only
# Outputs: dict[str, object] - the full split report: seed/cap/ratio/output settings, per-split
#          per-class image counts, exact-duplicate and mixed-label-cluster stats, the independent
#          scene audit (count, size distribution, straddle count), and the evaluation-validity
#          warning
# Description: Combines plan_binary_split (pure planning) with materialize_split (the only I/O
#              in this function besides directory creation) and assembles the full JSON-ready
#              report. Kept separate from main() so tests can call this directly with small
#              synthetic in-memory record pools -- no network, no real dataset required.
# Side Effects: Creates out_dir and its per-split/per-class subdirectories; writes normalized
#               JPEG files to disk for every kept record (via materialize_split); uses
#               locally-seeded RNG throughout (via plan_binary_split); does not touch global RNG
#               state.
def build_dataset(
    *,
    out_dir: Path,
    records: Sequence[HfRecord],
    seed: int = DEFAULT_SEED,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    per_class_cap: int | None = None,
) -> dict[str, object]:
    plan = plan_binary_split(records, seed=seed, phash_distance=phash_distance, ratios=ratios, per_class_cap=per_class_cap)
    written_counts = materialize_split(plan.records_by_split, out_dir)

    total_images_kept = sum(sum(v.values()) for v in written_counts.values())
    warning = build_evaluation_validity_warning(plan.total_clusters, plan.clusters_per_split_by_class, CLASS_NAMES)

    report: dict[str, object] = {
        "seed": seed,
        "phash_distance": phash_distance,
        "per_class_cap": per_class_cap,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "output_size": OUTPUT_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "class_names": list(CLASS_NAMES),
        "hf_dataset_id": HF_DATASET_ID,
        "counts": written_counts,
        "total_images_kept": total_images_kept,
        "total_rows_pooled": plan.total_rows_pooled,
        "exact_duplicates_dropped": plan.exact_duplicates_dropped,
        "mixed_label_clusters_excluded": {
            "cluster_count": plan.mixed_label_cluster_count,
            "images_excluded": plan.mixed_label_images_excluded,
        },
        "capped_out_cluster_count": plan.capped_out_cluster_count,
        "total_independent_scenes": plan.total_clusters,
        "largest_cluster_size": max(plan.cluster_sizes) if plan.cluster_sizes else 0,
        "cluster_size_distribution": summarize_cluster_sizes(plan.cluster_sizes),
        "clusters_per_split_by_class": plan.clusters_per_split_by_class,
        "clusters_straddling_hf_original_split": plan.straddling_clusters,
        "evaluation_validity_warning": warning,
    }
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed options (out, cache_dir, seed, phash_distance,
#          per_class_cap, force). Notable defaults: --seed DEFAULT_SEED (1337), --phash-distance
#          DEFAULT_PHASH_DISTANCE (matches training.ingest_roboflow_multi's default), no cap by
#          default.
# Description: Defines and parses the CLI for building the binary classification dataset.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"Output dataset dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"Cache dir for downloaded HF parquet files (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed for deterministic sampling/splitting (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=DEFAULT_PHASH_DISTANCE,
        help=f"dhash Hamming distance at or below which two images are one independent scene (default: {DEFAULT_PHASH_DISTANCE})",
    )
    parser.add_argument("--per-class-cap", type=int, default=None, help="Upper bound on images kept per class, applied whole-cluster (default: no cap)")
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild --out if it already exists")
    return parser.parse_args(argv)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Downloads and reads the Hugging Face source's parquet files
#              across all three of its own splits, decodes every row once to fingerprint it,
#              then dedupes/clusters/audits/splits/normalizes/writes the binary dataset via
#              build_dataset and prints the full split report (including the leakage-audit
#              headline numbers and the evaluation-validity warning).
# Side Effects: Makes HTTP requests to huggingface.co and downloads parquet files to
#               --cache-dir (unless already cached, see download_hf_split_parquets); decodes
#               every source image once during gathering; optionally wipes --out with
#               shutil.rmtree when --force is passed; creates --out and its class/split
#               subdirectories; writes normalized JPEG images to disk for every kept record;
#               writes --out/split_report.json; prints extensive progress and a full summary
#               report to stdout.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out

    print(f"[build_bin] Fetching Hugging Face '{HF_DATASET_ID}' parquet files into '{args.cache_dir}' ...")
    parquet_paths = download_hf_split_parquets(args.cache_dir)
    for split, path in sorted(parquet_paths.items()):
        print(f"  {split}: {path}")

    print("[build_bin] Decoding rows and computing content/perceptual hashes (reads every image once) ...")
    records = gather_hf_records(parquet_paths)
    print(f"  pooled {len(records)} rows across {len(parquet_paths)} HF splits")
    label_counts = Counter(r.label for r in records)
    for label in sorted(label_counts):
        print(f"    label {label} ({class_name_for_label(label)}): {label_counts[label]}")

    if out_dir.exists():
        if args.force:
            print(f"[build_bin] --force: removing existing '{out_dir}' ...")
            shutil.rmtree(out_dir)
        else:
            print(f"[build_bin] '{out_dir}' already exists. Pass --force to rebuild it. Proceeding to (over)write into it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_bin] Deduplicating, clustering near-duplicates, auditing HF's own split, and re-splitting by scene ...")
    report = build_dataset(
        out_dir=out_dir,
        records=records,
        seed=args.seed,
        phash_distance=args.phash_distance,
        per_class_cap=args.per_class_cap,
    )

    report_path = out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("BINARY DATASET BUILD -- SPLIT REPORT")
    print("=" * 78)
    print(f"Seed: {args.seed}   phash distance: {args.phash_distance}   per-class cap: {args.per_class_cap}")
    print(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}   JPEG quality: {JPEG_QUALITY}   Ratios: {DEFAULT_SPLIT_RATIOS}")
    print()
    header = f"{'split':<10}{'failure':>10}{'normal':>10}{'total':>10}"
    print(header)
    print("-" * len(header))
    for split in SPLIT_NAMES:
        counts = report["counts"][split]  # type: ignore[index]
        total = sum(counts.values())
        print(f"{split:<10}{counts['failure']:>10}{counts['normal']:>10}{total:>10}")
    print()
    print(f"Total images kept: {report['total_images_kept']}   Total rows pooled: {report['total_rows_pooled']}")
    print(f"Exact duplicates dropped: {report['exact_duplicates_dropped']}")
    mlc = report["mixed_label_clusters_excluded"]  # type: ignore[index]
    print(f"Mixed-label clusters excluded: {mlc['cluster_count']} clusters ({mlc['images_excluded']} images)")
    print(f"Total independent scenes (clusters): {report['total_independent_scenes']}")
    print(f"Largest cluster size: {report['largest_cluster_size']}   Cluster-size distribution: {report['cluster_size_distribution']}")
    print(f"Clusters per split per class: {report['clusters_per_split_by_class']}")
    print()
    print("!" * 78)
    print(f"CLUSTERS STRADDLING HUGGING FACE'S OWN ORIGINAL TRAIN/TEST BOUNDARY: {report['clusters_straddling_hf_original_split']}")
    print("!" * 78)
    print()
    print(report["evaluation_validity_warning"])
    print()
    print(f"Full report written to: {report_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
