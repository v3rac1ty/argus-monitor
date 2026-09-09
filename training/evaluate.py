"""Evaluates a trained checkpoint on the held-out TEST split (untouched by training/threshold selection, unlike the original archive's leaky train/valid split). Per class: precision/recall/AP50 at max-F1, a confidence-threshold sweep, and the lowest threshold reaching ``--target-precision`` (default 0.95).

PER-SOURCE-PHOTO DEDUPLICATION (on by default, ``--no-dedup-sources`` to disable). The test images are NOT independent samples: they are Roboflow augmentation variants of a much smaller pool of source photographs, and the variant counts are wildly uneven (one photo can carry 30+ variants while the median carries 3). The split is group-aware so there is no train/test leakage, but a raw per-image mAP is a *weighted* average in which a heavily-augmented photo casts 30 votes and a singleton casts 1 -- i.e. it measures whichever handful of photographs Roboflow happened to augment most, not per-photo performance. This module therefore groups test images by their ``.rf.<hash>``-stripped stem (``prepare_dataset.extract_source_id``, the same source-photo identity the splitter used), evaluates ONE deterministically-chosen representative per source photo, and reports that as the headline number. The raw per-image number is still computed and printed beside it: the gap between the two is a real diagnostic (a large gap means the raw score was being carried by augmentation-heavy photos) and is never hidden.

EFFECTIVE SAMPLE SIZE. Per-class support is reported as the number of unique SOURCE PHOTOS containing the class, not just the instance count -- an AP resting on 12 photographs is not the same measurement as one resting on 400, even if both show thousands of "instances". Where ``manifest.jsonl`` is available, the ingest's own independent-scene group count is reported beside it: a group merges near-duplicate and timelapse siblings too, so it is a tighter bound still, and several photographs of one print job stop counting as several independent observations.

SEVERITY. ``spaghetti``/``layer_separation``/``bed_adhesion``/``blob_of_death`` are CATASTROPHIC -- the only classes allowed to stop a print, so their precision at the chosen threshold is the real false-positive rate. ``warping``/``stringing``/``error_extrusion`` are COSMETIC: logged, never acted on.

``--min-recall`` rejects Ultralytics' precision=1.0-at-zero-predictions convention: a class that never fires can't be a false positive but also can't catch a real failure, so that's not a usable operating point.

SOURCE-CONFOUND DIAGNOSTIC. The dataset merges two Roboflow exports (``atco``, ``stereovision``) copied byte-for-byte, so resolution/framing/JPEG statistics still identify the origin of every image. Recall is therefore also reported split by source dataset -- a large gap means the model has partly learned "which dataset is this" rather than "what does a failure look like". Provenance comes from ``manifest.jsonl`` (written by ``training/ingest_roboflow_multi.py --source-tag``), with the ``<source>__`` filename prefix as fallback.

Usage: python training/evaluate.py --weights runs/train/argus_multi_yolo26n/weights/best.pt [--target-precision 0.95] [--no-dedup-sources]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# Run-from-anywhere bootstrap: this module reuses training/prepare_dataset.py, which is
# only importable once the repo root is on sys.path (pytest already does this; running
# `python training/evaluate.py` does not).
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.prepare_dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    extract_source_id,
    group_by_source,
    parse_yolo_label_classes,
)

DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "argus_multi" / "data.yaml"
DEFAULT_OUT_PATH = REPO_ROOT / "runs" / "evaluation.json"

#: Only these may drive an automated pause, so their precision IS the false-positive
#: rate the system exposes to a running print. Everything else is logged, never acted on.
CATASTROPHIC_CLASSES = ("spaghetti", "layer_separation", "bed_adhesion", "blob_of_death")
COSMETIC_CLASSES = ("warping", "stringing", "error_extrusion")

#: A class whose AP rests on fewer unique source photographs than this is flagged LOW-N:
#: the point estimate is dominated by a handful of scenes regardless of instance count.
LOW_SUPPORT_PHOTOS = 50

#: Relative mAP50 gap between the deduped and raw-per-image numbers that gets a loud
#: callout rather than a quiet line (0.10 = the raw number is 10% off the honest one).
MAP_GAP_RELATIVE_WARN = 0.10

#: Per-source recall gap above which the source-confound diagnostic escalates: the model
#: is measurably better on one origin dataset than the other.
SOURCE_RECALL_GAP_WARN = 0.15

#: Fewest images a per-source subset needs before its recall is worth reporting at all.
MIN_SOURCE_SUBSET_IMAGES = 20

REPRESENTATIVE_STRATEGIES = ("seeded", "first")


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed evaluation options (weights, data, manifest, imgsz,
#          batch, device, nms_iou, target_precision, min_recall, sweep_start/end/step,
#          dedup_sources, representative, seed, out). Notable defaults:
#          --target-precision 0.95, --min-recall 0.05 (the vacuous-precision floor: rejects
#          thresholds that only "achieve" target precision via Ultralytics' precision=1.0
#          convention for zero surviving predictions), --nms-iou 0.7, per-source-photo
#          dedup ON (--no-dedup-sources disables), --representative seeded, --seed 1337.
# Description: Defines and parses the CLI for evaluating a detection checkpoint on the test split.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help=f"Path to data.yaml (default: {DEFAULT_DATA_YAML})")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the ingest manifest.jsonl carrying each image's source dataset, for the "
        "source-confound diagnostic (default: manifest.jsonl next to --data). Falls back to the "
        "'<source>__' output-filename prefix when the manifest is absent.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold used for NMS during val (default: 0.7, Ultralytics default)")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.05,
        help=(
            "A candidate threshold must also reach at least this much recall to count as "
            "'reaching' --target-precision (default: 0.05). Ultralytics' precision curve reports "
            "precision=1.0 by convention wherever a class has zero surviving predictions (0/0), "
            "which is a statistically meaningless 'perfect' score, not a usable operating point -- "
            "a detector that never fires offers no protection at all. This floor rejects that "
            "vacuous case so the recommended threshold is backed by real detections."
        ),
    )
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-end", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument(
        "--no-dedup-sources",
        dest="dedup_sources",
        action="store_false",
        default=True,
        help=(
            "Disable per-source-photo deduplication and make the RAW per-image numbers the "
            "headline. Off by default because the test images are Roboflow augmentation variants "
            "of a much smaller photo pool with wildly uneven variant counts, so a per-image mean "
            "silently weights each source photograph by how many variants it happened to receive."
        ),
    )
    parser.add_argument(
        "--representative",
        type=str,
        choices=REPRESENTATIVE_STRATEGIES,
        default="seeded",
        help="How to pick the one evaluated image per source photo: 'seeded' (seeded random "
        "choice among that photo's variants, default) or 'first' (lexicographically first "
        "variant). Both are deterministic and independent of filesystem iteration order.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Seed for the 'seeded' representative choice (default: 1337)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Where to write the full JSON report (default: {DEFAULT_OUT_PATH})")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Pure logic -- per-source-photo dedup, support counting, provenance.
# Unit-tested on synthetic filename lists; no model, no GPU, no dataset on disk.
# --------------------------------------------------------------------------


# tuple[list[str], dict[str, list[str]], int] select_representatives(Iterable[str] filenames, int seed, str strategy)
# Inputs: Iterable[str] filenames - test-split image filenames (basenames, as emitted by
#         ingest_roboflow_multi.py: "<source>__<stem>.rf.<hash>.<ext>")
#         int seed - seed for the "seeded" strategy, e.g. the CLI's default 1337
#         str strategy - "seeded" (seeded random choice per source photo) or "first"
#         (lexicographically first variant); both are deterministic
# Outputs: tuple[list[str], dict[str, list[str]], int] - (representatives, groups,
#          no_rf_suffix_count): representatives is one sorted filename per source photo;
#          groups maps source-photo id -> its sorted variant filenames; no_rf_suffix_count is
#          how many filenames carried no ".rf.<hash>" suffix (each of those is its own group)
# Description: The core of the evaluation-weighting fix. Groups the test images by source-photo
#              identity (Roboflow's ".rf.<hash>" stripped -- the SAME key the splitter grouped
#              by) and picks exactly one image per photograph, so every source photograph
#              contributes one vote to precision/recall/AP instead of one vote per augmentation
#              variant. Selection is seeded per group id, never "whichever the filesystem
#              yielded first", so the chosen subset is reproducible across machines and runs.
# Side Effects: Raises ValueError if strategy is not one of REPRESENTATIVE_STRATEGIES. No I/O.
def select_representatives(
    filenames: Iterable[str], seed: int = 1337, strategy: str = "seeded"
) -> tuple[list[str], dict[str, list[str]], int]:
    if strategy not in REPRESENTATIVE_STRATEGIES:
        raise ValueError(f"strategy must be one of {REPRESENTATIVE_STRATEGIES}, got {strategy!r}")
    groups, no_rf_suffix_count = group_by_source(filenames)
    representatives: list[str] = []
    for source_id in sorted(groups):
        variants = groups[source_id]  # already sorted by group_by_source
        if strategy == "first":
            representatives.append(variants[0])
        else:
            # Seeded per GROUP, not per run: the pick depends only on (seed, source id,
            # that photo's variant names), so it is stable regardless of iteration order,
            # of which other photos are present, or of how many files the OS listed first.
            representatives.append(random.Random(f"{seed}:{source_id}").choice(variants))
    return sorted(representatives), groups, no_rf_suffix_count


# dict[str, float | int] variant_count_stats(Mapping[str, Sequence[str]] groups)
# Inputs: Mapping[str, Sequence[str]] groups - source-photo id -> its variant filenames, from
#         select_representatives
# Outputs: dict[str, float | int] - {"num_images", "num_source_photos", "min", "median",
#          "mean", "max", "p90", "num_photos_over_10_variants"}; all zeros/None-free for an
#          empty input (num_source_photos 0, the rest 0)
# Description: Summarizes how unevenly the augmentation variants are distributed over source
#              photographs. This spread IS the size of the measurement bug: if every photo had
#              the same variant count, per-image and per-photo means would agree.
# Side Effects: None
def variant_count_stats(groups: Mapping[str, Sequence[str]]) -> dict[str, float | int]:
    counts = sorted(len(v) for v in groups.values())
    if not counts:
        return {
            "num_images": 0,
            "num_source_photos": 0,
            "min": 0,
            "median": 0,
            "mean": 0.0,
            "max": 0,
            "p90": 0,
            "num_photos_over_10_variants": 0,
        }
    return {
        "num_images": int(sum(counts)),
        "num_source_photos": len(counts),
        "min": int(counts[0]),
        "median": float(statistics.median(counts)),
        "mean": float(sum(counts) / len(counts)),
        "max": int(counts[-1]),
        "p90": int(counts[min(len(counts) - 1, int(round(0.9 * (len(counts) - 1))))]),
        "num_photos_over_10_variants": int(sum(1 for c in counts if c > 10)),
    }


# dict[str, dict[str, int]] class_support(Mapping[str, Sequence[int]] labels_by_image, Sequence[str] class_names)
# Inputs: Mapping[str, Sequence[int]] labels_by_image - image filename -> the class indices in
#         that image's YOLO label file (empty list for a background/negative image)
#         Sequence[str] class_names - class names in data.yaml order; index i names class i
# Outputs: dict[str, dict[str, int]] - class name -> {"instances", "images", "source_photos"}:
#          total boxes, images containing at least one, and UNIQUE SOURCE PHOTOGRAPHS
#          containing at least one (the effective sample size)
# Description: Computes per-class effective sample size. "instances" and "images" both count
#              augmentation variants of the same photograph repeatedly; "source_photos" is the
#              number of genuinely independent scenes the class's AP actually rests on, which
#              is the number that decides whether a precision figure means anything.
# Side Effects: None (pure counting; unknown class indices are ignored rather than raising, so
#               a stale label file can't abort the whole evaluation)
def class_support(
    labels_by_image: Mapping[str, Sequence[int]], class_names: Sequence[str]
) -> dict[str, dict[str, int]]:
    instances = {c: 0 for c in class_names}
    images = {c: 0 for c in class_names}
    photos: dict[str, set[str]] = {c: set() for c in class_names}
    for filename, class_ids in labels_by_image.items():
        source_id, _ = extract_source_id(filename)
        seen: set[str] = set()
        for cid in class_ids:
            if not 0 <= int(cid) < len(class_names):
                continue
            cname = class_names[int(cid)]
            instances[cname] += 1
            seen.add(cname)
        for cname in seen:
            images[cname] += 1
            photos[cname].add(source_id)
    return {
        c: {"instances": instances[c], "images": images[c], "source_photos": len(photos[c])}
        for c in class_names
    }


# dict[str, int] class_group_support(Mapping[str, Sequence[int]] labels_by_image, Sequence[str] class_names, Mapping[str, str] group_of_image)
# Inputs: Mapping[str, Sequence[int]] labels_by_image - image filename -> class indices in its
#         label file
#         Sequence[str] class_names - class names in data.yaml order
#         Mapping[str, str] group_of_image - image filename -> leakage-group id, from
#         load_manifest_groups; images missing from the mapping are skipped
# Outputs: dict[str, int] - class name -> number of distinct leakage groups containing at least
#          one instance of it
# Description: Counts effective sample size at the STRICTER unit the ingest actually used to
#              guarantee split independence. A leakage group merges not just one photo's
#              augmentation variants but also its perceptual near-duplicates and timelapse
#              siblings, so it is a lower (more honest) bound on independent scenes than the
#              source-photo count: a class covering 85 photographs of the same few print jobs is
#              really covering only those jobs. Reported alongside, never instead of, the
#              source-photo count.
# Side Effects: None
def class_group_support(
    labels_by_image: Mapping[str, Sequence[int]],
    class_names: Sequence[str],
    group_of_image: Mapping[str, str],
) -> dict[str, int]:
    groups: dict[str, set[str]] = {c: set() for c in class_names}
    for filename, class_ids in labels_by_image.items():
        group = group_of_image.get(filename)
        if group is None:
            continue
        for cid in class_ids:
            if 0 <= int(cid) < len(class_names):
                groups[class_names[int(cid)]].add(group)
    return {c: len(groups[c]) for c in class_names}


# dict[str, str] resolve_image_sources(Iterable[str] filenames, Mapping[str, str] manifest_sources)
# Inputs: Iterable[str] filenames - test-split image basenames
#         Mapping[str, str] manifest_sources - basename -> source dataset key, from
#         load_manifest_sources (may be empty if no manifest was found)
# Outputs: dict[str, str] - filename -> source key ("atco", "stereovision", ... or "unknown")
# Description: Recovers each test image's origin dataset for the source-confound diagnostic.
#              The manifest written by ingest_roboflow_multi.py is authoritative; the
#              "<source>__" output-filename prefix that the same script writes under
#              --source-tag filename/both is the fallback, so the diagnostic still works on a
#              dataset copied without its manifest. Never guesses from image content.
# Side Effects: None
def resolve_image_sources(filenames: Iterable[str], manifest_sources: Mapping[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for fn in filenames:
        source = manifest_sources.get(fn)
        if source is None and "__" in fn:
            source = fn.split("__", 1)[0]
        resolved[fn] = source or "unknown"
    return resolved


# Optional[float] source_recall_gap(Mapping[str, Optional[float]] recall_by_source)
# Inputs: Mapping[str, Optional[float]] recall_by_source - source key -> that source's recall,
#         or None where the subset was too small to measure
# Outputs: Optional[float] - max recall minus min recall across the measurable sources, or
#          None if fewer than two sources have a recall
# Description: Reduces the per-source recalls to the single number the source-confound check
#              turns on: how much better the model is on its best origin dataset than its worst.
# Side Effects: None
def source_recall_gap(recall_by_source: Mapping[str, Optional[float]]) -> Optional[float]:
    values = [float(v) for v in recall_by_source.values() if v is not None]
    if len(values) < 2:
        return None
    return max(values) - min(values)


# dict[str, dict[str, object]] shared_class_recall_gaps(Mapping[str, Mapping[str, float]] recall_by_class_by_source)
# Inputs: Mapping[str, Mapping[str, float]] recall_by_class_by_source - source key -> {class
#         name: that source's recall for the class}; a class absent from a source's test subset
#         is simply missing from its inner mapping
# Outputs: dict[str, dict[str, object]] - class name -> {"by_source": {source: recall}, "gap":
#          max-min}, containing ONLY classes measured in two or more sources
# Description: Isolates the like-for-like half of the source-confound diagnostic. The aggregate
#              per-source recall is not comparable when the sources contribute different classes
#              (a source holding only easy classes scores higher for reasons that have nothing
#              to do with dataset origin), so the honest confound signal is the SAME class scored
#              on each origin. Classes present in only one source are excluded rather than
#              silently contributing a fake gap.
# Side Effects: None
def shared_class_recall_gaps(
    recall_by_class_by_source: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, object]]:
    classes: dict[str, dict[str, float]] = {}
    for source in sorted(recall_by_class_by_source):
        for cname, recall in recall_by_class_by_source[source].items():
            classes.setdefault(cname, {})[source] = float(recall)
    shared: dict[str, dict[str, object]] = {}
    for cname, by_source in classes.items():
        if len(by_source) < 2:
            continue
        values = list(by_source.values())
        shared[cname] = {"by_source": by_source, "gap": max(values) - min(values)}
    return shared


# str source_confound_conclusion(Mapping[str, Optional[float]] recall_by_source, Optional[float] gap, int unknown_count, Optional[Mapping[str, dict]] shared, float warn_gap)
# Inputs: Mapping[str, Optional[float]] recall_by_source - source key -> recall (or None)
#         Optional[float] gap - source_recall_gap's output (the AGGREGATE gap)
#         int unknown_count - test images whose origin could not be recovered
#         Optional[Mapping[str, dict]] shared - shared_class_recall_gaps' output: the per-class
#         like-for-like comparison, or None/empty when no class appears in two sources
#         float warn_gap - recall gap at or above which the model is called confounded,
#         default SOURCE_RECALL_GAP_WARN (0.15)
# Outputs: str - a one-paragraph verdict naming the numbers it is based on
# Description: Turns the per-source recall split into a stated conclusion rather than leaving
#              the reader to eyeball it. The verdict rests on the SHARED-CLASS gaps whenever any
#              class is measured on both origins, because the aggregate gap also moves with class
#              mix: if one source contributes the easy classes and the other the hard ones, the
#              aggregate differs for reasons unrelated to dataset origin. A large shared-class
#              gap means the model is partly keying on which dataset an image came from
#              (resolution, framing, JPEG artifacts) rather than on defect appearance, so
#              aggregate mAP is optimistic for any new printer/camera.
# Side Effects: None
def source_confound_conclusion(
    recall_by_source: Mapping[str, Optional[float]],
    gap: Optional[float],
    unknown_count: int,
    shared: Optional[Mapping[str, dict]] = None,
    warn_gap: float = SOURCE_RECALL_GAP_WARN,
) -> str:
    measured = {k: v for k, v in recall_by_source.items() if v is not None}
    if unknown_count:
        prefix = f"{unknown_count} test image(s) had unrecoverable provenance; the split below covers the rest. "
    else:
        prefix = ""
    if len(measured) < 2:
        return (
            prefix
            + "INCONCLUSIVE: fewer than two source datasets had a large enough test subset to "
            "measure recall separately, so a dataset-origin confound cannot be ruled out."
        )

    detail = ", ".join(f"{k}={v:.3f}" for k, v in sorted(measured.items()))
    aggregate = f"Aggregate recall by origin: {detail}"
    if gap is not None:
        aggregate += f" (gap {gap:.3f})"
    aggregate += "."

    if not shared:
        return (
            prefix
            + "AGGREGATE ONLY -- NOT A CLEAN CONFOUND TEST: no class is present in more than one "
            f"source's test subset, so the origins cannot be compared like for like. {aggregate} "
            "That gap reflects the two sources' different class mixes as much as any dataset-origin "
            "effect, and cannot be attributed to either. Do not read it as a confound measurement."
        )

    worst_class = max(shared, key=lambda c: float(shared[c]["gap"]))
    worst_gap = float(shared[worst_class]["gap"])
    per_class = "; ".join(
        f"{c}: " + ", ".join(f"{s}={r:.3f}" for s, r in sorted(shared[c]["by_source"].items()))
        + f" (gap {float(shared[c]['gap']):.3f})"
        for c in sorted(shared, key=lambda c: -float(shared[c]["gap"]))
    )
    n_shared = len(shared)
    mix_note = (
        f" {aggregate} The aggregate is NOT the confound measurement here -- only "
        f"{n_shared} {'class is' if n_shared == 1 else 'classes are'} scored on both origins, "
        "so it also moves with class mix."
    )

    if worst_gap >= warn_gap:
        return (
            prefix
            + f"CONFOUND LIKELY: on classes measured on BOTH origins, recall differs by up to "
            f"{worst_gap:.3f} ({worst_class}), at or above the {warn_gap:.2f} flag. Like-for-like: "
            f"{per_class}. The images are copied byte-for-byte, so resolution/framing/JPEG "
            "statistics still identify the origin dataset and the model can partly be keying on "
            "that instead of on defect appearance. Treat aggregate mAP as optimistic for a new "
            "printer/camera." + mix_note
        )
    return (
        prefix
        + f"NO STRONG CONFOUND: on classes measured on BOTH origins, recall differs by at most "
        f"{worst_gap:.3f} ({worst_class}), below the {warn_gap:.2f} flag. Like-for-like: "
        f"{per_class}. The model performs comparably on both origins for the classes that can be "
        "compared, which is what a defect-feature (rather than dataset-origin) detector should do."
        + mix_note
    )


# dict[str, object] map_gap_note(float deduped_map50, float raw_map50, float warn_relative)
# Inputs: float deduped_map50 - mAP@50 over one representative image per source photograph
#         float raw_map50 - mAP@50 over every test image (the old, augmentation-weighted number)
#         float warn_relative - relative gap at or above which the difference is called
#         significant, default MAP_GAP_RELATIVE_WARN (0.10)
# Outputs: dict[str, object] - {"deduped_map50", "raw_map50", "absolute", "relative",
#          "raw_is_optimistic", "significant", "message"}; relative is None when deduped is 0
# Description: Quantifies how much the raw per-image mAP was being moved by the uneven
#              augmentation weighting, and states which direction. A raw score above the deduped
#              one means the photographs that received the most augmentation variants are also
#              the ones the model does best on -- exactly the flattering artifact the dedup
#              exists to remove.
# Side Effects: None
def map_gap_note(
    deduped_map50: float, raw_map50: float, warn_relative: float = MAP_GAP_RELATIVE_WARN
) -> dict[str, object]:
    absolute = float(raw_map50) - float(deduped_map50)
    relative = (absolute / deduped_map50) if deduped_map50 > 0 else None
    significant = relative is not None and abs(relative) >= warn_relative
    if significant:
        direction = "OPTIMISTIC" if absolute > 0 else "PESSIMISTIC"
        message = (
            f"RAW PER-IMAGE mAP@50 IS {direction} BY {abs(absolute):.4f} "
            f"({abs(relative) * 100:.1f}% of the deduped value). The per-image mean weights each "
            "source photograph by how many Roboflow augmentation variants it happened to receive, "
            "so this gap is the size of that weighting artifact -- not a real change in model "
            "quality. Quote the deduped number."
        )
    else:
        message = (
            f"Raw per-image and per-source-photo mAP@50 agree within {abs(absolute):.4f} "
            "-- the uneven augmentation weighting is not materially moving the headline number "
            "(it can still move individual classes; check the per-class table)."
        )
    return {
        "deduped_map50": float(deduped_map50),
        "raw_map50": float(raw_map50),
        "absolute": absolute,
        "relative": relative,
        "raw_is_optimistic": absolute > 0,
        "significant": significant,
        "message": message,
    }


# list[float] sweep_thresholds(float start, float end, float step)
# Inputs: float start - first confidence threshold in the sweep
#         float end - last confidence threshold in the sweep (inclusive)
#         float step - increment between thresholds
# Outputs: list[float] - confidence thresholds from start to end (inclusive) in steps of step,
#          rounded to 10 decimal places to avoid float accumulation artifacts
# Description: Builds the list of confidence thresholds used for the per-class sweep table.
# Side Effects: None
def sweep_thresholds(start: float, end: float, step: float) -> list[float]:
    n_steps = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n_steps) if start + i * step <= end + 1e-9]


# tuple[Optional[dict[str, float]], Optional[dict[str, float]]] find_lowest_threshold_for_precision(np.ndarray px, np.ndarray p_curve, np.ndarray r_curve, float target_precision, float min_recall)
# Inputs: np.ndarray px - confidence values (x-axis) for the precision/recall curves
#         np.ndarray p_curve - precision at each confidence value in px, for one class
#         np.ndarray r_curve - recall at each confidence value in px, for one class
#         float target_precision - minimum precision a threshold must reach
#         float min_recall - minimum recall a threshold must also reach, to reject
#         Ultralytics' vacuous precision=1.0-at-zero-predictions convention
# Outputs: tuple[Optional[dict[str, float]], Optional[dict[str, float]]] - (result,
#          vacuous_example): result is {"threshold", "precision", "recall"} for the first
#          (lowest) threshold meeting both target_precision and min_recall, or None if none do;
#          vacuous_example is set (only when result is None) to the first point that met
#          target_precision but not min_recall, so callers can report why nothing was accepted.
# Description: Scans confidence thresholds ascending and returns the first (lowest) one whose
#              precision meets target_precision AND whose recall is at least min_recall. The
#              min_recall floor exists because Ultralytics' precision curve reports precision=1.0
#              by convention wherever zero predictions survive for a class at that confidence (a
#              vacuous 0/0-style "perfect" score) -- a detector that never fires can't be a false
#              positive, but it also can't ever catch a real failure, so this floor rejects that
#              statistically meaningless operating point.
# Side Effects: None (pure numeric scan)
def find_lowest_threshold_for_precision(
    px: np.ndarray,
    p_curve: np.ndarray,
    r_curve: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> tuple[Optional[dict[str, float]], Optional[dict[str, float]]]:
    """`min_recall` rejects Ultralytics' precision=1.0-at-zero-predictions convention (see
    ap_per_class's left=1 fill) -- a class that never fires can't be a false positive but also
    can't catch a real failure."""
    idxs = np.argsort(px)  # px is already ascending (linspace) but be defensive
    vacuous_example: Optional[dict[str, float]] = None
    for i in idxs:
        if p_curve[i] >= target_precision:
            if r_curve[i] >= min_recall:
                return {"threshold": float(px[i]), "precision": float(p_curve[i]), "recall": float(r_curve[i])}, None
            if vacuous_example is None:
                vacuous_example = {"threshold": float(px[i]), "precision": float(p_curve[i]), "recall": float(r_curve[i])}
    return None, vacuous_example


# dict best_supported_point(list[dict] sweep, float min_recall)
# Inputs: list[dict] sweep - per-threshold {"threshold", "precision", "recall"} dicts for one
#         class, from build_report's confidence sweep
#         float min_recall - the recall floor a sweep point must clear to count as "supported"
# Outputs: dict - the chosen sweep point ({"threshold", "precision", "recall"})
# Description: Picks the highest-precision sweep point that still clears min_recall (i.e.
#              backed by a meaningful number of real detections, not the vacuous
#              0-predictions/precision=1.0 artifact). Falls back to the single highest-recall
#              point if the class never clears min_recall anywhere in the sweep.
# Side Effects: None
def best_supported_point(sweep: list[dict], min_recall: float) -> dict:
    supported = [pt for pt in sweep if pt["recall"] >= min_recall]
    if supported:
        return max(supported, key=lambda pt: pt["precision"])
    return max(sweep, key=lambda pt: pt["recall"])


# --------------------------------------------------------------------------
# I/O -- dataset listing, subset materialization, model inference.
# --------------------------------------------------------------------------


# Path resolve_split_images_dir(Mapping[str, object] data_cfg, Path data_yaml, str split)
# Inputs: Mapping[str, object] data_cfg - the parsed data.yaml mapping
#         Path data_yaml - path to the data.yaml itself, used as the root when it declares no
#         "path:" key
#         str split - split name, e.g. "test"
# Outputs: Path - the resolved directory holding that split's images
# Description: Resolves data.yaml's split entry the same way Ultralytics does (join onto
#              "path:", falling back to the yaml's own directory) so the dedup subset is built
#              from exactly the images the raw validation pass will see.
# Side Effects: Raises KeyError if the split is absent from data.yaml and FileNotFoundError if
#               the resolved directory doesn't exist. Read-only otherwise.
def resolve_split_images_dir(data_cfg: Mapping[str, object], data_yaml: Path, split: str) -> Path:
    if split not in data_cfg or not data_cfg[split]:
        raise KeyError(f"data.yaml '{data_yaml}' has no '{split}:' entry")
    root = Path(str(data_cfg.get("path") or data_yaml.parent))
    images_dir = (root / str(data_cfg[split])).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Split '{split}' images directory not found: {images_dir}")
    return images_dir


# list[str] list_image_filenames(Path images_dir)
# Inputs: Path images_dir - a split's images directory
# Outputs: list[str] - sorted basenames of every image file directly inside it
# Description: Lists a split's images deterministically (sorted, not filesystem order) so the
#              dedup grouping and representative selection are reproducible.
# Side Effects: Read-only filesystem traversal.
def list_image_filenames(images_dir: Path) -> list[str]:
    return sorted(p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# dict[str, list[int]] read_split_labels(Path images_dir, Sequence[str] filenames)
# Inputs: Path images_dir - a split's images directory (its sibling "labels" dir holds the
#         YOLO .txt files)
#         Sequence[str] filenames - image basenames to read labels for
# Outputs: dict[str, list[int]] - filename -> class indices in its label file; an empty list
#          for a deliberate background/negative image and for a missing label file
# Description: Reads the split's YOLO labels so per-class support (instances, images and unique
#              SOURCE PHOTOS) can be counted without running the model. Background images with
#              empty label files are kept as empty lists, never dropped -- they are the
#              negatives that hold the false-positive rate down.
# Side Effects: Reads label files from disk. A missing label file is treated as background
#               rather than raising, matching Ultralytics' own behaviour.
def read_split_labels(images_dir: Path, filenames: Sequence[str]) -> dict[str, list[int]]:
    labels_dir = images_dir.parent / "labels"
    out: dict[str, list[int]] = {}
    for fn in filenames:
        label_path = labels_dir / (Path(fn).stem + ".txt")
        if label_path.is_file():
            out[fn] = parse_yolo_label_classes(label_path.read_text(encoding="utf-8"))
        else:
            out[fn] = []
    return out


# dict[str, str] load_manifest_sources(Optional[Path] manifest_path)
# Inputs: Optional[Path] manifest_path - path to ingest_roboflow_multi.py's manifest.jsonl, or
#         None / a nonexistent path
# Outputs: dict[str, str] - output image basename -> its source dataset key ("atco",
#          "stereovision", ...); empty dict when there is no readable manifest
# Description: Loads the authoritative image->origin mapping for the source-confound
#              diagnostic. Using the manifest rather than re-deriving provenance from filenames
#              means the diagnostic reports what the ingest actually recorded.
# Side Effects: Reads manifest.jsonl line by line. Malformed lines are skipped rather than
#               raising, so a partially-written manifest degrades to the filename-prefix
#               fallback instead of aborting the evaluation.
def load_manifest_sources(manifest_path: Optional[Path]) -> dict[str, str]:
    return _load_manifest_field(manifest_path, "source")


# dict[str, str] load_manifest_groups(Optional[Path] manifest_path)
# Inputs: Optional[Path] manifest_path - path to ingest_roboflow_multi.py's manifest.jsonl, or
#         None / a nonexistent path
# Outputs: dict[str, str] - output image basename -> its leakage-group id; empty dict when
#          there is no readable manifest
# Description: Loads the leakage-group assignment the ingest used to keep every near-duplicate
#              and timelapse sibling inside one split. Used to report effective sample size at
#              the strictest available unit alongside the source-photo count.
# Side Effects: Reads manifest.jsonl. Malformed lines are skipped rather than raising.
def load_manifest_groups(manifest_path: Optional[Path]) -> dict[str, str]:
    return _load_manifest_field(manifest_path, "group")


# dict[str, str] _load_manifest_field(Optional[Path] manifest_path, str field)
# Inputs: Optional[Path] manifest_path - path to manifest.jsonl, or None / a nonexistent path
#         str field - the record field to index by output basename, e.g. "source" or "group"
# Outputs: dict[str, str] - output image basename -> that record's field value; records missing
#          either "output" or the field are skipped
# Description: Shared reader behind load_manifest_sources / load_manifest_groups.
# Side Effects: Reads manifest.jsonl line by line. Malformed JSON lines are skipped rather than
#               raising, so a partially-written manifest degrades gracefully instead of
#               aborting the evaluation.
def _load_manifest_field(manifest_path: Optional[Path], field: str) -> dict[str, str]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    values: dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            output = rec.get("output")
            value = rec.get(field)
            if output and value:
                values[Path(str(output)).name] = str(value)
    return values


# Path write_subset_dataset(Mapping[str, object] data_cfg, Path images_dir, Sequence[str] filenames, Path work_dir, str tag)
# Inputs: Mapping[str, object] data_cfg - the parsed data.yaml mapping (class names/nc reused
#         verbatim so the subset is scored against the identical class set)
#         Path images_dir - the split's images directory
#         Sequence[str] filenames - the subset's image basenames
#         Path work_dir - scratch directory for the generated list + yaml
#         str tag - short name for the subset ("dedup", "src_atco", ...), used in filenames
# Outputs: Path - the generated data.yaml whose "test:" points at the subset's image list
# Description: Materializes an evaluation subset as an Ultralytics image-list dataset (a .txt
#              of absolute image paths plus a data.yaml pointing at it) so the deduped and
#              per-source passes go through Ultralytics' own val()/ap_per_class rather than a
#              reimplemented matcher. Nothing under datasets/ is copied, moved or deleted.
# Side Effects: Creates work_dir if needed and writes two files into it. Raises ValueError on
#               an empty filename list.
def write_subset_dataset(
    data_cfg: Mapping[str, object], images_dir: Path, filenames: Sequence[str], work_dir: Path, tag: str
) -> Path:
    if not filenames:
        raise ValueError(f"Refusing to build an empty evaluation subset for '{tag}'")
    work_dir.mkdir(parents=True, exist_ok=True)
    list_path = work_dir / f"{tag}_images.txt"
    list_path.write_text("\n".join((images_dir / fn).as_posix() for fn in filenames) + "\n", encoding="utf-8")

    import yaml

    subset_cfg = {
        "path": str(data_cfg.get("path") or images_dir.parent.parent),
        "train": data_cfg.get("train"),
        "val": data_cfg.get("val"),
        "test": list_path.as_posix(),
        "nc": data_cfg.get("nc"),
        "names": data_cfg.get("names"),
    }
    yaml_path = work_dir / f"{tag}_data.yaml"
    yaml_path.write_text(yaml.safe_dump(subset_cfg, sort_keys=False), encoding="utf-8")
    return yaml_path


# Any load_model(Path weights)
# Inputs: Path weights - path to the trained best.pt checkpoint
# Outputs: Any - the loaded Ultralytics YOLO model, reused across every validation pass
# Description: Loads the checkpoint once so the raw, deduped and per-source passes all score
#              the exact same weights (and so a checkpoint rewritten mid-evaluation by a still-
#              running training job can't silently change the numbers between passes).
# Side Effects: Imports ultralytics.YOLO lazily; reads the checkpoint from disk into memory.
def load_model(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


# Any run_validation(Any model, Path data, int imgsz, int batch, str device, float nms_iou, Optional[Path] run_dir, str run_name)
# Inputs: Any model - a loaded Ultralytics YOLO model from load_model
#         Path data - path to data.yaml describing the dataset (its "test" split may be a
#         directory or a generated image-list .txt)
#         int imgsz - validation image size
#         int batch - validation batch size
#         str device - CUDA device index, list, or "cpu"
#         float nms_iou - IoU threshold used for NMS during validation
#         Optional[Path] run_dir - scratch project dir for Ultralytics' own run artifacts
#         str run_name - subdirectory name under run_dir
# Outputs: Any - the Ultralytics DetMetrics object returned by model.val(), carrying the full
#          precision/recall-vs-confidence curves used by the rest of this module
# Description: Runs Ultralytics val() on the TEST split (held out from training and threshold
#              selection) at low confidence (conf=0.001) so the full precision/recall-vs-
#              confidence curve is available afterward, using Ultralytics' own IoU=0.5
#              ap_per_class matching rather than reimplemented logic.
# Side Effects: Runs a full GPU/CPU inference pass over the referenced split (plots=False,
#               save_json=False). Ultralytics creates its run directory under run_dir when
#               given, keeping the repo's runs/detect/ clean.
def run_validation(
    model,
    data: Path,
    imgsz: int,
    batch: int,
    device: str,
    nms_iou: float,
    run_dir: Optional[Path] = None,
    run_name: str = "val",
):
    kwargs = {}
    if run_dir is not None:
        kwargs["project"] = str(run_dir)
        kwargs["name"] = run_name
        kwargs["exist_ok"] = True
    return model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=nms_iou,
        plots=False,
        save_json=False,
        verbose=False,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Report construction.
# --------------------------------------------------------------------------


# dict metrics_summary(Any metrics)
# Inputs: Any metrics - an Ultralytics DetMetrics object from run_validation
# Outputs: dict - {"map50", "map50_95", "mean_precision_at_max_f1", "mean_recall_at_max_f1",
#          "per_class": {name: {"ap50", "ap50_95", "precision_at_max_f1", "recall_at_max_f1",
#          "num_instances"}}}
# Description: Flattens a validation pass into the small JSON-serializable summary used for
#              every pass that isn't the headline one (the raw per-image pass and the
#              per-source confound passes), so all three are reported in the same shape.
# Side Effects: None
def metrics_summary(metrics) -> dict:
    box = metrics.box
    names: dict[int, str] = metrics.names
    per_class: dict[str, dict] = {}
    for row, cls_idx in enumerate(list(box.ap_class_index)):
        cname = names.get(int(cls_idx), f"class_{cls_idx}")
        per_class[cname] = {
            "ap50": float(box.ap50[row]),
            "ap50_95": float(box.ap[row]),
            "precision_at_max_f1": float(box.p[row]),
            "recall_at_max_f1": float(box.r[row]),
            "num_instances": int(metrics.nt_per_class[cls_idx]) if metrics.nt_per_class is not None else None,
        }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mean_precision_at_max_f1": float(box.mp),
        "mean_recall_at_max_f1": float(box.mr),
        "per_class": per_class,
    }


# dict build_report(Any metrics, tuple[str, ...] class_names, argparse.Namespace args, dict dedup, Optional[dict] raw_summary, dict support, dict source_confound)
# Inputs: Any metrics - the DetMetrics object for the HEADLINE pass (deduped unless
#         --no-dedup-sources was given)
#         tuple[str, ...] class_names - full class-name list from data.yaml, in config order
#         argparse.Namespace args - parsed CLI args (target_precision, min_recall, sweep
#         bounds, weights, data paths, seed, representative strategy, etc.)
#         dict dedup - the dedup block: whether it was applied, seed/strategy, image and
#         source-photo counts, and the variant-count spread from variant_count_stats
#         Optional[dict] raw_summary - metrics_summary of the raw per-image pass, or None
#         dict support - class_support output over the FULL raw test split
#         dict source_confound - the per-source recall diagnostic block
# Outputs: dict - full JSON-serializable evaluation report: the headline (deduped) overall
#          mAP/precision/recall, the raw per-image overall block beside it, the deduped-vs-raw
#          gap note, per-class metrics (AP50, AP50-95, precision/recall at max-F1, effective
#          sample size in unique source photos, the confidence sweep, the lowest threshold
#          reaching target_precision or an explicit vacuous/unreachable marker, the raw
#          per-image counterpart, and a severity flag), classes with no test instances, the
#          severity policy, and the source-confound diagnostic
# Description: Builds the full per-class and overall evaluation report. The headline numbers
#              are per-SOURCE-PHOTO (one representative image per photograph) so every
#              photograph carries equal weight; the raw per-image numbers -- what a naive
#              evaluation reports, weighting each photograph by its Roboflow augmentation count
#              -- are carried alongside, never in place of, so the gap stays visible. Runs the
#              confidence sweep and the min_recall-guarded target-precision search
#              (find_lowest_threshold_for_precision) per class that had instances.
# Side Effects: None (pure computation over the metrics objects; no I/O)
def build_report(
    metrics,
    class_names: tuple[str, ...],
    args: argparse.Namespace,
    dedup: dict,
    raw_summary: Optional[dict],
    support: dict,
    source_confound: dict,
) -> dict:
    box = metrics.box
    names: dict[int, str] = metrics.names  # {idx: name}, full class set from data.yaml
    ap_class_index = list(box.ap_class_index)  # which classes had instances, row order into p/r/ap50/ap/p_curve/r_curve
    px = np.asarray(box.px)

    thresholds = sweep_thresholds(args.sweep_start, args.sweep_end, args.sweep_step)
    raw_per_class = (raw_summary or {}).get("per_class", {})

    per_class: dict[str, dict] = {}
    for row, cls_idx in enumerate(ap_class_index):
        cname = names.get(int(cls_idx), f"class_{cls_idx}")
        p_curve_row = np.asarray(box.p_curve[row])
        r_curve_row = np.asarray(box.r_curve[row])

        sweep = [
            {
                "threshold": t,
                "precision": float(np.interp(t, px, p_curve_row)),
                "recall": float(np.interp(t, px, r_curve_row)),
            }
            for t in thresholds
        ]

        target_result, vacuous_example = find_lowest_threshold_for_precision(
            px, p_curve_row, r_curve_row, args.target_precision, args.min_recall
        )

        class_support_entry = support.get(cname, {"instances": 0, "images": 0, "source_photos": 0})
        n_photos = int(class_support_entry["source_photos"])
        n_groups = class_support_entry.get("independent_groups")

        per_class[cname] = {
            "class_index": int(cls_idx),
            # Instances in the EVALUATED (headline) subset -- deduped unless --no-dedup-sources.
            "num_test_instances": int(metrics.nt_per_class[cls_idx]) if metrics.nt_per_class is not None else None,
            # Effective sample size: independent photographs, not augmentation variants.
            "effective_sample_size": {
                "source_photos_with_class": n_photos,
                # Stricter unit: the ingest's leakage group, which also merges perceptual
                # near-duplicates and timelapse siblings. None when no manifest was readable.
                "independent_groups_with_class": n_groups,
                "raw_images_with_class": int(class_support_entry["images"]),
                "raw_instances": int(class_support_entry["instances"]),
                "low_support": n_photos < LOW_SUPPORT_PHOTOS,
                "low_support_groups": (n_groups is not None and n_groups < LOW_SUPPORT_PHOTOS),
                "low_support_floor": LOW_SUPPORT_PHOTOS,
            },
            "precision_at_max_f1": float(box.p[row]),
            "recall_at_max_f1": float(box.r[row]),
            "ap50": float(box.ap50[row]),
            "ap50_95": float(box.ap[row]),
            # The same class scored per-image (augmentation-weighted), for side-by-side reading.
            "raw_per_image": raw_per_class.get(cname),
            "confidence_sweep": sweep,
            "target_precision": args.target_precision,
            "min_recall": args.min_recall,
            "lowest_threshold_for_target_precision": target_result,  # None => unreachable (with meaningful recall)
            "target_reachable": target_result is not None,
            # Set only when target_precision is met SOLELY by near-zero-recall points (the
            # vacuous "0 predictions -> precision=1.0" artifact) -- i.e. the number looks
            # perfect but the class effectively never fires there.
            "vacuous_precision_only": vacuous_example,
            "severity": "catastrophic" if cname in CATASTROPHIC_CLASSES else "cosmetic",
            "is_catastrophic": cname in CATASTROPHIC_CLASSES,
        }

    # Any configured class with zero test instances (shouldn't happen with a
    # reasonable split, but call it out rather than silently omitting it).
    missing_classes = [c for c in class_names if c not in per_class]

    headline_label = (
        "per_source_photo_deduped" if dedup.get("applied") else "raw_per_image (dedup disabled via --no-dedup-sources)"
    )
    overall_deduped = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mean_precision_at_max_f1": float(box.mp),
        "mean_recall_at_max_f1": float(box.mr),
    }

    report = {
        "weights": str(args.weights),
        "data_yaml": str(args.data),
        "test_split": "test",
        "headline_metric": headline_label,
        "dedup": dedup,
        # HEADLINE: one vote per source photograph.
        "overall_deduped_per_source_photo": overall_deduped,
        # What the old (unweighted-mean-over-images) evaluation reported: one vote per file,
        # so a photo with 30 augmentation variants outvotes a photo with 1 by 30 to 1.
        "overall_raw_per_image": raw_summary,
        "deduped_vs_raw": (
            map_gap_note(overall_deduped["map50"], float(raw_summary["map50"]))
            if (raw_summary and dedup.get("applied"))
            else None
        ),
        # Back-compat alias: "overall" always means the headline number.
        "overall": overall_deduped,
        "target_precision": args.target_precision,
        "min_recall": args.min_recall,
        "per_class": per_class,
        "class_support_raw_test_split": support,
        "classes_with_no_test_instances": missing_classes,
        "severity_policy": {
            "catastrophic": list(CATASTROPHIC_CLASSES),
            "cosmetic": list(COSMETIC_CLASSES),
            "note": "Only catastrophic classes may drive an automated pause; cosmetic classes are logged only.",
        },
        "catastrophic_classes": list(CATASTROPHIC_CLASSES),
        "source_confound": source_confound,
    }
    return report


# dict build_source_confound(dict[str, Any] metrics_by_source, dict[str, int] images_by_source, int unknown_count, str provenance_method, list[str] skipped)
# Inputs: dict[str, Any] metrics_by_source - source key -> that source's DetMetrics object
#         dict[str, int] images_by_source - source key -> how many evaluated images it
#         contributed (representatives, when dedup is on)
#         int unknown_count - test images whose origin could not be recovered
#         str provenance_method - human-readable description of where provenance came from
#         list[str] skipped - source keys whose subset was too small to evaluate
# Outputs: dict - {"method", "provenance_recoverable", "unknown_provenance_count",
#          "sources_skipped_too_small", "by_source", "aggregate_recall_gap",
#          "shared_class_recall", "worst_shared_class_gap", "classes_shared_across_sources",
#          "class_coverage_disjoint", "gap_flag", "conclusion", "recall_gap"}
# Description: Assembles the SOURCE-CONFOUND DIAGNOSTIC: recall (and mAP50) measured separately
#              per origin dataset, the aggregate gap, the like-for-like per-class gaps for the
#              classes present on both origins, and a stated verdict. A model that scores well
#              on one origin and poorly on the other for the SAME class has partly learned
#              dataset identity -- resolution, framing, JPEG statistics -- rather than defect
#              features, which will not transfer to the user's own printer and camera. The
#              aggregate gap alone is not that measurement: it also moves with class mix when
#              the sources contribute different classes, so it is reported but not relied on.
# Side Effects: None (pure assembly over already-computed metrics objects)
def build_source_confound(
    metrics_by_source: dict,
    images_by_source: dict,
    unknown_count: int,
    provenance_method: str,
    skipped: list[str],
) -> dict:
    by_source: dict[str, dict] = {}
    for src in sorted(metrics_by_source):
        summary = metrics_summary(metrics_by_source[src])
        by_source[src] = {
            "num_images_evaluated": int(images_by_source.get(src, 0)),
            "recall_at_max_f1": summary["mean_recall_at_max_f1"],
            "precision_at_max_f1": summary["mean_precision_at_max_f1"],
            "map50": summary["map50"],
            "per_class_recall": {c: v["recall_at_max_f1"] for c, v in summary["per_class"].items()},
            "per_class_ap50": {c: v["ap50"] for c, v in summary["per_class"].items()},
        }
    recall_by_source = {k: v["recall_at_max_f1"] for k, v in by_source.items()}
    gap = source_recall_gap(recall_by_source)
    shared = shared_class_recall_gaps({k: v["per_class_recall"] for k, v in by_source.items()})
    worst_shared = max((float(v["gap"]) for v in shared.values()), default=None)
    all_classes = {c for blk in by_source.values() for c in blk["per_class_recall"]}
    return {
        "method": provenance_method,
        "provenance_recoverable": unknown_count == 0,
        "unknown_provenance_count": int(unknown_count),
        "sources_skipped_too_small": sorted(skipped),
        "by_source": by_source,
        # Aggregate gap: reported for completeness, but it also moves with class mix.
        "aggregate_recall_gap": gap,
        "recall_gap": gap,  # back-compat alias
        # The like-for-like signal: same class, both origins.
        "shared_class_recall": shared,
        "classes_shared_across_sources": sorted(shared),
        "worst_shared_class_gap": worst_shared,
        "class_coverage_disjoint": bool(all_classes) and not shared,
        "gap_flag": SOURCE_RECALL_GAP_WARN,
        "conclusion": source_confound_conclusion(recall_by_source, gap, unknown_count, shared),
    }


# str format_yaml_block(dict report, tuple[str, ...] class_names)
# Inputs: dict report - evaluation report produced by build_report
#         tuple[str, ...] class_names - full class-name list from data.yaml, in config order
# Outputs: str - a "class_thresholds:" YAML block ready to paste into config.example.yaml /
#          config.yaml, with inline comments explaining each recommended threshold, including
#          loud warnings when target precision is only vacuously reachable or not reachable at
#          all (falling back to best_supported_point in those cases)
# Description: Formats the per-class recommended confidence thresholds as a pasteable YAML
#              block for the runtime config. Thresholds come from the HEADLINE (deduped)
#              curves, so each source photograph counted once when the operating point was
#              chosen.
# Side Effects: None (pure string formatting)
def format_yaml_block(report: dict, class_names: tuple[str, ...]) -> str:
    lines = ["class_thresholds:"]
    for cname in class_names:
        entry = report["per_class"].get(cname)
        key = f'"{cname}"' if " " in cname else cname
        if entry is None:
            lines.append(f"  {key}: 0.95  # NO TEST INSTANCES for this class -- cannot evaluate, do not trust")
            continue
        target = report["target_precision"]
        result = entry["lowest_threshold_for_target_precision"]
        ess = entry["effective_sample_size"]
        n_photos = ess["source_photos_with_class"]
        n_groups = ess.get("independent_groups_with_class")
        n_desc = f"{n_photos} source photos" + (f" / {n_groups} independent scenes" if n_groups is not None else "")
        low_n = f"  [LOW-N: only {n_desc}]" if (ess["low_support"] or ess.get("low_support_groups")) else ""
        if result is not None:
            lines.append(
                f"  {key}: {result['threshold']:.2f}  "
                f"# precision={result['precision']:.3f} recall={result['recall']:.3f} "
                f"(target precision {target:.2f} MET, n={n_desc}){low_n}"
            )
        elif entry["vacuous_precision_only"] is not None:
            # Only "met" via Ultralytics' precision=1.0-at-zero-predictions convention;
            # recommend the best real (recall-backed) point instead.
            v = entry["vacuous_precision_only"]
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            lines.append(
                f"  {key}: {best_p['threshold']:.2f}  "
                f"# WARNING: target precision {target:.2f} only reachable at near-zero recall "
                f"(precision={v['precision']:.3f} but recall={v['recall']:.3f} at conf={v['threshold']:.2f} -- "
                f"model effectively never fires there). Falling back to the best REAL operating point: "
                f"precision={best_p['precision']:.3f} recall={best_p['recall']:.3f}. "
                f"Do NOT deploy this class's threshold as-is; needs more data/training.{low_n}"
            )
        else:
            # Precision never reaches target at any threshold, real or vacuous:
            # fall back to the best recall-backed sweep point and say so loudly
            # rather than silently emitting a threshold that looks fine but isn't.
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            lines.append(
                f"  {key}: 0.95  "
                f"# WARNING: target precision {target:.2f} NOT reachable at any threshold -- "
                f"best real (recall>={report['min_recall']:.2f}) precision={best_p['precision']:.3f} "
                f"at conf={best_p['threshold']:.2f}. Do NOT deploy this class's threshold as-is; "
                f"needs more data/training.{low_n}"
            )
    return "\n".join(lines)


# None print_report(dict report, tuple[str, ...] class_names)
# Inputs: dict report - evaluation report produced by build_report
#         tuple[str, ...] class_names - full class-name list from data.yaml, in config order
# Outputs: None
# Description: Prints the human-readable evaluation summary to stdout: the deduped headline mAP
#              beside the raw per-image mAP with an explicit gap callout, the augmentation-
#              variant spread the dedup corrects for, a per-class metrics table carrying each
#              class's effective sample size in unique source photographs and its severity, the
#              recommended confidence threshold per class with vacuous/unreachable warnings, the
#              source-confound (per-origin recall) diagnostic, and the pasteable YAML block from
#              format_yaml_block.
# Side Effects: Prints the full evaluation report to stdout. No filesystem writes.
def print_report(report: dict, class_names: tuple[str, ...]) -> None:
    dedup = report["dedup"]
    raw = report.get("overall_raw_per_image")
    gap = report.get("deduped_vs_raw")

    print()
    print("=" * 100)
    print("EVALUATION -- TEST SPLIT (held out from training and threshold selection)")
    print("=" * 100)
    print(f"Weights: {report['weights']}")
    print(f"Data:    {report['data_yaml']}")
    print()

    if dedup.get("applied"):
        print(
            f"Sampling unit: SOURCE PHOTOGRAPH. {dedup['num_raw_images']} test images collapse to "
            f"{dedup['num_source_photos']} unique source photos "
            f"(Roboflow '.rf.<hash>' variants: min {dedup['variant_counts']['min']}, median "
            f"{dedup['variant_counts']['median']:.0f}, mean {dedup['variant_counts']['mean']:.1f}, "
            f"max {dedup['variant_counts']['max']}; {dedup['variant_counts']['num_photos_over_10_variants']} "
            f"photos carry >10 variants)."
        )
        print(
            f"  One representative per photo, strategy='{dedup['representative_strategy']}' seed={dedup['seed']} "
            f"(deterministic; not filesystem order)."
        )
        if dedup.get("num_independent_groups"):
            print(
                f"  NOTE: those {dedup['num_source_photos']} photos come from only "
                f"{dedup['num_independent_groups']} independent-scene groups (the ingest's own unit, merging "
                "near-duplicates and timelapse siblings), so even the deduped N is an upper bound on "
                "independence. See the 'groups' column below."
            )
    else:
        print("Sampling unit: IMAGE FILE (--no-dedup-sources). Each source photograph is weighted by")
        print("  how many Roboflow augmentation variants it happened to receive. Headline numbers below")
        print("  are NOT per-photo estimates.")
    print()

    print("-" * 100)
    print("OVERALL")
    print("-" * 100)
    print(f"{'':<46}{'mAP@50':>12}{'mAP@50-95':>12}{'P(maxF1)':>12}{'R(maxF1)':>12}")
    o = report["overall_deduped_per_source_photo"]
    label_head = "DEDUPED (per source photo)  <-- HEADLINE" if dedup.get("applied") else "RAW (per image)  <-- HEADLINE"
    print(
        f"{label_head:<46}{o['map50']:>12.4f}{o['map50_95']:>12.4f}"
        f"{o['mean_precision_at_max_f1']:>12.4f}{o['mean_recall_at_max_f1']:>12.4f}"
    )
    if raw is not None and dedup.get("applied"):
        print(
            f"{'RAW (per image, augmentation-weighted)':<46}{raw['map50']:>12.4f}{raw['map50_95']:>12.4f}"
            f"{raw['mean_precision_at_max_f1']:>12.4f}{raw['mean_recall_at_max_f1']:>12.4f}"
        )
    if gap is not None:
        print()
        banner = "!" * 100 if gap["significant"] else "-" * 100
        print(banner)
        print(f"DEDUPED vs RAW: {gap['message']}")
        print(banner)
    print()

    print("-" * 100)
    print("PER CLASS -- headline (deduped) metrics, with effective sample size")
    print("-" * 100)
    header = (
        f"{'class':<18}{'sev':>6}{'AP50':>8}{'AP50-95':>9}{'P(maxF1)':>10}{'R(maxF1)':>10}"
        f"{'photos':>8}{'groups':>8}{'rawImgs':>9}{'rawInst':>9}{'rawAP50':>9}"
    )
    print(header)
    print("-" * len(header))
    for cname in class_names:
        entry = report["per_class"].get(cname)
        sev = "CAT" if cname in CATASTROPHIC_CLASSES else "cos"
        supp = report["class_support_raw_test_split"].get(cname, {"instances": 0, "images": 0, "source_photos": 0})
        if entry is None:
            groups_cell = supp.get("independent_groups")
            print(
                f"{cname:<18}{sev:>6}{'--':>8}{'--':>9}{'--':>10}{'--':>10}"
                f"{supp['source_photos']:>8}{(groups_cell if groups_cell is not None else '--'):>8}"
                f"{supp['images']:>9}{supp['instances']:>9}{'--':>9}"
            )
            continue
        ess = entry["effective_sample_size"]
        raw_c = entry.get("raw_per_image")
        raw_ap = f"{raw_c['ap50']:.3f}" if raw_c else "--"
        n_groups = ess.get("independent_groups_with_class")
        flag = "  <-- LOW-N" if (ess["low_support"] or ess.get("low_support_groups")) else ""
        print(
            f"{cname:<18}{sev:>6}{entry['ap50']:>8.3f}{entry['ap50_95']:>9.3f}"
            f"{entry['precision_at_max_f1']:>10.3f}{entry['recall_at_max_f1']:>10.3f}"
            f"{ess['source_photos_with_class']:>8}{(n_groups if n_groups is not None else '--'):>8}"
            f"{ess['raw_images_with_class']:>9}{ess['raw_instances']:>9}{raw_ap:>9}{flag}"
        )
    print()
    print(f"  sev:     CAT = CATASTROPHIC ({', '.join(CATASTROPHIC_CLASSES)}) -- may pause a print.")
    print(f"           cos = COSMETIC ({', '.join(COSMETIC_CLASSES)}) -- logged, never acted on.")
    print("  photos:  UNIQUE SOURCE PHOTOGRAPHS in the test split containing this class -- the")
    print("           effective sample size, and the unit the headline metrics are weighted by.")
    print("  groups:  the ingest's INDEPENDENT-SCENE groups containing this class (near-duplicate and")
    print("           timelapse siblings merged). Lower than 'photos', and the more honest N: several")
    print("           photographs of one print job are not several independent observations.")
    print(f"           Either count under {LOW_SUPPORT_PHOTOS} is flagged LOW-N -- that class's AP/precision is an")
    print("           estimate over a handful of scenes, not a population figure.")
    print("  rawImgs/rawInst: images and boxes counting every augmentation variant (inflated by design).")
    print("  rawAP50: the same class scored per-image, i.e. what the un-deduped evaluation reports.")

    low_n = [
        c
        for c in class_names
        if (report["per_class"].get(c) or {})
        .get("effective_sample_size", {})
        .get("low_support")
        or (report["per_class"].get(c) or {}).get("effective_sample_size", {}).get("low_support_groups")
    ]
    if low_n:
        print()
        print(f"  LOW-N CLASSES: {', '.join(low_n)} -- do not quote their precision as a population rate.")

    target = report["target_precision"]
    print()
    print("-" * 100)
    print(f"RECOMMENDED CONFIDENCE THRESHOLDS (precision >= {target:.2f}, from the DEDUPED curves)")
    print("-" * 100)
    for cname in class_names:
        entry = report["per_class"].get(cname)
        if entry is None:
            print(f"  {cname:<20} NO TEST INSTANCES -- cannot evaluate")
            continue
        result = entry["lowest_threshold_for_target_precision"]
        tag = " [CATASTROPHIC -- governs false-positive rate]" if cname in CATASTROPHIC_CLASSES else ""
        if result is not None:
            print(
                f"  {cname:<20} threshold={result['threshold']:.2f}  "
                f"precision={result['precision']:.3f}  recall={result['recall']:.3f}{tag}"
            )
        elif entry["vacuous_precision_only"] is not None:
            v = entry["vacuous_precision_only"]
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            print(
                f"  {cname:<20} TARGET ONLY MET AT NEAR-ZERO RECALL (vacuous: precision={v['precision']:.3f} "
                f"recall={v['recall']:.3f} @ conf={v['threshold']:.2f} -- model effectively never fires there). "
                f"Best real point: precision={best_p['precision']:.3f} recall={best_p['recall']:.3f} "
                f"@ conf={best_p['threshold']:.2f}{tag}"
            )
        else:
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            print(
                f"  {cname:<20} TARGET NOT REACHABLE at any threshold "
                f"(best real precision={best_p['precision']:.3f} @ conf={best_p['threshold']:.2f}){tag}"
            )

    print()
    print("-" * 100)
    print("SOURCE-CONFOUND DIAGNOSTIC -- recall split by origin dataset")
    print("-" * 100)
    sc = report["source_confound"]
    print(f"Provenance: {sc['method']}")
    if sc["unknown_provenance_count"]:
        print(f"  WARNING: {sc['unknown_provenance_count']} evaluated image(s) had no recoverable origin.")
    if sc["sources_skipped_too_small"]:
        print(f"  Skipped (subset too small to measure): {', '.join(sc['sources_skipped_too_small'])}")
    if sc["by_source"]:
        sub_header = f"{'source':<18}{'images':>9}{'R(maxF1)':>11}{'P(maxF1)':>11}{'mAP@50':>10}"
        print(sub_header)
        print("-" * len(sub_header))
        for src, blk in sc["by_source"].items():
            print(
                f"{src:<18}{blk['num_images_evaluated']:>9}{blk['recall_at_max_f1']:>11.3f}"
                f"{blk['precision_at_max_f1']:>11.3f}{blk['map50']:>10.3f}"
            )
        print()
        print("Per-class recall by origin ('--' = that class has no instances in that origin's test subset):")
        print(f"{'class':<18}" + "".join(f"{src[:12]:>14}" for src in sc["by_source"]) + f"{'shared gap':>14}")
        for cname in class_names:
            cells = "".join(
                f"{blk['per_class_recall'][cname]:>14.3f}" if cname in blk["per_class_recall"] else f"{'--':>14}"
                for blk in sc["by_source"].values()
            )
            shared_entry = sc.get("shared_class_recall", {}).get(cname)
            gap_cell = f"{float(shared_entry['gap']):>14.3f}" if shared_entry else f"{'n/a':>14}"
            print(f"{cname:<18}{cells}{gap_cell}")
        print()
        if sc.get("aggregate_recall_gap") is not None:
            print(
                f"Aggregate recall gap between origins: {sc['aggregate_recall_gap']:.3f} "
                f"-- NOT the confound measurement (it also moves with class mix)."
            )
        if sc.get("worst_shared_class_gap") is not None:
            print(
                f"Worst LIKE-FOR-LIKE gap (classes scored on both origins: "
                f"{', '.join(sc['classes_shared_across_sources'])}): "
                f"{sc['worst_shared_class_gap']:.3f} (flag at {sc['gap_flag']:.2f})"
            )
        elif sc.get("class_coverage_disjoint"):
            print("No class appears in both origins' test subsets -- no like-for-like comparison is possible.")
    print(sc["conclusion"])

    print()
    print("-" * 100)
    print("Paste into config.example.yaml / config.yaml:")
    print("-" * 100)
    print(format_yaml_block(report, class_names))
    print("=" * 100)


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Loads class names from data.yaml, groups the test split by
#              source photograph and picks one deterministic representative per photo, runs
#              validation on the raw split AND the deduped subset AND each origin dataset's
#              subset, builds the combined report, prints it, and writes the full JSON to disk.
# Side Effects: Raises FileNotFoundError if --weights or --data don't exist; reads data.yaml,
#               the test split's labels and (if present) manifest.jsonl; runs several GPU/CPU
#               validation passes (see run_validation); writes the generated subset image-lists
#               and data.yamls into a temporary directory that is deleted afterward; prints the
#               evaluation report to stdout; creates --out's parent directory and writes the
#               full JSON report to --out (default runs/evaluation.json). Nothing under
#               datasets/ is modified.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    import yaml

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = tuple(data_cfg["names"])

    images_dir = resolve_split_images_dir(data_cfg, args.data, "test")
    filenames = list_image_filenames(images_dir)
    if not filenames:
        raise FileNotFoundError(f"No test images found under {images_dir}")

    labels_by_image = read_split_labels(images_dir, filenames)
    support = class_support(labels_by_image, class_names)
    n_background = sum(1 for v in labels_by_image.values() if not v)

    representatives, groups, no_rf = select_representatives(filenames, args.seed, args.representative)
    variant_counts = variant_count_stats(groups)

    manifest_path = args.manifest if args.manifest is not None else args.data.parent / "manifest.jsonl"
    group_of_image = load_manifest_groups(manifest_path)
    if group_of_image:
        group_support = class_group_support(labels_by_image, class_names, group_of_image)
        for cname, n in group_support.items():
            support[cname]["independent_groups"] = n
    n_independent_groups = len({g for fn, g in group_of_image.items() if fn in labels_by_image}) or None

    manifest_sources = load_manifest_sources(manifest_path)
    sources = resolve_image_sources(filenames, manifest_sources)
    provenance_method = (
        f"manifest.jsonl 'source' field ({manifest_path})"
        if manifest_sources
        else "'<source>__' output-filename prefix (no manifest.jsonl found)"
    )

    evaluated = representatives if args.dedup_sources else filenames
    dedup_block = {
        "applied": bool(args.dedup_sources),
        "grouping": "filename stem with Roboflow '.rf.<hash>' suffix stripped (prepare_dataset.extract_source_id)",
        "representative_strategy": args.representative,
        "seed": args.seed,
        "num_raw_images": len(filenames),
        "num_source_photos": len(groups),
        # The ingest's own independence unit (near-duplicate/timelapse clusters). Lower than
        # the photo count, so per-photo dedup is a floor on independence, not a ceiling.
        "num_independent_groups": n_independent_groups,
        "num_images_evaluated": len(evaluated),
        "num_filenames_without_rf_suffix": no_rf,
        "num_background_images_raw": n_background,
        "background_fraction_raw": (n_background / len(filenames)) if filenames else 0.0,
        "variant_counts": variant_counts,
    }

    print(
        f"[evaluate] TEST split: {len(filenames)} images -> {len(groups)} unique source photos "
        f"(dedup={'ON' if args.dedup_sources else 'OFF'}, strategy={args.representative}, seed={args.seed})"
    )
    print(
        f"[evaluate] Background (unlabeled negative) images kept: {n_background} "
        f"({100.0 * n_background / len(filenames):.1f}% of the raw split)"
    )

    model = load_model(args.weights)

    with tempfile.TemporaryDirectory(prefix="argus_eval_", ignore_cleanup_errors=True) as tmp:
        work_dir = Path(tmp)

        print(f"[evaluate] Pass 1/{'3+' if args.dedup_sources else '2+'}: RAW per-image over all {len(filenames)} test images ...")
        raw_metrics = run_validation(
            model, args.data, args.imgsz, args.batch, args.device, args.nms_iou, work_dir, "val_raw"
        )
        raw_summary = metrics_summary(raw_metrics)

        if args.dedup_sources:
            print(f"[evaluate] Pass 2: DEDUPED over {len(representatives)} representatives (one per source photo) ...")
            dedup_yaml = write_subset_dataset(data_cfg, images_dir, representatives, work_dir, "dedup")
            headline_metrics = run_validation(
                model, dedup_yaml, args.imgsz, args.batch, args.device, args.nms_iou, work_dir, "val_dedup"
            )
        else:
            headline_metrics = raw_metrics

        metrics_by_source: dict = {}
        images_by_source: dict[str, int] = {}
        skipped: list[str] = []
        source_keys = sorted({sources[fn] for fn in evaluated})
        for src in source_keys:
            subset = [fn for fn in evaluated if sources[fn] == src]
            images_by_source[src] = len(subset)
            if src == "unknown" or len(subset) < MIN_SOURCE_SUBSET_IMAGES:
                skipped.append(src)
                continue
            print(f"[evaluate] Source-confound pass: '{src}' over {len(subset)} images ...")
            src_yaml = write_subset_dataset(data_cfg, images_dir, subset, work_dir, f"src_{src}")
            metrics_by_source[src] = run_validation(
                model, src_yaml, args.imgsz, args.batch, args.device, args.nms_iou, work_dir, f"val_{src}"
            )

        unknown_count = sum(1 for fn in evaluated if sources[fn] == "unknown")
        source_confound = build_source_confound(
            metrics_by_source, images_by_source, unknown_count, provenance_method, skipped
        )

        report = build_report(
            headline_metrics,
            class_names,
            args,
            dedup_block,
            raw_summary if args.dedup_sources else None,
            support,
            source_confound,
        )

    print_report(report, class_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[evaluate] Full report written to: {args.out}")


if __name__ == "__main__":
    main()
