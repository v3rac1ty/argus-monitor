"""GPU-resident dataset pipeline for fine-tuning the whole-frame classifier on
tinygrad's NV backend, over a Thunderbolt link whose host->GPU bandwidth is
only ~2.62 GB/s (no resizable BAR) even though on-card compute is unaffected.
The dataset is decoded from JPEG to uint8 numpy ONCE, moved to the GPU ONCE
(see `move_to_device_resident`), and kept resident in VRAM for the rest of
training; every augmentation is a tinygrad tensor op run on-device so the
slow host link never appears inside the training loop.

------------------------------------------------------------------------
THE most important correctness contract in this file
------------------------------------------------------------------------
`argus.detectors.classifier.preprocess_classify` is what actually runs at
inference time on the printer. It does, in order: resize short side ->
center crop -> BGR-to-RGB -> HWC-to-CHW -> divide by 255.0 -> add batch dim.
There is NO ImageNet mean/std normalization anywhere in that path.

Every function here that touches pixel values (`apply_augmentation`,
`normalize_to_model_input`) must produce output that is statistically
identical to what `preprocess_classify` produces: RGB channel order,
CHW layout, and a plain /255.0 scale with no mean subtraction or std
division. Diverging here -- even just leaving images in BGR order, or
sneaking in a mean/std normalization because "that's what most training
recipes do" -- would silently skew every confidence the deployed model
produces and make every configured `class_thresholds` value in config wrong,
without a single test or error to catch it. `normalize_to_model_input`'s
tests are the regression guard for the /255-only half of this contract; the
BGR->RGB channel flip is folded into `apply_augmentation`'s initial
HWC->CHW reshape (see the comment at that call site).

------------------------------------------------------------------------
Deliberate simplification: crop window is shared across the whole batch
------------------------------------------------------------------------
`apply_augmentation` samples ONE random-crop window per BATCH, not one per
sample. A true per-sample random crop needs a hand-built vectorized gather
(each sample reading from a different window of its own source image), which
is real implementation risk on this pinned tinygrad branch for a benefit --
crop diversity -- that epoch-to-epoch reshuffling (a fresh batch-shared crop
every step, over many epochs) already recovers most of. This is a conscious
trade-off, not an oversight: see the comment in `apply_augmentation`.

------------------------------------------------------------------------
Deliberate simplification: no resident float copy
------------------------------------------------------------------------
The dataset is kept resident on the GPU as uint8 (`move_to_device_resident`)
and NEVER as a resident float32 copy: uint8 -> float32 is a 4x memory blowup
for zero benefit, since every batch gets cast to float on the fly inside
`apply_augmentation` anyway. Do not "optimize" this by pre-casting the whole
resident tensor to float once -- that defeats the entire point of fitting
the dataset in the RTX 5060 Ti's 16 GB of VRAM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

# Run-from-anywhere bootstrap: this module reuses training/evaluate_classifier.py's
# list_split_images, which is only importable once the repo root is on sys.path
# (pytest already does this via pyproject.toml's pythonpath/rootdir handling; running
# `python training/tg_data.py`-style scripts, or `sys.path.insert(0, "training")` followed
# by `import tg_data`, does not).
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.evaluate_classifier import list_split_images  # noqa: E402  (reuse, see module docstring)

from tinygrad import Tensor, dtypes  # noqa: E402

# Fixed dataset layout contract: datasets/argus_bin/{split}/{class}/*.jpg. Class order is
# ALWAYS sorted() folder names -- never hardcode a different order (see discover_class_names).


# --------------------------------------------------------------------------
# Dataset discovery / loading (pure filesystem + numpy; no tinygrad).
# --------------------------------------------------------------------------


# tuple[str, ...] discover_class_names(Path data_dir, str split)
# Inputs: Path data_dir - classification dataset root (e.g. datasets/argus_bin)
#         str split - split name whose subdirectories name the classes, default "train"
# Outputs: tuple[str, ...] - class names in sorted() order (index 0 is "failure", index 1 is
#          "normal", per the fixed dataset contract) -- ALWAYS derived from sorted(), never a
#          hardcoded literal, so this stays correct if the class set ever changes
# Description: Lists the immediate subdirectories of data_dir/split and returns their names
#              sorted alphabetically. This sorted order is the class-index order used
#              everywhere else in this module (load_split_to_arrays, ClassBalancedSampler).
# Side Effects: Raises FileNotFoundError if data_dir/split doesn't exist, or if it exists but
#               contains no subdirectories. Read-only filesystem traversal otherwise.
def discover_class_names(data_dir: Path, split: str = "train") -> tuple[str, ...]:
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not names:
        raise FileNotFoundError(f"No class subdirectories found under {split_dir}")
    return tuple(names)


# tuple[np.ndarray, np.ndarray, list[str]] load_split_to_arrays(Path data_dir, str split, Sequence[str] class_names, int size)
# Inputs: Path data_dir - classification dataset root (e.g. datasets/argus_bin)
#         str split - split name, "train", "val", or "test"
#         Sequence[str] class_names - class names in index order (see discover_class_names) --
#                 index into this sequence becomes the integer label
#         int size - expected spatial size of every image (dataset contract: 512x512 already
#                 center-cropped JPEGs); a decoded image of a different size is a dataset
#                 contract violation and raises rather than being silently resized
# Outputs: tuple[np.ndarray, np.ndarray, list[str]] - (images, labels, filenames):
#          images is uint8 (N, size, size, 3) in BGR order (as cv2.imread produces -- NOT
#          converted to RGB here, see apply_augmentation for where that happens);
#          labels is int32 (N,) with labels[i] == class_names.index(<i's class>);
#          filenames is the list of N image basenames, aligned index-for-index with images/labels
# Description: Loads every image for `split` (via list_split_images, reused from
#              training.evaluate_classifier) into one dense in-memory uint8 array, in
#              deterministic (sorted-by-class-then-filename) order, ready to be handed to
#              move_to_device_resident once and never touched by the slow host->GPU link again.
# Side Effects: Reads every image file under data_dir/split/<class>/ from disk (decodes each
#               JPEG via cv2.imread). Raises FileNotFoundError if the split directory is missing
#               (via list_split_images) or if it contains zero images. Raises ValueError if a
#               file fails to decode, or decodes to a shape other than (size, size, 3).
def load_split_to_arrays(
    data_dir: Path, split: str, class_names: Sequence[str], size: int = 512
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = list_split_images(Path(data_dir), split, class_names)
    if not records:
        raise FileNotFoundError(
            f"No images found under {Path(data_dir) / split} for class_names={list(class_names)}"
        )

    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    images = np.empty((len(records), size, size, 3), dtype=np.uint8)
    labels = np.empty((len(records),), dtype=np.int32)
    filenames: list[str] = []

    for i, (path, class_name) in enumerate(records):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        if image.shape != (size, size, 3):
            raise ValueError(
                f"{path}: expected a ({size}, {size}, 3) image per the dataset contract, got "
                f"{image.shape} -- the dataset builder should already have center-cropped this"
            )
        images[i] = image
        labels[i] = name_to_index[class_name]
        filenames.append(path.name)

    return images, labels, filenames


# Tensor move_to_device_resident(np.ndarray images, Optional[str] device)
# Inputs: np.ndarray images - uint8 image batch/array to move onto the GPU, any shape (e.g. the
#                 full-split (N, size, size, 3) array from load_split_to_arrays)
#         Optional[str] device - tinygrad device string (e.g. "NV"), or None to use
#                 tinygrad's Device.DEFAULT
# Outputs: Tensor - a realized (materialized) tinygrad Tensor holding `images` resident on
#          `device`, still uint8
# Description: The one and only place the slow ~2.62 GB/s host->GPU Thunderbolt link should be
#              used for image data during training: decode once on the host (load_split_to_arrays),
#              move once (this function), then never again. Deliberately keeps the resident copy
#              uint8 rather than pre-casting to float32 -- that would quadruple the VRAM
#              footprint of the resident dataset for no benefit, since apply_augmentation casts
#              to float per-batch anyway. Raises if `images` is not uint8, since silently
#              accepting (and thereby residently storing) a float array here is exactly the
#              VRAM-quadrupling mistake this function exists to prevent.
# Side Effects: Allocates GPU (or other device) memory and copies `images` across the host<->device
#               link; blocks until the transfer and any pending realize() complete.
def move_to_device_resident(images: np.ndarray, device: Optional[str] = None) -> Tensor:
    if images.dtype != np.uint8:
        raise ValueError(
            f"move_to_device_resident expects uint8 images (got dtype={images.dtype}); keeping a "
            "resident float copy would quadruple VRAM usage for no benefit -- cast per batch in "
            "apply_augmentation instead"
        )
    tensor = Tensor(images, device=device) if device is not None else Tensor(images)
    return tensor.realize()


# --------------------------------------------------------------------------
# Class-balanced sampling (pure numpy -- deliberately decoupled from tinygrad
# so it is unit-testable without a GPU or even tinygrad importable).
# --------------------------------------------------------------------------


class ClassBalancedSampler:
    """Yields batches of global dataset indices with (as close to) equal
    per-class representation as an integer batch_size allows, regardless of
    how imbalanced the underlying `labels` are (this dataset is roughly 1487
    normal vs 684 failure). Balance is achieved by cycling deterministically
    through class indices 0..num_classes-1 as each slot in a batch is filled
    (so per-class draw frequency is exactly 1/num_classes over any window of
    draws that's a multiple of num_classes, and very close otherwise) while
    *which* example is drawn for a given class is independently shuffled per
    class and reshuffled whenever that class's pool is exhausted. Pure numpy
    and seeded via `np.random.default_rng`, so it is fully unit-testable
    without tinygrad or a GPU.
    """

    # None __init__(np.ndarray labels, int num_classes, int batch_size, int seed)
    # Inputs: np.ndarray labels - int labels for the whole dataset/split, shape (N,)
    #         int num_classes - number of distinct classes (labels must be in [0, num_classes))
    #         int batch_size - number of indices yielded per batch by __iter__
    #         int seed - seed for this sampler's own np.random.default_rng, so two samplers
    #                 built with the same (labels, num_classes, batch_size, seed) yield
    #                 identical index sequences
    # Outputs: None
    # Description: Builds one shuffled index pool per class (the set of global dataset indices
    #              belonging to that class) and a per-class draw cursor into that pool, plus a
    #              global round-robin counter that decides which class supplies the next drawn
    #              index.
    # Side Effects: Mutates the new instance's state. Raises ValueError if batch_size or
    #               num_classes is not positive, or if any class in [0, num_classes) has zero
    #               examples in `labels` (a class-balanced sampler cannot draw from an empty
    #               class).
    def __init__(self, labels: np.ndarray, num_classes: int, batch_size: int, seed: int) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

        self._num_classes = num_classes
        self._batch_size = batch_size
        self._n = int(labels.shape[0])
        self._rng = np.random.default_rng(seed)

        self._class_indices: list[np.ndarray] = []
        for c in range(num_classes):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                raise ValueError(f"ClassBalancedSampler: class {c} has zero examples in labels")
            self._class_indices.append(idx)

        self._pools: list[np.ndarray] = [self._rng.permutation(idx) for idx in self._class_indices]
        self._cursors: list[int] = [0] * num_classes
        self._draw_count = 0

    # int _draw_one(int class_id)
    # Inputs: int class_id - which class's pool to draw the next global index from
    # Outputs: int - a global dataset index belonging to `class_id`
    # Description: Pops the next index off class_id's shuffled pool, reshuffling that class's
    #              pool (a fresh np.random.permutation of its full index set) whenever the pool
    #              is exhausted -- this is the "reshuffle on exhaustion" behavior, applied
    #              independently per class so a minority class cycles through its own examples
    #              far more often than a majority class does.
    # Side Effects: Mutates self._pools[class_id] and self._cursors[class_id].
    def _draw_one(self, class_id: int) -> int:
        cursor = self._cursors[class_id]
        pool = self._pools[class_id]
        if cursor >= len(pool):
            pool = self._rng.permutation(self._class_indices[class_id])
            self._pools[class_id] = pool
            cursor = 0
        self._cursors[class_id] = cursor + 1
        return int(pool[cursor])

    # int steps_per_epoch()
    # Inputs: None (operates on self)
    # Outputs: int - number of batches __iter__ yields per call, floor(N / batch_size) with a
    #          floor of 1 so a dataset smaller than one batch still yields something
    # Description: Defines "one epoch" as roughly one pass over the dataset's total size N, for
    #              scheduling/logging purposes -- since sampling is class-balanced with
    #              per-class reshuffling rather than a single global shuffle, this is a nominal
    #              length rather than a guarantee every example is seen exactly once.
    # Side Effects: None
    def steps_per_epoch(self) -> int:
        return max(1, self._n // self._batch_size)

    # Iterator[np.ndarray] __iter__()
    # Inputs: None (operates on self)
    # Outputs: Iterator[np.ndarray] - yields steps_per_epoch() batches, each an int64 array of
    #          shape (batch_size,) of global dataset indices
    # Description: Fills each batch slot by round-robining through class ids 0..num_classes-1
    #              using a counter that persists across __iter__ calls (so consecutive epochs
    #              continue the round-robin and each class's pool continues shuffling/reshuffling
    #              rather than resetting), guaranteeing near-exact 1/num_classes per-class draw
    #              frequency regardless of the underlying class imbalance.
    # Side Effects: Advances self._draw_count and the per-class pools/cursors (via _draw_one).
    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.steps_per_epoch()):
            batch = np.empty(self._batch_size, dtype=np.int64)
            for i in range(self._batch_size):
                class_id = self._draw_count % self._num_classes
                self._draw_count += 1
                batch[i] = self._draw_one(class_id)
            yield batch


# --------------------------------------------------------------------------
# Augmentation parameter sampling (PURE NUMPY, no tinygrad import at all --
# this seam is what makes augmentation deterministic and unit-testable
# without a GPU; see apply_augmentation for the tinygrad half).
# --------------------------------------------------------------------------


# tuple[float, float] -- module-level type alias comment for readability only.
_Range = tuple[float, float]


@dataclass(frozen=True)
class AugConfig:
    """Ranges/probabilities for one batch's sampled augmentation. Defaults
    are moderate (kept mild enough that a 'failure' image is never
    augmented into looking like a plausible 'normal' one, or vice versa):
    a mild random crop/zoom, standard L/R flip, mild photometric jitter, an
    occasional cutout patch, and rare/light blur or Gaussian pixel noise to
    approximate camera-focus and sensor-noise variation.
    """

    #: Fraction of the source image's side kept by the shared per-batch random crop before it
    #: is resized up to train_size. 1.0 means no crop (use the full frame).
    crop_scale_range: _Range = (0.75, 1.0)
    #: Probability that any given sample in the batch is horizontally flipped.
    hflip_prob: float = 0.5
    #: Per-sample multiplicative brightness jitter range (1.0 == unchanged).
    brightness_range: _Range = (0.8, 1.2)
    #: Per-sample multiplicative contrast jitter range (1.0 == unchanged), applied about the
    #: sample's own mean pixel value.
    contrast_range: _Range = (0.8, 1.2)
    #: Per-sample multiplicative saturation jitter range (1.0 == unchanged; 0.0 == grayscale),
    #: applied by blending toward the sample's own per-pixel channel mean.
    saturation_range: _Range = (0.8, 1.2)
    #: Probability that any given sample gets one cutout patch (a solid block zeroed out).
    cutout_prob: float = 0.3
    #: Cutout patch side length, as a fraction of train_size.
    cutout_size_range: _Range = (0.05, 0.25)
    #: Probability that any given sample is box-blurred (fixed 3x3 kernel).
    blur_prob: float = 0.1
    #: Probability that any given sample gets additive Gaussian pixel noise.
    noise_prob: float = 0.1
    #: Additive Gaussian noise std-dev range, in 0..255 pixel-value units.
    noise_std_range: _Range = (0.0, 8.0)

    # None __post_init__()
    # Inputs: None (operates on self; validates its own range/probability fields)
    # Outputs: None
    # Description: Fails loudly on a malformed config (a reversed range, or a probability
    #              outside [0, 1]) at construction time rather than producing silently-wrong
    #              samples later inside sample_augmentation_params.
    # Side Effects: None (read-only validation of self's own fields).
    def __post_init__(self) -> None:
        for name in ("crop_scale_range", "brightness_range", "contrast_range", "saturation_range",
                     "cutout_size_range", "noise_std_range"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"AugConfig.{name} must be (low <= high), got {(lo, hi)}")
        for name in ("hflip_prob", "cutout_prob", "blur_prob", "noise_prob"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"AugConfig.{name} must be in [0, 1], got {p}")


@dataclass
class AugParams:
    """Sampled numpy arrays/scalars for exactly one batch, produced by
    `sample_augmentation_params` and consumed by `apply_augmentation`. The
    crop_* fields are batch-shared scalars (see the module docstring's
    per-batch-crop trade-off); everything else is per-sample, shape (B,).
    """

    #: Shared crop window (see apply_augmentation): fraction of source side kept, and the
    #: fractional offset (0 == top/left-aligned, 1 == bottom/right-aligned) of that window.
    crop_scale: float
    crop_top_frac: float
    crop_left_frac: float

    #: Per-sample bool (B,): whether to horizontally flip that sample.
    hflip: np.ndarray
    #: Per-sample float32 (B,): brightness/contrast/saturation multipliers.
    brightness: np.ndarray
    contrast: np.ndarray
    saturation: np.ndarray

    #: Per-sample cutout: whether to apply it, its center (as a fraction of train_size in
    #: [0, 1]), and its half-size (as a fraction of train_size).
    cutout_apply: np.ndarray
    cutout_cy_frac: np.ndarray
    cutout_cx_frac: np.ndarray
    cutout_half_frac: np.ndarray

    #: Per-sample bool (B,): whether to blur / add noise, and noise's std-dev (pixel units).
    blur_apply: np.ndarray
    noise_apply: np.ndarray
    noise_std: np.ndarray


# AugParams sample_augmentation_params(np.random.Generator rng, int batch_size, AugConfig cfg)
# Inputs: np.random.Generator rng - caller-owned, caller-seeded RNG (so callers control
#                 reproducibility explicitly rather than this function reseeding globally)
#         int batch_size - number of per-sample augmentation params to draw
#         AugConfig cfg - ranges/probabilities to draw from
# Outputs: AugParams - one batch's worth of sampled augmentation parameters, ready to hand to
#          apply_augmentation
# Description: Draws every random quantity `apply_augmentation` needs, entirely in numpy (no
#              tinygrad import anywhere in this function or its call graph) so augmentation
#              parameters are deterministic and unit-testable without a GPU: the same
#              (rng state, batch_size, cfg) always produces the same AugParams.
# Side Effects: Advances `rng`'s internal state (consumes random draws from it).
def sample_augmentation_params(rng: np.random.Generator, batch_size: int, cfg: AugConfig) -> AugParams:
    crop_scale = float(rng.uniform(cfg.crop_scale_range[0], cfg.crop_scale_range[1]))
    crop_top_frac = float(rng.uniform(0.0, 1.0))
    crop_left_frac = float(rng.uniform(0.0, 1.0))

    hflip = rng.random(batch_size) < cfg.hflip_prob
    brightness = rng.uniform(cfg.brightness_range[0], cfg.brightness_range[1], size=batch_size).astype(np.float32)
    contrast = rng.uniform(cfg.contrast_range[0], cfg.contrast_range[1], size=batch_size).astype(np.float32)
    saturation = rng.uniform(cfg.saturation_range[0], cfg.saturation_range[1], size=batch_size).astype(np.float32)

    cutout_apply = rng.random(batch_size) < cfg.cutout_prob
    cutout_cy_frac = rng.uniform(0.0, 1.0, size=batch_size).astype(np.float32)
    cutout_cx_frac = rng.uniform(0.0, 1.0, size=batch_size).astype(np.float32)
    cutout_half_frac = (
        rng.uniform(cfg.cutout_size_range[0], cfg.cutout_size_range[1], size=batch_size) / 2.0
    ).astype(np.float32)

    blur_apply = rng.random(batch_size) < cfg.blur_prob
    noise_apply = rng.random(batch_size) < cfg.noise_prob
    noise_std = rng.uniform(cfg.noise_std_range[0], cfg.noise_std_range[1], size=batch_size).astype(np.float32)

    return AugParams(
        crop_scale=crop_scale,
        crop_top_frac=crop_top_frac,
        crop_left_frac=crop_left_frac,
        hflip=hflip,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        cutout_apply=cutout_apply,
        cutout_cy_frac=cutout_cy_frac,
        cutout_cx_frac=cutout_cx_frac,
        cutout_half_frac=cutout_half_frac,
        blur_apply=blur_apply,
        noise_apply=noise_apply,
        noise_std=noise_std,
    )


# --------------------------------------------------------------------------
# The tinygrad half: GPU-resident augmentation + inference-matching
# normalization. Every op here runs on whatever device `images` already
# lives on (e.g. "NV"), so no host<->GPU traffic happens per batch.
# --------------------------------------------------------------------------


# Tensor apply_augmentation(Tensor images, AugParams params, int train_size)
# Inputs: Tensor images - a batch resident on the training device, uint8 or float, shape
#                 (B, H, W, 3), channel order BGR (as produced by load_split_to_arrays /
#                 move_to_device_resident -- cv2's native order)
#         AugParams params - one batch's sampled augmentation parameters, from
#                 sample_augmentation_params
#         int train_size - the square spatial size the model trains at (may differ from the
#                 source images' H/W -- the crop+interpolate step below reconciles that)
# Outputs: Tensor - float32, shape (B, 3, train_size, train_size), RGB channel order, CHW
#          layout, pixel values still in roughly [0, 255] (NOT yet divided by 255 -- that is
#          normalize_to_model_input's job, kept separate so augmentation always operates in
#          pixel-value space)
# Description: GPU-side augmentation pipeline, entirely tinygrad tensor ops (no numpy/host
#              round-trip once `images` is already resident): (1) HWC->CHW + BGR->RGB (folded
#              into one step: flipping the channel axis of a 3-channel BGR tensor produces
#              exactly RGB, since the middle "G" channel is fixed and only B/R swap -- this is
#              the training-side half of the "no ImageNet normalization, but channel order and
#              scale must still match preprocess_classify" correctness contract described in
#              this module's docstring); (2) a random crop window SHARED ACROSS THE WHOLE BATCH
#              (see the module docstring's "Deliberate simplification" section -- true per-sample
#              crop would need a hand-built vectorized gather, which is real risk on this pinned
#              tinygrad branch for a diversity benefit epoch reshuffling largely recovers
#              anyway), via Tensor.shrink; (3) Tensor.interpolate up/down to train_size; (4)
#              per-sample horizontal flip via a broadcast boolean mask and Tensor.where; (5)
#              per-sample brightness/contrast/saturation jitter via broadcast per-sample scalars;
#              (6) per-sample cutout via broadcast coordinate-comparison masks (no gather
#              needed: every pixel compares its own (row, col) against that sample's cutout box);
#              (7) optional per-sample box blur (Tensor.avg_pool2d) and additive Gaussian noise,
#              each gated by a per-sample boolean mask.
# Side Effects: None beyond ordinary tinygrad lazy-graph construction; the caller is expected to
#               call .realize() when the result is actually needed (this function does not).
def apply_augmentation(images: Tensor, params: AugParams, train_size: int) -> Tensor:
    if len(images.shape) != 4 or images.shape[3] != 3:
        raise ValueError(f"apply_augmentation expects (B, H, W, 3) images, got shape {images.shape}")

    batch_size, height, width = images.shape[0], images.shape[1], images.shape[2]

    # (1) HWC -> CHW, and BGR -> RGB in the same step: for a 3-channel image, reversing the
    # channel axis turns [B, G, R] into [R, G, B] -- exactly a BGR->RGB swap, since G (the
    # middle channel) maps to itself. Doing it here means every function downstream of this
    # line already matches preprocess_classify's channel order.
    x = images.permute(0, 3, 1, 2).cast(dtypes.float32).flip(1)

    # (2) Shared-per-batch random crop window (see module docstring: deliberately NOT
    # per-sample). crop_scale/top_frac/left_frac come from AugParams as plain Python floats,
    # so the actual pixel window is resolved here against this call's real H/W.
    crop_h = min(height, max(1, int(round(params.crop_scale * height))))
    crop_w = min(width, max(1, int(round(params.crop_scale * width))))
    top = int(round(params.crop_top_frac * (height - crop_h)))
    left = int(round(params.crop_left_frac * (width - crop_w)))
    x = x.shrink((None, None, (top, top + crop_h), (left, left + crop_w)))

    # (3) Resize the (possibly non-train_size) crop up/down to train_size.
    x = x.interpolate((train_size, train_size), mode="linear")

    device = x.device

    # (4) Per-sample horizontal flip.
    hflip_mask = Tensor(params.hflip.reshape(batch_size, 1, 1, 1), device=device)
    x = hflip_mask.where(x.flip(-1), x)

    # (5) Per-sample brightness / contrast / saturation jitter.
    brightness = Tensor(params.brightness.reshape(batch_size, 1, 1, 1), device=device)
    x = x * brightness

    contrast = Tensor(params.contrast.reshape(batch_size, 1, 1, 1), device=device)
    per_sample_mean = x.mean(axis=(1, 2, 3), keepdim=True)
    x = (x - per_sample_mean) * contrast + per_sample_mean

    saturation = Tensor(params.saturation.reshape(batch_size, 1, 1, 1), device=device)
    per_pixel_gray = x.mean(axis=1, keepdim=True)  # (B, 1, train_size, train_size)
    x = per_pixel_gray + (x - per_pixel_gray) * saturation

    x = x.clip(0.0, 255.0)

    # (6) Per-sample cutout via broadcast coordinate-comparison masks -- no gather needed:
    # every pixel's own (row, col) is compared against that SAMPLE's cutout box, and
    # broadcasting does the rest.
    rows = Tensor(np.arange(train_size, dtype=np.float32).reshape(1, 1, train_size, 1), device=device)
    cols = Tensor(np.arange(train_size, dtype=np.float32).reshape(1, 1, 1, train_size), device=device)
    cutout_cy = Tensor((params.cutout_cy_frac * train_size).reshape(batch_size, 1, 1, 1), device=device)
    cutout_cx = Tensor((params.cutout_cx_frac * train_size).reshape(batch_size, 1, 1, 1), device=device)
    cutout_half = Tensor((params.cutout_half_frac * train_size).reshape(batch_size, 1, 1, 1), device=device)
    cutout_apply = Tensor(params.cutout_apply.reshape(batch_size, 1, 1, 1), device=device)
    in_cutout_box = (
        (rows >= cutout_cy - cutout_half)
        & (rows < cutout_cy + cutout_half)
        & (cols >= cutout_cx - cutout_half)
        & (cols < cutout_cx + cutout_half)
    )
    cutout_mask = cutout_apply & in_cutout_box
    x = cutout_mask.where(Tensor.zeros_like(x), x)

    # (7) Optional per-sample box blur.
    blurred = x.avg_pool2d(kernel_size=(3, 3), stride=1, padding=1)
    blur_apply = Tensor(params.blur_apply.reshape(batch_size, 1, 1, 1), device=device)
    x = blur_apply.where(blurred, x)

    # (7 cont'd) Optional per-sample additive Gaussian noise, in pixel-value units.
    noise_std = Tensor(params.noise_std.reshape(batch_size, 1, 1, 1), device=device)
    noise_apply = Tensor(params.noise_apply.reshape(batch_size, 1, 1, 1), device=device)
    noise = Tensor.randn(batch_size, 1, train_size, train_size, device=device) * noise_std
    x = noise_apply.where(x + noise, x)

    return x.clip(0.0, 255.0)


# Tensor normalize_to_model_input(Tensor images)
# Inputs: Tensor images - a batch already in the layout preprocess_classify expects (RGB, CHW),
#                 pixel values in roughly [0, 255], uint8 or float, any device
# Outputs: Tensor - float32, same shape, values divided by 255.0 -- NO mean/std normalization
# Description: The single scaling step that MUST exactly match preprocess_classify's final
#              `/ 255.0`. Deliberately does nothing else: no ImageNet mean subtraction, no std
#              division, no clipping (input is assumed already clipped to [0, 255] by
#              apply_augmentation or by being raw uint8 pixel data). This is the regression
#              surface for this module's most important correctness contract -- see the module
#              docstring -- so resist the urge to "improve" this function.
# Side Effects: None (pure function of its input; does not call .realize()).
def normalize_to_model_input(images: Tensor) -> Tensor:
    return images.cast(dtypes.float32) / 255.0
