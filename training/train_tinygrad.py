"""Trains the binary (``failure``/``normal``) whole-frame classifier on tinygrad's NV
backend. Both splits are decoded to HOST numpy ONCE (``training/tg_data.py``'s
``load_split_to_arrays``) and stay there for the rest of the run -- every training step
and every validation chunk slices a small piece out of that host array and builds a
FRESH, small ``Tensor`` for just that slice (batch 32 @ 512x512x3 uint8 is ~25 MB). This
file deliberately does NOT hold either split resident on the GPU: an earlier version
that moved the whole 1.2 GB train / 260 MB val split to VRAM once via
``tg_data.move_to_device_resident`` reliably crashed this eGPU's virtual-memory
allocator partway into training (``AssertionError: PTE already mapped`` in
``tinygrad/runtime/support/memory.py``, via ``Buffer.allocate -> NVAllocator._alloc ->
system.py alloc -> memory.py valloc -> map_range``), almost certainly because this
card's Thunderbolt BAR1 window is only 256 MB (no resizable BAR over Thunderbolt) -- see
fact 8 below. Streaming per batch instead of once up front costs roughly 2% of the
measured per-step time (a ~25 MB transfer at the measured ~2.62 GB/s is ~10 ms against a
449 ms/step), so this is not a meaningful speed trade-off.

Usage:
    DEV=NV python training/train_tinygrad.py --epochs 1 --batch 32 --no-pretrained   # smoke test
    DEV=NV python training/train_tinygrad.py                                          # real run

------------------------------------------------------------------------
HARD-WON ENVIRONMENT FACTS (measured on this machine's RTX 5060 Ti eGPU, not guesses)
------------------------------------------------------------------------
1. TRAIN IN FP32. DO NOT set ``Context(DEFAULT_FLOAT=dtypes.float16)`` or otherwise push
   fp16 activations into the backward pass anywhere in this file. fp16 activations
   reaching backward fail on this tinygrad build with
   ``ValueError: Buffer size too small (0 instead of at least N bytes)`` -- confirmed to
   be a tinygrad bug (upstream ``extra/models/resnet.py`` fails identically under fp16
   backward on this same build), not something fixable from caller code. Forward-only
   fp16 inference works fine; fp16 training does not. Do not retry this.
2. The train step is wrapped in ``TinyJit`` -- measured ~2x speedup, because this eGPU
   is kernel-dispatch bound over Thunderbolt (many small op launches cost more in
   round-trip latency than compute). JIT requires a fixed batch shape every call, which
   ``ClassBalancedSampler`` guarantees (it always yields exactly ``batch_size`` indices).
   Deliberately NOT jitted: the validation pass (runs once per epoch, JIT's win doesn't
   apply, and its last batch is a ragged remainder so its shape isn't even fixed); and
   ``apply_augmentation``/``sample_augmentation_params`` (see below).
3. ``Context(TRAINING=1)`` is hoisted OUTSIDE the whole per-step loop (entered once for
   the entire training run), not entered fresh per step -- measured much slower the
   other way. Validation runs under ``Context(TRAINING=0)``, which has been empirically
   verified on this build to correctly switch BatchNorm to its running stats (eval-mode
   output mean matched the running stats to 3 decimal places), so this is safe.
4. Measured throughput at fp32/imgsz=320 with JIT: batch 32 = 449 ms/step (71 img/s);
   batch 64 = 973 ms/step. Batch 32 (this script's default) is both the fastest
   per-image AND the conservative choice. JIT warmup (first step, tracing + compiling
   the kernel graph) costs roughly 33s on top of steady-state per-step time.
5. NEVER exceed batch 64 (enforced by ``parse_args``, see MAX_BATCH_SIZE). Batch 128 at
   imgsz 320 previously wedged this eGPU's driver badly enough to require physically
   unplugging the Thunderbolt cable. Do not probe upward from 64.
6. Only one tinygrad ``DEV=NV`` process at a time -- concurrent runs collide with
   ``Failed to acquire lock file`` or spurious ``Buffer size too small`` errors.
7. ``dtypes.default_float`` has no setter on this build (read-only property); this file
   never needs to touch it since training stays fp32 throughout (see fact 1).
8. NEVER allocate a single GPU tensor larger than ~100 MB. A resident whole-split tensor
   (1.2 GB train, 260 MB val, uint8) is what produced the ``PTE already mapped`` crash
   described above. Every ``Tensor`` this file builds from a host numpy slice -- the
   per-step train batch, and each validation chunk in ``_evaluate_macro_f1`` -- must stay
   well under that ceiling. Do not reintroduce ``tg_data.move_to_device_resident`` (or
   any equivalent full-split upload) here.

------------------------------------------------------------------------
Design: where the TinyJit boundary is drawn, and why
------------------------------------------------------------------------
``training/tg_data.py``'s ``apply_augmentation``/``sample_augmentation_params`` are
deliberately NOT called inside the jitted train step. ``sample_augmentation_params``
draws fresh values from a host-side ``numpy.random.Generator`` every batch, and
``apply_augmentation`` wraps each of those numpy arrays in a brand-new ``Tensor(...)``
every call. If that construction happened *inside* a ``TinyJit``-wrapped function body,
those per-call Python-side values would only be captured correctly on tinygrad's own
native random ops (``Tensor.rand``/``Tensor.randint``, which JIT is specifically built to
re-execute with fresh draws on replay -- see tinygrad's own
``examples/hlb_cifar10.py``'s jitted ``augmentations()``); host-numpy-sourced tensors
constructed fresh from closured Python state on every call have no such guarantee and
risk silently baking in the FIRST call's random values as a constant for the rest of
training. So augmentation runs eagerly (a handful of tinygrad ops per batch, not the
dispatch-heavy part of a step) and its *output* -- an already-realized, fixed-shape
float32 tensor -- is what gets passed as a genuine argument into the jitted step, the
same way ``hlb_cifar10`` passes its own already-fetched ``X, Y`` batches into
``train_step_jitted``. The learning-rate and EMA-decay schedules are likewise computed
in plain Python each step and handed in as small ``Tensor`` arguments (mirroring how
``hlb_cifar10``'s ``modelEMA.update`` takes its decay as a ``Tensor`` argument) rather
than baked in as Python-float constants, so ``TinyJit`` correctly replays the compiled
step graph with a genuinely different learning rate / decay every call.

------------------------------------------------------------------------
Design: discriminative LR, warmup+cosine, label smoothing, class balance, EMA
------------------------------------------------------------------------
The backbone (ImageNet-pretrained) and the freshly-initialized 2-class ``fc`` head use
separate ``tinygrad.nn.optim.AdamW`` instances combined via ``OptimizerGroup`` (the
pattern tinygrad's own ``examples/hlb_cifar10.py`` uses for its bias/non-bias LR split),
so the head can learn faster than the pretrained backbone is nudged. Both LRs follow a
warmup+cosine schedule (``cosine_lr_with_warmup``). The training set is imbalanced
(1102 normal vs 445 failure), so batches are drawn via ``tg_data.ClassBalancedSampler``
rather than a plain shuffle. Label smoothing softens the binary cross-entropy target.

Weight EMA uses an ADAPTIVE decay schedule (``ema_decay_at_step``), not the fixed
``--ema-decay`` target directly: ``min(target, (1+step)/(10+step))``. A fixed 0.999 has
a ~1000-step time constant, but at ~48 steps/epoch x 40 epochs this run is only ~1900
steps total -- a fixed 0.999 EMA would still be dominated by the very first few epochs'
weights by the time training ends. The adaptive schedule starts low (heavily weighting
recent updates while the model is far from converged) and rises toward the target as
training progresses, the same warmup-EMA trick used in torchvision's/timm's EMA
implementations and tinygrad's own ``hlb_cifar10`` example.

BatchNorm's own ``running_mean``/``running_var``/``num_batches_tracked`` buffers are
copied directly from the live model every step rather than decay-blended (mirroring
``hlb_cifar10``'s ``modelEMA.update``, which excludes exactly these from its EMA math):
those buffers are already an exponential average of *activations* (see
``tg_models.BatchNorm``), so re-averaging them a second time under a *different* decay
schedule is redundant at best; and ``num_batches_tracked`` is a 0-d integer counter, not
a weight -- multiplying it by a decay tensor doesn't broadcast (it has no shape to
broadcast into) and wouldn't be meaningful even if it did.

Validation (and therefore early stopping / checkpointing) evaluates the EMA model, not
the raw model being optimized -- EMA weights are the actual deployment target, so val
macro-F1 on anything else would be measuring the wrong model.

------------------------------------------------------------------------
CHECKPOINT CONTRACT -- pinned, training/export_tinygrad_onnx.py reads exactly this
------------------------------------------------------------------------
``tinygrad.nn.state.safe_save(tensors, path, metadata=...)`` where ``tensors`` is
``get_state_dict(ema_model)`` (torchvision-identical key names, see ``tg_models.py``)
and ``metadata`` is ``build_checkpoint_metadata``'s ``str -> str`` dict: ``class_names``
(comma-separated, model index order), ``arch``, ``imgsz``, ``num_classes``, ``seed``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

# Run-from-anywhere bootstrap (matching training/tg_data.py and
# training/export_tinygrad_onnx.py): this module reuses training.tg_data,
# training.tg_models and training.evaluate_classifier, which are only importable once
# the repo root is on sys.path -- pytest already does this via pyproject.toml's
# pythonpath/rootdir handling; `python training/train_tinygrad.py` does not.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tinygrad import Tensor, TinyJit  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402
from tinygrad.nn.optim import AdamW, OptimizerGroup  # noqa: E402
from tinygrad.nn.state import get_state_dict, safe_save  # noqa: E402

from training.evaluate_classifier import build_confusion_matrix, per_class_prf1  # noqa: E402
from training.tg_data import (  # noqa: E402
    AugConfig,
    AugParams,
    ClassBalancedSampler,
    apply_augmentation,
    discover_class_names,
    load_split_to_arrays,
    normalize_to_model_input,
    sample_augmentation_params,
)
from training.tg_models import build_model  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "argus_bin"
DEFAULT_OUT_PATH = REPO_ROOT / "runs" / "train_tinygrad" / "cls_bin_v1" / "weights" / "best.safetensors"

#: The dataset builder (training/build_binary_dataset.py) already center-crops every
#: image to this fixed square size -- see tg_data.load_split_to_arrays' contract.
DATASET_IMAGE_SIZE = 512

#: Batch 128 at imgsz 320 previously wedged this eGPU's driver hard enough to require
#: physically unplugging the Thunderbolt cable -- see module docstring fact 5.
MAX_BATCH_SIZE = 64

#: safetensors metadata is str -> str only; keys pinned by training/export_tinygrad_onnx.py.
_HEAD_PARAM_PREFIX = "fc."

#: State-dict key substrings identifying BatchNorm's own bookkeeping buffers (running
#: stats + the batch counter) -- copied directly from the live model into the EMA model
#: every step rather than decay-blended, see module docstring.
_EMA_DIRECT_COPY_SUBSTRINGS = ("running_mean", "running_var", "num_batches_tracked")


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed training options: data, arch, epochs, batch, imgsz, seed,
#          patience, backbone_lr, head_lr, weight_decay, label_smoothing, warmup_epochs, ema_decay,
#          out, pretrained (True unless --no-pretrained is passed)
# Description: Defines and parses this script's CLI. Enforces MAX_BATCH_SIZE (64) on --batch here,
#              at parse time, rather than deep inside train() -- a batch size that could wedge the
#              eGPU's driver (see module docstring fact 5) must fail before any dataset loading or
#              GPU allocation happens, not after.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or GPU/network activity). Raises ValueError if --batch exceeds
#               MAX_BATCH_SIZE.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help=f"Binary classification dataset root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--arch", type=str, default="resnet18", help="Backbone architecture (default: resnet18; see training/tg_models.py for valid options)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32, help=f"Batch size (default: 32; hard max {MAX_BATCH_SIZE}, see module docstring fact 5)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience, in epochs with no val macro-F1 improvement")
    parser.add_argument("--backbone-lr", type=float, default=3e-4, help="Peak LR for the pretrained backbone (default: 3e-4)")
    parser.add_argument("--head-lr", type=float, default=3e-3, help="Peak LR for the freshly-initialized fc head (default: 3e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Epochs of linear LR warmup before the cosine decay begins")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Target EMA decay (see ema_decay_at_step's adaptive schedule -- this is a ceiling, not the decay used from step 0)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Checkpoint output path (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True, help="Skip ImageNet-pretrained init (for smoke tests -- no network access needed)")
    args = parser.parse_args(argv)

    if args.batch > MAX_BATCH_SIZE:
        raise ValueError(
            f"--batch {args.batch} exceeds the maximum of {MAX_BATCH_SIZE}. Batch 128 at imgsz 320 "
            "previously wedged this eGPU's driver hard enough to require physically unplugging the "
            "Thunderbolt cable -- do not raise this above 64."
        )
    return args


# None assert_nv_device(str actual_device)
# Inputs: str actual_device - a REALIZED tensor's .device string (e.g. Tensor.zeros(1).realize().device)
#         -- NOT tinygrad.Device.DEFAULT, which can report a device tinygrad never actually ran a
#         kernel on
# Outputs: None
# Description: Fails loudly, in milliseconds and before any image decoding, if training is not
#              actually running on tinygrad's NV backend (the RTX 5060 Ti). Without DEV=NV set in
#              the environment, tinygrad's default device on this machine resolves to METAL (the
#              MacBook's own GPU) -- training would silently run on the wrong hardware, producing a
#              real but meaningless checkpoint, unless this check catches it first.
# Side Effects: None (pure function of its input) beyond raising.
def assert_nv_device(actual_device: str) -> None:
    if actual_device != "NV":
        raise RuntimeError(
            f"tinygrad is running on device '{actual_device}', not 'NV'. Without DEV=NV set in the "
            "environment, tinygrad's default device on this machine resolves to whatever GPU backend "
            "is available locally (e.g. METAL on a Mac) -- training would silently run on the wrong "
            "hardware. Re-run with `DEV=NV python training/train_tinygrad.py ...`."
        )


# float cosine_lr_with_warmup(int step, int total_steps, int warmup_steps, float peak_lr)
# Inputs: int step - current global training step, 0-indexed
#         int total_steps - total number of training steps in the run
#         int warmup_steps - number of initial steps spent in linear warmup
#         float peak_lr - the learning rate reached at the end of warmup (and the start of decay)
# Outputs: float - the learning rate for this step: 0.0 at step 0 (when warmup_steps > 0), rising
#          linearly to peak_lr at step == warmup_steps, then following a cosine decay down to ~0.0
#          by step == total_steps
# Description: Pure LR-schedule function (no tinygrad, no state) so it's trivially unit-testable.
#              Linear warmup for step < warmup_steps (0 at step 0, peak_lr at the warmup boundary),
#              then a half-cosine decay from peak_lr to 0 over the remaining steps. Clamps decay
#              progress to [0, 1] so calling with step >= total_steps still returns a sane value
#              (0.0) rather than a negative or >peak_lr number.
# Side Effects: None
def cosine_lr_with_warmup(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    decay_span = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_span, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# float ema_decay_at_step(int step, float target)
# Inputs: int step - current global training step, 0-indexed
#         float target - the ceiling EMA decay this schedule rises toward, default 0.999
# Outputs: float - min(target, (1 + step) / (10 + step)): starts well below target (0.1 at step 0)
#          and rises monotonically toward target, bounded above by it
# Description: A fixed decay of 0.999 has a ~1000-step time constant; a short run (this project's
#              ~1900 total steps) would leave such an EMA dominated by its first few epochs' weights
#              by the time training ends. This adaptive schedule (the standard EMA-warmup trick, as
#              used in tinygrad's own examples/hlb_cifar10.py) starts responsive and only approaches
#              the fixed target asymptotically, so the EMA is never anchored to a barely-trained
#              initial model.
# Side Effects: None
def ema_decay_at_step(step: int, target: float = 0.999) -> float:
    return min(target, (1 + step) / (10 + step))


# float macro_f1(Sequence[str] y_true, Sequence[str] y_pred, Sequence[str] class_names)
# Inputs: Sequence[str] y_true - true class name per sample
#         Sequence[str] y_pred - predicted class name per sample, same length as y_true
#         Sequence[str] class_names - class names defining which classes are averaged over
# Outputs: float - the unweighted mean of per-class F1 scores across class_names
# Description: Composes training/evaluate_classifier.py's build_confusion_matrix and
#              per_class_prf1 (reused, not reimplemented) into the single macro-F1 number this
#              script uses for early stopping and checkpointing. Macro (not micro/weighted) so the
#              minority "failure" class's performance can't be masked by the majority "normal"
#              class the way overall accuracy would.
# Side Effects: None
def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], class_names: Sequence[str]) -> float:
    cm = build_confusion_matrix(y_true, y_pred, class_names)
    per_class = per_class_prf1(cm, class_names)
    f1_scores = [per_class[name]["f1"] for name in class_names]
    return float(sum(f1_scores) / len(f1_scores))


# dict[str, str] build_checkpoint_metadata(Sequence[str] class_names, str arch, int imgsz, int seed)
# Inputs: Sequence[str] class_names - class names in model output index order, e.g. ("failure", "normal")
#         str arch - architecture name, e.g. "resnet18"
#         int imgsz - square input spatial size the model was trained at
#         int seed - the training run's random seed
# Outputs: dict[str, str] - safetensors metadata (str -> str only, per the safetensors format):
#          class_names (comma-separated), arch, imgsz, num_classes, seed
# Description: Builds the exact metadata dict training/export_tinygrad_onnx.py's load_checkpoint
#              requires (see that module's pinned checkpoint contract and this module's docstring).
#              Every value is stringified here -- safetensors metadata cannot hold anything else --
#              so callers never have to remember to do that themselves.
# Side Effects: None
def build_checkpoint_metadata(class_names: Sequence[str], arch: str, imgsz: int, seed: int) -> dict[str, str]:
    return {
        "class_names": ",".join(class_names),
        "arch": str(arch),
        "imgsz": str(imgsz),
        "num_classes": str(len(class_names)),
        "seed": str(seed),
    }


# AugParams _deterministic_eval_params(int batch_size)
# Inputs: int batch_size - number of samples in the validation batch this call is for
# Outputs: AugParams - a fully "neutral" set of augmentation parameters: no crop (full frame kept),
#          no flip, brightness/contrast/saturation multipliers all 1.0, no cutout, no blur, no noise
# Outputs (cont'd): passing this to apply_augmentation reduces it to exactly HWC->CHW + BGR->RGB +
#          resize-to-train_size -- since this dataset's source images are already square
#          (DATASET_IMAGE_SIZE x DATASET_IMAGE_SIZE, per tg_data's dataset contract), that resize is
#          equivalent to argus.detectors.classifier.preprocess_classify's resize-short-side +
#          center-crop on an already-square input (both degenerate to a plain resize)
# Description: apply_augmentation (tg_data.py, reused not reimplemented) is the only available
#              tinygrad-side HWC->CHW/BGR->RGB/resize path, but it always requires an AugParams --
#              there is no separate "no augmentation" code path. This builds the neutral parameters
#              that make it deterministic, for use during validation (where random augmentation
#              would make macro-F1 noisy and non-reproducible across epochs).
# Side Effects: None
def _deterministic_eval_params(batch_size: int) -> AugParams:
    zeros_f = np.zeros(batch_size, dtype=np.float32)
    zeros_b = np.zeros(batch_size, dtype=bool)
    ones_f = np.ones(batch_size, dtype=np.float32)
    return AugParams(
        crop_scale=1.0,
        crop_top_frac=0.0,
        crop_left_frac=0.0,
        hflip=zeros_b,
        brightness=ones_f,
        contrast=ones_f,
        saturation=ones_f,
        cutout_apply=zeros_b,
        cutout_cy_frac=zeros_f,
        cutout_cx_frac=zeros_f,
        cutout_half_frac=zeros_f,
        blur_apply=zeros_b,
        noise_apply=zeros_b,
        noise_std=zeros_f,
    )


# float _evaluate_macro_f1(object model, np.ndarray images_np, np.ndarray labels, Sequence[str] class_names, int imgsz, int batch_size)
# Inputs: object model - a tg_models ResNet in evaluation use (called under Context(TRAINING=0) by
#         this function itself)
#         np.ndarray images_np - HOST uint8 (N, H, W, 3) images (e.g. the val split, straight from
#         load_split_to_arrays) -- never moved to the GPU as a whole; only one batch_size-sized
#         chunk at a time is uploaded (see Description and module docstring fact 8)
#         np.ndarray labels - int labels aligned with images_np, shape (N,)
#         Sequence[str] class_names - class names in label-index order
#         int imgsz - square input size to resize to (via _deterministic_eval_params + apply_augmentation)
#         int batch_size - chunk size to stream the pass in (typically 32-64; the last chunk may be
#         smaller -- this is exactly why this pass is not JIT'd, see module docstring)
# Outputs: float - macro_f1 over the whole of images_np/labels
# Description: Runs one deterministic (no augmentation) forward pass over the entire split, streamed
#              in batch_size-sized chunks -- each chunk is sliced from host numpy and uploaded as its
#              own small, freshly-built Tensor (never a whole-split resident tensor, see module
#              docstring fact 8) -- under Context(TRAINING=0) so BatchNorm uses its running stats
#              rather than this chunk's own statistics, accumulating predicted indices on the host
#              and returning the resulting macro-F1. Not JIT'd: runs once per epoch (JIT's per-step
#              win doesn't apply) and the final chunk is a ragged remainder whenever len(labels)
#              isn't a multiple of batch_size, so the shape isn't even fixed across chunks the way
#              TinyJit requires.
# Side Effects: Streams and runs GPU inference over the whole split, one small chunk at a time (no
#               writes; no state mutation of model).
def _evaluate_macro_f1(
    model,
    images_np: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    imgsz: int,
    batch_size: int,
) -> float:
    n = images_np.shape[0]
    y_true = [class_names[int(i)] for i in labels]
    predicted_indices: list[int] = []

    with Context(TRAINING=0):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # Upload just this chunk (<=64 * 512*512*3 uint8 bytes =~ 50 MB, well under the
            # ~100 MB ceiling) -- never the whole split at once, see module docstring fact 8.
            raw_batch = Tensor(images_np[start:end])
            eval_params = _deterministic_eval_params(end - start)
            aug_batch = apply_augmentation(raw_batch, eval_params, imgsz)
            images_f32 = normalize_to_model_input(aug_batch)
            logits = model(images_f32)
            predicted_indices.extend(int(i) for i in logits.argmax(axis=1).numpy())

    y_pred = [class_names[i] for i in predicted_indices]
    return macro_f1(y_true, y_pred, class_names)


# Path train(argparse.Namespace args)
# Inputs: argparse.Namespace args - parsed CLI options, see parse_args
# Outputs: Path - the checkpoint path (args.out) that was written for the best validation macro-F1
#          epoch seen before training stopped (early or by exhausting --epochs)
# Description: The full training loop -- see module docstring for the design rationale (TinyJit
#              boundary, discriminative LR, warmup+cosine, class-balanced sampling, adaptive-decay
#              EMA, early stopping on EMA val macro-F1). Loads both splits fully into HOST memory
#              once (uint8), and NEVER onto the GPU as a whole -- every training batch and every
#              validation chunk is later streamed to the GPU as its own small, freshly-built Tensor
#              (see module docstring fact 8). Builds the backbone+head optimizers and an EMA shadow
#              model initialized from the live model's starting weights, then runs epochs of
#              class-balanced, augmented, JIT-compiled training steps followed by a deterministic
#              EMA-model validation pass, checkpointing whenever that epoch's val macro-F1 improves
#              and stopping early after --patience epochs without improvement.
# Side Effects: Reads every image in the train/val splits from disk once, into host memory;
#               allocates GPU memory for two full models (live + EMA) plus one small streamed
#               batch/chunk Tensor at a time (never a whole-split resident tensor, see module
#               docstring fact 8); writes a safetensors checkpoint to args.out (and creates its
#               parent directory) every time val macro-F1 improves; prints per-epoch progress to
#               stdout. When args.pretrained is True, also has ResNet.load_from_pretrained's
#               network/disk-cache side effects.
def train(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data)
    class_names = discover_class_names(data_dir, "train")
    num_classes = len(class_names)
    print(f"[train_tinygrad] arch={args.arch} classes(index order)={class_names} pretrained={args.pretrained}")

    print(f"[train_tinygrad] loading dataset from '{data_dir}' ...")
    train_images_np, train_labels_np, _ = load_split_to_arrays(data_dir, "train", class_names, size=DATASET_IMAGE_SIZE)
    val_images_np, val_labels_np, _ = load_split_to_arrays(data_dir, "val", class_names, size=DATASET_IMAGE_SIZE)
    print(
        f"[train_tinygrad] train={len(train_labels_np)} images, val={len(val_labels_np)} images "
        "(host memory; streamed per-batch to the GPU, never held resident -- see module docstring fact 8)"
    )

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained)
    ema_model = build_model(args.arch, num_classes=num_classes, pretrained=False)
    model_state = get_state_dict(model)
    ema_state = get_state_dict(ema_model)
    for name, ema_param in ema_state.items():
        ema_param.assign(model_state[name].detach())
    Tensor.realize(*model_state.values(), *ema_state.values())

    backbone_params = [t for name, t in model_state.items() if not name.startswith(_HEAD_PARAM_PREFIX)]
    head_params = [t for name, t in model_state.items() if name.startswith(_HEAD_PARAM_PREFIX)]
    opt_backbone = AdamW(backbone_params, lr=args.backbone_lr, weight_decay=args.weight_decay)
    opt_head = AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    opt = OptimizerGroup(opt_backbone, opt_head)

    sampler = ClassBalancedSampler(train_labels_np, num_classes=num_classes, batch_size=args.batch, seed=args.seed)
    steps_per_epoch = sampler.steps_per_epoch()
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    aug_rng = np.random.default_rng(args.seed)
    aug_cfg = AugConfig()

    # The jitted core of one training step -- forward, loss, backward, optimizer update, EMA
    # update. Everything data-dependent that varies call-to-call (the image/label batch, both
    # LRs, the EMA decay) arrives as a genuine Tensor argument (see module docstring for why
    # this boundary is drawn here and not around augmentation/sampling too).
    def train_step(images_f32: Tensor, labels_t: Tensor, lr_backbone: Tensor, lr_head: Tensor, ema_decay: Tensor) -> Tensor:
        opt_backbone.lr.assign(lr_backbone)
        opt_head.lr.assign(lr_head)

        logits = model(images_f32)
        loss = logits.sparse_categorical_crossentropy(labels_t, label_smoothing=args.label_smoothing)

        opt.zero_grad()
        loss.backward()

        for name, ema_param in ema_state.items():
            live_param = model_state[name]
            if any(substr in name for substr in _EMA_DIRECT_COPY_SUBSTRINGS):
                ema_param.assign(live_param.detach())
            else:
                ema_param.assign(ema_param.detach() * ema_decay + live_param.detach() * (1.0 - ema_decay))

        return loss.realize(*opt.schedule_step())

    train_step_jitted = TinyJit(train_step)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    global_step = 0

    print(
        f"[train_tinygrad] steps/epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps} "
        f"batch={args.batch} imgsz={args.imgsz}"
    )
    training_start = time.monotonic()

    with Context(TRAINING=1):
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            epoch_losses: list[float] = []

            for batch_idx in sampler:
                aug_params = sample_augmentation_params(aug_rng, args.batch, aug_cfg)
                # Stream just this step's batch to the GPU as a small, freshly-built Tensor --
                # never index into a whole-split resident tensor. batch 32 @ 512x512x3 uint8 is
                # ~25 MB, well under the ~100 MB single-allocation ceiling (module docstring
                # fact 8); a resident whole-split tensor here previously crashed the NV
                # allocator with "PTE already mapped".
                raw_batch = Tensor(train_images_np[batch_idx])
                aug_batch = apply_augmentation(raw_batch, aug_params, args.imgsz)
                images_f32 = normalize_to_model_input(aug_batch).realize()
                labels_t = Tensor(train_labels_np[batch_idx].astype(np.int64)).realize()

                lr_backbone = cosine_lr_with_warmup(global_step, total_steps, warmup_steps, args.backbone_lr)
                lr_head = cosine_lr_with_warmup(global_step, total_steps, warmup_steps, args.head_lr)
                decay = ema_decay_at_step(global_step, target=args.ema_decay)

                loss = train_step_jitted(
                    images_f32,
                    labels_t,
                    Tensor([lr_backbone]).realize(),
                    Tensor([lr_head]).realize(),
                    Tensor([decay]).realize(),
                )
                epoch_losses.append(float(loss.item()))
                global_step += 1

            train_loss = float(np.mean(epoch_losses))
            val_f1 = _evaluate_macro_f1(ema_model, val_images_np, val_labels_np, class_names, args.imgsz, args.batch)
            epoch_time = time.monotonic() - epoch_start
            print(
                f"[train_tinygrad] epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_macro_f1={val_f1:.4f}  ({epoch_time:.1f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_since_improvement = 0
                metadata = build_checkpoint_metadata(class_names, args.arch, args.imgsz, args.seed)
                safe_save(get_state_dict(ema_model), str(out_path), metadata=metadata)
                print(f"[train_tinygrad]   -> new best val_macro_f1={best_val_f1:.4f}, checkpoint saved to '{out_path}'")
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= args.patience:
                    print(
                        f"[train_tinygrad] early stopping at epoch {epoch}: no val_macro_f1 improvement for "
                        f"{args.patience} epochs (best={best_val_f1:.4f} @ epoch {best_epoch})"
                    )
                    break

    total_time = time.monotonic() - training_start
    print(
        f"[train_tinygrad] training complete in {total_time:.1f}s. best val_macro_f1={best_val_f1:.4f} @ "
        f"epoch {best_epoch}. checkpoint: '{out_path}'"
    )
    return out_path


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Parses args, then IMMEDIATELY asserts tinygrad is actually running
#              on the NV device (before any dataset loading or model construction) so a missing
#              DEV=NV fails in milliseconds rather than after minutes of accidental METAL/CPU
#              training, then runs train() and reports the resulting checkpoint path.
# Side Effects: Everything train() does (see its docstring); prints a completion summary to stdout;
#               raises RuntimeError immediately if not running on tinygrad's NV device.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assert_nv_device(Tensor.zeros(1).realize().device)

    out_path = train(args)

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Checkpoint: {out_path}  (exists: {out_path.is_file()})")
    print("=" * 72)


if __name__ == "__main__":
    main()
