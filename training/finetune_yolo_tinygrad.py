"""Fine-tunes the ONNX-exported YOLO26s whole-frame classifier (``models/yolo26s_cls_bin.onnx``,
83 initializers / 81 float) directly via tinygrad's ONNX *import* path
(``tinygrad.nn.onnx.OnnxRunner``), rather than retraining a torchvision-shaped model from
scratch the way ``training/train_tinygrad.py`` does. tinygrad can import an arbitrary ONNX
graph, run it forward, and -- because every op it lowers to is a normal autograd-tracked
tinygrad ``Tensor`` op -- also run it backward, so the exported Ultralytics weights can be
fine-tuned in place and written back into a new ONNX file with the exact same graph shape,
op set, and ``names`` class-metadata the runtime (``argus.detectors.classifier``) already reads.

------------------------------------------------------------------------
Three graph-shape facts about this specific ONNX export, discovered while writing this file
------------------------------------------------------------------------
1. **The exported graph's batch dimension is a hardcoded ``dim_value: 1``, not dynamic** (the
   export used ``dynamic=False``). Calling ``OnnxRunner`` on the file as exported only ever
   accepts a batch of exactly 1. Training at batch 32 therefore first requires graph surgery
   (``make_batch_dynamic``): the input's leading dim is switched to a ``dim_param`` (so
   ``OnnxRunner`` treats it as data-dependent, caching whatever batch size the first call
   provides -- see ``OnnxRunner._parse_input``), AND the attention block at ``model.9`` (a
   YOLO26 C2PSA-style block operating on the 10x10 feature map at imgsz=320) has exactly two
   ``Reshape`` nodes whose target-shape initializers hardcode the batch axis as literal ``1``
   (``[1, 4, 128, 100]`` and ``[1, 256, 10, 10]``) rather than the usual ``-1``/inferred-axis
   idiom -- these were found by grepping every ``Reshape`` node's shape-initializer in this
   file's model (only 3 Reshape nodes total, only these 2 distinct shape tensors), and both get
   their leading element flipped to ``-1`` so the reshape infers the real batch size instead of
   failing a shape-mismatch at batch>1. This graph surgery is applied ONLY to an in-memory,
   throwaway copy of the model used to build the training-time ``OnnxRunner`` -- the file this
   script eventually writes to ``--out`` is built by patching trained weight VALUES into a
   pristine reload of the ORIGINAL ``--onnx`` file (see ``write_finetuned_onnx``), so the
   deployed graph's shape/op contract is completely unchanged from what
   ``argus.detectors.classifier`` already expects (static batch=1, final ``Softmax`` intact).
2. **The graph's final op is ``Softmax`` (not raw logits).** ``Tensor.cross_entropy`` expects
   logits and applies its own ``log_softmax`` internally -- feeding it already-softmaxed output
   would double-softmax the loss. ``find_pre_softmax_output_name`` walks the graph to find the
   node that produces the declared output; if it's a ``Softmax``, this returns that softmax's
   OWN input name (the ``Gemm`` output, real logits) instead. During training, that pre-softmax
   tensor is read out of ``OnnxRunner.graph_values`` (a plain dict of every intermediate tensor
   the runner has computed so far -- see ``tinygrad/nn/onnx.py``'s ``__call__``, which updates
   it after every node) right after the forward call, and cross-entropy runs on THAT. Validation
   argmax is invariant to the final softmax (a monotonic per-row transform), so validation just
   uses the graph's normal declared output.
3. **This export has no ``BatchNormalization`` or ``Dropout`` node anywhere** (BN has been
   fused into the preceding ``Conv``'s weight/bias at export time, standard for an
   inference-optimized export) -- confirmed by counting op types in the graph. So unlike
   ``training/train_tinygrad.py``'s torchvision-shaped model, there is no running-stats
   train/eval mode distinction for this specific graph; wrapping the training loop in
   ``Context(TRAINING=1)`` and validation in ``Context(TRAINING=0)`` is still done below (both
   for consistency with this project's other tinygrad training script and in case a future
   export re-introduces BN/Dropout in unfused form), but it is a documented no-op for the
   current graph, not something this script's correctness currently depends on.

------------------------------------------------------------------------
Which tensor gets no gradient (measured, not guessed)
------------------------------------------------------------------------
Setting ``requires_grad=True`` on all 81 float tensors in ``get_state_dict(OnnxRunner(...))``,
then running one forward+backward pass, leaves exactly ONE of them with ``.grad is None``:
``model.9.m.0.attn.Constant_1_output_0`` (bare ONNX initializer name, i.e. the state-dict key
``graph_values./model.9/m/m.0/attn/Constant_1_output_0``), a rank-0 (scalar) tensor whose value
is ``0.17677669 == 1/sqrt(32)`` -- the attention block's fixed query/key scale factor (32 =
128 channels / 4 heads), consumed by a plain ``Mul`` against the query split. It is a genuine
architectural constant, not a disconnected accident. ``train()`` runs exactly this
forward+backward detection pass once (on a throwaway dummy batch, before the optimizer is
built) and partitions the result via ``select_trainable_names``, which excludes whatever names
come back with no gradient (by name, generically -- not hardcoded to this one name) from the
optimizer rather than crashing when building the trainable parameter list, exactly as
instructed.

------------------------------------------------------------------------
Reused, not reimplemented
------------------------------------------------------------------------
Dataset loading/sampling/augmentation is 100% ``training/tg_data.py`` (``load_split_to_arrays``,
``ClassBalancedSampler``, ``sample_augmentation_params``, ``apply_augmentation``,
``normalize_to_model_input``) -- see that module's docstring for why its ``/255``-only, no
mean/std scaling exactly matches ``argus.detectors.classifier.preprocess_classify``. The LR
schedule (``cosine_lr_with_warmup``) and the early-stopping metric (``macro_f1``) are imported
from ``training/train_tinygrad.py`` rather than reimplemented.

------------------------------------------------------------------------
Environment facts inherited from training/train_tinygrad.py (see that module's docstring for
the measurements) -- fp32 only, never allocate a single GPU tensor >100 MB (stream from host
numpy per batch/chunk), TinyJit the train step only, Context(TRAINING=1) hoisted outside the
step loop, batch capped at 64, and only one tinygrad NV process at a time.
------------------------------------------------------------------------

Usage:
    python training/finetune_yolo_tinygrad.py --no-device-guard --epochs 1   # METAL smoke test
    DEV=NV python training/finetune_yolo_tinygrad.py --epochs 40 --patience 8  # real run
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

# Run-from-anywhere bootstrap (matching training/train_tinygrad.py and training/tg_data.py):
# this module reuses training.tg_data, training.train_tinygrad and argus.detectors.classifier,
# which are only importable once the repo root (and src/) are on sys.path -- pytest already does
# this via pyproject.toml's pythonpath/rootdir handling; `python training/finetune_yolo_tinygrad.py`
# does not.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tinygrad import Tensor, TinyJit, dtypes  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402
from tinygrad.nn.onnx import OnnxRunner  # noqa: E402
from tinygrad.nn.optim import AdamW  # noqa: E402
from tinygrad.nn.state import get_state_dict  # noqa: E402

from training.train_tinygrad import cosine_lr_with_warmup, macro_f1  # noqa: E402  (reuse, not reimplement)
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

DEFAULT_ONNX_PATH = REPO_ROOT / "models" / "yolo26s_cls_bin.onnx"
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "argus_bin"
DEFAULT_OUT_PATH = REPO_ROOT / "models" / "yolo26s_cls_bin_finetuned.onnx"

#: The dataset builder already center-crops every image to this fixed square size -- see
#: tg_data.load_split_to_arrays' contract (same constant as training/train_tinygrad.py).
DATASET_IMAGE_SIZE = 512

#: Batch 128 at imgsz 320 previously wedged this eGPU's driver hard enough to require
#: physically unplugging the Thunderbolt cable -- see training/train_tinygrad.py's docstring
#: fact 5. Same hard ceiling applies here; this script shares the same GPU.
MAX_BATCH_SIZE = 64

#: Fixed (non-CLI) training hyperparameters -- kept as module constants rather than extra flags
#: so this script's CLI surface matches exactly what was specified for it.
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_WEIGHT_DECAY = 0.01

#: Dim-param name substituted for the ONNX graph's hardcoded batch=1 input dimension while
#: building the training-time OnnxRunner (see make_batch_dynamic).
_DYNAMIC_BATCH_DIM_PARAM = "batch"

#: Number of real test-split images (per class, when available) used by the end-of-run
#: tinygrad-vs-onnxruntime parity check.
_VERIFY_SAMPLES_PER_CLASS = 4

#: `get_state_dict(OnnxRunner(...))` prefixes every key with this -- see
#: onnx_initializer_name_from_state_dict_key.
_STATE_DICT_PREFIX = "graph_values."


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed options: onnx, data, epochs, batch, imgsz, lr, seed,
#          patience, out, no_device_guard
# Description: Defines and parses this script's CLI. Enforces MAX_BATCH_SIZE (64) on --batch
#              here, at parse time, rather than deep inside train() -- a batch size that could
#              wedge the eGPU's driver must fail before any dataset loading or GPU allocation
#              happens, not after (same rationale as training/train_tinygrad.py's parse_args).
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or GPU/network activity). Raises ValueError if --batch exceeds
#               MAX_BATCH_SIZE.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help=f"Source ONNX classifier to fine-tune (default: {DEFAULT_ONNX_PATH})")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help=f"Binary classification dataset root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32, help=f"Batch size (default: 32; hard max {MAX_BATCH_SIZE})")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for AdamW (default: 1e-4 -- a fine-tune-scale LR, not a from-scratch one)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience, in epochs with no val macro-F1 improvement")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Fine-tuned ONNX output path (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--no-device-guard", dest="no_device_guard", action="store_true", default=False, help="Skip the NV-device assertion (for METAL smoke tests)")
    args = parser.parse_args(argv)

    if args.batch > MAX_BATCH_SIZE:
        raise ValueError(
            f"--batch {args.batch} exceeds the maximum of {MAX_BATCH_SIZE}. Batch 128 at imgsz 320 "
            "previously wedged this eGPU's driver hard enough to require physically unplugging the "
            "Thunderbolt cable -- do not raise this above 64."
        )
    return args


# None assert_device_guard(str actual_device)
# Inputs: str actual_device - a REALIZED tensor's .device string (e.g. Tensor.zeros(1).realize().device)
# Outputs: None
# Description: Fails loudly, before any ONNX loading or image decoding, if training is not
#              actually running on tinygrad's NV backend. Callers gate whether this is invoked
#              at all on --no-device-guard (see main()) -- this function itself always raises on
#              anything other than "NV", mirroring training/train_tinygrad.py's assert_nv_device.
# Side Effects: None (pure function of its input) beyond raising.
def assert_device_guard(actual_device: str) -> None:
    if actual_device != "NV":
        raise RuntimeError(
            f"tinygrad is running on device '{actual_device}', not 'NV'. Without DEV=NV set in the "
            "environment, tinygrad's default device on this machine resolves to whatever GPU backend "
            "is available locally (e.g. METAL on a Mac) -- training would silently run on the wrong "
            "hardware. Re-run with `DEV=NV python training/finetune_yolo_tinygrad.py ...`, or pass "
            "--no-device-guard for an intentional METAL smoke test."
        )


# str find_pre_softmax_output_name(onnx.ModelProto model)
# Inputs: onnx.ModelProto model - a loaded ONNX model with exactly one graph output
# Outputs: str - the tensor name to treat as raw logits for loss purposes: if the node producing
#          the declared output is a Softmax, this is that Softmax's OWN input name (the pre-
#          softmax logits); otherwise it's the declared output name itself (already logits)
# Description: Tensor.cross_entropy expects raw logits and applies its own log_softmax --
#              feeding it an already-softmaxed output would double-softmax the loss. Ultralytics'
#              classify export ends in a Softmax node (confirmed for models/yolo26s_cls_bin.onnx:
#              its final 5 ops are ...Gemm -> Softmax), so this walks the graph once to find
#              whichever tensor actually holds pre-softmax logits, generically (not hardcoded to
#              this model's specific node names) so it still works if re-exported.
# Side Effects: None (pure function of its input).
def find_pre_softmax_output_name(model: onnx.ModelProto) -> str:
    if len(model.graph.output) != 1:
        raise ValueError(
            f"expected exactly one graph output, got {len(model.graph.output)}: "
            f"{[o.name for o in model.graph.output]}"
        )
    output_name = model.graph.output[0].name
    producer = next((n for n in model.graph.node if output_name in n.output), None)
    if producer is None:
        raise ValueError(f"no node in the graph produces the declared output '{output_name}'")
    if producer.op_type == "Softmax":
        if len(producer.input) != 1:
            raise ValueError(f"Softmax node '{producer.name}' has {len(producer.input)} inputs, expected 1")
        return producer.input[0]
    return output_name


# list[str] make_batch_dynamic(onnx.ModelProto model, str dim_param)
# Inputs: onnx.ModelProto model - loaded ONNX model, MUTATED IN PLACE by this function
#         str dim_param - the symbolic dimension name to substitute for every graph input's
#                 hardcoded batch axis (default "batch")
# Outputs: list[str] - names of the Reshape target-shape initializers this call actually patched
#          (their leading element changed from literal 1 to -1); empty if none needed patching
# Description: Two-part graph surgery so an ONNX model exported with a static batch=1 input
#              (``dynamic=False``) can run at any batch size under tinygrad's OnnxRunner: (1)
#              every graph input's leading shape dimension is cleared and replaced with a
#              symbolic dim_param, so OnnxRunner._parse_input treats it as data-dependent rather
#              than asserting it equals a fixed literal; (2) any Reshape node's target-shape
#              initializer that hardcodes its OWN leading element as literal 1 (rather than the
#              usual -1/inferred-axis idiom) is patched to -1, so the reshape infers the batch
#              size from the actual tensor arriving at that point in the graph instead of
#              raising a shape-mismatch at batch>1. Only initializers actually used as a
#              Reshape's second input (its target-shape tensor) are ever touched -- this does
#              NOT rewrite every "starts with 1" integer array in the model, only the ones graph
#              topology identifies as reshape targets, so it can't accidentally corrupt an
#              unrelated int64 initializer that happens to start with 1.
# Side Effects: Mutates `model.graph.input` and any matching `model.graph.initializer` entries
#               in place.
def make_batch_dynamic(model: onnx.ModelProto, dim_param: str = _DYNAMIC_BATCH_DIM_PARAM) -> list[str]:
    for graph_input in model.graph.input:
        dims = graph_input.type.tensor_type.shape.dim
        if len(dims) == 0:
            continue
        dims[0].Clear()
        dims[0].dim_param = dim_param

    reshape_shape_names = {
        n.input[1] for n in model.graph.node if n.op_type == "Reshape" and len(n.input) > 1
    }
    patched: list[str] = []
    for init in model.graph.initializer:
        if init.name not in reshape_shape_names:
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim == 1 and arr.size > 0 and int(arr[0]) == 1:
            new_arr = arr.copy()
            new_arr[0] = -1
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_init)
            patched.append(init.name)
    return patched


# dict[str, Tensor] run_onnx_eager(OnnxRunner run, str input_name, Tensor images_f32)
# Inputs: OnnxRunner run - a batch-dynamic runner (see make_batch_dynamic)
#         str input_name - the graph's input tensor name
#         Tensor images_f32 - input batch, any batch size
# Outputs: dict[str, Tensor] - run's normal __call__ return value (its declared graph outputs)
# Description: OnnxRunner caches the FIRST batch size it ever observes for a dynamic dim_param
#              in `run.variable_dims` and then RAISES on any later eager call that uses a
#              different size (see tinygrad/nn/onnx.py's `_parse_input`: `dim_param` values are
#              resolved once via `setdefault` and never re-bound). This project's validation
#              loop deliberately streams ragged chunks (a split's size is rarely an exact
#              multiple of --batch) and the end-of-run onnxruntime-parity check calls at batch=1
#              after training has already locked the runner to --batch, so every EAGER call
#              site clears `run.variable_dims` immediately beforehand to re-bind it fresh to
#              whatever size THIS call actually uses. This must never be relied on inside the
#              TinyJit-wrapped train step for anything beyond its own one-time trace call: JIT
#              replay never re-executes Python code at all (it replays a fixed compiled kernel
#              schedule), so a fixed batch size there is both required (TinyJit needs identical
#              shapes every replay, which ClassBalancedSampler already guarantees) and
#              automatically safe without this helper.
# Side Effects: Clears and repopulates `run.variable_dims`; runs one forward pass.
def run_onnx_eager(run: OnnxRunner, input_name: str, images_f32: Tensor) -> dict:
    run.variable_dims.clear()
    return run({input_name: images_f32})


# tuple[list[str], list[str]] select_trainable_names(Mapping[str, bool] has_gradient)
# Inputs: Mapping[str, bool] has_gradient - tensor name -> whether it received a (non-None)
#                 gradient after a real forward+backward pass
# Outputs: tuple[list[str], list[str]] - (trainable_names, excluded_names), each sorted for
#          deterministic ordering; trainable_names is every name whose value was True
# Description: Pure partition of a name->got-gradient mapping into what the optimizer should
#              actually update versus what to exclude -- e.g. models/yolo26s_cls_bin.onnx's
#              attention-scale constant (see module docstring) receives no gradient and must be
#              excluded rather than crashing AdamW with a None-grad parameter. Deliberately
#              generic (excludes however many names come back False, not hardcoded to exactly
#              one) since a different --onnx model could have a different count.
# Side Effects: None
def select_trainable_names(has_gradient: Mapping[str, bool]) -> tuple[list[str], list[str]]:
    trainable = sorted(name for name, got_grad in has_gradient.items() if got_grad)
    excluded = sorted(name for name, got_grad in has_gradient.items() if not got_grad)
    return trainable, excluded


# str onnx_initializer_name_from_state_dict_key(str key)
# Inputs: str key - a key from `tinygrad.nn.state.get_state_dict(OnnxRunner(...))`, e.g.
#                 "graph_values.model.0.conv.weight"
# Outputs: str - the bare ONNX initializer name, e.g. "model.0.conv.weight" (confirmed to match
#          an entry in onnx.load(...).graph.initializer by name, for every float tensor in
#          models/yolo26s_cls_bin.onnx)
# Description: get_state_dict walks an OnnxRunner's `graph_values` dict and prefixes every key
#              with "graph_values." (the runner attribute's own name) -- this strips exactly
#              that prefix so the resulting name can be used directly with
#              replace_matching_initializers / write_finetuned_onnx. Raises rather than
#              silently mis-mapping if `key` doesn't actually have this prefix, since a wrong
#              initializer name here would silently write a trained tensor's value into the
#              wrong (or no) initializer.
# Side Effects: None
def onnx_initializer_name_from_state_dict_key(key: str) -> str:
    if not key.startswith(_STATE_DICT_PREFIX):
        raise ValueError(
            f"expected a state-dict key prefixed with '{_STATE_DICT_PREFIX}' (from "
            f"get_state_dict(OnnxRunner(...))), got '{key}'"
        )
    return key[len(_STATE_DICT_PREFIX):]


# list[str] replace_matching_initializers(onnx.ModelProto model, Mapping[str, np.ndarray] trained_arrays)
# Inputs: onnx.ModelProto model - loaded ONNX model, MUTATED IN PLACE by this function
#         Mapping[str, np.ndarray] trained_arrays - ONNX initializer name -> new value to write
#                 into that initializer
# Outputs: list[str] - the initializer names actually replaced, in the iteration order of
#          `trained_arrays`
# Description: The ONNX weight write-back step: for each (name, array) pair, finds the matching
#              initializer BY NAME, verifies the new array's shape matches the existing
#              initializer's declared shape exactly (a silent shape drift here would produce a
#              corrupt or misleadingly-shaped ONNX file), casts the array to the initializer's
#              OWN existing ONNX dtype (so a fine-tuned tensor that's still float32 on the
#              tinygrad side round-trips back into the model's declared TensorProto dtype
#              exactly, never silently promoting/demoting precision), and replaces it via
#              `onnx.numpy_helper.from_array` + `CopyFrom` -- which naturally preserves the
#              initializer's name (from_array is called with the same name) and leaves every
#              other part of `model` (metadata_props including the `names` class-metadata,
#              graph.output, every other node/initializer) completely untouched.
# Side Effects: Mutates matching entries of `model.graph.initializer` in place. Raises KeyError
#               if a name in `trained_arrays` has no matching initializer in `model`, or
#               ValueError if the shapes don't match.
def replace_matching_initializers(model: onnx.ModelProto, trained_arrays: Mapping[str, np.ndarray]) -> list[str]:
    name_to_initializer = {init.name: init for init in model.graph.initializer}
    replaced: list[str] = []
    for name, array in trained_arrays.items():
        if name not in name_to_initializer:
            raise KeyError(f"replace_matching_initializers: no initializer named '{name}' in the model")
        init = name_to_initializer[name]
        expected_shape = tuple(init.dims)
        arr = np.asarray(array)
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"replace_matching_initializers: shape mismatch for '{name}': existing initializer "
                f"is {expected_shape}, new array is {tuple(arr.shape)}"
            )
        target_dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        new_init = numpy_helper.from_array(arr.astype(target_dtype), name=name)
        init.CopyFrom(new_init)
        replaced.append(name)
    return replaced


# None write_finetuned_onnx(Path original_onnx_path, Path out_path, Mapping[str, np.ndarray] trained_arrays)
# Inputs: Path original_onnx_path - the pristine, never-graph-surgered --onnx source file
#         Path out_path - where to write the fine-tuned model
#         Mapping[str, np.ndarray] trained_arrays - ONNX initializer name -> fine-tuned value,
#                 for exactly the tensors the optimizer actually trained (see select_trainable_names)
# Outputs: None
# Description: Reloads `original_onnx_path` FRESH (never the in-memory, batch-surgered copy used
#              for training -- see module docstring point 1) so the file this writes has exactly
#              the original static-batch-1 shape contract and final Softmax that
#              argus.detectors.classifier already expects, then patches in the fine-tuned weight
#              VALUES via replace_matching_initializers and saves the result. Every part of the
#              original model this script never trained (metadata_props including `names`,
#              graph structure, the untrainable attention-scale constant) passes through
#              unchanged because only matching initializers are ever touched.
# Side Effects: Reads `original_onnx_path` from disk; writes `out_path` (creating its parent
#               directory if needed).
def write_finetuned_onnx(original_onnx_path: Path, out_path: Path, trained_arrays: Mapping[str, np.ndarray]) -> None:
    model = onnx.load(str(original_onnx_path))
    replace_matching_initializers(model, trained_arrays)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))


# AugParams _neutral_aug_params(int batch_size)
# Inputs: int batch_size - number of samples in the validation/verification batch this call is for
# Outputs: AugParams - a fully "neutral" set of augmentation parameters (no crop, no flip, no
#          jitter, no cutout, no blur, no noise) -- see training/train_tinygrad.py's
#          _deterministic_eval_params, which this mirrors exactly (kept as a local copy rather
#          than importing that module's underscore-prefixed helper across module boundaries).
# Side Effects: None
def _neutral_aug_params(batch_size: int) -> AugParams:
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


# float _evaluate_macro_f1_onnx(OnnxRunner run, str input_name, str output_name, np.ndarray images_np, np.ndarray labels, Sequence[str] class_names, int imgsz, int batch_size)
# Inputs: OnnxRunner run - the (batch-dynamic) training-time runner, called forward-only here
#         str input_name - the graph's input tensor name
#         str output_name - the graph's declared output name (post-softmax probabilities;
#                 argmax is invariant to the softmax so this is fine for prediction)
#         np.ndarray images_np - HOST uint8 (N, H, W, 3) images (e.g. the val split), never moved
#                 to the GPU as a whole -- only one batch_size-sized chunk at a time is uploaded
#         np.ndarray labels - int labels aligned with images_np, shape (N,)
#         Sequence[str] class_names - class names in label-index order
#         int imgsz - square input size to resize to
#         int batch_size - chunk size to stream the pass in (the last chunk may be smaller)
# Outputs: float - macro_f1 over the whole of images_np/labels
# Description: Streamed, deterministic (no augmentation) forward pass over an entire split,
#              structurally mirroring training/train_tinygrad.py's _evaluate_macro_f1 but calling
#              through an OnnxRunner instead of a tg_models model -- the actual macro-F1 math is
#              still the imported `macro_f1` (not reimplemented here).
# Side Effects: Streams and runs GPU inference over the whole split, one small chunk at a time.
def _evaluate_macro_f1_onnx(
    run: OnnxRunner,
    input_name: str,
    output_name: str,
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
            raw_batch = Tensor(images_np[start:end])
            eval_params = _neutral_aug_params(end - start)
            aug_batch = apply_augmentation(raw_batch, eval_params, imgsz)
            images_f32 = normalize_to_model_input(aug_batch)
            outputs = run_onnx_eager(run, input_name, images_f32)
            predicted_indices.extend(int(i) for i in outputs[output_name].argmax(axis=1).numpy())

    y_pred = [class_names[i] for i in predicted_indices]
    return macro_f1(y_true, y_pred, class_names)


# dict[str, int] verify_onnx_matches_tinygrad(Path finetuned_onnx_path, OnnxRunner run, str input_name, str output_name, Sequence[Path] image_paths, int imgsz)
# Inputs: Path finetuned_onnx_path - the just-written --out file (loaded fresh via onnxruntime,
#                 proving the SAVED FILE -- not just the in-memory tinygrad state -- behaves
#                 correctly)
#         OnnxRunner run - the live, just-trained tinygrad runner (same object used for
#                 training, holding the actual fine-tuned weight values)
#         str input_name - the graph's input tensor name
#         str output_name - the graph's declared output name
#         Sequence[Path] image_paths - real image files to check (never synthetic -- the whole
#                 point is proving the round-trip through a real ONNX file)
#         int imgsz - square input size
# Outputs: dict[str, int] - {"matched": n, "total": len(image_paths)}
# Description: For each real image, builds ONE shared preprocessed blob via
#              argus.detectors.classifier.preprocess_classify (the actual production
#              preprocessing path -- reused, not reimplemented) and feeds that SAME blob to (a)
#              onnxruntime against the freshly-saved --out file and (b) the tinygrad runner that
#              was actually trained, comparing argmax class predictions between the two. This is
#              the true end-to-end check: does serializing tinygrad's trained weights into ONNX
#              and reloading them via a completely different runtime (onnxruntime, what the
#              printer actually runs) preserve the model's predictions.
# Side Effects: Imports onnxruntime, cv2, and argus.detectors.classifier lazily. Reads
#               `finetuned_onnx_path` and every path in `image_paths` from disk. Runs one
#               inference pass per image on both backends.
def verify_onnx_matches_tinygrad(
    finetuned_onnx_path: Path,
    run: OnnxRunner,
    input_name: str,
    output_name: str,
    image_paths: Sequence[Path],
    imgsz: int,
) -> dict[str, int]:
    import cv2
    import onnxruntime as ort

    from argus.detectors.classifier import preprocess_classify

    session = ort.InferenceSession(str(finetuned_onnx_path), providers=["CPUExecutionProvider"])
    ort_input_name = session.get_inputs()[0].name

    matched = 0
    with Context(TRAINING=0):
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"verify_onnx_matches_tinygrad: failed to read image: {path}")
            blob = preprocess_classify(image, imgsz)  # (1, 3, imgsz, imgsz) float32 NCHW in [0, 1]

            ort_probs = session.run(None, {ort_input_name: blob})[0]
            ort_pred = int(np.argmax(ort_probs, axis=-1).reshape(-1)[0])

            tg_outputs = run_onnx_eager(run, input_name, Tensor(blob))
            tg_pred = int(tg_outputs[output_name].argmax(axis=1).numpy()[0])

            if ort_pred == tg_pred:
                matched += 1

    return {"matched": matched, "total": len(image_paths)}


# Path train(argparse.Namespace args)
# Inputs: argparse.Namespace args - parsed CLI options, see parse_args
# Outputs: Path - args.out, the fine-tuned ONNX checkpoint written for the best validation
#          macro-F1 epoch seen before training stopped (early or by exhausting --epochs)
# Description: The full fine-tuning loop -- see module docstring for the graph-surgery,
#              pre-softmax-logits, and no-gradient-tensor design notes. Loads both dataset
#              splits fully into HOST memory once (uint8) and never onto the GPU as a whole, the
#              same streaming discipline as training/train_tinygrad.py. Builds a batch-dynamic
#              OnnxRunner from a throwaway graph-surgered copy of --onnx, detects which float
#              tensors actually receive a gradient (excluding the rest rather than crashing),
#              trains those with AdamW under a cosine+warmup LR schedule and label-smoothed
#              cross-entropy on class-balanced batches, validates each epoch's macro-F1 on the
#              val split, and writes the fine-tuned weights back into a fresh ONNX file (built
#              from a pristine reload of the ORIGINAL --onnx, see write_finetuned_onnx) every
#              time val macro-F1 improves, early-stopping after --patience epochs without
#              improvement. Finally verifies the saved file against the live tinygrad model on a
#              handful of real test-split images.
# Side Effects: Reads every image in the train/val/test splits from disk once, into host memory.
#               Allocates GPU memory for the OnnxRunner's weights plus one small streamed
#               batch/chunk Tensor at a time. Writes an ONNX file to args.out (and creates its
#               parent directory) every time val macro-F1 improves. Prints per-epoch progress
#               and a final verification summary to stdout.
def train(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data)
    class_names = discover_class_names(data_dir, "train")
    num_classes = len(class_names)
    print(f"[finetune_yolo_tinygrad] onnx={args.onnx} classes(index order)={class_names}")

    original_model = onnx.load(str(args.onnx))
    input_name = original_model.graph.input[0].name
    output_name = original_model.graph.output[0].name
    logits_name = find_pre_softmax_output_name(original_model)
    print(f"[finetune_yolo_tinygrad] input='{input_name}' output='{output_name}' pre-softmax logits='{logits_name}'")

    with tempfile.TemporaryDirectory(prefix="finetune_yolo_tinygrad_") as tmp_dir:
        dynamic_model = onnx.load(str(args.onnx))  # fresh copy -- keep original_model pristine
        patched = make_batch_dynamic(dynamic_model)
        print(f"[finetune_yolo_tinygrad] graph surgery: dynamic batch dim + patched reshape targets {patched}")
        dynamic_path = Path(tmp_dir) / "dynamic_batch.onnx"
        onnx.save(dynamic_model, str(dynamic_path))
        run = OnnxRunner(str(dynamic_path))
    # OnnxRunner loads all initializer/external data into memory in its constructor, so the
    # temp file is safe to delete (via the TemporaryDirectory context exit) once it returns.

    print(f"[finetune_yolo_tinygrad] loading dataset from '{data_dir}' ...")
    train_images_np, train_labels_np, _ = load_split_to_arrays(data_dir, "train", class_names, size=DATASET_IMAGE_SIZE)
    val_images_np, val_labels_np, _ = load_split_to_arrays(data_dir, "val", class_names, size=DATASET_IMAGE_SIZE)
    print(
        f"[finetune_yolo_tinygrad] train={len(train_labels_np)} images, val={len(val_labels_np)} images "
        "(host memory; streamed per-batch to the GPU, never held resident)"
    )

    state_dict = get_state_dict(run)
    float_items = {name: t for name, t in state_dict.items() if t.dtype == dtypes.float32}
    print(f"[finetune_yolo_tinygrad] {len(state_dict)} total tensors, {len(float_items)} float")
    for t in float_items.values():
        t.requires_grad = True

    # Detect which float tensors actually receive a gradient with a throwaway dummy batch --
    # this is a structural property of the graph, not of the data, so random noise is enough
    # (see module docstring: exactly one attention-scale constant is expected to get none).
    # No optimizer has been built yet and nothing here touches tensor VALUES (only .grad), so
    # this is safe to run directly on the real weights before training starts. Deliberately
    # sized at args.batch (not some smaller probe size): OnnxRunner permanently caches the
    # FIRST batch size it ever sees for the dynamic batch dim (see run_onnx_eager's docstring),
    # and the very next call after this one is train_step_jitted's real, JIT-tracing call at
    # args.batch -- using any other size here would make THAT call collide with a stale cached
    # value. run_onnx_eager still clears the cache defensively either way.
    dummy_images = Tensor(np.random.default_rng(args.seed).random((args.batch, 3, args.imgsz, args.imgsz)).astype(np.float32))
    dummy_labels = Tensor(np.arange(args.batch, dtype=np.int64) % num_classes)
    _ = run_onnx_eager(run, input_name, dummy_images)
    dummy_logits = run.graph_values[logits_name]
    dummy_loss = dummy_logits.cross_entropy(dummy_labels, label_smoothing=DEFAULT_LABEL_SMOOTHING)
    dummy_loss.backward()
    has_gradient = {name: (t.grad is not None) for name, t in float_items.items()}
    trainable_names, excluded_names = select_trainable_names(has_gradient)
    print(f"[finetune_yolo_tinygrad] {len(trainable_names)} trainable float tensors, {len(excluded_names)} excluded (no gradient): {excluded_names}")
    for name in excluded_names:
        float_items[name].requires_grad = False
    for t in float_items.values():
        t.grad = None  # discard the dummy pass's gradients before real training starts

    trainable_params = [float_items[name] for name in trainable_names]
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY)

    sampler = ClassBalancedSampler(train_labels_np, num_classes=num_classes, batch_size=args.batch, seed=args.seed)
    steps_per_epoch = sampler.steps_per_epoch()
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = DEFAULT_WARMUP_EPOCHS * steps_per_epoch
    aug_rng = np.random.default_rng(args.seed)
    aug_cfg = AugConfig()

    def train_step(images_f32: Tensor, labels_t: Tensor, lr_t: Tensor) -> Tensor:
        opt.lr.assign(lr_t)
        outputs = run({input_name: images_f32})
        logits = run.graph_values[logits_name]
        loss = logits.cross_entropy(labels_t, label_smoothing=DEFAULT_LABEL_SMOOTHING)
        opt.zero_grad()
        loss.backward()
        return loss.realize(*opt.schedule_step())

    train_step_jitted = TinyJit(train_step)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    global_step = 0

    print(
        f"[finetune_yolo_tinygrad] steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup_steps={warmup_steps} batch={args.batch} imgsz={args.imgsz} lr={args.lr}"
    )
    training_start = time.monotonic()

    with Context(TRAINING=1):
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            epoch_losses: list[float] = []

            for batch_idx in sampler:
                aug_params = sample_augmentation_params(aug_rng, args.batch, aug_cfg)
                raw_batch = Tensor(train_images_np[batch_idx])
                aug_batch = apply_augmentation(raw_batch, aug_params, args.imgsz)
                images_f32 = normalize_to_model_input(aug_batch).realize()
                labels_t = Tensor(train_labels_np[batch_idx].astype(np.int64)).realize()

                lr = cosine_lr_with_warmup(global_step, total_steps, warmup_steps, args.lr)
                loss = train_step_jitted(images_f32, labels_t, Tensor([lr]).realize())
                epoch_losses.append(float(loss.item()))
                global_step += 1

            train_loss = float(np.mean(epoch_losses))
            val_f1 = _evaluate_macro_f1_onnx(run, input_name, output_name, val_images_np, val_labels_np, class_names, args.imgsz, args.batch)
            epoch_time = time.monotonic() - epoch_start
            print(
                f"[finetune_yolo_tinygrad] epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_macro_f1={val_f1:.4f}  ({epoch_time:.1f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_since_improvement = 0
                trained_arrays = {
                    onnx_initializer_name_from_state_dict_key(name): float_items[name].numpy()
                    for name in trainable_names
                }
                write_finetuned_onnx(args.onnx, out_path, trained_arrays)
                print(f"[finetune_yolo_tinygrad]   -> new best val_macro_f1={best_val_f1:.4f}, ONNX written to '{out_path}'")
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= args.patience:
                    print(
                        f"[finetune_yolo_tinygrad] early stopping at epoch {epoch}: no val_macro_f1 improvement for "
                        f"{args.patience} epochs (best={best_val_f1:.4f} @ epoch {best_epoch})"
                    )
                    break

    total_time = time.monotonic() - training_start
    print(
        f"[finetune_yolo_tinygrad] training complete in {total_time:.1f}s. best val_macro_f1={best_val_f1:.4f} @ "
        f"epoch {best_epoch}. checkpoint: '{out_path}'"
    )

    if out_path.is_file():
        test_records = []
        for class_name in class_names:
            class_dir = data_dir / "test" / class_name
            if class_dir.is_dir():
                test_records.extend(sorted(class_dir.iterdir())[:_VERIFY_SAMPLES_PER_CLASS])
        if test_records:
            verification = verify_onnx_matches_tinygrad(out_path, run, input_name, output_name, test_records, args.imgsz)
            print(
                f"[finetune_yolo_tinygrad] onnxruntime-vs-tinygrad parity on {verification['total']} real test "
                f"images: {verification['matched']}/{verification['total']} argmax matches"
            )

    return out_path


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Parses args, then (unless --no-device-guard) IMMEDIATELY asserts
#              tinygrad is actually running on the NV device -- before any ONNX/dataset loading
#              -- so a missing DEV=NV fails in milliseconds. --no-device-guard exists
#              specifically for the mandatory METAL smoke test this project's workflow requires
#              before ever running on the real eGPU.
# Side Effects: Everything train() does; prints a completion summary; raises RuntimeError
#               immediately if device guard is active and not running on NV.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.no_device_guard:
        assert_device_guard(Tensor.zeros(1).realize().device)

    out_path = train(args)

    print()
    print("=" * 72)
    print("FINE-TUNING COMPLETE")
    print("=" * 72)
    print(f"Output ONNX: {out_path}  (exists: {out_path.is_file()})")
    print("=" * 72)


if __name__ == "__main__":
    main()
