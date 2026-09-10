"""Exports a tinygrad-trained binary (``failure``/``normal``) print-failure classifier
checkpoint to ONNX for deployment on a Raspberry Pi running ONNX Runtime only (no
tinygrad, no torch on the Pi).

tinygrad can *import* ONNX but cannot *export* it, so the bridge out of tinygrad is:

    tinygrad safetensors checkpoint -> torchvision ResNet -> torch.onnx.export

This works only because ``training/tg_models.py`` is written so that
``tinygrad.nn.state.get_state_dict(model)`` produces exactly the same key set (names
and shapes) as the corresponding ``torchvision.models.resnet{18,34}`` state dict (see
that module's docstring; verified for both depths in ``tests/test_tg_models.py``). That
means loading a tinygrad checkpoint into a real torchvision module is a plain
``load_state_dict(strict=True)`` with zero key remapping. torch is used ONLY as an
export vehicle here -- it never trains anything and never ships to the Pi.

------------------------------------------------------------------------
Pinned checkpoint contract
------------------------------------------------------------------------
A ``.safetensors`` file written by ``tinygrad.nn.state.safe_save(tensors, path,
metadata=...)``, where:

* ``tensors`` is ``tinygrad.nn.state.get_state_dict(model)`` with torchvision key names.
* ``metadata`` (safetensors metadata is ``str -> str`` only) carries:
    - ``class_names``: comma-separated, in model output index order, e.g.
      ``"failure,normal"``.
    - ``arch``: ``"resnet18"`` or ``"resnet34"``.
    - ``imgsz``: e.g. ``"320"``.
    - ``num_classes``: e.g. ``"2"``.
    - ``seed``: e.g. ``"1337"``.

``arch``, ``imgsz`` and ``class_names`` are read from this metadata, never from CLI
flags -- that structurally eliminates the train/export mismatch bug class (exporting at
the wrong input size, or with class names in the wrong order).

------------------------------------------------------------------------
Raw logits, not softmax
------------------------------------------------------------------------
This script exports raw logits (torchvision's ``ResNet.fc`` output, unmodified) rather
than adding a softmax layer to the exported graph. The runtime's
``argus.detectors.classifier.probabilities_from_output`` already auto-detects whether a
classifier's raw output looks like a probability distribution and only applies softmax
when it doesn't, specifically so it can support both kinds of exported classifiers
without double-softmax-ing one of them. Emitting logits here routes correctly through
that auto-detection; emitting our own softmax would risk either a double-softmax (if
detection somehow still fired) or a subtly wrong axis, for zero benefit.

------------------------------------------------------------------------
ONNX class-name metadata
------------------------------------------------------------------------
``argus.detectors.classifier.class_names_from_onnx_metadata`` reads the model's
training-time class order from ``session.get_modelmeta().custom_metadata_map["names"]``
via ``ast.literal_eval``, expecting a Python dict-repr string like ``"{0: 'failure', 1:
'normal'}"``. After ``torch.onnx.export`` this script loads the file with ``onnx``,
appends exactly that key/value to ``model.metadata_props``, and saves it back
(``format_class_names_metadata`` / ``export_onnx``). Skipping this step means the
deployed detector falls back to (unverified) config class names -- or refuses to start
if none are configured -- so it is not optional.

------------------------------------------------------------------------
Class order
------------------------------------------------------------------------
Class order comes entirely from the checkpoint metadata (which in turn comes from
sorted training-folder names -> ``("failure", "normal")``, i.e. index 0 is ``failure``).
This script never hardcodes or re-sorts that order.

------------------------------------------------------------------------
Static shape, CPU-only
------------------------------------------------------------------------
Export uses the legacy TorchScript exporter (``torch.onnx.export(..., dynamo=False)``,
the well-trodden path for static opset-12 export) with a fixed ``(1, 3, imgsz, imgsz)``
input and ``(1, num_classes)`` output -- no dynamic axes. Nothing in this script
requires a GPU or a particular tinygrad device: the tinygrad forward pass used for
parity verification runs on whatever ``Device.DEFAULT`` resolves to on the machine
running this script.

Usage: python training/export_tinygrad_onnx.py --checkpoint runs/train_tg/bin_v1/best.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import load_state_dict as tg_load_state_dict
from tinygrad.nn.state import safe_load, safe_load_metadata

# Run-from-anywhere bootstrap (matching training/evaluate.py): this module reuses
# training/tg_models.py and argus.detectors.classifier, which are only importable once
# the repo root (and src/) are on sys.path -- pytest already does this via
# pyproject.toml's pythonpath/rootdir insertion; `python training/export_tinygrad_onnx.py`
# does not.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from argus.detectors.classifier import preprocess_classify  # noqa: E402
from training.tg_models import build_model as build_tg_model  # noqa: E402

DEFAULT_OUT_PATH = REPO_ROOT / "models" / "argus_bin.onnx"
DEFAULT_TEST_DATA_DIR = REPO_ROOT / "datasets" / "argus_bin" / "test"
DEFAULT_OPSET = 12
DEFAULT_SAMPLES_PER_CLASS = 5
DEFAULT_ATOL = 2e-2

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: Required safetensors metadata keys per the pinned checkpoint contract (see module
#: docstring). All four must be present and non-empty or export refuses to proceed.
_REQUIRED_METADATA_KEYS: tuple[str, ...] = ("class_names", "arch", "imgsz", "num_classes")

#: arch name -> torchvision.models constructor attribute name. Mirrors
#: training/tg_models.py's _ARCH_BUILDERS -- the only two depths this project trains.
_TORCHVISION_ARCH_ATTR: dict[str, str] = {"resnet18": "resnet18", "resnet34": "resnet34"}


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed --checkpoint (required), --out (default
#          models/argus_bin.onnx), --opset (default 12), --test-data (default
#          datasets/argus_bin/test), --samples-per-class (default 5), --atol (default 2e-2)
# Description: Defines and parses the CLI for exporting a tinygrad checkpoint to ONNX and
#              verifying tinygrad/ONNX Runtime parity. Deliberately has no --arch/--imgsz/
#              --class-names flags: those come from the checkpoint's own safetensors metadata
#              (see module docstring) so they cannot drift from what was actually trained.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity).
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the tinygrad safetensors checkpoint (see module docstring's pinned contract)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"ONNX output path (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help=f"ONNX opset version (default: {DEFAULT_OPSET})")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATA_DIR,
        help=f"Binary classification test-split directory (test_data/<class>/*.jpg) used for the "
        f"tinygrad/ONNX Runtime parity check (default: {DEFAULT_TEST_DATA_DIR})",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=f"How many real test images per class to check for parity (default: {DEFAULT_SAMPLES_PER_CLASS}); "
        "falls back to deterministic synthetic input if none are found",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Max-abs-diff tolerance reported against (default: {DEFAULT_ATOL}); argmax agreement, not this "
        "tolerance, is what export actually fails on -- see module docstring",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------


# str _require_metadata_value(Mapping[str, str] metadata, str key)
# Inputs: Mapping[str, str] metadata - the checkpoint's parsed safetensors metadata
#         str key - the required metadata key to fetch
# Outputs: str - metadata[key]
# Description: Fetches a required metadata value, raising a clear, actionable ValueError
#              (naming the missing key, the keys that ARE present, and the full required set)
#              if it is absent or empty, instead of letting a bare KeyError surface deep inside
#              export logic.
# Side Effects: None (pure function of its inputs).
def _require_metadata_value(metadata: Mapping[str, str], key: str) -> str:
    value = metadata.get(key)
    if not value:
        raise ValueError(
            f"tinygrad checkpoint safetensors metadata is missing required key {key!r} (present keys: "
            f"{sorted(metadata.keys())}). A checkpoint written by the training script must carry metadata "
            f"for all of {list(_REQUIRED_METADATA_KEYS)} -- see training/export_tinygrad_onnx.py's module "
            "docstring for the pinned checkpoint contract."
        )
    return value


# tuple[dict, dict[str, str]] load_checkpoint(Path path)
# Inputs: Path path - path to the tinygrad safetensors checkpoint
# Outputs: tuple[dict, dict[str, str]] - (tensors, metadata): tensors is
#          tinygrad.nn.state.safe_load(path)'s {name: Tensor} state dict; metadata is the
#          checkpoint's custom str->str safetensors metadata (the "__metadata__" block), with
#          presence of every key in _REQUIRED_METADATA_KEYS already validated
# Description: Loads a tinygrad safetensors checkpoint's tensors and metadata, and fails loudly
#              (via _require_metadata_value) if any of _REQUIRED_METADATA_KEYS is missing or
#              empty -- this is the single point where a malformed/incomplete checkpoint is
#              caught, before any downstream code assumes those keys exist.
# Side Effects: Reads `path` from disk (memory-mapped by tinygrad's safe_load/safe_load_metadata).
#               Raises FileNotFoundError if `path` does not exist; raises ValueError if required
#               metadata is missing.
def load_checkpoint(path: Path) -> tuple[dict, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"tinygrad checkpoint not found at '{path}' -- has training been run yet?")

    tensors = safe_load(str(path))
    _, _, raw_meta = safe_load_metadata(str(path))
    metadata: dict[str, str] = dict(raw_meta.get("__metadata__") or {})

    for key in _REQUIRED_METADATA_KEYS:
        _require_metadata_value(metadata, key)

    return tensors, metadata


# tuple[str, ...] parse_class_names(Mapping[str, str] metadata)
# Inputs: Mapping[str, str] metadata - checkpoint safetensors metadata containing a
#                 comma-separated "class_names" entry, in model output index order
# Outputs: tuple[str, ...] - class names, in model output index order
# Description: Parses the checkpoint's "class_names" metadata entry (comma-separated, e.g.
#              "failure,normal") into an ordered tuple. Whitespace around each name is
#              stripped. Raises ValueError (naming the raw value) if the key is missing/empty
#              or splits into any empty name.
# Side Effects: None (pure function of its input).
def parse_class_names(metadata: Mapping[str, str]) -> tuple[str, ...]:
    raw = _require_metadata_value(metadata, "class_names")
    names = tuple(part.strip() for part in raw.split(","))
    if not names or any(not name for name in names):
        raise ValueError(
            f"checkpoint metadata 'class_names'={raw!r} did not parse into a clean list of non-empty, "
            "comma-separated names."
        )
    return names


# str format_class_names_metadata(Sequence[str] class_names)
# Inputs: Sequence[str] class_names - class names in model output index order
# Outputs: str - a Python dict-repr string, e.g. "{0: 'failure', 1: 'normal'}"
# Description: Formats class_names into exactly the string format
#              argus.detectors.classifier.class_names_from_onnx_metadata expects in an ONNX
#              model's custom_metadata_map["names"] entry (it parses this value with
#              ast.literal_eval). This is the inverse of that parser: feeding this function's
#              output through class_names_from_onnx_metadata (via a session whose
#              custom_metadata_map["names"] holds it) must return class_names back unchanged --
#              see tests/test_export_tinygrad_onnx.py.
# Side Effects: None (pure function of its input).
def format_class_names_metadata(class_names: Sequence[str]) -> str:
    return repr({index: name for index, name in enumerate(class_names)})


# --------------------------------------------------------------------------
# tinygrad checkpoint -> torchvision module
# --------------------------------------------------------------------------


# torch.nn.Module build_torch_resnet(dict state_dict, str arch, int num_classes)
# Inputs: dict state_dict - tinygrad {name: Tensor} state dict (torchvision key names), as
#                 returned by load_checkpoint
#         str arch - "resnet18" or "resnet34"
#         int num_classes - number of output classes for the final fc layer
# Outputs: torch.nn.Module - a torchvision resnet{18,34} in eval() mode, with state_dict loaded
#          via strict=True (zero key remapping -- see module docstring's key-parity contract)
# Description: Builds an untrained torchvision ResNet matching `arch`/`num_classes`, then loads
#              `state_dict` into it. Every tinygrad tensor is upcast to float32 numpy before
#              conversion to a torch tensor -- regardless of its training-time dtype (e.g. fp16
#              under mixed-precision training on the eGPU) -- because the deployment target (a
#              Raspberry Pi CPU running ONNX Runtime) wants fp32 weights, not fp16.
# Side Effects: Imports torch and torchvision (lazily, since they are only needed for this
#               export path, not for training or the Pi runtime). Raises ValueError for an
#               unrecognized arch; raises RuntimeError (via load_state_dict's strict=True) if
#               state_dict's keys/shapes don't match the constructed torchvision module exactly.
def build_torch_resnet(state_dict, arch: str, num_classes: int):
    import torch
    import torchvision

    attr = _TORCHVISION_ARCH_ATTR.get(arch)
    if attr is None:
        raise ValueError(f"unknown arch {arch!r}; valid options are {sorted(_TORCHVISION_ARCH_ATTR)}")
    builder = getattr(torchvision.models, attr)

    torch_state = {
        key: torch.from_numpy(np.asarray(tensor.numpy(), dtype=np.float32)) for key, tensor in state_dict.items()
    }

    torch_model = builder(weights=None, num_classes=num_classes)
    torch_model.load_state_dict(torch_state, strict=True)
    torch_model.eval()
    return torch_model


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------


# Path export_onnx(torch.nn.Module torch_model, int imgsz, int opset, Sequence[str] class_names, Path out)
# Inputs: torch.nn.Module torch_model - the torchvision ResNet to export (see build_torch_resnet)
#         int imgsz - square input spatial size, from the checkpoint's "imgsz" metadata
#         int opset - ONNX opset version to export with
#         Sequence[str] class_names - class names in model output index order
#         Path out - destination .onnx path
# Outputs: Path - `out`, unchanged (returned for convenient chaining)
# Description: Exports `torch_model` to a static-shape ONNX graph -- input "images" shaped
#              (1, 3, imgsz, imgsz), output "logits" shaped (1, num_classes), RAW LOGITS (no
#              softmax appended; see module docstring) -- using the legacy TorchScript exporter
#              (dynamo=False, the well-trodden path for static opset-12 export) with
#              dynamic_axes=None. Then reloads the file with `onnx`, appends a "names"
#              metadata_props entry (format_class_names_metadata(class_names)) so
#              argus.detectors.classifier.class_names_from_onnx_metadata can recover the
#              training-time class order at inference time, and saves it back in place.
# Side Effects: Imports torch and onnx (lazily). Creates `out`'s parent directory. Writes (and
#               then rewrites, to add metadata) the ONNX file at `out`. Runs one forward pass of
#               `torch_model` (traced by the exporter) on an all-zero dummy input.
def export_onnx(torch_model, imgsz: int, opset: int, class_names: Sequence[str], out: Path) -> Path:
    import onnx
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch_model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            torch_model,
            (dummy,),
            str(out),
            export_params=True,
            opset_version=opset,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=None,
            dynamo=False,
        )

    onnx_model = onnx.load(str(out))
    meta = onnx_model.metadata_props.add()
    meta.key = "names"
    meta.value = format_class_names_metadata(class_names)
    onnx.save(onnx_model, str(out))

    return out


# --------------------------------------------------------------------------
# Parity verification
# --------------------------------------------------------------------------


# float max_abs_diff(np.ndarray a, np.ndarray b)
# Inputs: np.ndarray a - first array (e.g. tinygrad model output)
#         np.ndarray b - second array, same shape as a (e.g. ONNX Runtime output)
# Outputs: float - the largest absolute elementwise difference between a and b
# Description: Computes max(|a - b|) after upcasting both to float64, so the comparison is
#              robust regardless of a/b's original dtypes.
# Side Effects: None (pure function of its inputs).
def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# tuple[bool, float] arrays_agree(np.ndarray a, np.ndarray b, float atol)
# Inputs: np.ndarray a - first array
#         np.ndarray b - second array, same shape as a
#         float atol - absolute-difference tolerance
# Outputs: tuple[bool, float] - (max_abs_diff(a, b) <= atol, max_abs_diff(a, b))
# Description: Thin wrapper around max_abs_diff that also reports the pass/fail verdict against
#              `atol`, so callers get both the raw number and the decision in one call.
# Side Effects: None (pure function of its inputs).
def arrays_agree(a: np.ndarray, b: np.ndarray, atol: float) -> tuple[bool, float]:
    diff = max_abs_diff(a, b)
    return diff <= atol, diff


# tuple[list[np.ndarray], bool] _collect_parity_blobs(Path test_data_dir, Sequence[str] class_names, int samples_per_class, int imgsz)
# Inputs: Path test_data_dir - classification test-split directory (test_data_dir/<class>/*.jpg)
#         Sequence[str] class_names - class names to sample real images for, in model output order
#         int samples_per_class - how many real images to pick per class (sorted by filename,
#                 first N -- deterministic)
#         int imgsz - square input spatial size the model expects
# Outputs: tuple[list[np.ndarray], bool] - a list of (1, 3, imgsz, imgsz) float32 blobs, ready to
#          feed to both the tinygrad model and the ONNX Runtime session, plus whether the
#          deterministic synthetic fallback was used (True) instead of real images (False)
# Description: Preprocesses real test images (via argus.detectors.classifier.preprocess_classify
#              -- the actual production preprocessing, never a reimplementation) found under
#              test_data_dir/<class>/ for each class in class_names. If no usable images are
#              found anywhere (missing directory, empty class subdirectories, or every image
#              failing to decode), falls back to `max(samples_per_class, 1)` deterministic
#              synthetic blobs drawn from `np.random.default_rng(1337)`, so the parity check
#              always runs.
# Side Effects: Reads image files from disk via cv2.imread (only when test_data_dir has usable
#               images); imports cv2 lazily.
def _collect_parity_blobs(
    test_data_dir: Path, class_names: Sequence[str], samples_per_class: int, imgsz: int
) -> tuple[list[np.ndarray], bool]:
    blobs: list[np.ndarray] = []

    if test_data_dir.is_dir():
        import cv2

        for cname in class_names:
            class_dir = test_data_dir / cname
            if not class_dir.is_dir():
                continue
            images = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
            for path in images[:samples_per_class]:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                blobs.append(preprocess_classify(image, imgsz))

    if blobs:
        return blobs, False

    rng = np.random.default_rng(1337)
    synthetic = [rng.random((1, 3, imgsz, imgsz), dtype=np.float32) for _ in range(max(samples_per_class, 1))]
    return synthetic, True


# dict verify_parity(dict tinygrad_state_dict, str arch, int num_classes, int imgsz, Path onnx_path, Path test_data_dir, Sequence[str] class_names, int samples_per_class, float atol)
# Inputs: dict tinygrad_state_dict - tinygrad {name: Tensor} state dict, as returned by
#                 load_checkpoint (loaded fresh into a new tinygrad model here, independent of
#                 build_torch_resnet's already-converted torch copy)
#         str arch - "resnet18" or "resnet34"
#         int num_classes - number of output classes
#         int imgsz - square input spatial size
#         Path onnx_path - path to the just-exported ONNX file
#         Path test_data_dir - classification test-split directory for real parity images
#         Sequence[str] class_names - class names in model output order (used only to know which
#                 test_data_dir subdirectories to sample)
#         int samples_per_class - how many real images per class to check (or synthetic samples,
#                 if none are found)
#         float atol - absolute-difference tolerance reported against
# Outputs: dict - {"num_samples": int, "used_synthetic_fallback": bool, "max_abs_diff": float,
#          "atol": float, "within_atol": bool, "argmax_agreement": float in [0, 1]}
# Description: Runs the same real input blobs through (a) a fresh tinygrad model with
#              `tinygrad_state_dict` loaded (tinygrad.nn.state.load_state_dict, on whatever
#              Device.DEFAULT this machine resolves to -- no device guard) and (b) an
#              onnxruntime.InferenceSession on `onnx_path`, and compares the two raw-logit
#              outputs per sample. Reports both a max-abs-diff (informational: two independent
#              math stacks drift slightly even when both are correct, so this is not, by itself,
#              a pass/fail signal) and an argmax-agreement rate (the decision-relevant check --
#              this is what actually determines whether the exported model classifies inputs the
#              same way the source tinygrad model does).
# Side Effects: Imports onnxruntime (lazily). Constructs a tinygrad model and an
#               onnxruntime.InferenceSession (loads `onnx_path` from disk). Runs one forward pass
#               per sample through each backend. May read test images from disk (via
#               _collect_parity_blobs).
def verify_parity(
    tinygrad_state_dict: dict,
    arch: str,
    num_classes: int,
    imgsz: int,
    onnx_path: Path,
    test_data_dir: Path,
    class_names: Sequence[str],
    samples_per_class: int,
    atol: float,
) -> dict:
    import onnxruntime as ort

    tg_model = build_tg_model(arch, num_classes=num_classes, pretrained=False)
    tg_load_state_dict(tg_model, dict(tinygrad_state_dict), verbose=False)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    blobs, used_synthetic_fallback = _collect_parity_blobs(test_data_dir, class_names, samples_per_class, imgsz)

    diffs: list[float] = []
    agreements: list[bool] = []
    for blob in blobs:
        tg_out = tg_model(Tensor(blob)).numpy()
        onnx_out = session.run(None, {input_name: blob})[0]
        _, diff = arrays_agree(tg_out, onnx_out, atol)
        diffs.append(diff)
        agreements.append(int(np.argmax(tg_out, axis=-1)[0]) == int(np.argmax(onnx_out, axis=-1)[0]))

    overall_max_diff = max(diffs) if diffs else 0.0
    argmax_agreement = (sum(agreements) / len(agreements)) if agreements else 1.0

    return {
        "num_samples": len(blobs),
        "used_synthetic_fallback": used_synthetic_fallback,
        "max_abs_diff": overall_max_diff,
        "atol": atol,
        "within_atol": overall_max_diff <= atol,
        "argmax_agreement": argmax_agreement,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Loads the tinygrad checkpoint (failing loudly if required
#              metadata is missing), builds a torchvision module from it, exports that to ONNX
#              with embedded class-name metadata, runs the tinygrad/ONNX Runtime parity check,
#              and prints a verification report. Raises AssertionError if argmax agreement is
#              not 100% -- that is the one condition this script treats as an export failure,
#              since it means the exported model would classify at least one input differently
#              than the source tinygrad model (see verify_parity's docstring for why max-abs-diff
#              alone isn't used as the pass/fail gate).
# Side Effects: Reads the checkpoint from disk; writes the exported (and metadata-patched) ONNX
#               file to `--out`, creating its parent directory; constructs a tinygrad model and
#               an onnxruntime.InferenceSession; may read real test images from disk for the
#               parity check; prints export progress and a verification report to stdout; raises
#               AssertionError if argmax agreement is not 100%.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tensors, metadata = load_checkpoint(args.checkpoint)
    class_names = parse_class_names(metadata)
    arch = metadata["arch"]
    imgsz = int(metadata["imgsz"])
    num_classes = int(metadata["num_classes"])

    if num_classes != len(class_names):
        raise ValueError(
            f"checkpoint metadata inconsistency: num_classes={num_classes} but class_names has "
            f"{len(class_names)} entries {list(class_names)} -- these must match exactly."
        )

    print(
        f"[export_tinygrad_onnx] Loaded checkpoint '{args.checkpoint}': arch={arch} imgsz={imgsz} "
        f"num_classes={num_classes} class_names={list(class_names)}"
    )

    torch_model = build_torch_resnet(tensors, arch, num_classes)
    print(f"[export_tinygrad_onnx] Loaded {arch} torchvision module with matching state dict (strict=True) -- OK")

    out_path = export_onnx(torch_model, imgsz, args.opset, class_names, args.out)
    print(
        f"[export_tinygrad_onnx] Exported ONNX (opset={args.opset}, static 1x3x{imgsz}x{imgsz}, raw logits, "
        f"class-name metadata embedded) -> {out_path}"
    )

    print(f"[export_tinygrad_onnx] Verifying tinygrad/ONNX Runtime parity against '{args.test_data}' ...")
    report = verify_parity(
        tensors, arch, num_classes, imgsz, out_path, args.test_data, class_names, args.samples_per_class, args.atol
    )

    print()
    print("=" * 88)
    print("TINYGRAD -> ONNX EXPORT PARITY VERIFICATION")
    print("=" * 88)
    print(f"Model path:       {out_path}")
    print(f"Samples checked:  {report['num_samples']} "
          f"({'synthetic fallback (no test images found)' if report['used_synthetic_fallback'] else 'real test images'})")
    print(
        f"Max abs diff:     {report['max_abs_diff']:.6g}  (atol={args.atol:g}) -- "
        f"{'within tolerance' if report['within_atol'] else 'EXCEEDS TOLERANCE (see note below)'}"
    )
    print(f"Argmax agreement: {report['argmax_agreement'] * 100:.1f}%")
    print("=" * 88)

    if report["argmax_agreement"] < 1.0:
        num_disagreements = round((1.0 - report["argmax_agreement"]) * report["num_samples"])
        raise AssertionError(
            f"tinygrad and ONNX Runtime disagree on the predicted class for {num_disagreements} of "
            f"{report['num_samples']} sample(s). The exported ONNX model would classify at least one input "
            "differently than the source tinygrad model -- this must be 100% before shipping to the Pi."
        )

    if not report["within_atol"]:
        print(
            f"[export_tinygrad_onnx] NOTE: max abs diff {report['max_abs_diff']:.6g} exceeds --atol={args.atol:g}, "
            "but argmax predictions agreed on every sample (see above) -- this is expected numeric drift between "
            "two independent math stacks (tinygrad vs. PyTorch/ONNX Runtime), not a class-order or export bug."
        )


if __name__ == "__main__":
    main()
