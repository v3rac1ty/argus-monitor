"""Hailo-8 (AI HAT+) detector backend for the trained whole-frame failure
classifier: runs a compiled `.hef` on the Hailo-8 NPU via HailoRT instead of
ONNX Runtime on the CPU.

**CRITICAL -- input contract differs from the ONNX path.** `ClassifierDetector`
(argus.detectors.classifier) feeds the model float32 NCHW pixels scaled to
[0, 1]. A Hailo HEF instead expects **uint8 NHWC** pixels straight off the
camera (0-255, no host-side scaling) -- `preprocess_classify_hailo` below
performs the exact same resize-short-side + center-crop geometry as
`argus.detectors.classifier.preprocess_classify` (both call the shared
`argus.detectors.classifier.resize_and_center_crop`), but stops short of the
/255 scaling and NCHW transpose that function does. **The /255 normalization
MUST instead be compiled into the `.hef` itself, as a normalization layer in
the Hailo Dataflow Compiler's model script** (see docs/hailo-deployment.md).
If the HEF was compiled WITHOUT that normalization layer, this detector will
still run without error -- it will just feed the model pixels 255x too large,
which silently and systematically skews every confidence value the model
produces. There is no runtime check that can catch this from the Python side;
it can only be caught by re-deriving the confidence threshold on-device (see
docs/hailo-deployment.md) and noticing the model behaves nothing like its
fp32 ONNX counterpart, or by inspecting the compiled HEF/model script
directly.

**`hailo_platform` (HailoRT's Python bindings) is imported lazily**, inside
`HailoDetector.__init__` only, never at module import time. It is not
installable on macOS (Hailo only ships Linux wheels, x86_64 for the
Dataflow Compiler host and aarch64 for the Pi runtime), so importing it
eagerly at module scope would break `import argus.detectors.hailo` -- and
therefore `import argus.detectors` (see `argus/detectors/__init__.py`) --
on every machine without HailoRT installed, including this development Mac
and the CI/test environment. `HailoDetector` can only actually be
*constructed* on a machine with `hailo_platform` installed and a Hailo
device attached; everything else in this module (the pure pre/post-processing
functions) has no such dependency and is unit-tested directly.

**Output side is NOT reimplemented.** `probabilities_from_output` and
`postprocess_classify` are reused unmodified from `argus.detectors.classifier`
-- `HailoDetector.predict_proba` runs the HEF's raw output through
`probabilities_from_output` (exactly the same raw-logits-vs-already-softmaxed
auto-detection the ONNX path uses), and `HailoDetector.infer` hands that
straight to `postprocess_classify`, exactly like `ClassifierDetector.infer`
does with the ONNX session's output, so per-class thresholds, severity
mapping, and the "normal short-circuits to zero detections" rule are
byte-identical between the two backends. `predict_proba` is also what
`training/evaluate_classifier.py`'s Hailo backend calls to get the full
per-class probability vector for its confidence-threshold sweep, instead of
just the single argmax-gated Detection `infer` produces -- see that script
for the exact command to re-derive `class_thresholds.failure` on a quantized
HEF (docs/hailo-deployment.md). `class_names` cannot be recovered from HEF
metadata the way `ClassifierDetector` recovers it from ONNX
`custom_metadata_map` -- a compiled HEF simply does not carry an equivalent
field -- so `HailoDetector` takes `class_names` from `cfg.class_names` only,
and fails loudly at
construction if it is empty (see `argus.config._KINDS_REQUIRING_CLASS_NAMES`,
which also enforces this at config-load time).

**VERIFIED vs UNVERIFIED (read this before trusting this module on real
hardware).** This module was written on a Mac with no Hailo device attached
and no way to install `hailo_platform` (macOS-only development machine; see
the repo's eGPU/tinygrad constraints for why -- unrelated hardware, same
"no way to test on this machine" situation). Nothing HailoRT-specific in
this file has been executed against the real `hailo_platform` package or
real Hailo-8 hardware. Specifically UNVERIFIED, and written from HailoRT's
publicly documented Python API shape (`hailo_platform.VDevice`, `HEF`,
`ConfigureParams`, `InputVStreamParams`/`OutputVStreamParams`,
`InferVStreams`, `HailoStreamInterface`, `FormatType`, and the
`network_group.activate(...)` context manager) rather than confirmed
against a running system:
  - The exact class/function names and call signatures above -- a HailoRT
    version bump can rename or reshape any of these.
  - Whether `InputVStreamParams.make(..., quantized=True,
    format_type=FormatType.UINT8)` is the correct combination for feeding
    already-uint8, already-normalized-by-the-HEF pixels (as opposed to
    `quantized=False` with host-side float32 and letting HailoRT quantize).
    This is the single riskiest guess in this file.
  - Whether `HailoStreamInterface.PCIe` is the right interface enum for the
    AI HAT+ (a PCIe-attached HAT, per README.md's hardware table) as opposed
    to some other value.
  - The exact shape HailoRT's `InferVStreams.infer(...)` returns for a
    single-input, single-output classification HEF -- `_flatten_hailo_output`
    below defensively flattens to 1-D rather than assuming a specific rank,
    but that defensiveness itself is unverified against a real output.
  - Whether holding `VDevice`/`InferVStreams`/`network_group.activate(...)`
    open for the object's entire lifetime (rather than per-call) is
    supported and performant for this HEF -- it is the architecturally
    sensible choice (activating per-frame would add latency on every tick),
    but has not been measured.
Confirm every one of these against the actual installed `hailo_platform`
version and a real Hailo-8 device on the Pi before trusting this in
production. If any HailoRT call in `__init__`/`infer`/`close` raises
`AttributeError` or a signature-mismatch `TypeError`, that is this list
turning out to be wrong in a specific, fixable way -- not a sign the overall
approach is unsound.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from argus.config import DetectorConfig
from argus.detectors.base import Detector
from argus.detectors.classifier import (
    postprocess_classify,
    probabilities_from_output,
    resize_and_center_crop,
)
from argus.types import DetectionResult, Frame

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Pure pre/post-processing functions -- no hailo_platform dependency, fully
# unit-testable on any machine.
# --------------------------------------------------------------------------


# np.ndarray preprocess_classify_hailo(np.ndarray image, int input_size)
# Inputs: np.ndarray image - source BGR frame (HxWx3, uint8) to preprocess
#         int input_size - target square spatial size the compiled HEF expects
# Outputs: np.ndarray - a (1, input_size, input_size, 3) uint8 blob, RGB, channel-last (NHWC)
# Description: Crops via the SAME `resize_and_center_crop` geometry
#              `argus.detectors.classifier.preprocess_classify` uses for the ONNX path, converts
#              BGR->RGB, and adds a batch dimension -- but, unlike the ONNX path, does NOT scale
#              to [0, 1] or transpose to channel-first. Hailo HEF models are compiled for uint8
#              NHWC input, with the /255 normalization baked into the HEF itself as a
#              normalization layer (see this module's docstring and
#              docs/hailo-deployment.md) -- applying /255 here as well would normalize twice and
#              silently wreck every confidence the model produces.
# Side Effects: None (pure function of its inputs).
def preprocess_classify_hailo(image: np.ndarray, input_size: int) -> np.ndarray:
    """Same crop as `argus.detectors.classifier.preprocess_classify`
    (resize-short-side + center-crop via the shared `resize_and_center_crop`),
    but emits uint8 NHWC with NO /255 scaling -- that normalization must be
    baked into the compiled HEF instead, or every confidence is silently
    255x too small a scale relative to what the model was trained on."""
    cropped = resize_and_center_crop(image, input_size)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    blob = np.ascontiguousarray(rgb, dtype=np.uint8)
    return blob[np.newaxis, ...]


# np.ndarray _flatten_hailo_output(Any raw)
# Inputs: Any raw - the raw array HailoRT's InferVStreams.infer(...) returns for the HEF's
#                 single output vstream (exact rank/shape UNVERIFIED against real hardware --
#                 see module docstring)
# Outputs: np.ndarray - a 1-D array of length num_classes
# Description: Defensively flattens whatever shape HailoRT hands back (observed HailoRT
#              examples return shapes like (num_classes,), (1, num_classes), or
#              (1, 1, 1, num_classes) depending on how the HEF's output tensor was compiled) down
#              to the flat (N,) row `postprocess_classify` expects. Assumes batch size 1, which
#              matches `HailoDetector.infer`'s one-frame-per-call usage; a real batched caller
#              would need a different reshape.
# Side Effects: None (pure function of its input).
def _flatten_hailo_output(raw: Any) -> np.ndarray:
    """Assumes batch size 1 (HailoDetector.infer's only usage) -- reshapes
    whatever rank HailoRT returns down to a flat (num_classes,) row rather
    than assuming a specific shape, since that shape is unverified against
    real hardware (see module docstring)."""
    return np.asarray(raw).reshape(-1)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class HailoDetector(Detector):
    """`Detector` implementation backed by a Hailo-8 HEF running through
    HailoRT (`hailo_platform`). See this module's docstring for the
    uint8-NHWC input contract, the output-side reuse of
    `argus.detectors.classifier.postprocess_classify`, and -- importantly --
    which parts of the HailoRT call sequence below are verified versus
    written from documentation only.
    """

    # None __init__(DetectorConfig cfg)
    # Inputs: DetectorConfig cfg - detector configuration: HEF model path, input size, class
    #                 names, thresholds, and severity map (`providers` is unused by this backend
    #                 -- it is an onnxruntime-specific concept)
    # Outputs: None
    # Description: Validates `cfg.class_names` is non-empty (a compiled HEF carries no
    #              equivalent to ONNX's class-name metadata for `ClassifierDetector` to fall back
    #              on), confirms the `.hef` file exists, lazily imports `hailo_platform`, and
    #              brings up the full HailoRT lifecycle: loads the HEF, opens a `VDevice`,
    #              configures a network group from the HEF, and activates input/output vstreams
    #              in uint8 (input) / float32 (output) format, held open for this instance's
    #              lifetime so `infer()` does not pay activation cost on every call.
    # Side Effects: Reads `cfg.model_path` from disk; raises `FileNotFoundError` if it doesn't
    #               exist. Raises `ValueError` if `cfg.class_names` is empty. Raises
    #               `ImportError` if `hailo_platform` is not installed. Allocates real HailoRT
    #               resources (VDevice, network group, input/output vstreams, an activated
    #               network group) -- see this module's docstring for which of these calls are
    #               unverified against real hardware. Logs an info message on success. Mutates
    #               the new instance's state.
    def __init__(self, cfg: DetectorConfig) -> None:
        if not cfg.class_names:
            raise ValueError(
                "HailoDetector requires cfg.class_names to be a non-empty tuple. Unlike "
                "ClassifierDetector, there is no ONNX-metadata-equivalent embedded in a "
                "compiled .hef for this to fall back on or cross-check against -- set "
                "detector.class_names in config to the model's real training-time class order "
                "(the same order used when the model was exported to ONNX, before HEF "
                "compilation -- see docs/hailo-deployment.md)."
            )

        model_path = Path(cfg.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hailo HEF model not found at '{model_path}' -- has it been compiled (on x86_64 "
                "Linux/WSL2, see docs/hailo-deployment.md) and copied to the Pi yet?"
            )

        try:
            import hailo_platform as hpf
        except ImportError as exc:
            raise ImportError(
                "hailo_platform (HailoRT's Python bindings) is not installed. HailoDetector can "
                "only be constructed on a machine with HailoRT installed and a Hailo device "
                "attached (the Raspberry Pi with the AI HAT+, in production) -- see "
                "docs/hailo-deployment.md for the install steps. It cannot run on macOS."
            ) from exc

        self._cfg = cfg
        self._class_names = tuple(cfg.class_names)
        self._input_size = cfg.input_size

        self._hef = hpf.HEF(str(model_path))
        self._vdevice = hpf.VDevice()
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef=self._hef, interface=hpf.HailoStreamInterface.PCIe
        )
        network_groups = self._vdevice.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        network_group_params = self._network_group.create_params()

        input_infos = self._hef.get_input_vstream_infos()
        output_infos = self._hef.get_output_vstream_infos()
        if len(input_infos) != 1 or len(output_infos) != 1:
            raise RuntimeError(
                "HailoDetector only supports a single-input, single-output classification HEF "
                f"-- got {len(input_infos)} input(s) and {len(output_infos)} output(s)"
            )
        self._input_name = input_infos[0].name
        self._output_name = output_infos[0].name

        # Input: raw uint8 pixels, already in the format the HEF's own
        # compiled-in normalization layer expects (see module docstring) --
        # `quantized=True` tells HailoRT this data is already in the model's
        # native uint8 domain, not host-side float32 for it to quantize
        # itself. Output: FLOAT32, so postprocess_classify always sees real
        # (dequantized) probabilities/logits regardless of the HEF's
        # internal on-chip precision.
        input_vstreams_params = hpf.InputVStreamParams.make(
            self._network_group, quantized=True, format_type=hpf.FormatType.UINT8
        )
        output_vstreams_params = hpf.OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        self._infer_pipeline = hpf.InferVStreams(
            self._network_group, input_vstreams_params, output_vstreams_params
        )
        self._infer_pipeline.__enter__()

        try:
            self._activation: Optional[Any] = self._network_group.activate(network_group_params)
            self._activation.__enter__()
        except Exception:
            # Don't leak the pipeline if activation fails.
            self._infer_pipeline.__exit__(None, None, None)
            raise

        logger.info(
            "HailoDetector: loaded %s, input_size=%d, classes=%s",
            model_path,
            self._input_size,
            self._class_names,
        )

    # np.ndarray predict_proba(np.ndarray image)
    # Inputs: np.ndarray image - source BGR frame (HxWx3, uint8) to classify
    # Outputs: np.ndarray - the full per-class probability vector (shape (len(class_names),)),
    #          with NO threshold/severity gating applied
    # Description: Runs one frame through the Hailo-8: `preprocess_classify_hailo` builds the
    #              uint8 NHWC input blob (same crop geometry as the ONNX path, no /255 scaling --
    #              see module docstring), HailoRT's activated infer pipeline runs it through the
    #              HEF, `_flatten_hailo_output` normalizes whatever shape comes back to a flat
    #              row, and `probabilities_from_output` (imported unmodified from
    #              `argus.detectors.classifier`) converts it to a valid probability distribution.
    #              Exists as its own method (rather than being inlined into `infer`) so
    #              `training/evaluate_classifier.py`'s Hailo backend can get the full vector for
    #              its confidence-threshold sweep, not just the single argmax-gated Detection
    #              `infer` produces.
    # Side Effects: Runs Hailo-8 NPU inference via `self._infer_pipeline.infer(...)`. Raises
    #               `RuntimeError` if the detector has already been `close()`d.
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        if self._infer_pipeline is None:
            raise RuntimeError("HailoDetector is closed")

        blob = preprocess_classify_hailo(image, self._input_size)
        raw_outputs = self._infer_pipeline.infer({self._input_name: blob})
        raw = _flatten_hailo_output(raw_outputs[self._output_name])
        return probabilities_from_output(raw)

    # DetectionResult infer(Frame frame)
    # Inputs: Frame frame - the captured camera frame to classify
    # Outputs: DetectionResult - zero or one Detection (see `postprocess_classify`), plus the
    #          measured inference time in milliseconds
    # Description: Runs the full classify pipeline for one frame on the Hailo-8: `predict_proba`
    #              gets the per-class probability vector, and `postprocess_classify` (imported
    #              unmodified from `argus.detectors.classifier`) turns it into a `DetectionResult`
    #              -- identical severity mapping, per-class thresholds, and normal-short-circuit
    #              behavior to the ONNX path. Passing already-normalized probabilities through
    #              `postprocess_classify` (which itself calls `probabilities_from_output` again)
    #              is safe and a no-op: that function detects an already-valid distribution and
    #              returns it unchanged rather than double-softmaxing it.
    # Side Effects: Runs Hailo-8 NPU inference (via `predict_proba`); reads
    #               `time.perf_counter()` to measure elapsed time. Raises `RuntimeError` if the
    #               detector has already been `close()`d.
    def infer(self, frame: Frame) -> DetectionResult:
        start = time.perf_counter()
        probs = self.predict_proba(frame.image)

        detections = postprocess_classify(
            probs,
            self._class_names,
            self._cfg.class_thresholds,
            self._cfg.default_threshold,
            self._cfg.severity,
            frame.image.shape[:2],
        )
        inference_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(detections=detections, inference_ms=inference_ms)

    # None close()
    # Inputs: None
    # Outputs: None
    # Description: Releases the HailoRT lifecycle in reverse acquisition order: deactivates the
    #              network group, tears down the infer vstream pipeline, then releases the
    #              VDevice. Safe to call multiple times and safe to call even if construction
    #              only partially completed. Subsequent `infer()` calls will raise `RuntimeError`.
    # Side Effects: Releases real HailoRT resources (deactivation, vstream pipeline teardown,
    #               VDevice release). Logs (via logger.exception) rather than raising if any
    #               individual step fails, so one failure doesn't prevent releasing the rest --
    #               matching `ArgusService.close()`'s own resilience pattern. Mutates instance
    #               state so a second `close()` call is a no-op.
    def close(self) -> None:
        activation = getattr(self, "_activation", None)
        if activation is not None:
            try:
                activation.__exit__(None, None, None)
            except Exception:
                logger.exception("HailoDetector: error deactivating network group")
            self._activation = None

        infer_pipeline = getattr(self, "_infer_pipeline", None)
        if infer_pipeline is not None:
            try:
                infer_pipeline.__exit__(None, None, None)
            except Exception:
                logger.exception("HailoDetector: error tearing down infer vstream pipeline")
            self._infer_pipeline = None

        vdevice = getattr(self, "_vdevice", None)
        if vdevice is not None:
            try:
                release = getattr(vdevice, "release", None)
                if release is not None:
                    release()
            except Exception:
                logger.exception("HailoDetector: error releasing VDevice")
            self._vdevice = None
