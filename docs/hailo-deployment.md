# Hailo-8 (AI HAT+) deployment

This records how to get the trained failure classifier running on the
Raspberry Pi 5's AI HAT+ (Hailo-8, 26 TOPS), end to end, up to the one step
that cannot happen on either of your machines: the actual HEF compile. That
step needs x86_64 Linux and runs under WSL2 on your Windows PC (see below).
Everything before and after it -- the ONNX re-export, the calibration set,
the Pi-side install and config, and the mandatory post-quantization
threshold re-derivation -- is covered here.

**Primary model: YOLO26s-cls, fine-tuned in tinygrad**
(`models/yolo26s_cls_bin.onnx` -> `models/yolo26s_cls_bin_finetuned.onnx` ->
`models/yolo26s_cls_bin_finetuned_opset11.onnx`). This is the model this
document targets end to end. Rather than retraining the Ultralytics
classification head from scratch, `training/finetune_yolo_tinygrad.py`
imports the exported ONNX graph directly via tinygrad's own ONNX runner and
fine-tunes it in place (batch 32, lr 1e-4, early-stopping patience 8 on
validation macro-F1 -- best val macro-F1 0.9675 at epoch 9 of 40; see
`runs/logs/finetune_yolo_tinygrad_metal.log`), writing the trained weights
back into a new ONNX file with the exact same graph shape, opset, and
`names` metadata the runtime already reads. On the held-out test split (331
images, 95 `failure` / 236 `normal`) this measures top-1 **0.9849** (326/331)
and, critically, the widest, most usable zero-false-positive operating point
of every variant tried:

| Model | Test top-1 | Failure P / R | False positives / 236 normal | Zero-FP threshold | Recall there |
|---|---|---|---|---|---|
| ResNet18 (`models/argus_bin_opset11.onnx`) | 0.9637 | 0.988 / 0.884 | 1 | 0.75 | 0.674 |
| YOLO26s Ultralytics retrain (`yolo26s_cls_bin_v2`, `models/yolo26s_cls_bin_opset11.onnx`) | 0.9789 | 0.968 / 0.958 | 3 | 0.99723 (0.00005-wide window) | 0.779 |
| YOLO26s tinygrad fine-tune, patience 30 (`models/yolo26s_cls_bin_ft_p30.onnx`) | higher val F1 | -- | -- | 0.95 | 0.242 (rejected: recall collapses) |
| **YOLO26s tinygrad fine-tune, patience 8 (this doc)** | **0.9849** | **0.979 / 0.968** | **2** | **~0.78 (0.775-0.805, a 30-point-wide window)** | **0.968** |

Two other candidates were measured and rejected on this same test split
before settling on the patience-8 fine-tune: the Ultralytics retrain above
(`yolo26s_cls_bin_v2`, formerly this document's primary model) reaches zero
false positives only in a 0.00005-wide window pinned against the softmax
ceiling -- too narrow for INT8's ~256 quantization levels to reliably land
in -- and a longer tinygrad fine-tune (patience 30) has a higher validation
F1 but only reaches zero false positives at recall 0.242, which throws away
most of its ability to actually catch failures. The patience-8 fine-tune
beats both on the metric that matters for an automated print-pause signal:
a zero-FP threshold with real recall and real margin. **If the Hailo-8's
quantized zero-FP threshold re-derivation below doesn't hold up, ResNet18
remains the documented fallback** -- see the mandatory re-derivation section
for exactly when to reach for it.

Read `argus.detectors.hailo`'s module docstring before going further. It
documents the input-format mismatch this whole page exists to get right (Hailo
wants uint8 NHWC pixels with normalization baked into the HEF, not the ONNX
path's float32 NCHW), and it is explicit about which parts of the HailoRT call
sequence are verified against real hardware and which are written from
Hailo's published API and still need on-device confirmation. Nothing in this
repo has been run against a real Hailo-8 device -- this Mac cannot install
`hailo_platform`, and there is no Hailo device attached to it. Treat the
HailoRT-specific and Dataflow-Compiler-specific commands below the same way:
as the best-documented path, not a proven one, and verify each one against
your actual installed SDK versions as you go.

---

## Why the compile can't happen on the Mac or the Pi

The Hailo Dataflow Compiler (DFC) -- the tool that quantizes a floating-point
model to INT8 and compiles it into a `.hef` -- is published for **Ubuntu
20.04 or 22.04, x86_64, with 16+ GB of RAM.** There is no macOS build and no
ARM build. The Raspberry Pi 5 is ARM64; this development Mac is
Apple Silicon. Both are valid **deployment** targets for the compiled `.hef`
(the Pi to run it, in principle even an x86_64 dev box to test it), but
neither can run the compiler itself.

Your Windows PC gets there via **WSL2 running Ubuntu 22.04**. Install WSL2 if
you haven't (`wsl --install -d Ubuntu-22.04` from an elevated PowerShell,
then reboot when prompted), then do everything in this document's "compile"
section (Step 3) inside that Ubuntu 22.04 environment, not in Windows
itself. The Hailo Dataflow Compiler and `hailo_model_zoo` (`hailomz`) are
installed from Hailo's Developer Zone
(<https://hailo.ai/developer-zone/>, free account required) as a Python
wheel matched to your WSL2 Ubuntu version and Python version -- follow
Hailo's own install instructions for the wheel you download, since the exact
package name and supported Python versions change between DFC releases.

---

## Three ways to compile: which one to actually use

Three distinct paths get a `.hef` out of this model. **None of the three has
been run in this repo** -- there is no Hailo hardware or DFC install
reachable from this Mac (see above) -- so all three are written from
published documentation, with sources cited, not from a successful local
run. Pick based on what your WSL2 setup already has installed.

1. **Ultralytics' own `format="hailo"` export (newest, likely simplest --
   UNVERIFIED, added after this repo's `ultralytics==8.4.146` pin, and
   NOT DIRECTLY APPLICABLE to the fine-tuned model this doc targets -- see
   caveat below).** Ultralytics added a one-command export that owns the
   entire pipeline (`.pt` -> ONNX -> Hailo parse -> INT8 optimize -> HEF
   compile) behind `model.export(format="hailo", name="hailo8")`, announced
   by Hailo's own community forum on 2026-08-20 and documented at
   <https://docs.ultralytics.com/integrations/hailo>. Classification models
   -- explicitly including YOLO26-cls -- are on its validated list. This
   path starts from a `.pt` checkpoint, e.g.:
   ```
   pip install ultralytics
   pip install /path/to/hailo_dataflow_compiler-*.whl   # from Hailo's Developer Zone, DFC v3.x for Hailo-8
   yolo export model=runs/classify/runs/train/yolo26s_cls_bin/weights/best.pt \
       format=hailo name=hailo8 imgsz=320 data=datasets/argus_bin
   ```
   **Caveat specific to this doc's primary model:** `training/finetune_yolo_tinygrad.py`
   fine-tunes ONNX initializer values directly via tinygrad's own ONNX
   runner -- there is no `.pt` checkpoint that carries the fine-tuned
   weights, only `models/yolo26s_cls_bin_finetuned.onnx` itself. The command
   above, run against the pre-fine-tune `.pt`, would compile the WEAKER base
   checkpoint (test top-1 lower than the fine-tuned model's 0.9849, see the
   comparison table above), not the model this document targets. Path 3
   below (the DFC's own Python API, working from the ONNX graph directly) is
   the one that actually operates on the fine-tuned weights, which is the
   other reason it is this document's primary path.
   `data=` points it at a classification dataset in the same
   `<root>/train/<class>/*.jpg` layout `training/build_classification_dataset.py`
   already produces, for its own internal calibration sampling -- **UNVERIFIED
   here** whether it accepts `datasets/argus_bin` directly in that role or
   expects a `data.yaml` manifest; confirm against whatever version you
   install. One caveat straight from Hailo's own announcement thread, worth
   taking seriously: as of that August 2026 post, quantized accuracy from
   this Ultralytics path was reported as measurably behind the Hailo Model
   Zoo path specifically for YOLO26 models -- so treat this as the fastest
   thing to *try first*, not an automatic replacement for Step 3's more
   hands-on route if the accuracy comparison (Step "MANDATORY" below) comes
   out worse.

2. **`hailomz compile` (Hailo Model Zoo's own CLI wrapper).** Works cleanly
   when an official model-zoo YAML config already matches your architecture.
   As of this writing, no official `hailo_model_zoo` config for a YOLO26-cls
   (or plain 2-class) classifier was found in the model zoo's public
   `cfg/networks/` directory (<https://github.com/hailo-ai/hailo_model_zoo>)
   -- **UNVERIFIED, may have changed since**; check your installed
   `hailo_model_zoo` version's own `cfg/networks/` for a `yolo26*-cls` or
   generic classifier entry before assuming you need to write one from
   scratch. If none exists, this route requires hand-writing a custom YAML
   inheriting from `hailo_model_zoo/cfg/base/`, which is enough extra work
   that path 3 below is more direct for a one-off custom-trained model like
   this one.

3. **The DFC's own Python API (`hailo_sdk_client.ClientRunner`) --
   documented in full below.** The most direct path for a custom-trained
   model like this one: it works from the ONNX graph directly, no model-zoo
   catalog entry required. This is the path with a complete script below,
   and the one this document treats as primary for that reason -- not
   because it's necessarily the best-quantizing option (see point 1's
   caveat), but because it's the one fully specified here rather than
   deferred to a tool version you haven't installed yet.

---

## Step 1 -- re-export the ONNX model at opset 11

**Opset 11 is mandatory. This is the single most common way this whole
pipeline fails.** `models/yolo26s_cls_bin_finetuned.onnx` (the fine-tuned
checkpoint, see above) was written at opset 12 -- the opset the base
Ultralytics export (`models/yolo26s_cls_bin.onnx`) used before tinygrad
fine-tuned its weights in place. Hailo's ONNX parser requires **opset 11**.
Feeding the DFC an opset-12 graph typically fails during translation,
sometimes with an error that doesn't obviously point at the opset at all --
if the DFC's `translate_onnx_model` step fails in a way that doesn't make
sense, check the opset first.

`models/yolo26s_cls_bin_finetuned_opset11.onnx` already exists in this repo,
produced with:

```
python3 -c "
import onnx
from onnx import version_converter

m = onnx.load('models/yolo26s_cls_bin_finetuned.onnx')
converted = version_converter.convert_version(m, 11)
onnx.checker.check_model(converted)
onnx.save(converted, 'models/yolo26s_cls_bin_finetuned_opset11.onnx')
"
```

`onnx.version_converter` operates on the graph in place and, unlike a fresh
Ultralytics re-export, does not need the original `.pt` checkpoint or a
GPU/MPS device -- and it preserves `metadata_props` (including the `names`
entry `argus.detectors.classifier` reads) untouched. Verified on all 331
held-out test images through onnxruntime CPU: **100% argmax agreement, 0.0
max absolute difference** against the opset-12 model it was converted
from -- the two files are numerically identical on every image in the test
split, not just close. If `version_converter` ever fails or produces a graph
onnxruntime can't run for some future fine-tune, the fallback is to export
the base Ultralytics checkpoint
(`runs/classify/runs/train/yolo26s_cls_bin/weights/best.pt`) at opset 11 via
`training/export_classifier_onnx.py --opset 11 --imgsz 320`, then copy the
fine-tuned initializer values from `models/yolo26s_cls_bin_finetuned.onnx`
into it by matching initializer names/shapes/dtypes -- but for this
checkpoint the direct version-convert route worked cleanly and is what
`models/yolo26s_cls_bin_finetuned_opset11.onnx` actually is.

Three details read straight from the exported model's own ONNX metadata
(`onnxruntime.InferenceSession.get_modelmeta()`), not guessed -- you'll need
them for Step 3:

- **Input node: `images`**, shape `(1, 3, 320, 320)`, NCHW, float32, scaled
  to `[0, 1]`.
- **Output node: `output0`**, shape `(1, 2)`. **Unlike the ResNet18/tinygrad
  ONNX path (`models/argus_bin_opset11.onnx`, output node `logits`, raw
  logits), this graph's own final op is a `Softmax`, so `output0` is
  ALREADY a valid probability distribution** -- verified by running a random
  input through it and confirming the two output values sum to exactly 1.0.
  `argus.detectors.classifier.probabilities_from_output` (used by both
  `ClassifierDetector` and `HailoDetector.predict_proba`) auto-detects
  raw-logits-vs-already-softmaxed either way, so this doesn't require any
  code change -- but it matters for Step 3's `translate_onnx_model` call,
  where the DFC needs to know it's translating a graph that already ends in
  a Softmax.
- **Class order: `{0: 'failure', 1: 'normal'}`**, read from
  `custom_metadata_map["names"]` -- alphabetical, and happens to match
  `config.trident.yaml`'s ResNet18 order, but confirmed independently rather
  than assumed from that coincidence.

---

## Step 2 -- build the calibration set

```
/opt/anaconda3/bin/python3 training/build_hailo_calibration.py \
    --out datasets/hailo_calib \
    --n 256 \
    --input-size 320
```

This step is **unchanged from, and shared with, the ResNet18 path** --
`training/build_hailo_calibration.py` samples 256 images (128 `failure`,
128 `normal` -- class-balanced, seeded, deterministic; see the script's
module docstring for why both of those properties matter to quantization
quality) from `datasets/argus_bin/train`, resizes and center-crops each one
with the exact same geometry `argus.detectors.classifier.preprocess_classify`
uses at inference time (via the shared `resize_and_center_crop`), and writes
them as flat JPEGs into `datasets/hailo_calib/`. Nothing about this step is
model-specific -- both models share the same 320x320 input geometry and the
same source dataset -- so if you already built this directory for the
ResNet18 path, reuse it unmodified; there is no need to rebuild it for
YOLO26s. Copy the directory over to the WSL2 side along with the opset-11
ONNX file from Step 1 -- both are needed for Step 3.

---

## Step 3 -- quantize and compile (inside WSL2 Ubuntu 22.04, with the DFC installed)

This is path 3 from the comparison above: the DFC's own Python API
(`hailo_sdk_client.ClientRunner`), the most direct route for a
custom-trained model with no matching model-zoo catalog entry.
**The exact API below is written from Hailo's published DFC documentation,
not run against a real installation** -- confirm class/method names and
signatures against the DFC User Guide that ships alongside your installed
SDK version (`hailo_sdk_client` is versioned; method signatures have changed
across DFC releases) before trusting this script.

```python
# compile_hailo.py -- run inside WSL2 Ubuntu 22.04, with the Hailo DFC installed.
import glob
import numpy as np
import cv2
from hailo_sdk_client import ClientRunner

ONNX_PATH = "yolo26s_cls_bin_finetuned_opset11.onnx"
CALIB_DIR = "hailo_calib"
HEF_OUT = "yolo26s_cls_bin.hef"
INPUT_SIZE = 320

runner = ClientRunner(hw_arch="hailo8")  # NOT "hailo8l" -- see warning below

# Translate the opset-11 ONNX graph. Node names and static shape are read
# straight from the exported model's own metadata (Step 1), not guessed.
# end_node_names=["output0"] includes the graph's own final Softmax --
# UNVERIFIED whether your installed DFC's ONNX parser accepts a Softmax op
# at the very end of the graph without complaint. If translate_onnx_model
# rejects it, the alternative is to stop one node earlier, at the Softmax's
# own input (the pre-softmax Gemm output) -- inspect the graph directly
# (e.g. `onnx.load("yolo26s_cls_bin_finetuned_opset11.onnx").graph.node`, or
# open it in https://netron.app) to find that node's real output name, since
# it is not reproduced here to avoid stating a name that may not match your
# exact export. Either choice works with this repo's own evaluation and runtime
# code unmodified -- probabilities_from_output auto-detects raw-logits-vs
# -already-softmaxed either way (see Step 1) -- so this is purely a DFC
# compatibility question, not a correctness one.
runner.translate_onnx_model(
    ONNX_PATH,
    "yolo26s_cls_bin",
    start_node_names=["images"],
    end_node_names=["output0"],
    net_input_shapes={"images": [1, 3, INPUT_SIZE, INPUT_SIZE]},
)

# CRITICAL: bake the /255 normalization into the HEF here, as a model-script
# command, so HailoDetector's uint8 NHWC input (raw 0-255 pixels, no
# host-side scaling -- see argus.detectors.hailo's module docstring) lands
# on the model exactly the way the fp32 ONNX path's [0, 1]-scaled input did.
# The mean/std argument order and exact command name below are the part of
# this script most likely to need adjusting for your installed DFC version
# -- check the "normalization" command in your DFC User Guide's model-script
# reference. Getting this step wrong or skipping it does not raise an error
# anywhere in this pipeline; it just silently and systematically skews every
# confidence the compiled model produces relative to the fp32 ONNX model
# runs/eval_yolo26s_finetuned_opset11.json was measured against.
runner.load_model_script(
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
)

# Load the calibration set built in Step 2, in the same geometry the
# deployed model will see (already resized/cropped to 320x320 by
# training/build_hailo_calibration.py) -- uint8 NHWC, RGB.
calib_paths = sorted(glob.glob(f"{CALIB_DIR}/*.jpg"))
calib_dataset = np.stack(
    [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in calib_paths]
).astype(np.uint8)

runner.optimize(calib_dataset)  # INT8 post-training quantization
hef = runner.compile()

with open(HEF_OUT, "wb") as f:
    f.write(hef)
print(f"wrote {HEF_OUT}")
```

Run it:

```
python3 compile_hailo.py
```

**`hw_arch="hailo8"`, not `"hailo8l"`.** The AI HAT+ in this project's
hardware table is the **26 TOPS Hailo-8**. The 13 TOPS variant is a
*different chip*, **Hailo-8L**, used on the lower-power AI HAT (no plus) and
some other Hailo-8L-based accessories. `hw_arch="hailo8l"` will compile
without error and produce a `.hef` -- it will simply refuse to run (or run
against the wrong chip architecture) on this hardware. There is no runtime
check on the Pi that catches a `hailo8`/`hailo8l` mixup before you try to
load the HEF; confirm which HAT you actually have before compiling. (Note
that Ultralytics' own `format="hailo"` path from option 1 above defaults to
`hailo8l` if you don't pass `name="hailo8"` explicitly -- the same mixup
risk applies there too.)

After compiling, sanity-check the HEF's own reported I/O contract with
HailoRT's own inspection tool (installed alongside HailoRT in Step 4) before
ever pointing a live camera at it:

```
hailortcli parse-hef yolo26s_cls_bin.hef
```

Confirm it reports a uint8 input and the shape/order you expect. If it
doesn't, that's a real signal something upstream (the model script, the
translate step's node names) needs revisiting -- better to catch it here
than from a silently miscalibrated confidence stream on the printer.

If you'd rather use the `hailomz compile` CLI wrapper instead (option 2
above), the command shape is:

```
hailomz compile --ckpt yolo26s_cls_bin_finetuned_opset11.onnx \
    --calib-path datasets/hailo_calib \
    --yaml path/to/your-classifier.yaml \
    --hw-arch hailo8
```

but, per the comparison above, this requires either an exact matching
`hailo_model_zoo` YAML config (none was confirmed to exist for a YOLO26-cls
or generic 2-class classifier as of this writing -- check your installed
version's own `cfg/networks/` directory) or writing a custom one inheriting
from `hailo_model_zoo/cfg/base/`. Consult `hailo_model_zoo`'s own
documentation for the flags your installed version supports; they have
changed across releases and are not fully reproduced here to avoid stating a
command that turns out to be version-specific and wrong.

---

## Step 4 -- install HailoRT on the Pi and copy the HEF over

On Raspberry Pi OS (Bookworm or later), the AI HAT+'s runtime, firmware, and
`hailortcli` install via:

```
sudo apt update
sudo apt install hailo-all
sudo reboot
```

This is Raspberry Pi's own documented install path for the AI HAT+; consult
<https://www.raspberrypi.com/documentation/> for the current instructions if
`hailo-all` isn't available on your OS image (package names and the
supported OS version have moved before and may again). Confirm the device is
visible afterward:

```
hailortcli fw-control identify
```

Copy the compiled HEF from the WSL2 side to the Pi (adjust host/path):

```
scp yolo26s_cls_bin.hef pi@<printer-host>:~/argus-monitor/models/yolo26s_cls_bin.hef
```

---

## Step 5 -- config change

`config.trident.yolo26s.yaml` already ships this as a commented-out block --
uncomment it and comment out the `kind: "classification"` block above it
once the HEF is compiled, copied, and its threshold re-derived (next
section):

```yaml
detector:
  kind: "hailo"
  model_path: "models/yolo26s_cls_bin.hef"
  input_size: 320
  class_names: ["failure", "normal"]   # same order as the opset-11 ONNX export's own
                                        # metadata (Step 1) -- NOT recoverable from the
                                        # HEF itself (see below)
  default_threshold: 0.50
  class_thresholds:
    failure: 0.78   # PLACEHOLDER -- SEE THE WARNING BELOW, this value is the fp32
                     # measurement and is NOT valid for the quantized model until
                     # re-derived on the compiled HEF
    normal: 0.50
  severity:
    failure: catastrophic
    normal: cosmetic
```

Unlike the ONNX path, `class_names` here is not cross-checked against
anything embedded in the model file -- a compiled HEF carries no equivalent
of ONNX's `custom_metadata_map`, so `HailoDetector` trusts `cfg.class_names`
outright and fails loudly at startup only if it's empty, not if it's wrong.
Get the order wrong and every prediction is silently mislabeled with no
error. The correct order is whatever the opset-11 ONNX export's own metadata
says in Step 1 -- for this model that's `["failure", "normal"]`.

---

## MANDATORY: re-derive the confidence threshold before trusting this

**INT8 quantization shifts the confidence distribution the model produces --
this is true of every model it quantizes, and this one is not exempt just
because its fp32 margin is comfortable.**

> **This model's zero-FP threshold sits mid-range, not fragile -- but that's
> not a reason to skip re-deriving it.** YOLO26s-cls's own measured
> zero-false-positive threshold is **~0.78** (precisely: precision reaches
> 1.000 at 0.775 and recall 0.968 holds through 0.805 -- see
> `runs/eval_yolo26s_finetuned_zerofp_pinpoint.json`), a **30-point-wide**
> window, nowhere near the softmax ceiling of 1.0. INT8's roughly 256
> quantization levels across a tensor's dynamic range can represent a window
> this wide without difficulty -- unlike the earlier Ultralytics-retrain
> model's 0.99723, a 0.00005-wide window pinned against that ceiling, which
> INT8 realistically could not have represented. That difference is exactly
> why this model was chosen over that one for the Hailo path. **It does
> NOT mean the threshold survives quantization unexamined.** INT8 still
> shifts the whole confidence distribution the model produces -- by how
> much, in which direction, is not knowable without measuring it on the
> actual compiled HEF. A window being wide only means it is more likely to
> still contain a usable operating point after that shift, not that the
> specific value 0.78 will still be correct. **Do not deploy
> `class_thresholds.failure: 0.78` (or any value carried over unmodified
> from the fp32 measurement) against the compiled HEF.** Re-derive it for
> real, below, before this config is trusted for anything beyond generating
> notifications you personally review.

Re-run the real evaluation script, on the Pi (or any Linux box with
`hailo_platform` installed and the HEF reachable), against the compiled HEF,
on the same held-out test split the fp32 number came from:

```
PYTHONPATH=src python3 training/evaluate_classifier.py \
    --weights models/yolo26s_cls_bin.hef \
    --data datasets/argus_bin --split test \
    --imgsz 320 \
    --catastrophic-class failure --class-names failure,normal \
    --sweep-start 0.60 --sweep-end 0.95 --sweep-step 0.005 \
    --out runs/eval_yolo26s_hailo_hef.json
```

(`--data datasets/argus_bin` matters -- the script's own default,
`datasets/argus_cls`, is the unrelated 6-class dataset. The
`--sweep-start`/`--sweep-end` range is centered on the fp32 model's own
zero-FP window (0.775-0.805) with margin on both sides, not the script's
ordinary default of 0.05-0.95 in coarse 0.025 steps -- there is no reason to
assume the quantized model's own zero-FP point, if it has one, lands exactly
where the fp32 model's did, so scan around it rather than only at it.)

This runs the exact same confusion matrix / per-class P-R / confidence-sweep
report `training/evaluate_classifier.py` has always produced, just routed
through the Hailo backend (`run_inference_hailo`, added alongside this
deployment doc) that drives `HailoDetector.predict_proba` under the hood --
so the sweep measures precisely the code path that will run in production,
not a reimplementation of it. `--class-names` is required here in a way it
isn't for the ONNX backend: a HEF has no embedded class-name metadata for
the script to read the way it reads ONNX's `custom_metadata_map`, so there's
no fallback if you get this flag wrong or omit it -- the script refuses to
run without it rather than guessing.

Read the printed `[4] p_failure ANALYSIS` sweep the same way `0.78` was
chosen for the fp32 model above: find the lowest confidence threshold that
reaches zero false positives (precision 1.000) at a real, non-vacuous
recall, and put that number -- not `0.78` unmodified -- into
`class_thresholds.failure` in your Hailo config, along with a comment
recording the HEF's own measured recall at that threshold the same way this
document records the fp32 number.

**If the quantized model can't reach precision 1.000 at any usable recall,
that is real information, not a bug to work around.** This model's fp32
margin is comfortable (a 30-point-wide window), so that outcome would be
more surprising here than it would have been for the rejected 0.99723
candidate -- but it is still not guaranteed, and must be measured, not
assumed. If it happens: **`models/argus_bin_opset11.onnx` (the ResNet18
model) is the documented fallback and remains available** --
its own zero-FP threshold, 0.75, was measured with real margin on both
sides (precision is already 0.988 flat from confidence 0.05 to 0.70, only
climbing to 1.000 at 0.75 -- see `runs/eval_bin_test.json`), which is
exactly the kind of headroom this YOLO26s operating point lacks. Re-deriving
ResNet18's own threshold on ITS quantized HEF is still required before
trusting it either -- INT8 shifts every model's confidence distribution,
not just this one's -- but it starts from a much less fragile place.

---

## Preprocessing contract -- get this wrong and every confidence is silently skewed

Ultralytics classification models (this one included) preprocess with
resize-short-side + center-crop, converting to float32 `[0, 1]` NCHW --
that's what `training/build_classification_dataset.py` trains against, what
`argus.detectors.classifier.preprocess_classify` reproduces exactly for the
ONNX path, and what the exported graph's `images` input (Step 1) expects.

A Hailo HEF's native input is **uint8 NHWC** -- raw 0-255 pixel values,
channel-last, no host-side floating-point scaling. Getting from one contract
to the other requires baking the `/255` normalization into the HEF itself
as a compiled normalization layer (Step 3's `load_model_script` call), not
applying it in Python before or after inference. `HailoDetector` (see
`argus.detectors.hailo`) is written on that assumption:
`preprocess_classify_hailo` performs the exact same resize/crop geometry as
the ONNX path (via the shared `resize_and_center_crop`) but deliberately
stops short of the `/255` scale-and-transpose, because that normalization is
supposed to already be inside the HEF by the time a frame reaches it.

**If the HEF is compiled without that normalization layer, nothing in this
pipeline raises an error.** `HailoDetector` will run, produce a full
probability distribution that sums to 1.0, and look completely normal --
it will just have been computed on pixels 255x too large a scale relative to
what the model was actually trained on, which silently and systematically
skews every confidence value it ever produces. The only way to catch this
after the fact is the mandatory threshold re-derivation above behaving
nothing like the fp32 number (or the model's predictions looking randomly
distributed rather than confidently peaked near 0 or 1), or inspecting the
compiled HEF/model script directly. There is no runtime check that can catch
it from the Python side.

---

## The actual benefit

A 320px binary classifier running once every `tick_interval_s` (1 second by
default) is not a demanding workload -- YOLO26s-cls is 5.4M parameters and
12.3 GFLOPs at this input size, and the README is explicit that inference
already runs comfortably on the Pi 5's own CPU via ONNX Runtime. The AI
HAT+'s 26 TOPS is mainly headroom rather than necessary throughput for this
model. The real win of moving inference onto the Hailo-8 isn't raw speed,
it's **freeing the Pi 5's CPU for Klipper's own real-time work** -- step
timing and motion planning are latency-sensitive in a way that competing
with a CPU-bound inference loop, even an infrequent one, is exactly the kind
of interference you'd rather not introduce onto a machine that's also
running the printer. Offloading this classifier to dedicated silicon is a
correctness-adjacent property (protecting Klipper's real-time guarantees),
not a benchmark chase.
