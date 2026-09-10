# Hailo-8 (AI HAT+) deployment

This records how to get the trained failure classifier running on the
Raspberry Pi 5's AI HAT+ (Hailo-8, 26 TOPS), end to end, up to the one step
that cannot happen on either of your machines: the actual HEF compile. That
step needs x86_64 Linux and runs under WSL2 on your Windows PC (see below).
Everything before and after it -- the ONNX re-export, the calibration set,
the Pi-side install and config, and the mandatory post-quantization
threshold re-derivation -- is covered here.

**Primary model: YOLO26s-cls** (`runs/train/yolo26s_cls_bin_v2/weights/best.pt`
-> `models/yolo26s_cls_bin_opset11.onnx`). This is the model this document
targets end to end. It is the winner of three `training/train.py` retrains
(see `runs/eval_yolo26s_final.json` for the full evaluation and the README's
training log for how the three variants compared) -- measured test-split
top-1 0.9789 (324/331), 3 false positives out of 236 held-out `normal`
images. That is a marginal, not a decisive, improvement over the incumbent:

| Model | Test top-1 | Failure P / R | False positives / 236 normal | Zero-FP threshold | Recall there |
|---|---|---|---|---|---|
| **ResNet18** (`models/argus_bin_opset11.onnx`) | 0.9637 | 0.988 / 0.884 | **1** | 0.75 | 0.674 |
| YOLO26s, first run (undertrained, `--patience 8`) | 0.9789 | 0.958 / 0.968 | 4 | ~0.998 | 0.811 |
| **YOLO26s, this doc** (`yolo26s_cls_bin_v2`) | 0.9789 | 0.968 / 0.958 | 3 | 0.99723 | 0.779 |

Properly training YOLO26s (`--patience 20` instead of the first run's buggy
`--patience 8`) traded one false positive for one false negative relative to
the first run -- same top-1, one fewer false alarm -- but ResNet18 still has
the best false-positive count of the three by a comfortable margin, and by
far the most comfortable (least fragile) zero-FP threshold. **If the
Hailo-8's quantized zero-FP threshold re-derivation below doesn't hold up,
ResNet18 remains the documented fallback** -- see the mandatory
re-derivation section for exactly when to reach for it.

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
   UNVERIFIED, added after this repo's `ultralytics==8.4.146` pin).**
   Ultralytics added a one-command export that owns the entire pipeline
   (`.pt` -> ONNX -> Hailo parse -> INT8 optimize -> HEF compile) behind
   `model.export(format="hailo", name="hailo8")`, announced by Hailo's own
   community forum on 2026-08-20 and documented at
   <https://docs.ultralytics.com/integrations/hailo>. Classification models
   -- explicitly including YOLO26-cls -- are on its validated list. If this
   is available in whatever `ultralytics` version you install in your WSL2
   Ubuntu environment (check `pip show ultralytics`; this repo's own
   `yolo26s-cls.pt` training happened on `ultralytics==8.4.146`, and it is
   not confirmed here whether that version or a newer one is what first
   shipped this feature), it collapses Steps 1 and 3 below into:
   ```
   pip install ultralytics
   pip install /path/to/hailo_dataflow_compiler-*.whl   # from Hailo's Developer Zone, DFC v3.x for Hailo-8
   yolo export model=runs/train/yolo26s_cls_bin_v2/weights/best.pt \
       format=hailo name=hailo8 imgsz=320 data=datasets/argus_bin
   ```
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
pipeline fails.** The winning checkpoint,
`runs/train/yolo26s_cls_bin_v2/weights/best.pt`, was trained with
Ultralytics' default export opset (12) the one time it was auto-exported
during training; Hailo's ONNX parser requires **opset 11**. Feeding the DFC
an opset-12 graph typically fails during translation, sometimes with an
error that doesn't obviously point at the opset at all -- if the DFC's
`translate_onnx_model` step fails in a way that doesn't make sense, check
the opset first.

Re-export a **separate** opset-11 ONNX file on this Mac (do not overwrite
any opset-12 `.onnx` already in `models/` -- nothing else in this repo reads
`models/yolo26s_cls_bin_opset11.onnx` unless you point it there explicitly,
so bringing up the Hailo path can't regress anything already working):

```
PYTHONPATH=src /opt/anaconda3/bin/python3 training/export_classifier_onnx.py \
    --weights runs/train/yolo26s_cls_bin_v2/weights/best.pt \
    --imgsz 320 --opset 11 \
    --out models/yolo26s_cls_bin_opset11.onnx \
    --test-data datasets/argus_bin/test
```

This is the same export-and-verify script that produced
`models/yolo26s_cls_bin_opset11.onnx` for this document's own numbers above:
it re-exports via Ultralytics' own ONNX exporter, then verifies the output
shape with a raw onnxruntime pass AND runs the real `ClassifierDetector`
against real held-out test images end to end, so a broken export fails loud
here rather than silently on the Pi.

Four details read straight from the exported model's own ONNX metadata
(`onnxruntime.InferenceSession.get_modelmeta()`), not guessed -- you'll need
them for Step 3:

- **Input node: `images`**, shape `(1, 3, 320, 320)`, NCHW, float32, scaled
  to `[0, 1]`.
- **Output node: `output0`**, shape `(1, 2)`. **Unlike the ResNet18/tinygrad
  ONNX path (`models/argus_bin_opset11.onnx`, output node `logits`, raw
  logits), this graph's own final op is a `Softmax`** -- verified by running
  a random input through it and confirming the two output values sum to
  exactly 1.0. `argus.detectors.classifier.probabilities_from_output` (used
  by both `ClassifierDetector` and `HailoDetector.predict_proba`)
  auto-detects raw-logits-vs-already-softmaxed either way, so this doesn't
  require any code change -- but it matters for Step 3's `translate_onnx_model`
  call, where the DFC needs to know it's translating a graph that already
  ends in a Softmax.
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

ONNX_PATH = "yolo26s_cls_bin_opset11.onnx"
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
# (e.g. `onnx.load("yolo26s_cls_bin_opset11.onnx").graph.node`, or open it in
# https://netron.app) to find that node's real output name, since it is not
# reproduced here to avoid stating a name that may not match your exact
# export. Either choice works with this repo's own evaluation and runtime
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
# runs/eval_yolo26s_final.json was measured against.
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
hailomz compile --ckpt yolo26s_cls_bin_opset11.onnx \
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
    failure: 0.99723   # PLACEHOLDER -- SEE THE WARNING BELOW, this value is NOT valid
                        # for the quantized model yet, and is especially unlikely to
                        # survive unchanged given how close it sits to 1.0
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
and this model's operating point is unusually exposed to that shift.**

> **PROMINENT WARNING, specific to this model.** YOLO26s-cls's own measured
> zero-false-positive threshold is **0.99723** -- extremely close to the
> softmax ceiling of 1.0 (see `runs/eval_yolo26s_final_finesweep.json`; a
> finer local sweep pinned the exact false-positive boundary to
> `(0.997205, 0.99726)`, a window barely 0.00005 wide). INT8 gives you
> roughly 256 distinct quantization levels across a tensor's whole dynamic
> range, and the levels available in the last thousandth of a softmax output
> approaching 1.0 are sparse to begin with even before quantization error is
> considered -- there is almost no resolution left up there for INT8 to
> represent faithfully. This is NOT a hypothetical concern the way it might
> be phrased for a model with more headroom: it is entirely plausible that
> quantization moves this specific model's zero-FP boundary by more than
> 0.00005, collapses the gap between "zero FP" and "no positive predictions
> at all," or removes the zero-FP operating point at a usable recall
> entirely. **Do not deploy `class_thresholds.failure: 0.99723` (or any
> value carried over unmodified from the fp32 measurement) against the
> compiled HEF.** Re-derive it for real, below, before this config is
> trusted for anything beyond generating notifications you personally
> review.

Re-run the real evaluation script, on the Pi (or any Linux box with
`hailo_platform` installed and the HEF reachable), against the compiled HEF,
on the same held-out test split the fp32 number came from:

```
PYTHONPATH=src python3 training/evaluate_classifier.py \
    --weights models/yolo26s_cls_bin.hef \
    --data datasets/argus_bin --split test \
    --imgsz 320 \
    --catastrophic-class failure --class-names failure,normal \
    --sweep-start 0.90 --sweep-end 0.9999 --sweep-step 0.0005 \
    --out runs/eval_yolo26s_hailo_hef.json
```

(`--data datasets/argus_bin` matters -- the script's own default,
`datasets/argus_cls`, is the unrelated 6-class dataset. The wide
`--sweep-start`/`--sweep-end` matters too: the fp32 model's own zero-FP
point sits above the script's ordinary default sweep range of 0.05-0.95, and
there is no reason to assume the quantized model's own zero-FP point, if it
has one, lands somewhere more convenient.)

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

Read the printed `[4] p_failure ANALYSIS` sweep the same way `0.99723` was
chosen for the fp32 model above: find the lowest confidence threshold that
reaches zero false positives (precision 1.000) at a real, non-vacuous
recall, and put that number -- not `0.99723` unmodified -- into
`class_thresholds.failure` in your Hailo config, along with a comment
recording the HEF's own measured recall at that threshold the same way this
document records the fp32 number.

**If the quantized model can't reach precision 1.000 at any usable recall,
that is real information, not a bug to work around.** Given how thin this
model's fp32 margin already is (a 0.00005-wide window), that outcome
would not be surprising. If it happens: **`models/argus_bin_opset11.onnx`
(the ResNet18 model) is the documented fallback and remains available** --
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
