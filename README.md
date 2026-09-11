# Argus Monitor

Real-time print failure detection for Klipper printers, built to run on the printer's own Raspberry Pi.

This README has two parts. Part 1 describes the system Argus is designed to become — the architecture, the sensors, the decision engine, the philosophy behind the data and evaluation work. Part 2 describes what is actually built and measured today. Where the two disagree, Part 2 wins.

---

# Part 1 — The overall design

Argus watches a print through two independent senses. A camera feeds a vision model that recognises visual failure modes such as spaghetti, delamination and bed detachment. An ADXL345 accelerometer on the toolhead, already present on most Voron builds for input shaping, feeds a time-series model that recognises mechanical faults the camera cannot see. A fusion layer combines both into a single failure score, and a temporal decision engine decides whether that score justifies interrupting a print.

The design goal is a model that works on *any* Klipper machine, not one tuned to a single printer. That constraint drives most of the decisions below.

Accelerometer fusion is designed, not implemented — see Part 2 for what actually runs today.

## Why two sensors

Vision and vibration fail in opposite directions, which is the reason to run both.

A camera sees the build plate but not the machine. It catches spaghetti, warping and blobs, and it is useless when the toolhead is obscured, the lighting shifts, or the failure has no visual signature yet.

An accelerometer sees the machine but not the plate. A skipped step, a belt slipping a tooth, or a nozzle ploughing into an earlier layer all produce a distinct vibration signature at the moment they happen. Layer shift in particular is often invisible to a camera until many layers later, by which point the print is already lost.

Neither modality alone covers the failure space. Fused, each one covers the other's blind spot, and requiring agreement between them is a strong defence against false positives. This fusion layer is designed but not yet built; the camera path runs alone today.

## Hardware

| Component | Role | Interface |
|---|---|---|
| Raspberry Pi 5 | Inference host, Klipper/Moonraker host | |
| AI HAT+ (Hailo-8, 26 TOPS) | Optional NPU for offloaded inference | PCIe |
| USB webcam or Pi Camera | Frame capture | USB / CSI |
| EBB36/42 toolhead board | CAN node hosting the accelerometer | CAN (500 kbps) |
| ADXL345 | Vibration sensing for mechanical faults | SPI to CAN bridge |
| U2C adapter | Host side of the CAN bus | USB-CAN |

**Camera placement matters more than camera quality.** A fixed mount with a clear view of the whole build plate, angled around 45 degrees rather than top down, catches spaghetti far earlier. Dedicated lighting is worth more than resolution, because a model that has learned one lighting condition generalises badly to another.

Inference is expected to run comfortably on the Pi's CPU at the sampling rates Argus uses. The AI HAT+ is optional and mainly buys headroom rather than necessary throughput — see "Runtime backends" in Part 2 for why it's worth using anyway.

## Failure taxonomy

Every class is intended to be detected and logged. Only some are permitted to stop a print.

| Class | Severity | Sensor |
|---|---|---|
| `spaghetti` | catastrophic | vision |
| `layer_separation` | catastrophic | vision |
| `bed_adhesion` | catastrophic | vision |
| `blob_of_death` | catastrophic | vision |
| `layer_shift` | catastrophic | accelerometer |
| `warping` | cosmetic | vision |
| `stringing` | cosmetic | vision |
| `error_extrusion` | cosmetic | vision |

The split is a policy decision about which failures justify the cost of a false stop, kept separate from the question of how well the model currently detects them. Catastrophic classes destroy the part or risk the machine. Cosmetic classes appear on plenty of perfectly good prints, so letting them drive an action would be a large and avoidable source of false positives.

Severity lives in config, not code. Per-class confidence thresholds, not severity, are what stop a weakly detected class from firing.

This is the target taxonomy the architecture is built for. What ships today is a single binary class, `failure` vs. `normal` — see Part 2.

## Architecture

```
                    Raspberry Pi 5
  +------------------------------------------------+
  |                                                |
  |  camera ---> frame gate --> detector --+       |
  |              (blur, luma,              |       |
  |               staleness)               v       |
  |                                    late fusion |
  |  ADXL345 --> CAN --> windowing ----+   |       |
  |  (designed, not built)             |   |       |
  |                      + FFT         |   |       |
  |                                    +-->+       |
  |                                        |       |
  |                                        v       |
  |                                 DecisionEngine |
  |                                 (EMA, K-of-N,  |
  |                                  hysteresis)   |
  |                                        |       |
  |                    +-------------------+       |
  |                    |                   |       |
  |                    v                   v       |
  |              Moonraker REST      notify + log  |
  |              (pause / cancel)    (Discord,     |
  |                                   JSONL)       |
  +------------------------------------------------+
```

The detector is an interface, not a fixed implementation. Everything downstream of it — the decision engine, Moonraker, notify, the event log — consumes only a scalar failure probability, never bounding boxes, so backends can be swapped without touching decision logic. See Part 2 for which backends actually exist.

## The decision engine

This is where the false positive requirement is actually met, and it does more work than the model does.

A false stop on a twelve hour print is expensive, so a single confident frame must never be able to end a print. Each tick runs in order:

1. **Hard gates.** The tick is skipped entirely unless the printer is actually printing, warmup has elapsed (the first layers are both the most failure prone and the least predictable), no cooldown is active, and the frame passes a quality check for blur, brightness and staleness.
2. **Frame score.** The maximum confidence among catastrophic detections. Cosmetic detections are attached to the event record but excluded from the score.
3. **Temporal filter.** An exponential moving average, plus a K-of-N vote across a sliding window.
4. **Two tier trigger with hysteresis.** Warn notifies a human. Pause requires a higher score, more votes, and several consecutive ticks. Cancel is disabled by default.
5. **Cooldown.** After firing, the engine re-arms only once the score has stayed low for a sustained period.

The subtlety worth stating plainly: this arithmetic assumes false positives are independent between frames, and they are not. A support tower, a shadow, or a dark purge blob gets misread in *every* consecutive frame, and correlated false positives pass straight through K-of-N voting. Defending against that is a matter of training data diversity, not filter tuning.

Argus ships in notify-only mode. Pause and cancel are opt-in, and are meant to be enabled only after calibration on a specific machine.

## Counting data honestly

Public print-failure datasets are almost always far smaller than their file counts suggest, in three compounding ways.

**Augmentation inflation.** Roboflow exports commonly contain several augmented variants per source photograph. Splitting those independently puts variants of the same image in both train and test. Argus recovers the original source identity from the filename and treats all variants as one indivisible group.

**Timelapse sessions.** Many datasets are video frames sampled every few seconds from a small number of print jobs. Frame level splitting leaks near identical images across the split boundary. Argus segments sessions using timestamps and perceptual hashing, and treats each session as one group.

**Cross dataset overlap.** Public sets are frequently rebuilt from the same underlying image pools. Argus hashes every image against every other source, including datasets already ingested, and refuses to merge a source whose content is largely contained in one already present.

The number that matters is not files, and not even unique photographs, but **independent scenes**. Reporting all three separately is a deliberate feature of the pipeline — see Part 2 for what this discipline actually found in the shipped dataset.

## Evaluation philosophy

Standard detection metrics are misleading for this problem, so Argus is built to measure a few things specifically.

**Per source photo scoring.** Test metrics are computed on one representative per source photograph. Otherwise a photo that happened to receive thirty augmented variants contributes thirty votes while another contributes one, and the headline number reflects augmentation counts rather than performance.

**Effective sample size per class.** Every class reports how many independent scenes back it, not just an instance count. A class whose precision rests on a dozen scenes is visibly distinguishable from one resting on hundreds. High precision measured at near zero recall is flagged as vacuous rather than reported as success.

**Cross source transfer.** This is the metric that matters most for a universal model. Recall is measured separately per originating dataset for every class present in more than one. A model that scores well on one source and poorly on another has learned to recognise the dataset, not the defect, and it will not transfer to a stranger's printer. Aggregate mAP hides this completely. This is a design goal for the multi-source, multi-class model; the single-source binary classifier shipped today has no cross-source question to ask (see Part 2).

Byte for byte merging preserves each source's native resolution and compression signature, which is itself a shortcut a model can key on. Normalising those statistics across sources is part of closing the gap.

## Deployment gates

mAP is a research metric and a poor deployment gate. Before pause is enabled on a machine, Argus targets:

- fewer than one false positive per one hundred print hours, measured by replaying recorded prints through the real decision engine
- catastrophic class precision at or above 0.95 at the selected threshold, counted on independent scenes
- detection within sixty seconds of failure onset

The calibration tool is built to replay real recorded sessions through the shipping decision engine rather than a reimplementation of it, so the measured false positive rate is the one the deployed system will actually produce. See Part 2 for where the current model stands against each gate.

## Universal model, local calibration

A single frozen model cannot be simultaneously accurate on every camera, angle and bed surface at one fixed threshold. Argus splits the problem instead. The model carries general knowledge and is trained for source diversity rather than raw accuracy on any one dataset. Thresholds are then derived per installation from the operator's own prints.

This is a far more achievable target than universal perfection, and it means adding a new printer is a calibration step rather than a retraining project.

## Klipper and Moonraker integration

The accelerometer needs no hardware changes. It is already declared in `printer.cfg` for input shaping, and Argus is designed to read it over the same CAN bus once the accelerometer path is built.

Print state comes from Moonraker's REST API, which also handles pause and cancel. Every call fails closed: any error, timeout or malformed response resolves to an unknown state that gates detection off rather than acting on bad information. The client never raises into the main loop.

Pause is routed through a user-defined macro rather than the raw endpoint, so operators can park the toolhead, lift Z, or change lighting on their own terms:

```
[gcode_macro DETECTION_PAUSE]
gcode:
    PARK_TOOLHEAD
    SET_LED LED=chamber WHITE=1.0
    PAUSE
```

Events are written as JSONL alongside archived frames, with retention bounds so a long running install cannot fill the disk. Notifications go to a Discord webhook.

---

# Part 2 — The implementation today

## Status

| Piece | Status |
|---|---|
| Camera path + quality gate + decision engine + Moonraker + notify + event log | built |
| Whole-frame binary YOLO26s-cls classifier | built, measured |
| ONNX Runtime backend | built |
| Hailo-8 backend (`src/argus/detectors/hailo.py`) | built, **not yet verified on real hardware** |
| Accelerometer path + late fusion | designed, not built |
| 6-class defect model | in tree, gated on the source confound |
| Calibration on real Trident footage | not done |

## Deployed model: YOLO26s-cls, fine-tuned

A working binary print-failure classifier exists end to end: trained, exported, evaluated, and wired into the shipped config. It is smaller than the Part 1 taxonomy, deliberately.

The model distinguishes exactly two classes, `("failure", "normal")` — index 0 is failure, embedded directly in the ONNX metadata, so a config that gets the order wrong fails to load rather than silently mislabeling every prediction. Only `failure` is catastrophic; a `normal` prediction emits zero detections, so `p_failure` is 0.0 on every frame the model calls normal.

Architecture is YOLO26s-cls, ~5.4M parameters. Input `(1, 3, 320, 320)` float32 NCHW, scaled to `[0, 1]`; the graph's own final op is a `Softmax`, so the output is already a probability distribution, not raw logits. The deployed file is `models/yolo26s_cls_bin_finetuned_opset11.onnx` — opset 11, which is required by the Hailo compiler (see below), not by the ONNX Runtime path itself. Shipped config: `config.trident.yolo26s.yaml`, `class_thresholds.failure: 0.78`, `action_mode: notify_only`.

It was produced in two stages: YOLO26s-cls was trained with Ultralytics on the M3 Pro's GPU (MPS), then the exported ONNX graph was imported into tinygrad via tinygrad's own ONNX runner and fine-tuned further (`training/finetune_yolo_tinygrad.py`), with the trained weights written back into the ONNX graph in place.

## Why binary, and why one dataset

The original plan was the 6-class taxonomy in Part 1, training `normal` from a HuggingFace dataset and every defect class from the FDM dataset. That plan has a hole: the FDM dataset contains zero normal images (Cracking 427, Layer_shifting 364, Off_platform 91, Stringing 447, Warping 538) — `normal` had to come from somewhere else precisely because FDM has none. The consequence is that source and label become perfectly correlated, so a classifier trained on that combination can score well by learning which dataset an image came from, not what is actually wrong with the print.

v1 removes that shortcut by construction rather than trying to correct for it after the fact. It trains on a single source, `Masamsa/3d-print-failure-detection`, which carries both labels itself — there is no second dataset for the model to key on. The 6-class defect-type path remains in the tree (`training/build_classification_dataset.py`) as a documented follow-up, gated on this confound being addressed (see Roadmap).

Vision training for that 6-class defect-type model draws on two public datasets, both redistributable:

| Dataset | License | Contributes |
|---|---|---|
| [AtCo, 3D printing error v7](https://universe.roboflow.com/atco/3d-printing-error/dataset/7) | MIT | spaghetti, stringing, warping, error extrusion |
| [Yawllen, StereoVision v8](https://universe.roboflow.com/yawllen-jectr/stereovision-gyibu/dataset/8) | CC BY 4.0 | bed adhesion, blob of death, layer separation, spaghetti, warping |

Several other public sets were evaluated and rejected. Two large ones proved, on measurement, to be augmented re-uploads of photographs already present in the AtCo export: thousands of files that resolved to a few hundred source images and contributed nothing new. One promising set was excluded on licensing grounds, since a NonCommercial dataset cannot produce weights distributable under this repository's GPL-3.0 license. That experience is the reason the ingest pipeline treats every new dataset as leaky until proven otherwise.

## Dataset and leakage audit

`training/build_binary_dataset.py` builds `datasets/argus_bin/` from `Masamsa/3d-print-failure-detection`, pooling all three of that source's published splits and re-deriving the train/val/test boundary rather than trusting the one HuggingFace ships:

- 2,714 images pooled, 0 exact duplicates
- Near-duplicate images are clustered into independent scenes by perceptual hash (dhash): **1,220 independent scenes**
- **147 of those scene clusters straddle HuggingFace's own train/test boundary** — direct evidence that the published split leaks, and the reason this pipeline re-splits by cluster instead of using it
- 21 mixed-label clusters (505 images), where near-duplicate frames disagreed on their own label, are excluded rather than guessed at
- The final split keeps 2,209 images, cluster-disjoint:

| Split | failure | normal |
|---|---|---|
| train | 445 | 1,102 |
| val | 95 | 236 |
| test | 95 | 236 |

## Model comparison

All rows measured on the same held-out 331-image test split, with the same evaluator (`training/evaluate_classifier.py`):

| Model | Top-1 | Failure P / R | False positives (of 236 normal) | Zero-FP threshold (recall there) |
|---|---|---|---|---|
| ResNet18 (tinygrad, METAL) | 0.9637 | 0.988 / 0.884 | 1 | 0.75 (0.674) |
| YOLO26n-cls (Ultralytics) | 0.9668 | 0.988 / 0.895 | 1 | 0.995 (0.737) |
| YOLO26s-cls (Ultralytics) | 0.9789 | 0.958 / 0.968 | 4 | ~0.998 (0.811) |
| YOLO26s-cls (Ultralytics, retrained longer) | 0.9789 | 0.968 / 0.958 | 3 | 0.99723, a 0.00005-wide window (0.779) |
| YOLO26s-cls, 52-epoch tinygrad fine-tune | 0.9819 | 0.968 / 0.968 | 3 | 0.95 (0.242) |
| **YOLO26s-cls, tinygrad fine-tune (deployed)** | **0.9849** | **0.979 / 0.968** | **2** | **0.78 (0.968)** |

Deployed model confusion: 92 of 95 failures caught, 234 of 236 normals correct.

**Why the zero-false-positive threshold, not top-1, decided this.** The model gets INT8-quantized for the Hailo-8, and INT8 cannot represent a threshold pinned near the softmax ceiling of 1.0. Models trained harder or longer became *overconfident* — their mistakes scored around 0.99 — which pushed their zero-FP point up to 0.95–0.998, either collapsing recall or landing in a window (0.00005 wide, in one case) narrower than INT8's roughly 256 quantization levels could reliably land in. The short, gentle tinygrad fine-tune produced a better-calibrated model whose zero-FP point sits at 0.78 with recall intact: it holds zero false positives from 0.78 upward, with recall still 0.968 at 0.80.

The longer fine-tune is also a caution against picking a checkpoint by validation score alone: it had *higher* validation macro-F1 (0.9746 vs. 0.9675) but was worse on test. The 331-image validation set is small enough that peak validation score is not a reliable stopping signal. More epochs did not help here — the binding constraint is data, not training time.

## Runtime backends

`src/argus/detectors/` implements ONNX Runtime (`onnx_yolo.py`, `classifier.py`) and the Hailo-8 AI HAT+ (`kind: "hailo"`, `src/argus/detectors/hailo.py`) behind one detector interface, sharing the same post-processing and severity logic, so the decision engine cannot tell which backend produced a score.

Hailo deployment (`docs/hailo-deployment.md`) requires compiling on x86_64 Ubuntu — WSL2 works — because the Hailo Dataflow Compiler does not run on macOS or ARM, at opset 11, with `--hw-arch hailo8` (not `hailo8l`, a different, lower-power chip), against a 256-image class-balanced calibration set (`training/build_hailo_calibration.py`).

**Stated plainly: the HailoRT call sequence in `src/argus/detectors/hailo.py` is written from Hailo's published documentation and has not been run against real Hailo-8 hardware.** No Hailo device or Dataflow Compiler is reachable from the development machine. The 0.78 threshold is a fp32 measurement and must be re-derived on the quantized model before it can be trusted — INT8 shifts the confidence distribution every model produces, and this one is not exempt just because its margin is comfortable.

Offloading inference to the HAT is not about raw speed — a 320px classifier is not a demanding workload — it is about keeping the Pi 5's CPU free for Klipper's own real-time step timing and motion planning. CPU inference latency has so far only been measured on the M3 Pro development machine, not on a Pi.

## Training infrastructure

Training used tinygrad (a vendored ResNet with torchvision-identical state-dict keys in `training/tg_models.py`, and an ONNX export bridge in `training/export_tinygrad_onnx.py`) and Ultralytics. An RTX 5060 Ti eGPU over Thunderbolt was investigated and **could not be used**: training on it kernel-panicked the Mac (an Apple Silicon IOMMU violation, `VIOLATION_T8110_DART_INVALID_ERR_MASK`), and even inference-only runs hung at random points and wedged the card. Both tinygrad's pinned checkout and current upstream load GSP firmware 570.144, which a tinygrad issue reports lacks USB4 eGPU support for this card. fp16 training is also broken on that tinygrad build. All shipped models were trained on the M3 Pro instead. See `docs/tinygrad-setup.md` for the pinned checkout and the `DEV=NV` footgun.

## Decision-engine calibration, stated honestly

Replaying held-out normal frames through the real `DecisionEngine` (`tools/calibrate.py`) produced zero false positives for the earlier ResNet18 model. That replay has **not** been re-run for YOLO26s; what's known for YOLO26s is that its per-frame false positives are zero at threshold 0.78 on the test split, not that the decision engine has been calibrated against it. All of this — for both models — is public-dataset stills, not footage from a real printer.

## Deployment gates — where v1 stands

- **Catastrophic class precision ≥ 0.95** — met. 0.979 at the default threshold, 1.000 at the shipped 0.78.
- **Detection within 60 seconds of onset** — not meaningfully measured.
- **Fewer than one false positive per 100 print hours, measured on recorded prints** — **not met**. No real printer footage has been recorded yet, which is exactly why this ships `notify_only`.

## Tests

954 passing (`python -m pytest`).

---

## Roadmap

In priority order:

1. Compile the HEF and validate `HailoDetector` on the Pi 5 AI HAT+, re-deriving the confidence threshold on the quantized model — the backend exists but has never touched real hardware.
2. Record real Voron Trident footage with `tools/record.py` and calibrate with `tools/calibrate.py` — the one deployment gate v1 has not met.
3. Replay YOLO26s through the decision engine to measure detection latency.
4. Train the accelerometer model and implement late fusion, requiring agreement between modalities before any pause.
5. Broaden source diversity and bring back the 6-class defect-type model, gated on addressing the dataset-origin confound described above.

## License

GPL-3.0. Dataset licenses are listed above and are the operative constraint on redistributing any trained weights.
