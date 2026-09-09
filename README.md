# Argus Monitor

Real-time print failure detection for Klipper printers, built to run on the printer's own Raspberry Pi.

Argus watches a print through two independent senses. A camera feeds a YOLO26 object detector that recognises visual failure modes such as spaghetti, delamination and bed detachment. An ADXL345 accelerometer on the toolhead, already present on most Voron builds for input shaping, feeds a time-series model that recognises mechanical faults the camera cannot see. A fusion layer combines both into a single failure score, and a temporal decision engine decides whether that score justifies interrupting a print.

The design goal is a model that works on *any* Klipper machine, not one tuned to a single printer. That constraint drives most of the decisions below.

---

## Why two sensors

Vision and vibration fail in opposite directions, which is the reason to run both.

A camera sees the build plate but not the machine. It catches spaghetti, warping and blobs, and it is useless when the toolhead is obscured, the lighting shifts, or the failure has no visual signature yet.

An accelerometer sees the machine but not the plate. A skipped step, a belt slipping a tooth, or a nozzle ploughing into an earlier layer all produce a distinct vibration signature at the moment they happen. Layer shift in particular is often invisible to a camera until many layers later, by which point the print is already lost.

Neither modality alone covers the failure space. Fused, each one covers the other's blind spot, and requiring agreement between them is a strong defence against false positives.

---

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

Inference runs comfortably on the Pi's CPU at the sampling rates Argus uses. The AI HAT+ is optional and mainly buys headroom rather than necessary throughput.

---

## Failure taxonomy

Every class is detected and logged. Only some are permitted to stop a print.

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

---

## Architecture

```
                    Raspberry Pi 5
  +------------------------------------------------+
  |                                                |
  |  camera ---> frame gate ---> YOLO26 ---+       |
  |              (blur, luma,              |       |
  |               staleness)               v       |
  |                                    late fusion |
  |  ADXL345 --> CAN --> windowing ----+   |       |
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

The detector is an interface, not a fixed implementation. ONNX Runtime on CPU, a Hailo-compiled model on the AI HAT+, and a whole-frame classifier all satisfy the same contract, so the runtime can switch backends without touching the decision logic. The decision engine only ever consumes a scalar failure probability, never bounding boxes.

ONNX Runtime is the only inference dependency at runtime. Training pulls in Ultralytics, which is AGPL-3.0, but nothing shipped to the printer does.

---

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

---

## Training data

Vision training uses two public datasets, both redistributable:

| Dataset | License | Contributes |
|---|---|---|
| [AtCo, 3D printing error v7](https://universe.roboflow.com/atco/3d-printing-error/dataset/7) | MIT | spaghetti, stringing, warping, error extrusion |
| [Yawllen, StereoVision v8](https://universe.roboflow.com/yawllen-jectr/stereovision-gyibu/dataset/8) | CC BY 4.0 | bed adhesion, blob of death, layer separation, spaghetti, warping |

Several other public sets were evaluated and rejected. Two large ones proved, on measurement, to be augmented re-uploads of photographs already present in the AtCo export: thousands of files that resolved to a few hundred source images and contributed nothing new. One promising set was excluded on licensing grounds, since a NonCommercial dataset cannot produce weights distributable under this repository's GPL-3.0 license.

That experience is the reason the ingest pipeline treats every new dataset as leaky until proven otherwise.

### Counting data honestly

Public print-failure datasets are almost always far smaller than their file counts suggest, in three compounding ways.

**Augmentation inflation.** Roboflow exports commonly contain several augmented variants per source photograph. Splitting those independently puts variants of the same image in both train and test. Argus recovers the original source identity from the filename and treats all variants as one indivisible group.

**Timelapse sessions.** Many datasets are video frames sampled every few seconds from a small number of print jobs. Frame level splitting leaks near identical images across the split boundary. Argus segments sessions using timestamps and perceptual hashing, and treats each session as one group.

**Cross dataset overlap.** Public sets are frequently rebuilt from the same underlying image pools. Argus hashes every image against every other source, including datasets already ingested, and refuses to merge a source whose content is largely contained in one already present.

The number that matters is not files, and not even unique photographs, but **independent scenes**. Reporting all three separately is a deliberate feature of the pipeline.

---

## Evaluation

Standard detection metrics are misleading for this problem, so Argus measures a few things specifically.

**Per source photo scoring.** Test metrics are computed on one representative per source photograph. Otherwise a photo that happened to receive thirty augmented variants contributes thirty votes while another contributes one, and the headline number reflects augmentation counts rather than performance.

**Effective sample size per class.** Every class reports how many independent scenes back it, not just an instance count. A class whose precision rests on a dozen scenes is visibly distinguishable from one resting on hundreds. High precision measured at near zero recall is flagged as vacuous rather than reported as success.

**Cross source transfer.** This is the metric that matters most for a universal model. Recall is measured separately per originating dataset for every class present in more than one. A model that scores well on one source and poorly on another has learned to recognise the dataset, not the defect, and it will not transfer to a stranger's printer. Aggregate mAP hides this completely.

Byte for byte merging preserves each source's native resolution and compression signature, which is itself a shortcut a model can key on. Normalising those statistics across sources is part of closing the gap.

---

## Deployment gates

mAP is a research metric and a poor deployment gate. Before pause is enabled on a machine, Argus targets:

- fewer than one false positive per one hundred print hours, measured by replaying recorded prints through the real decision engine
- catastrophic class precision at or above 0.95 at the selected threshold, counted on independent scenes
- detection within sixty seconds of failure onset

The calibration tool replays real recorded sessions through the shipping decision engine rather than a reimplementation of it, so the measured false positive rate is the one the deployed system will actually produce.

### Universal model, local calibration

A single frozen model cannot be simultaneously accurate on every camera, angle and bed surface at one fixed threshold. Argus splits the problem instead. The model carries general knowledge and is trained for source diversity rather than raw accuracy on any one dataset. Thresholds are then derived per installation from the operator's own prints.

This is a far more achievable target than universal perfection, and it means adding a new printer is a calibration step rather than a retraining project.

---

## Klipper and Moonraker integration

The accelerometer needs no hardware changes. It is already declared in `printer.cfg` for input shaping, and Argus reads it over the same CAN bus.

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

## Roadmap

- Train the accelerometer model and implement late fusion, requiring agreement between modalities before any pause
- Broaden source diversity per class, which is the binding constraint on universality
- Normalise image statistics across sources to remove dataset origin as a learnable shortcut
- Add a Hailo backend so the AI HAT+ is usable without changing the decision path
- Record real print sessions and calibrate thresholds against measured false positive rates
- Graduate from notify-only to pause on a per machine basis, gated on the criteria above

## License

GPL-3.0. Dataset licenses are listed above and are the operative constraint on redistributing any trained weights.
