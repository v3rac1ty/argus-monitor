# AI Print Failure Monitor
## Overview

Real-time print failure detection running on the Voron Trident's onboard Raspberry Pi 5. Two sensing modalities — a camera (vision model) and the ADXL345 accelerometer over CAN bus (time-series model) — feed a fusion layer that triggers print control actions on detection.

The system uses a **C++/Python hybrid architecture**. C++ owns the hot path: frame capture, ONNX inference, accel windowing, and fusion. Python owns the orchestration layer: Moonraker REST calls, logging, and Telegram notifications. The two processes communicate over a Unix domain socket. This keeps inference latency tight without rewriting the surrounding glue code.

---

## Hardware Layer

| Component | Role | Interface |
|---|---|---|
| Raspberry Pi 5 | Inference host, Klipper/Moonraker host | — |
| USB webcam / Pi Camera | Frame capture for vision model | USB / CSI |
| EBB36/42 toolhead board | CAN bus node; hosts ADXL345 | CAN (500kbps) |
| ADXL345 | Accelerometer for layer shift detection | SPI → CAN bridge |
| CAN bus (U2C adapter) | Toolhead communication | USB-CAN |

**Camera placement:** Fixed mount with a clear sightline to the full build plate. Side-angle (45°) is better than top-down for catching spaghetti early. Consistent lighting matters — use a dedicated LED strip so inference isn't sensitive to ambient changes.

---

## Firmware / Software Integration Points

### Klipper
- ADXL345 is already registered in `printer.cfg` for input shaper. No hardware changes needed.
- Add a persistent `ACCELEROMETER_MEASURE` stream during prints, or poll via the Klipper socket.
- The detection service reads Klipper's Unix domain socket at `/tmp/klippy_uds` to query printer state (printing, paused, idle) before acting on detections.

### Moonraker
- REST API at `http://localhost:7125` used for all print control actions.
- Endpoints used:
  - `POST /printer/print/pause` — pause on detection
  - `POST /printer/print/cancel` — cancel on high-confidence detection
  - `GET /printer/objects/query?print_stats` — poll print state
  - `GET /server/info` — health check

### Klipper Macros (optional)
Define a `DETECTION_PAUSE` macro in `printer.cfg` that runs custom gcode on pause (e.g. park the toolhead, turn on lights). The detection service calls this macro instead of the raw pause endpoint so behavior is user-configurable.

```
[gcode_macro DETECTION_PAUSE]
gcode:
    PARK_TOOLHEAD
    SET_LED LED=chamber WHITE=1.0
    PAUSE
```

---

## Software Architecture

Two processes, one Unix socket between them.

```
┌──────────────────────────────────────────────────────────┐
│                     Raspberry Pi 5                        │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              C++ Inference Process                 │  │
│  │                                                   │  │
│  │  ┌─────────────┐      ┌────────────────────────┐  │  │
│  │  │ Camera Feed │      │  CAN Socket Reader     │  │  │
│  │  │ (V4L2/CSI)  │      │  (linux/can.h)         │  │  │
│  │  │ OpenCV C++  │      └───────────┬────────────┘  │  │
│  │  └──────┬──────┘                  │               │  │
│  │         │                ┌────────▼─────────┐     │  │
│  │  ┌──────▼──────┐         │  Accel Windowing │     │  │
│  │  │  YOLOv8n   │         │  + FFT (FFTW3)   │     │  │
│  │  │  ORT C++   │         └────────┬─────────┘     │  │
│  │  └──────┬──────┘                  │               │  │
│  │         │                ┌────────▼─────────┐     │  │
│  │         │                │  1D CNN  ORT C++ │     │  │
│  │         │                └────────┬─────────┘     │  │
│  │         │                         │               │  │
│  │  ┌──────▼─────────────────────────▼───────────┐  │  │
│  │  │  Late Fusion + Threshold Logic (C++)        │  │  │
│  │  └──────────────────────┬─────────────────────┘  │  │
│  └─────────────────────────┼─────────────────────── ┘  │
│                             │ Unix domain socket         │
│                    JSON event payload                    │
│                    {class, confidence, frame_path}       │
│                             │                           │
│  ┌──────────────────────────▼──────────────────────┐   │
│  │              Python Orchestration Process        │   │
│  │                                                  │   │
│  │   Moonraker REST  │  Logger  │  Telegram notify  │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Why this split:** ONNX Runtime's C++ API has zero binding overhead and OpenCV frame capture + preprocessing is faster without the GIL. The orchestration side (REST calls, disk I/O, notifications) is I/O-bound so Python is fine and saves a lot of boilerplate.

---

## Data Pipeline

### Camera (C++)
1. Capture frames via OpenCV C++ (`cv::VideoCapture`) at 5–10 fps during active printing.
2. Resize to 416x416, normalize to [0,1] — all in-process with no copy to Python.
3. Run ONNX inference via ORT C++ API → bounding boxes + class confidences.
4. Classes: `nominal`, `spaghetti`, `adhesion_failure`, `blob`.
5. Confidence score fed to fusion layer in the same process.

### Accelerometer (C++)
1. Read ADXL345 data directly from the CAN socket via `linux/can.h` — no Klipper socket intermediary.
2. Buffer into 0.5s windows (50Hz sample rate → 25 samples/window) in a ring buffer.
3. Compute FFT via FFTW3 on each axis → frequency feature vector.
4. Run 1D CNN via ORT C++ API → class probabilities for `nominal`, `layer_shift`, `resonance_anomaly`.
5. Confidence score fed to fusion layer.

### Print State Gating
The C++ process polls Moonraker's REST API (via libcurl) once per second to check `print_stats.state`. Both models only run when state is `"printing"`. This avoids false positives during homing, bed mesh, and other pre-print routines.

### IPC: C++ → Python
On each detection event, the C++ process writes a JSON payload to the Unix domain socket:
```json
{
  "class": "spaghetti",
  "confidence": 0.87,
  "consecutive_ticks": 3,
  "frame_path": "/tmp/pfm/frame_1234.jpg",
  "timestamp": 1716912345.2
}
```
The Python process reads this and decides whether to call Moonraker and send a notification.

---

## Model Details

### Vision Model (YOLOv8n)
- **Input:** 416x416 RGB frame
- **Backbone:** CSPDarknet (pretrained ImageNet weights)
- **Training:**
  - Base dataset: Obico open-source dataset (Roboflow) — ~2000 labeled frames
  - Fine-tune on your own prints — aim for 50–100 frames per failure class
  - Augmentations: brightness ±30%, contrast ±20%, horizontal flip, ±10% crop
  - Loss: YOLO composite (box + cls + dfl)
  - Freeze backbone for 10 epochs, unfreeze and fine-tune at LR=1e-4 for 20 epochs
- **Export:** ONNX (opset 12), FP16 quantization for Pi 5 speedup
- **Inference:** ~80–120ms per frame on Pi 5 CPU; ~40ms with ONNX Runtime optimizations

### Accelerometer Model (1D CNN)
- **Input:** 25-sample window × 3 axes (X, Y, Z) = shape (25, 3)
- **Architecture:**
  ```
  Conv1D(32, kernel=5) → ReLU → MaxPool(2)
  Conv1D(64, kernel=3) → ReLU → MaxPool(2)
  Flatten → Dense(64) → Dropout(0.3) → Dense(3, softmax)
  ```
- **Training:**
  - Collect during known-good prints and intentional layer shift events
  - Segment by phase (travel vs. perimeter vs. infill) using Klipper move timestamps
  - Weighted cross-entropy for class imbalance (layer shifts are rare)
  - Train in PyTorch, export to ONNX

### Late Fusion
```cpp
float P_final = 0.65f * P_vision + 0.35f * P_accel;
```
Weights are tunable. Vision is weighted higher because it has richer signal; accel catches shifts that happen before the camera can see them. Start with these defaults and calibrate on real data.

---

## Detection Service

Two processes managed by the same systemd unit group.

### C++ Inference Process
Three `std::thread`s sharing a mutex-protected state struct:

- **Thread 1:** Camera capture loop (5–10 fps) — blocks on `cv::VideoCapture::read()`
- **Thread 2:** CAN socket reader — blocks on `read()` from the CAN socket fd, pushes samples into a ring buffer
- **Thread 3:** Inference + fusion loop — ticks every 1s, pulls latest frame and latest accel window, runs both models, evaluates thresholds, writes to Unix socket on detection

### Python Orchestration Process
Single-threaded async loop (`asyncio`) listening on the Unix socket. On receiving a detection event it calls Moonraker, writes to the log, and fires the Telegram notification concurrently with `asyncio.gather`.

**Thresholds (enforced in C++):**
| Condition | Action |
|---|---|
| P_final(failure) > 0.75 for 3 consecutive ticks | Emit event → Python pauses print |
| P_final(failure) > 0.92 for 1 tick | Emit event → Python cancels print |
| P_final(failure) < 0.75 | No event; C++ logs internally |

**Logging:**
- C++ saves the triggering frame as JPEG to `/tmp/pfm/` and includes the path in the socket payload
- Python writes the full JSON event to `/var/log/print_failure_detection/` and moves the frame there

---

## Notification System

Handled entirely in the Python process via a Discord webhook. No bot token or server permissions needed — just a webhook URL from your Discord channel settings.

```python
async def handle_event(event: dict):
    await asyncio.gather(
        moonraker_action(event),
        discord_notify(event),
        log_event(event),
    )

async def discord_notify(event: dict):
    async with aiohttp.ClientSession() as session:
        # Send embed with detection info
        payload = {
            "embeds": [{
                "title": f"Print failure detected: {event['class']}",
                "color": 0xFF4444,
                "fields": [
                    {"name": "Confidence", "value": f"{event['confidence']:.2%}", "inline": True},
                    {"name": "Action", "value": "Paused" if event["confidence"] < 0.92 else "Cancelled", "inline": True},
                    {"name": "Timestamp", "value": str(event["timestamp"]), "inline": False},
                ],
            }]
        }
        await session.post(DISCORD_WEBHOOK_URL, json=payload)

        # Send frame as a follow-up file upload to the same webhook
        if event.get("frame_path"):
            form = aiohttp.FormData()
            form.add_field("file", open(event["frame_path"], "rb"), filename="frame.jpg")
            await session.post(DISCORD_WEBHOOK_URL, data=form)
```

Set `DISCORD_WEBHOOK_URL` in `config.py`. The embed fires first with the detection metadata, then the frame image uploads as a second request to the same webhook. Notification includes detection class, confidence, action taken, and timestamp for post-hoc review and retraining data collection.

---

## Systemd Services

Two units, both in the same target group so they start and stop together.

```ini
# /etc/systemd/system/pfm-inference.service
[Unit]
Description=Print Failure Monitor — C++ Inference
After=network.target moonraker.service

[Service]
ExecStart=/usr/local/bin/pfm_inference
Restart=on-failure
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/pfm-orchestrator.service
[Unit]
Description=Print Failure Monitor — Python Orchestrator
After=pfm-inference.service
Requires=pfm-inference.service

[Service]
ExecStart=/usr/bin/python3 /home/pi/print_monitor/orchestrator.py
Restart=on-failure
RestartSec=5
User=pi
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

---

## Development Phases

| Phase | Deliverable | Est. Time |
|---|---|---|
| 1 | C++ camera capture + frame logging during prints | 1 week |
| 2 | Label dataset, fine-tune YOLOv8n in Python/PyTorch, export to ONNX | 2–3 weeks |
| 3 | Integrate YOLOv8n into C++ ORT session, validate inference output | 1 week |
| 4 | CAN socket accel reader + FFTW3 windowing + 1D CNN ORT session in C++ | 1–2 weeks |
| 5 | Fusion layer + Unix socket IPC + Python orchestrator + systemd services | 1 week |
| 6 | Telegram notifications + JSON logging + frame archival | 3–5 days |
| 7 | Threshold tuning on real prints, iterative retraining | Ongoing |

---

## File Structure

```
print_monitor/
├── inference/                   # C++ — compiled to pfm_inference binary
│   ├── main.cpp                 # Entry point, thread management
│   ├── config.h                 # Thresholds, paths, socket path
│   ├── camera/
│   │   ├── capture.cpp          # OpenCV V4L2 frame capture
│   │   └── preprocess.cpp       # Resize, normalize → ORT tensor
│   ├── accel/
│   │   ├── can_reader.cpp       # linux/can.h socket reader
│   │   ├── ring_buffer.h        # Lock-free ring buffer for accel samples
│   │   └── preprocess.cpp       # Windowing + FFTW3
│   ├── models/
│   │   ├── vision.cpp           # ORT C++ YOLOv8n session wrapper
│   │   ├── accel.cpp            # ORT C++ 1D CNN session wrapper
│   │   └── fusion.cpp           # Late fusion + threshold logic
│   ├── ipc/
│   │   └── socket_writer.cpp    # Unix domain socket event emitter
│   └── CMakeLists.txt
│
└── orchestrator/                # Python — runs as pfm-orchestrator.service
    ├── orchestrator.py          # asyncio entry point, socket listener
    ├── moonraker.py             # aiohttp Moonraker REST calls
    ├── notify.py                # Discord webhook notifications
    └── logger.py                # JSON event log + frame archival
```
