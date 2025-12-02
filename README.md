# Grenat-MVP-runtime

Runtime for the Grenat MVP tiny-object sorter that runs on NVIDIA Jetson Orin modules. This is the sister project of [Grenat-MVP-training](https://github.com/josue-scheidegger/Grenat-MVP-training): the training repo produces the YOLOv11 weights, this repo ingests two MIPI streams, performs real-time inference, and drives 64 ejector fingers over CAN while staying in sync with the conveyor and hopper controls.

## System Overview
- Two Sony IMX900 MIPI cameras cover left/right halves of the conveyor.
- Hopper feeds parts; gate opening controls throughput (fixed or encoder-driven feedback).
- Encoder wheel reports conveyor speed to a companion microcontroller; microcontroller and Jetson exchange data via SPI.
- YOLOv11 model (Ultralytics) detects the target class; everything else is rejected.
- 64 ejector fingers (4 CAN modules x 16 solenoids) knock defects off the belt; CAN addresses set via PCB DIP switches.
- IoT/mobile app supervises the runtime (start/stop, mode, config, telemetry).

Data-flow sketch:
```
Cameras (IMX900 x2) --> Frame ingest --> Preprocess --> YOLOv11 inference
    --> Postprocess (NMS, stitching) --> Decision & timing (conveyor speed)
    --> Actuation over CAN (64 solenoids) --> Telemetry/IoT
                             ^                       |
                             |--- SPI to MCU (encoder, hopper)
```

## Current Minimal Structure
Start simple and grow toward the full system:
- Single entrypoint script to wire config, load the model, grab frames, and print/placeholder-actuate results.
- One YAML config file (human-friendly, comments allowed) for camera IDs, model weights path, and minimal IO settings.

Suggested starting tree:
```
src/
  main.py
configs/
  runtime.yaml
data/
  samples/
    images/      # dry-run input frames (filenames include capture timestamp)
logs/            # dry-run CAN emulation outputs (e.g., can_emulator.log)
models/  (gitignored, holds YOLO weights)
```

## Future Architecture (Python)
**Runtime services**
- `app.py`: entrypoint wiring configuration, lifecycle (init -> run -> graceful stop), health checks.
- `config/`: load/validate runtime config (YAML/ENV), hardware map, camera ROI settings, timing offsets.
- `pipelines/`: end-to-end pipeline orchestration (ingest -> inference -> decision -> actuation).
- `vision/`: camera drivers, pre/post-processing, detector abstraction wrapping Ultralytics YOLOv11, stitching for dual-camera layout, calibration helpers.
- `control/`: conveyor speed tracker, delay calculator (distance from cameras to ejectors), hopper gate controller, CAN actuation scheduler, SPI exchange with MCU.
- `interfaces/`: IoT/mobile API (MQTT/REST), operator commands, telemetry export, Prometheus/logging sinks.
- `monitoring/`: structured logging, metrics, heartbeat, watchdogs, model performance stats (FPS, latency, drop counts).

**I/O adapters**
- `io/camera.py`: IMX900 capture via libargus/v4l2; synchronized frames, ROI cropping.
- `io/can_bus.py`: CAN interface (socketcan/can-utils), address discovery (DIP), batch writes to 4 modules.
- `io/spi_bridge.py`: SPI protocol with MCU (encoder ticks, gate position commands/acks).
- `io/encoder.py`: compute belt speed from ticks, debounce, low-pass filtering.

**Scheduling & timing**
- `control/scheduler.py`: maps detections to finger indices and fire times using belt speed and camera-to-ejector offset; handles overlap and priority.
- `control/hopper.py`: regulates hopper opening based on target FPS/throughput; supports manual and closed-loop modes.

**Testing & tooling**
- `tests/`: unit tests with hardware abstractions mocked.
- `scripts/`: deployment, flashing, model download, camera calibration utilities.
- `models/`: YOLO weights managed separately (not committed); download helper scripts.
- `configs/`: example runtime configs for dev/bench/prod.

Proposed tree:
```
src/
  app.py
  config/
  pipelines/
  vision/
  control/
  interfaces/
  monitoring/
  io/
tests/
scripts/
configs/
models/  (gitignored)
```

## Setup
- Target: Jetson Orin (JetPack with CUDA/cuDNN/TensorRT). Use Python 3.10+.
- Install Ultralytics and I/O deps (socketcan, spidev, v4l2/libargus bindings).
- Typical workflow:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt  # to be added alongside the first code
  ```
- Place YOLOv11 weights in `models/` (or configure a download step); keep them out of git.

## Running (expected)
```bash
python -m src.app --config configs/runtime.yaml
```
Suggested config keys:
- `cameras`: device ids, exposure, gain, ROI per side.
- `model`: weights path, input size, NMS thresholds.
- `timing`: camera-to-ejector distance, belt speed source (fixed/encoder), fire lead time.
- `can`: interface (e.g., `can0`), module addresses, solenoid mapping.
- `spi`: bus/device, protocol version, retry/backoff.
- `hopper`: mode (manual/auto), gate min/max, PID or step rules for auto.
- `iot`: MQTT/REST endpoints, auth, topics, telemetry rate.
- `logging/metrics`: level, sinks, retention.

## Operational Notes
- Calibrate cameras (intrinsics/align ROIs) and measure conveyor speed to compute correct fire delays.
- Keep a dry-run/sim mode (no CAN/SPI) for development and CI.
- Expose health endpoints for the mobile app to surface errors (camera down, MCU offline, model missing).
- Measure per-stage latency (capture, preprocess, inference, scheduling, CAN send) to maintain real-time guarantees.

## Relationship to Training Repo
- Training happens in `Grenat-MVP-training` (data prep, labeling, augmentation, YOLOv11 training).
- This runtime consumes the exported weights and runs inference + actuation in real time.

## Contributing
- Prefer small, test-backed changes. Keep hardware dependencies behind interfaces so tests can run without devices.
- Add configs and scripts for both lab benches and on-line machines (Jetson, MCU, CAN stack).
