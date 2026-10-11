# Usage Guide

This guide explains how to prepare, start, operate, and stop the
face-following robot. The desktop implementation is self-contained in
`esp_app.py`.

## 1. Power up the robot

Power the ESP32 and motors with a stable supply. Make sure the TB6612FNG
driver and motor wiring match the definitions in `espcam.cpp`.

The firmware starts a Wi-Fi access point and serves the camera stream at:

```text
http://192.168.4.1/stream
```

The TCP command listener is also available at:

```text
192.168.4.1:8080
```

## 2. Connect your computer to the robot network

Join the Wi-Fi network created by the ESP32, typically:

```text
XIAO-ESP32S3
```

The default password is set in `espcam.cpp`:

```cpp
const char *AP_PASS = "12345678";
```

## 3. Install Python dependencies

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install opencv-python pillow ultralytics
```

For optional ArcFace-based familiar-face recognition, also install:

```bash
python -m pip install insightface onnxruntime
```

If InsightFace is not installed or cannot initialize, the app uses a
lower-accuracy image-statistics embedding fallback.

Supported model files are `yolo26n-face.pt`, `yolov8n-face.pt`, and
`yolov10n-face.pt`. They can be placed in `yolo/` manually or downloaded from
the app.

## 4. Start the desktop app

```bash
python3 esp_app.py
```

On launch, the app initializes the UI and starts loading the default YOLO
model in the background. It does not connect to the robot until you press
`Connect`.

The model status must show `Ready` before the app can connect.

## 5. Select the target face

When the live camera feed is visible:

1. Click on the detected face you want the robot to follow.
2. The app chooses the largest detected face containing the click.
3. A session profile is created from the face geometry and appearance
   embedding.
4. The selected face is persisted as `memory/target.json`.

The robot begins making movement decisions after the target is selected.

## 6. Behavior while tracking

The worker continuously:

1. reads a new MJPEG frame from the ESP32
2. runs YOLO face detection
3. finds the target by overlap or proximity
4. falls back to the learned profile and stored-face matching when needed
5. measures the target position and size
6. decides whether to move forward, backward, turn, or stop
7. sends the command to the robot

Short detection gaps retain the target profile for reacquisition. If the
target is missing for too long, tracking is released.

## 7. Controls in the GUI

- **Model dropdown** — choose a supported YOLO face model
- **Load / Unload** — load or release the selected model
- **Confidence slider** — tune the detector threshold
- **Target card** — show target memory status and similarity samples
- **Connect / Disconnect** — start or stop the stream and tracking worker
- **Stop robot** — immediately stop movement and clear the target
- **Esc** — same as Stop robot

Changing the model requires disconnecting, unloading/loading the desired model,
and reconnecting.

## 8. Persistent face memory

Face memories are JSON files under `memory/`:

- `target.json` — the current selected target
- other JSON files — named familiar faces when present

Memories contain embeddings, geometry statistics, confidence, and tracking
counters. The app updates confirmed familiar-face memories incrementally.

## 9. Troubleshooting

### App says stream unavailable

- confirm the computer is connected to the ESP32 Wi-Fi network
- check the board serial console for boot and camera errors
- verify `http://192.168.4.1/stream` is reachable

### Model not found

- place a supported model in `yolo/`
- or use the model selector to download it
- wait for the model status to show `Ready` before pressing `Connect`

### No face detected

- lower the confidence threshold
- try another supported YOLO model
- improve lighting and keep the face fully visible

### Familiar-face recognition is unavailable

- install `insightface` and `onnxruntime`
- check that the InsightFace model can initialize on the host
- target tracking still works with the built-in fallback embedding

### Robot does not respond

- verify the command port is open on `192.168.4.1:8080`
- check the motor driver, power supply, and pin assignments
- watch the ESP32 serial output for command logs

### Target drifts or loses lock

- reduce the confidence threshold if detections are unstable
- improve lighting and keep the face in view
- select a clear target near the center of the image

## 10. Stop the app safely

Press `Stop robot` or `Esc` before closing the window. Closing the app stops
the worker, releases the camera stream, sends a stop command, and exits.

## 11. Extending the system

Possible future improvements include:

- multiple-person tracking
- obstacle avoidance
- autonomous search when the face disappears
- PID-style motion control
- richer camera and robot diagnostics
