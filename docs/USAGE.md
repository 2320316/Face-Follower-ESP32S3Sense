# Usage Guide

This guide explains how to run the robot and use the face-following workflow in practice.

## 1. Power up the robot

Power the ESP32 and motors with a stable supply. Make sure the TB6612FNG driver and motor wiring match the definitions in `espcam.cpp`.

The firmware starts a Wi‑Fi access point and serves the camera stream at:

```text
http://192.168.4.1/stream
```

The TCP command listener is also available at:

```text
192.168.4.1:8080
```

## 2. Connect your Mac to the robot network

On the Mac, join the robot SSID created by the ESP32, typically:

```text
XIAO-ESP32S3
```

The default password is set in `espcam.cpp`:

```cpp
const char *AP_PASS = "12345678";
```

## 3. Install Python dependencies

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python pillow ultralytics
```

If your system does not already have the model file, you can download it through the app or place one in the `yolo/` folder.

## 4. Start the desktop app

```bash
python esp_app.py
```

On launch, the app:

- initializes the UI
- loads the default YOLO face model
- connects to the camera stream
- prepares the robot command client

## 5. Select the target face

When the live camera feed is visible:

- click on the face you want the robot to follow
- the app picks the face containing the click and stores a profile for it
- the target profile is based on the selected face dimensions and color histogram

The robot begins making motion decisions after the target is selected.

## 6. Behavior while tracking

The detection loop continuously does the following:

1. reads a new MJPEG frame from the ESP32
2. runs YOLO face detection
3. finds the selected face or a similar face nearby
4. measures how far the face is from the center of the frame
5. decides whether to move forward, backward, or turn
6. sends the correct command to the robot

This is all implemented in `espyolo.py` and the worker thread in `esp_app.py`.

## 7. Controls in the GUI

The desktop app exposes a simple control interface:

- Model dropdown — choose a YOLO face model
- Load / Unload — load or release the selected model
- Confidence slider — tune how sensitive detection is
- Open target view — show a closer crop of the tracked face
- Stop robot — immediately issue a stop command
- Esc key — same as stop

## 8. Typical operating flow

A normal tracking loop looks like this:

- camera is streaming
- a face is selected by click
- the robot centers the face horizontally
- if the face is too close or too far, the robot moves toward/away from it
- if the target is lost briefly, it retains target memory and tries to reacquire it
- if the target is missing for too long, tracking is released

## 9. Troubleshooting

### App says stream unavailable

- confirm the robot is on Wi‑Fi
- check the board serial console for boot and camera status
- make sure `http://192.168.4.1/stream` is reachable from the laptop

### Model not found

- put a supported file in `yolo/`
- or use the built-in model picker to download a compatible file

### Robot does not respond

- verify the command port is open on `192.168.4.1:8080`
- check the motor driver, power supply, and pin assignments in the firmware
- watch the serial output from the ESP32 for command logs

### Target drifts or loses lock

- reduce the detection confidence if the model is overfitting to noise
- improve lighting and keep the face in view
- select a clear target in the center of the image first

## 10. Extending the system

Good next improvements include:

- multiple-person tracking
- obstacle avoidance
- autonomous search when the face disappears
- more stable PID-style motion control
- a richer GUI for camera tuning and robot diagnostics
