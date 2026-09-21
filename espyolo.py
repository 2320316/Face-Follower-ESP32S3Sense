"""YOLO face detection and distance/alignment classification."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    if exc.name != "typing_extensions":
        raise
    subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions"])
    from ultralytics import YOLO


DEFAULT_MODEL = "yolo26n-face.pt"
MODEL_DIR = Path(__file__).with_name("yolo")
MODEL_RELEASE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/{}"
FACE_MODEL_OPTIONS = ("yolo26n-face.pt", "yolov8n-face.pt", "yolov10n-face.pt")
DEFAULT_ALIGN_MARGIN = 0.06
DEFAULT_DISTANCE_MARGIN = 0.06
# Pivot-turn speed (PWM 0-255) by how far the target is from the frame center.
TURN_SPEED_LITTLE = 100  # just outside the center zone
TURN_SPEED_MEDIUM = 150  # halfway to the frame edge
TURN_SPEED_BIG = 220     # at the frame edge
TURN_SPEED_STEP = 10     # rounding step, so the command only changes when the speed really does


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self):
        return self.x + self.width // 2, self.y + self.height // 2

    def contains(self, point):
        px, py = point
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


@dataclass
class FaceDecision:
    label: str
    command: str
    speed: int
    box: FaceBox | None


@dataclass
class TargetProfile:
    width_ratio: float
    height_ratio: float
    aspect_ratio: float
    appearance: np.ndarray


def find_model_path(model_name=DEFAULT_MODEL):
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base_dir in (MODEL_DIR, os.getcwd(), script_dir):
        path = os.path.join(base_dir, model_name)
        if os.path.exists(path):
            return path
    return model_name


def download_model(model_name=DEFAULT_MODEL, progress=None):
    if model_name not in FACE_MODEL_OPTIONS:
        raise ValueError(f"Unsupported face model: {model_name}")

    MODEL_DIR.mkdir(exist_ok=True)
    destination = MODEL_DIR / model_name
    url = MODEL_RELEASE_URL.format(model_name)
    with urlopen(url, timeout=30) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if progress and total:
                progress(received / total)
    return str(destination)


class FaceDetector:
    def __init__(self, model_name=DEFAULT_MODEL, confidence=0.4,
                 align_margin=DEFAULT_ALIGN_MARGIN,
                 distance_margin=DEFAULT_DISTANCE_MARGIN):
        self.model_path = find_model_path(model_name)
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.align_margin = align_margin
        self.distance_margin = distance_margin

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(FaceBox(
                x1, y1, x2 - x1, y2 - y1, float(result.conf[0])
            ))
        return detections

    def choose(self, detections, point):
        containing = [face for face in detections if face.contains(point)]
        if not containing:
            return None
        return max(containing, key=lambda face: face.width * face.height)

    def match_selected(self, detections, selected, profile=None, frame=None):
        if selected is None or not detections:
            return None
        selected_cx, selected_cy = selected.center
        candidates = detections
        if profile is not None and frame is not None:
            candidates = [
                face for face in detections
                if self.profile_score(face, profile, frame) <= 0.9
            ]
            if not candidates:
                return None
        candidate = min(candidates, key=lambda face: (
            face.center[0] - selected_cx) ** 2 +
            (face.center[1] - selected_cy) ** 2
        )
        distance = ((candidate.center[0] - selected_cx) ** 2 +
                    (candidate.center[1] - selected_cy) ** 2) ** 0.5
        max_distance = max(selected.width, selected.height) * 2.5
        return candidate if distance <= max_distance else None

    @staticmethod
    def profile(face, frame, frame_width, frame_height):
        crop = frame[face.y:face.y + face.height, face.x:face.x + face.width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        appearance = cv2.normalize(
            histogram, histogram, 0, 1, cv2.NORM_MINMAX,
        ).flatten()
        return TargetProfile(
            width_ratio=face.width / frame_width,
            height_ratio=face.height / frame_height,
            aspect_ratio=face.width / max(face.height, 1),
            appearance=appearance,
        )

    def profile_score(self, face, profile, frame):
        frame_height, frame_width = frame.shape[:2]
        crop = frame[face.y:face.y + face.height, face.x:face.x + face.width]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        appearance = cv2.normalize(
            histogram, histogram, 0, 1, cv2.NORM_MINMAX,
        ).flatten()
        geometry = (
            0.2 * abs(face.width / frame_width - profile.width_ratio) / max(profile.width_ratio, 0.001)
            + 0.2 * abs(face.height / frame_height - profile.height_ratio) / max(profile.height_ratio, 0.001)
            + abs(face.width / max(face.height, 1) - profile.aspect_ratio) / max(profile.aspect_ratio, 0.001)
        )
        appearance_difference = 1.0 - cv2.compareHist(
            profile.appearance.astype(np.float32), appearance.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        return geometry + max(appearance_difference, 0.0)

    def match_profile(self, detections, profile, frame):
        if not detections or profile is None:
            return None
        candidate = min(detections, key=lambda face: self.profile_score(face, profile, frame))
        if self.profile_score(candidate, profile, frame) > 0.9:
            return None
        return candidate

    def turn_speed(self, offset, frame_width):
        """Little-to-medium just outside the center zone, medium-to-big toward the frame edge."""
        margin = frame_width * self.align_margin
        ratio = min(max((abs(offset) - margin) / (frame_width / 2.0 - margin), 0.0), 1.0)
        if ratio < 0.5:
            speed = TURN_SPEED_LITTLE + (TURN_SPEED_MEDIUM - TURN_SPEED_LITTLE) * ratio * 2
        else:
            speed = TURN_SPEED_MEDIUM + (TURN_SPEED_BIG - TURN_SPEED_MEDIUM) * (ratio - 0.5) * 2
        return int(round(speed / TURN_SPEED_STEP)) * TURN_SPEED_STEP

    def classify(self, face, frame_width, frame_height):
        if face is None:
            return FaceDecision("NO TARGET", "p", 0, None)

        center_x = face.center[0]
        center_y = face.center[1]
        midpoint_x = frame_width / 2.0
        midpoint_y = frame_height / 2.0
        align_margin = frame_width * self.align_margin
        distance_margin = frame_height * self.distance_margin
        if center_x < midpoint_x - align_margin:
            return FaceDecision("LEFT", "d", self.turn_speed(center_x - midpoint_x, frame_width), face)
        if center_x > midpoint_x + align_margin:
            return FaceDecision("RIGHT", "a", self.turn_speed(center_x - midpoint_x, frame_width), face)
        if center_y < midpoint_y - distance_margin and center_x >= midpoint_x - align_margin and center_x <= midpoint_x + align_margin:
            return FaceDecision("CLOSE", "s", 150, face)
        if center_y > midpoint_y + distance_margin and center_x >= midpoint_x - align_margin and center_x <= midpoint_x + align_margin:
            return FaceDecision("FAR", "w", 150, face)
        return FaceDecision("GOOD", "p", 0, face)

    def draw(self, frame, detections, selected, decision, profile=None):
        frame_height, frame_width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for face in detections:
            color = (0, 165, 255) if selected is not face else (0, 255, 0)
            if selected is face:
                cx, cy = face.center
                radius = max(face.width, face.height) // 2 + 8
                diamond = np.array([
                    [cx, cy - radius], [cx + radius, cy],
                    [cx, cy + radius], [cx - radius, cy],
                ], dtype=np.int32)
                cv2.polylines(frame, [diamond], True, color, 3)
            else:
                cv2.rectangle(
                    frame, (face.x, face.y),
                    (face.x + face.width, face.y + face.height), color, 2
                )

            face_decision = decision if selected is face else self.classify(
                face, frame.shape[1], frame.shape[0]
            )
            detail = f"{face_decision.label} | conf {face.confidence:.2f} | {face.width}x{face.height}"
            font_scale = 0.42
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                detail, font, font_scale, thickness
            )
            detail_x = min(face.x + face.width, frame.shape[1] - 4) - text_width
            detail_x = max(detail_x, 4)
            detail_y = min(face.y + face.height + text_height + baseline + 4, frame.shape[0] - 4)
            background_top = detail_y - text_height - baseline - 4
            cv2.rectangle(
                frame, (detail_x - 2, background_top),
                (detail_x + text_width + 2, detail_y + 2), (20, 20, 20), -1
            )
            cv2.putText(frame, detail, (detail_x, detail_y), font,
                        font_scale, color, thickness, cv2.LINE_AA)

        if selected is not None:
            face = selected
            cx, cy = face.center
            radius = max(face.width, face.height) // 2 + 8
            diamond = np.array([
                [cx, cy - radius], [cx + radius, cy],
                [cx, cy + radius], [cx - radius, cy],
            ], dtype=np.int32)
            guide_color = (0, 255, 255)
            cv2.polylines(frame, [diamond], True, guide_color, 2)
            profile_text = "PROFILE MEMORY"
            if selected in detections and profile is not None:
                profile_text = f"PROFILE {self.profile_score(selected, profile, frame):.2f}"
            if selected not in detections:
                detail = f"TARGET LOST | conf {face.confidence:.2f} | {face.width}x{face.height}"
                cv2.putText(
                    frame, detail,
                    (max(4, min(face.x + face.width, frame.shape[1] - 4) - 190),
                     min(face.y + face.height + 18, frame.shape[0] - 4)),
                    font, 0.42, guide_color, 1, cv2.LINE_AA
                )

            top_tip = (cx, max(cy - radius, 0))
            right_tip = (min(cx + radius, frame_width - 1), cy)
            bottom_tip = (cx, min(cy + radius, frame_height - 1))
            left_tip = (max(cx - radius, 0), cy)
            (profile_width, profile_height), _ = cv2.getTextSize(
                profile_text, font, 0.42, 1
            )
            profile_x = min(right_tip[0] + 6, frame_width - profile_width - 4)
            profile_x = max(profile_x, 4)
            profile_y = max(cy - 8, profile_height + 4)
            cv2.putText(
                frame, profile_text, (profile_x, profile_y),
                font, 0.42, guide_color, 1, cv2.LINE_AA
            )
            guides = (
                (top_tip, (cx, 0), f"{top_tip[1]} px", (cx + 6, 18), (top_tip[0] + 6, top_tip[1] + 14)),
                (right_tip, (frame_width - 1, cy), f"{frame_width - 1 - right_tip[0]} px",
                 (frame_width - 70, max(cy - 6, 18)), (right_tip[0] - 42, max(right_tip[1] - 6, 18))),
                (bottom_tip, (cx, frame_height - 1), f"{frame_height - 1 - bottom_tip[1]} px",
                 (cx + 6, frame_height - 8), (bottom_tip[0] + 6, bottom_tip[1] - 6)),
                (left_tip, (0, cy), f"{left_tip[0]} px", (6, max(cy - 6, 18)),
                 (left_tip[0] + 6, max(left_tip[1] - 6, 18))),
            )
            for tip, edge, label, label_point, origin_point in guides:
                cv2.line(frame, tip, edge, guide_color, 1, cv2.LINE_AA)
                cv2.circle(frame, tip, 3, guide_color, -1)
                cv2.putText(frame, " ", origin_point, font, 0.36,
                            guide_color, 1, cv2.LINE_AA)
                cv2.putText(frame, label, label_point, font, 0.42,
                            guide_color, 1, cv2.LINE_AA)

        if decision.box is not None and selected is decision.box:
            face = decision.box
            cv2.putText(
                frame, f"TARGET: {decision.label}",
                (face.x, max(face.y - 8, 18)), font, 0.65, (0, 255, 0), 2
            )
        return frame