"""Desktop control app for the XIAO ESP32S3 face-following robot.

Features:
- Live MJPEG camera stream from the ESP32
- Click-to-select face tracking with persistent memory
- YOLO face detection with adaptive profile learning
- Stored face recognition — familiar faces are identified by name
- Closed-loop movement control for centering and following a face
"""

import os
import json
import math
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
import urllib.error
import urllib.request

import cv2
from PIL import Image, ImageTk
import numpy as np
try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


class MjpegStream:
    MAX_BUFFER = 2 * 1024 * 1024  # 2 MB cap to prevent unbounded growth

    def __init__(self, url, connect_timeout=1.0, read_timeout=0.5):
        self.response = None
        self._closed = False
        self.buffer = bytearray()
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "XIAO-face-tracker"}
            )
            self.response = urllib.request.urlopen(request, timeout=connect_timeout)
            try:
                self.response.fp.raw._sock.settimeout(read_timeout)
            except AttributeError:
                pass
        except (OSError, urllib.error.URLError):
            self.release()

    def is_opened(self):
        return self.response is not None

    def read(self):
        if self.response is None or self._closed:
            return False, None

        try:
            while True:
                start = self.buffer.find(b"\xff\xd8")
                end = self.buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    latest_end = end
                    latest_start = start
                    while True:
                        next_start = self.buffer.find(b"\xff\xd8", latest_end + 2)
                        if next_start < 0:
                            break
                        next_end = self.buffer.find(b"\xff\xd9", next_start + 2)
                        if next_end < 0:
                            break
                        latest_start = next_start
                        latest_end = next_end

                    jpeg = bytes(self.buffer[latest_start:latest_end + 2])
                    del self.buffer[:latest_end + 2]
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        return True, frame

                chunk = self.response.read(4096)
                if not chunk:
                    self.release()
                    return False, None
                self.buffer.extend(chunk)
                # Cap buffer to discard stale intermediate frames
                while len(self.buffer) > self.MAX_BUFFER:
                    # Find next frame boundary after current position
                    marker = self.buffer.find(b"\xff\xd8", 1024)
                    if marker > 0:
                        del self.buffer[:marker]
                    else:
                        del self.buffer[:len(self.buffer) // 2]
        except (OSError, TimeoutError, urllib.error.URLError):
            self.release()
            return False, None

    def release(self):
        self._closed = True
        if self.response is not None:
            self.response.close()
            self.response = None


class RobotCommandClient:
    VALID_COMMANDS = {"w", "s", "a", "d", "p"}

    def __init__(self, host="192.168.4.1", port=8080, timeout=0.25):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.last_command = None
        self.last_speed = None

    @staticmethod
    def format_payload(command, speed):
        """Return one complete firmware command frame."""
        return f"{command}, {int(speed)}\r\n".encode("utf-8")

    def send(self, command, speed=100, force=False):
        if command not in self.VALID_COMMANDS:
            raise ValueError(f"Unsupported robot command: {command}")

        speed = int(speed)
        if not force and command == self.last_command and speed == self.last_speed:
            return True, "duplicate suppressed"

        payload = self.format_payload(command, speed)
        try:
            with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout) as sock:
                sock.sendall(payload)
        except OSError as exc:
            return False, str(exc)

        self.last_command = command
        self.last_speed = speed
        return True, f"sent {command}, {speed}"

    def stop(self):
        return self.send("p", 0)
# ── insightface face embeddings ────────────────────────────────────────────────

try:
    from insightface.app import FaceAnalysis
    _FACE_EMBEDDER_READY = False
    _face_embedder: Optional[FaceAnalysis] = None

    def _init_face_embedder():
        """Lazy-load the ArcFace embedding model."""
        global _face_embedder, _FACE_EMBEDDER_READY
        if _face_embedder is not None:
            return
        try:
            _face_embedder = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            _face_embedder.prepare(ctx_id=0, det_size=(640, 640))
            _FACE_EMBEDDER_READY = True
        except Exception as exc:
            print(f"[FaceEmbedding] Could not load ArcFace: {exc}")
            _FACE_EMBEDDER_READY = False

except ImportError:
    _FACE_EMBEDDER_READY = False
    _face_embedder = None


# ── defaults ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "yolo26n-face.pt"
MODEL_DIR = Path(__file__).with_name("yolo")
MODEL_RELEASE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/{}"
FACE_MODEL_OPTIONS = ("yolo26n-face.pt", "yolov8n-face.pt", "yolov10n-face.pt")

DEFAULT_ALIGN_MARGIN = 0.06
DEFAULT_DISTANCE_MARGIN = 0.06

# Pivot-turn speed (PWM 0-255) by how far the target is from the frame center.
TURN_SPEED_LITTLE = 100
TURN_SPEED_MEDIUM = 150
TURN_SPEED_BIG = 220
TURN_SPEED_STEP = 10

# Recognition tuning
DEFAULT_RECOGNITION_RADIUS_MULTIPLIER = 3.0
DEFAULT_PROFILE_LEARNING_RATE = 0.08
CONFIDENCE_DECAY_PER_FRAME = 0.005
CONFIDENCE_MIN_FOR_TRACK = 0.05
DEFAULT_RECOGNITION_THRESHOLD = 0.65  # cosine similarity threshold for matching

# Persistence
MEMORY_DIR = Path(__file__).with_name("memory")  # stores learned face memories on disk


# ── data classes ────────────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    """Detected face bounding box with confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def center(self):
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def y2(self):
        return self.y + self.height

    @property
    def x2(self):
        return self.x + self.width

    def contains(self, point):
        px, py = point
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def io_union(self, other):
        """Intersection over union with another box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        self_area = self.width * self.height
        other_area = other.width * other.height
        union = self_area + other_area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FaceDecision:
    """Result of movement classification for the tracked face."""
    label: str
    command: str
    speed: int
    box: Optional[FaceBox] = None


@dataclass
class FaceDetection:
    """Lightweight per-session face detection representation.

    Holds geometric and appearance info for the actively tracked face.
    Uses ArcFace embedding (512-dim) as the primary recognition feature.
    """
    width_ratio: float
    height_ratio: float
    aspect_ratio: float
    embedding: np.ndarray          # ArcFace 512-dim, L2-normalized
    confidence_score: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    age_frames: int = 0
    successful_matches: int = 0
    frame_width: int = 0           # for distance normalization
    frame_height: int = 0          # for distance normalization
    stored_name: str | None = None  # name in FaceDetector.stored_faces
    # Score tracking for adaptive learning
    score_history: list[float] = field(default_factory=list)  # recent match scores
    last_update_frame: int = 0

    def size_scale(self) -> float:
        return (self.width_ratio + self.height_ratio) / 2.0

    def record_score(self, score: float, frame_idx: int = -1):
        """Record a recognition score with frame index for smoothing."""
        self.score_history.append(score)
        if len(self.score_history) > 20:
            self.score_history = self.score_history[-20:]
        self.last_update_frame = frame_idx

    def avg_score(self) -> float:
        """Average of recent recognition scores."""
        return float(np.mean(self.score_history)) if self.score_history else 0.0


@dataclass
class FaceMemory:
    """Persistent, learnable memory with multi-sample ensemble.

    Stores one or more ArcFace embedding vectors per person. Each time the
    same person is re-identified, their embedding is blended into the stored
    set via exponential moving average, building a robust multi-angle
    representation over time.

    Uses cosine similarity for matching (robust to lighting and scale).
    """
    name: str
    # Core: the stored embedding vectors (multiple samples for ensemble)
    mean_embedding: np.ndarray = field(default_factory=lambda: np.zeros(512))
    # Geometry (running averages for size-aware matching)
    mean_width_ratio: float = 0.0
    mean_height_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    # Variance (for adaptive thresholding)
    var_width_ratio: float = 0.0
    var_height_ratio: float = 0.0
    # Metadata
    sample_count: int = 0
    confidence: float = 0.0
    created: str = ""
    created_ts: int = 0
    last_seen_frames_ago: int = 0
    total_tracking_time_frames: int = 0
    unknown_faces_seen: int = 0
    # Ensemble config
    best_embeddings: list = field(default_factory=list)  # top-k similar stored embeddings
    max_samples_in_memory: int = 5                        # max stored embeddings to keep

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "mean_width_ratio": self.mean_width_ratio,
            "mean_height_ratio": self.mean_height_ratio,
            "mean_aspect_ratio": self.mean_aspect_ratio,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "var_width_ratio": self.var_width_ratio,
            "var_height_ratio": self.var_height_ratio,
            "created": self.created,
            "created_ts": self.created_ts,
            "last_seen_frames_ago": self.last_seen_frames_ago,
            "total_tracking_time_frames": self.total_tracking_time_frames,
            "unknown_faces_seen": self.unknown_faces_seen,
            "max_samples_in_memory": self.max_samples_in_memory,
            "mean_embedding": np.asarray(self.mean_embedding,
                                          dtype=np.float32).tolist(),
        }
        # Persist ensemble embeddings for cross-session persistence
        d["best_embeddings"] = [np.asarray(emb, dtype=np.float32).tolist()
                                 for emb in self.best_embeddings]
        d["embedding_dim"] = int(self.mean_embedding.size)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FaceMemory":
        if not isinstance(d, dict) or not isinstance(d.get("name"), str):
            raise ValueError("memory record must contain a name")
        dim = int(d.get("embedding_dim", 512))
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")

        def vector(value, label):
            if not isinstance(value, list):
                raise ValueError(f"{label} must be a JSON list")
            result = np.asarray(value, dtype=np.float32)
            if result.ndim != 1 or result.size != dim or not np.isfinite(result).all():
                raise ValueError(f"{label} has invalid dimension or values")
            return result

        raw_embeddings = d.get("best_embeddings", [])
        if not isinstance(raw_embeddings, list):
            raise ValueError("best_embeddings must be a JSON list")
        embeddings = [vector(value, "best embedding")
                      for value in raw_embeddings]
        mean_value = d.get("mean_embedding")
        mean = (vector(mean_value, "mean embedding")
                if mean_value is not None
                else (embeddings[0].copy()
                      if embeddings else np.zeros(dim, dtype=np.float32)))

        p = cls(name=d["name"], mean_embedding=mean)
        for field_name in ("mean_width_ratio", "mean_height_ratio",
                           "mean_aspect_ratio", "sample_count", "confidence",
                           "var_width_ratio", "var_height_ratio", "created_ts",
                           "last_seen_frames_ago", "total_tracking_time_frames",
                           "unknown_faces_seen"):
            if field_name in d:
                setattr(p, field_name, d[field_name])
        p.created = d.get("created", "")
        p.max_samples_in_memory = int(d.get("max_samples_in_memory", 5))
        p.best_embeddings = (embeddings[-p.max_samples_in_memory:]
                             if p.max_samples_in_memory else [])
        return p

    def add_sample_embedding(self, embedding: np.ndarray, learning_rate: float = 0.05):
        """Blend a new ArcFace embedding into the stored mean via EMA."""
        if self.sample_count == 0:
            self.mean_embedding = embedding.copy()
        else:
            self.mean_embedding = (
                self.mean_embedding * (1 - learning_rate) +
                embedding * learning_rate
            )
        self.sample_count += 1
        self.confidence = min(1.0, self.confidence + 0.02)

        # Keep top-k embeddings for ensemble matching
        self._store_embedding(embedding)

    def _store_embedding(self, embedding: np.ndarray):
        """Store an embedding in the ensemble buffer."""
        self.best_embeddings.append(embedding.copy())
        if len(self.best_embeddings) > self.max_samples_in_memory:
            # Remove oldest when buffer is full
            self.best_embeddings = self.best_embeddings[-self.max_samples_in_memory:]

    def cosine_similarity(self, embedding: np.ndarray) -> float:
        """Compute cosine similarity between stored mean and a query embedding."""
        if self.sample_count == 0:
            return 0.0
        a = self.mean_embedding / (np.linalg.norm(self.mean_embedding) + 1e-8)
        b = embedding / (np.linalg.norm(embedding) + 1e-8)
        return float(np.dot(a, b))

    def ensemble_similarity(self, embedding: np.ndarray, top_k: int = 3) -> float:
        """Compute ensemble similarity using top-k most similar stored embeddings.

        More robust than single-mean matching because it considers the
        diversity of stored samples. Returns the average similarity of
        the top-k stored embeddings to the query.
        """
        if self.sample_count == 0:
            return 0.0

        # Normalize query
        query = embedding / (np.linalg.norm(embedding) + 1e-8)

        if len(self.best_embeddings) <= top_k:
            # If fewer samples than top_k, just use mean
            return self.cosine_similarity(embedding)

        # Compute similarity to all stored embeddings
        sims = []
        for stored in self.best_embeddings:
            stored_norm = stored / (np.linalg.norm(stored) + 1e-8)
            sim = float(np.dot(query, stored_norm))
            sims.append(sim)

        # Average of top-k
        sims.sort(reverse=True)
        return float(np.mean(sims[:top_k]))


# ── persistence helpers ─────────────────────────────────────────────────────────

def _load_memory_file(name: str) -> Optional[dict]:
    """Load a stored memory JSON file, or return None."""
    path = MEMORY_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return raw
    except (json.JSONDecodeError, OSError):
        return None


def _save_memory_file(name: str, memory: FaceMemory) -> bool:
    """Persist a memory to disk."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{name}.json"
    try:
        data = memory.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def list_stored_faces() -> list[str]:
    """Return names of all stored memories."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p.stem for p in MEMORY_DIR.glob("*.json")])


def delete_memory(name: str) -> bool:
    """Remove a stored memory."""
    path = MEMORY_DIR / f"{name}.json"
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


# ── face embedding extraction ───────────────────────────────────────────────────

def _extract_face_embedding(frame: np.ndarray, box: FaceBox) -> Optional[np.ndarray]:
    """Extract an ArcFace 512-dim embedding from a face crop using InsightFace.

    Returns an L2-normalized numpy array of shape (512,) or None if extraction fails.
    """
    if _face_embedder is None or not _FACE_EMBEDDER_READY:
        return None

    # Crop the face region with some padding
    h, w = frame.shape[:2]
    pad_x = int(box.width * 0.3)
    pad_y = int(box.height * 0.35)
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.x2 + pad_x)
    y2 = min(h, box.y2 + pad_y)
    crop = frame[y1:y2, x1:x2].copy()

    # Detect faces in the crop and get the largest one
    faces = _face_embedder.get(crop)
    if not faces:
        return None

    # Use the highest-confidence face
    face = max(faces, key=lambda f: f.det_score)
    return face.embedding  # 512-dim, already normalized


def _extract_face_embedding_fallback(frame: np.ndarray, box: FaceBox) -> np.ndarray:
    """Fallback embedding when ArcFace is unavailable.

    Uses a simple pixel-statistics embedding. Much less accurate but provides
    a baseline for comparison.
    """
    h, w = frame.shape[:2]
    pad_x = int(box.width * 0.3)
    pad_y = int(box.height * 0.35)
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.x2 + pad_x)
    y2 = min(h, box.y2 + pad_y)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(512, dtype=np.float32)

    # Resize to fixed size and extract statistics
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    rgb = resized.reshape(-1, 3).astype(np.float64) / 255.0
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64) / [180, 255, 255]

    # Statistics per channel
    features = np.concatenate([
        rgb.mean(axis=0),        # 3
        rgb.std(axis=0),         # 3
        rgb.flatten()[:36],      # 36 (patch means)
        hsv.mean(axis=0),        # 3
        hsv.std(axis=0),         # 3
    ])

    # Pad or trim to fixed size
    target_dim = 512
    if len(features) < target_dim:
        features = np.pad(features, (0, target_dim - len(features)))
    else:
        features = features[:target_dim]

    return features / (np.linalg.norm(features) + 1e-8)


def _create_face_embedding(frame: np.ndarray, box: FaceBox) -> np.ndarray:
    """Create a face embedding, trying ArcFace first, falling back to stats."""
    embedding = _extract_face_embedding(frame, box)
    if embedding is not None:
        return embedding
    return _extract_face_embedding_fallback(frame, box)


# ── profile creation & learning ────────────────────────────────────────────────

def _create_face_detection(face: FaceBox, frame: np.ndarray,
                           frame_w: int, frame_h: int) -> FaceDetection:
    """Create a new FaceDetection (session-level target) from a face detection."""
    embedding = _create_face_embedding(frame, face)
    return FaceDetection(
        width_ratio=face.width / frame_w,
        height_ratio=face.height / frame_h,
        aspect_ratio=face.width / max(face.height, 1),
        embedding=embedding,
        confidence_score=face.confidence,
        center_x=face.center[0] / frame_w,
        center_y=face.center[1] / frame_h,
        frame_width=frame_w,
        frame_height=frame_h,
    )


def _learn_from_detection(detection: FaceDetection, face: FaceBox,
                          frame: np.ndarray, frame_w: int, frame_h: int,
                          learning_rate: float = 0.08):
    """Incrementally update the face detection from a confirmed detection.

    Uses exponential moving average for geometry and embedding blending.
    """
    wr = face.width / frame_w
    hr = face.height / frame_h
    ar = face.width / max(face.height, 1)
    cx = face.center[0] / frame_w
    cy = face.center[1] / frame_h

    detection.width_ratio = detection.width_ratio * (1 - learning_rate) + wr * learning_rate
    detection.height_ratio = detection.height_ratio * (1 - learning_rate) + hr * learning_rate
    detection.aspect_ratio = detection.aspect_ratio * (1 - learning_rate) + ar * learning_rate
    detection.center_x = detection.center_x * (1 - learning_rate) + cx * learning_rate
    detection.center_y = detection.center_y * (1 - learning_rate) + cy * learning_rate
    detection.age_frames += 1
    detection.successful_matches += 1
    detection.frame_width = frame_w
    detection.frame_height = frame_h

    # Blend embedding
    new_emb = _create_face_embedding(frame, face)
    if new_emb is not None:
        detection.embedding = (
            detection.embedding * (1 - learning_rate) +
            new_emb * learning_rate
        )

    # Normalize embedding
    detection.embedding = detection.embedding / (np.linalg.norm(detection.embedding) + 1e-8)
    detection.confidence_score = min(1.0, detection.confidence_score + 0.02)


def _create_face_memory(name: str, face: FaceBox, frame: np.ndarray,
                        frame_w: int, frame_h: int) -> FaceMemory:
    """Create a persistent memory from initial detection."""
    embedding = _create_face_embedding(frame, face)
    wr = face.width / frame_w
    hr = face.height / frame_h
    ar = face.width / max(face.height, 1)

    now = datetime.now().isoformat()
    ts = int(now[:10].replace("-", ""))

    mem = FaceMemory(
        name=name,
        mean_embedding=embedding.copy() if embedding is not None else np.zeros(512),
        mean_width_ratio=wr,
        mean_height_ratio=hr,
        mean_aspect_ratio=ar,
        sample_count=1,
        confidence=face.confidence,
        created=now,
        created_ts=ts,
        total_tracking_time_frames=0,
    )
    _save_memory_file(name, mem)
    return mem


def _update_face_memory(memory: FaceMemory, face: FaceBox, frame: np.ndarray,
                        frame_w: int, frame_h: int,
                        learning_rate: float = 0.05):
    """Incrementally update a persistent memory from a confirmed detection."""
    embedding = _create_face_embedding(frame, face)
    if embedding is None:
        return

    wr = face.width / frame_w
    hr = face.height / frame_h
    ar = face.width / max(face.height, 1)

    # Update geometry means with EMA
    memory.mean_width_ratio = memory.mean_width_ratio * (1 - learning_rate) + wr * learning_rate
    memory.mean_height_ratio = memory.mean_height_ratio * (1 - learning_rate) + hr * learning_rate
    memory.mean_aspect_ratio = memory.mean_aspect_ratio * (1 - learning_rate) + ar * learning_rate

    # Running variance (simplified)
    diff_w = wr - memory.mean_width_ratio
    memory.var_width_ratio = memory.var_width_ratio * (1 - learning_rate) + diff_w * diff_w * learning_rate
    diff_h = hr - memory.mean_height_ratio
    memory.var_height_ratio = memory.var_height_ratio * (1 - learning_rate) + diff_h * diff_h * learning_rate

    # Blend embedding
    memory.add_sample_embedding(embedding, learning_rate)
    memory.total_tracking_time_frames += 1
    memory.confidence = min(1.0, memory.confidence + 0.01)
    memory.last_seen_frames_ago = 0

    # Save periodically
    if memory.sample_count % 3 == 0:
        _save_memory_file(memory.name, memory)


# ── face matching & recognition ─────────────────────────────────────────────────

def _adaptive_threshold(detection: FaceDetection) -> float:
    """Dynamic matching threshold based on detection confidence.

    Higher confidence = stricter threshold (lower = more matches accepted).
    For ArcFace cosine similarity: higher is better, so threshold is in [0.3, 0.7].
    """
    base = 0.60
    offset = (detection.confidence_score - 0.5) * 0.30
    return max(0.30, min(0.75, base - offset))


def _memory_threshold(memory: FaceMemory) -> float:
    """Dynamic threshold for persistent memory.

    Uses embedding variance (actual arcface similarity spread) for thresholding.
    With more samples, we know the face well → stricter threshold.
    ArcFace cosine similarity: typically 0.6-0.9 for same person.
    """
    if memory.sample_count < 2:
        return 0.70  # cautious with few samples

    if memory.sample_count < 5:
        return 0.65

    # Compute actual embedding variance from stored samples
    if len(memory.best_embeddings) < 2:
        return 0.65

    # Cosine similarities between stored samples give us embedding spread
    sims = []
    embeddings = memory.best_embeddings
    n = min(len(embeddings), 10)  # cap for performance
    for i in range(n):
        for j in range(i + 1, n):
            a = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
            b = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
            sim = float(np.dot(a, b))
            sims.append(max(0.0, sim))

    if not sims:
        return 0.65

    # Low variance among stored samples = high confidence → stricter threshold
    spread = float(np.std(sims))
    mean_sim = float(np.mean(sims))

    # Threshold: base 0.50, lowered by high mean similarity, raised by high spread
    mean_bonus = min((mean_sim - 0.6) * 0.15, 0.10)  # up to +0.10 for good mean
    spread_penalty = min(max(spread - 0.05, 0.0), 0.10)  # up to +0.10 for high spread
    threshold = 0.50 - mean_bonus + spread_penalty
    return max(0.40, min(0.65, threshold))


def match_profile_face(detections, detection: FaceDetection,
                       frame: np.ndarray,
                       frame_idx: int = -1) -> Optional[FaceBox]:
    """Find the face in detections that best matches the stored detection.

    Returns None if no face exceeds the adaptive threshold.
    Uses ArcFace cosine similarity for matching.
    Records the match score into the detection for smoothing.
    """
    if not detections or detection is None:
        return None

    threshold = _adaptive_threshold(detection)
    best_face = None
    best_score = -1.0  # cosine similarity: higher is better

    for face in detections:
        score = detection_cosine_similarity(face, detection, frame)
        if score > best_score:
            best_score = score
            best_face = face

    if best_score >= threshold and best_face is not None:
        detection.record_score(best_score, frame_idx)
        return best_face
    return None


def match_stored_face(detections, memory: FaceMemory,
                      frame: np.ndarray,
                      frame_idx: int = -1) -> tuple[Optional[FaceBox], float]:
    """Match detections against a persistent memory.

    Uses ensemble similarity (top-k stored embeddings) for robust matching.
    Returns (best_matching_face, cosine_similarity) or (None, 0.0).
    """
    if not detections or memory is None:
        return None, 0.0

    threshold = _memory_threshold(memory)
    best_face = None
    best_sim = 0.0

    for face in detections:
        embedding = _create_face_embedding(frame, face)
        if embedding is None:
            continue
        # Use ensemble similarity for more robust matching
        sim = memory.ensemble_similarity(embedding, top_k=min(3, max(1, memory.sample_count)))
        if sim > best_sim:
            best_sim = sim
            best_face = face

    if best_sim >= threshold and best_face is not None:
        return best_face, best_sim
    return None, best_sim


def detection_cosine_similarity(face: FaceBox, detection: FaceDetection,
                                frame: np.ndarray) -> float:
    """Compute combined score between a detected face and stored detection.

    Combines:
    1. ArcFace embedding cosine similarity (primary, weighted 0.80)
    2. Geometry similarity (secondary, weighted 0.20)

    Returns 0.0 (no match) to 1.0 (perfect match).

    ArcFace cosine similarity on L2-normalized embeddings is naturally
    in [0, 1] for face pairs, so we use it directly without the
    problematic (x+1)/2 conversion.
    """
    # 1. Embedding cosine similarity (primary signal)
    face_embedding = _create_face_embedding(frame, face)
    if face_embedding is None:
        return 0.0

    det_emb = detection.embedding / (np.linalg.norm(detection.embedding) + 1e-8)
    face_emb = face_embedding / (np.linalg.norm(face_embedding) + 1e-8)
    raw_cosine = float(np.dot(det_emb, face_emb))

    # ArcFace embeddings are L2-normalized; cosine is naturally [-1,1].
    # For face-to-face matching we clamp negative to 0 and use as-is.
    emb_score = max(0.0, raw_cosine)

    # 2. Geometry similarity (secondary, scale-aware)
    frame_w, frame_h = detection.frame_width, detection.frame_height
    if frame_w == 0 or frame_h == 0:
        frame_w, frame_h = frame.shape[1], frame.shape[0]

    wr = face.width / frame_w
    hr = face.height / frame_h
    ar = face.width / max(face.height, 1)

    # Use relative deviation from tracked geometry
    scale = max(detection.size_scale(), 0.001)

    # Width/height deviation as fraction of tracked size
    geo_w = abs(wr - detection.width_ratio) / max(scale, 0.001)
    geo_h = abs(hr - detection.height_ratio) / max(scale, 0.001)

    # Aspect ratio deviation (normalized by detected aspect ratio)
    geo_ar = abs(ar - detection.aspect_ratio) / max(detection.aspect_ratio, 0.01)

    # Clamp individual geometry deviations to 1.0 before weighting
    # This prevents one outlier from dominating
    geo_score = max(0.0, 1.0 - min(1.0, geo_w * 0.4 + geo_h * 0.4 + geo_ar * 0.2))

    # Combined score
    return 0.80 * emb_score + 0.20 * geo_score


# ── movement classification ─────────────────────────────────────────────────────

def classify(face: Optional[FaceBox], frame_w: int, frame_h: int,
             align_margin: float = DEFAULT_ALIGN_MARGIN,
             distance_margin: float = DEFAULT_DISTANCE_MARGIN) -> FaceDecision:
    """Classify the robot's next movement based on face position.

    Returns LEFT/RIGHT/CLOSE/FAR/GOOD/NO TARGET with corresponding command.
    """
    if face is None:
        return FaceDecision("NO TARGET", "p", 0, None)

    center_x = face.center[0]
    center_y = face.center[1]
    midpoint_x = frame_w / 2.0
    midpoint_y = frame_h / 2.0
    align_margin_px = frame_w * align_margin
    distance_margin_px = frame_h * distance_margin

    # Horizontal alignment
    if center_x < midpoint_x - align_margin_px:
        return FaceDecision("LEFT", "a", turn_speed(center_x - midpoint_x, frame_w, align_margin_px), face)
    if center_x > midpoint_x + align_margin_px:
        return FaceDecision("RIGHT", "d", turn_speed(center_x - midpoint_x, frame_w, align_margin_px), face)

    # Vertical distance (only if horizontally aligned)
    if center_y < midpoint_y - distance_margin_px:
        return FaceDecision("CLOSE", "s", 150, face)
    if center_y > midpoint_y + distance_margin_px:
        return FaceDecision("FAR", "w", 150, face)

    return FaceDecision("GOOD", "p", 0, face)


def turn_speed(offset: float, frame_width: float, align_margin: float) -> int:
    """Speed curve: low near center, high at edges, stepped to avoid jitter."""
    margin = align_margin
    range_margin = frame_width / 2.0 - margin
    if range_margin <= 0:
        return TURN_SPEED_MEDIUM
    ratio = min(max((abs(offset) - margin) / range_margin, 0.0), 1.0)
    if ratio < 0.5:
        speed = TURN_SPEED_LITTLE + (TURN_SPEED_MEDIUM - TURN_SPEED_LITTLE) * ratio * 2
    else:
        speed = TURN_SPEED_MEDIUM + (TURN_SPEED_BIG - TURN_SPEED_MEDIUM) * (ratio - 0.5) * 2
    return int(round(speed / TURN_SPEED_STEP)) * TURN_SPEED_STEP


# ── visualization ───────────────────────────────────────────────────────────────

def _boxes_overlap(box1: FaceBox, box2: FaceBox, radius: float = 30.0) -> bool:
    """Check if two boxes are near each other by center distance."""
    cx1, cy1 = box1.x + box1.width // 2, box1.y + box1.height // 2
    cx2, cy2 = box2.x + box2.width // 2, box2.y + box2.height // 2
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    return dist <= max(radius, 30.0)


def draw(frame: np.ndarray, detections, selected, decision,
         detection: Optional[FaceDetection] = None,
         face_memory: Optional[FaceMemory] = None,
         active_recognition_score: float = 0.0):
    """Draw bounding boxes, guides, and status overlays on the frame."""
    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw all detected faces
    for face in detections:
        color = (0, 255, 0) if selected is face else (0, 165, 255)
        if selected is face:
            # Diamond outline for tracked face
            cx, cy = face.center
            radius = max(face.width, face.height) // 2 + 8
            diamond = np.array([
                [cx, cy - radius], [cx + radius, cy],
                [cx, cy + radius], [cx - radius, cy],
            ], dtype=np.int32)
            cv2.polylines(frame, [diamond], True, color, 3)
        else:
            cv2.rectangle(frame, (face.x, face.y),
                          (face.x + face.width, face.y + face.height), color, 2)

        # Label per face
        face_decision = decision if selected is face else classify(
            face, frame.shape[1], frame.shape[0])
        # Compute similarity if we have a stored target detection to match against
        sim_score = None
        if detection is not None:
            sim_score = detection_cosine_similarity(face, detection, frame)
        if sim_score is not None and sim_score > 0.0:
            detail = f"sim {sim_score:.2f} | {face.width}x{face.height}"
        else:
            detail = f"conf {face.confidence:.2f} | {face.width}x{face.height}"
        font_scale = 0.42
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            detail, font, font_scale, thickness)
        detail_x = min(face.x + face.width, frame_width - 4) - text_width
        detail_x = max(detail_x, 4)
        detail_y = min(face.y + face.height + text_height + baseline + 4, frame_height - 4)
        bg_top = detail_y - text_height - baseline - 4
        cv2.rectangle(frame, (detail_x - 2, bg_top),
                      (detail_x + text_width + 2, detail_y + 2), (20, 20, 20), -1)
        cv2.putText(frame, detail, (detail_x, detail_y), font,
                    font_scale, color, thickness, cv2.LINE_AA)

    # Draw target selection overlay
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

        # Profile status text
        # Check if any detection closely matches the selected face's position
        selected_nearby = any(
            _boxes_overlap(selected, det, radius=30)
            for det in detections
        ) if detections else False
        if selected_nearby and detection is not None:
            score = detection_cosine_similarity(selected, detection, frame)
            profile_text = f"SIMILARITY {score:.2f}"
        elif face_memory is not None and active_recognition_score > 0:
            profile_text = f"FAMILIAR {active_recognition_score:.2f}"
        elif not selected_nearby:
            profile_text = "TARGET LOST"
        else:
            profile_text = "TARGET TRACKED"

        if detection is not None:
            profile_text += f" | C{detection.confidence_score:.2f}"

        (pw, ph), _ = cv2.getTextSize(profile_text, font, 0.42, 1)
        px = min(cx + radius + 6, frame_width - pw - 4)
        px = max(px, 4)
        py = max(cy - 8, ph + 4)
        cv2.putText(frame, profile_text, (px, py), font, 0.42, guide_color, 1, cv2.LINE_AA)

        # Distance guide lines
        guides = (
            ((cx, max(cy - radius, 0)), (cx, 0), f"{cy - radius}px", (cx + 6, 18), (cx + 6, 18)),
            ((min(cx + radius, frame_width - 1), cy), (frame_width - 1, cy), f"{frame_width - 1 - cx - radius}px",
             (frame_width - 70, max(cy - 6, 18)), (min(cx + radius, frame_width - 1) - 42, max(cy - 6, 18))),
            ((cx, min(cy + radius, frame_height - 1)), (cx, frame_height - 1),
             f"{frame_height - 1 - cy - radius}px", (cx + 6, frame_height - 8), (cx + 6, min(cy + radius, frame_height - 1) - 6)),
            ((max(cx - radius, 0), cy), (0, cy), f"{cx - radius}px", (6, max(cy - 6, 18)),
             (max(cx - radius, 0) + 6, max(cy - 6, 18))),
        )
        for tip, edge, label, _, _ in guides:
            cv2.line(frame, tip, edge, guide_color, 1, cv2.LINE_AA)
            cv2.circle(frame, tip, 3, guide_color, -1)

    # Top status
    if decision.box is not None and decision.box is selected:
        cv2.putText(frame, f"TARGET: {decision.label}",
                    (decision.box.x, max(decision.box.y - 8, 18)),
                    font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# ── YOLO detection class ────────────────────────────────────────────────────────

def find_model_path(model_name: str = DEFAULT_MODEL) -> str:
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base_dir in (MODEL_DIR, os.getcwd(), script_dir):
        path = os.path.join(base_dir, model_name)
        if os.path.exists(path):
            return path
    return model_name


def download_model(model_name: str = DEFAULT_MODEL, progress=None) -> str:
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
    def __init__(self, model_name: str = DEFAULT_MODEL, confidence: float = 0.4,
                 align_margin: float = DEFAULT_ALIGN_MARGIN,
                 distance_margin: float = DEFAULT_DISTANCE_MARGIN):
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is unavailable; install the project dependencies "
                "before loading a face model"
            )
        self.model_path = find_model_path(model_name)
        self.model = YOLO(self.model_path)
        self.confidence = confidence
        self.align_margin = align_margin
        self.distance_margin = distance_margin
        self.stored_faces: dict[str, FaceMemory] = {}
        self._load_stored_faces()

    def _load_stored_faces(self):
        """Load all stored face memories from disk."""
        for name in list_stored_faces():
            raw = _load_memory_file(name)
            if raw is None:
                continue
            try:
                memory = FaceMemory.from_dict(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            self.stored_faces[name] = memory

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(FaceBox(
                int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(result.conf[0])
            ))
        return detections

    def choose(self, detections: list[FaceBox], point: tuple[int, int]) -> Optional[FaceBox]:
        """Select a face from detections by clicking a point in the frame."""
        containing = [face for face in detections if face.contains(point)]
        if not containing:
            return None
        return max(containing, key=lambda face: face.width * face.height)

    def match_selected(self, detections: list[FaceBox], selected: FaceBox,
                       detection: Optional[FaceDetection] = None, frame: Optional[np.ndarray] = None,
                       radius_multiplier: float = DEFAULT_RECOGNITION_RADIUS_MULTIPLIER) -> Optional[FaceBox]:
        """Find the currently tracked face among detections."""
        if selected is None or not detections:
            return None

        # Step 1: IOU check
        for face in detections:
            if face.io_union(selected) > 0.5:
                return face

        # Step 2: radius-based search
        search_radius = max(selected.width, selected.height) * radius_multiplier

# ── theme ───────────────────────────────────────────────────────────────────

PANEL, SURFACE, STAGE = "#eef1f4", "#ffffff", "#111820"
INK, MUTED, LINE = "#1c2630", "#5b6770", "#cfd6dc"
ACCENT, STOP, PREVIEW = "#0b6e99", "#c62828", "#42d9ff"
TONE = {"idle": "#7b8794", "ok": "#1a8f4b", "busy": ACCENT, "warn": "#b7791f", "bad": STOP}
LOG_COLORS = {"warn": "#f0b429", "bad": "#ff7b72", "ok": "#1a8f4b"}

# ── timing constants ────────────────────────────────────────────────────────

TARGET_LOST_FRAMES = 45          # frames without target before release
NO_VIDEO_SECONDS = 1.5
PREVIEW_WIDTH = 220
LOG_LINES = 500
FAMILIAR_ALERT_FRAMES = 15       # frames to show "FAMILIAR" label after match
FAMILIAR_ALERT_DELAY = 3.0       # seconds to keep alert visible

# ── UI text maps ────────────────────────────────────────────────────────────

STATE_TEXT = {
    "NO TARGET": ("No target", "idle"),
    "GOOD": ("On target", "ok"),
    "FAR": ("Target too far", "busy"),
    "CLOSE": ("Target too close", "busy"),
    "LEFT": ("Target left of center", "busy"),
    "RIGHT": ("Target right of center", "busy"),
}
ACTION_TEXT = {"w": "Driving forward", "s": "Backing up", "a": "Turning left",
               "d": "Turning right", "p": "Stopped"}


# ── vision worker ───────────────────────────────────────────────────────────

class VisionWorker(threading.Thread):
    """Background thread: stream → detect → track → classify → command."""

    def __init__(self, settings, events):
        super().__init__(daemon=True)
        self.settings = settings
        self.events = events
        self.stop_event = threading.Event()
        self.stop_robot_event = threading.Event()
        self.stop_sent = False
        self.clear_event = threading.Event()
        self.detector = None
        self.selection_lock = threading.Lock()
        self.selection_point = None
        self.selected = None
        self.target_profile: FaceDetection | None = None
        self.target_missed_frames = 0
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        # Recognition state
        self.familiar_name: str | None = None
        self.familiar_score: float = 0.0
        self.familiar_alert_countdown = 0  # frames to show familiar alert

    # ── public API ───────────────────────────────────────────────────────

    def select(self, point):
        with self.selection_lock:
            self.selection_point = point

    def stop(self):
        self.stop_event.set()

    def stop_robot(self):
        self.stop_robot_event.set()

    def clear_selection(self):
        self.clear_event.set()

    def publish(self, event):
        self.events.put(event)

    def _request_face_name(self, face, frame, frame_w, frame_h):
        result = {"name": ""}
        done = threading.Event()
        self.publish(("name_prompt", face, frame.copy(), frame_w,
                      frame_h, result, done))
        while not done.wait(0.1):
            if self.stop_event.is_set():
                return ""
        return result["name"]

    def take_frame(self):
        """Latest rendered frame, or None. Older frames are dropped, never queued."""
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
        return frame

    # ── internal ─────────────────────────────────────────────────────────

    def _load_detector(self):
        try:
            model_path = find_model_path(self.settings["model"])
            if not os.path.exists(model_path):
                self.publish(("error",
                              f"Model not found locally: {self.settings['model']}; load it before connecting"))
                return None
            self.publish(("log", f"Loading {self.settings['model']}..."))
            detector = FaceDetector(model_path,
                                    confidence=self.settings["confidence"])
            self.detector = detector
            self.publish(("detector_ready", detector,
                          self.settings["model"]))
            self.publish(("log", f"YOLO ready: {detector.model_path}"))
            return detector
        except Exception as exc:
            self.publish(("error", f"YOLO could not load: {exc}"))
            return None

    def run(self):
        stream = None
        client = RobotCommandClient(self.settings["host"],
                                    self.settings["port"])
        try:
            detector = self._load_detector()
            if detector is None:
                return

            retrying = False
            link_ok = True
            fps = 0.0
            last_frame_time = None

            while not self.stop_event.is_set():
                # ── stop / clear commands ──────────────────────────
                if self.stop_robot_event.is_set():
                    if not getattr(self, "stop_sent", False):
                        sent, detail = client.send("p", 0, force=True)
                        self.publish(("log", "Stop command sent"
                                      if sent else f"Stop failed: {detail}"))
                        self.stop_sent = True

                if self.clear_event.is_set():
                    self.clear_event.clear()
                    self._clear_all()

                # ── stream handling ────────────────────────────────
                if stream is None or not stream.is_opened():
                    if not retrying:
                        self.publish(("status",
                                      "Connecting to XIAO stream...",
                                      "busy"))
                    stream = MjpegStream(self.settings["url"])
                    if not stream.is_opened():
                        if not retrying:
                            self.publish(("status",
                                          "Stream unavailable; retrying",
                                          "warn"))
                            retrying = True
                        self.stop_event.wait(1.0)
                        continue
                    retrying = False
                    self.publish(("status", "Stream connected", "ok"))

                ok, frame = stream.read()
                if not ok:
                    stream.release()
                    stream = None
                    last_frame_time = None
                    client.stop()
                    self.publish(("status",
                                  "Stream lost; target memory retained",
                                  "bad"))
                    continue

                now = time.monotonic()
                if last_frame_time is not None:
                    instant = 1.0 / max(now - last_frame_time, 1e-3)
                    fps = (instant if fps == 0
                           else fps * 0.9 + instant * 0.1)
                last_frame_time = now

                # ── detection ──────────────────────────────────────
                frame = cv2.flip(frame, -1)
                preview = cv2.cvtColor(
                    cv2.resize(
                        frame,
                        (PREVIEW_WIDTH,
                         PREVIEW_WIDTH * frame.shape[0] //
                         frame.shape[1]),
                        interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB)
                detector.confidence = self.settings["confidence"]
                detections = detector.detect(frame)

                frame_w, frame_h = frame.shape[1], frame.shape[0]

                # ── face selection (user click) ────────────────────
                with self.selection_lock:
                    point = self.selection_point
                    self.selection_point = None
                if point is not None:
                    chosen, profile, log_msg, face_emb = (
                        detector.select_face(
                            detections, point, frame, frame_w, frame_h,
                            prompt_name=False))
                    if chosen is None:
                        self.publish(("log", log_msg))
                    else:
                        # Auto-save selected face as target (with pre-computed embedding)
                        saved = detector._save_target(
                            chosen, frame, frame_w, frame_h,
                            profile, embedding=face_emb)
                        if saved is not None:
                            # Link profile to stored memory so heatmap data flows
                            profile.stored_name = "target"
                            self.publish(("log", "Target saved: target"))
                        else:
                            self.publish(("log", "Target selected (save failed)"))
                        self._select_target(chosen, profile,
                                            detections, frame,
                                            frame_w, frame_h)

                # ── face tracking & recognition ────────────────────
                active_selected = None
                recognition_name = None
                recognition_score = 0.0

                if self.selected is not None and detections:
                    # Phase 1: spatial match from previous selection
                    active_selected = detector.match_selected(
                        detections, self.selected,
                        self.target_profile, frame)

                    # Phase 2: profile-based match
                    if active_selected is None:
                        active_selected = detector.match_profile_face(
                            detections, self.target_profile, frame)
                        if active_selected is not None:
                            self.publish(("log",
                                          "Target reacquired by "
                                          "profile match"))
                    if active_selected is None:
                        rec_name, rec_face, rec_score = (
                            detector.match_stored_faces(
                                detections, frame))
                        if rec_name is not None:
                            recognition_name = rec_name
                            recognition_score = rec_score
                            active_selected = rec_face

                # ── update tracking state ────────────────────────
                if active_selected is not None:
                    # We have a lock on the target
                    self._update_target(active_selected,
                                        detections, frame,
                                        frame_w, frame_h)

                    # Update stored face profile if this is a known face
                    if recognition_name is not None:
                        detector.update_stored_face(
                            recognition_name, active_selected,
                            frame, frame_w, frame_h)
                        # Promote: this becomes the active tracked face now
                        self.familiar_name = recognition_name
                        self.familiar_score = recognition_score
                        self.familiar_alert_countdown = (
                            int(FAMILIAR_ALERT_FRAMES * (
                                frame_h / 480)))
                        self.publish(("log",
                                      f"Familiar face: "
                                      f"{recognition_name} "
                                      f"({recognition_score:.2f})"))
                    elif self.familiar_alert_countdown > 0:
                        self.familiar_alert_countdown -= 1
                else:
                    # Target lost
                    self.target_missed_frames += 1
                    if (self.target_missed_frames
                            == TARGET_LOST_FRAMES):
                        self.publish(("log",
                                      "Target lost; memory "
                                      "retained for "
                                      "reacquisition"))
                    if self.target_missed_frames > TARGET_LOST_FRAMES * 3:
                        # Give up after too many misses
                        self.publish(("log",
                                      "Target lost for too long; "
                                      "tracking released"))
                        self._clear_all()

                # ── classification & command ─────────────────────
                decision = detector.classify(
                    active_selected, frame_w, frame_h)
                if self.stop_robot_event.is_set():
                    continue
                sent, detail = client.send(decision.command,
                                           decision.speed)
                if sent != link_ok:
                    link_ok = sent
                    if sent:
                        self.publish(("status",
                                      "Robot link restored", "ok"))
                    else:
                        self.publish(("warning",
                                      f"Robot command failed: "
                                      f"{detail}"))

                # ── memory data for UI ───────────────────────────
                memory_data = None
                if self.target_profile is not None:
                    mem = getattr(self.target_profile, "stored_name", None)
                    if mem:
                        stored = detector.stored_faces.get(mem)
                        if stored:
                            sim = ensemble_similarity(stored)
                            # Compute per-embedding similarity scores for heatmap
                            target_mem = detector.stored_faces.get("target", None)
                            similarity_scores = []
                            if target_mem is not None:
                                for e in stored.best_embeddings:
                                    sim = float(target_mem.cosine_similarity(e))
                                    similarity_scores.append(sim)
                            memory_data = {
                                "name": mem,
                                "similarity": float(sim),
                                "samples": stored.sample_count,
                                "ensemble_size": len(stored.best_embeddings),
                                "confidence": stored.confidence,
                                "similarity_scores": similarity_scores,
                            }
                    else:
                        memory_data = {
                            "name": "Unlearned target",
                            "similarity": 0.0,
                            "samples": 0,
                            "ensemble_size": 0,
                            "confidence": 0.0,
                            "_embeddings": [],
                        }

                # ── render ───────────────────────────────────────
                target_crop = None
                if active_selected is not None:
                    padding = max(active_selected.width,
                                  active_selected.height)
                    x1 = max(active_selected.x - padding, 0)
                    y1 = max(active_selected.y - padding, 0)
                    x2 = (min(active_selected.x +
                              active_selected.width + padding,
                              frame_w))
                    y2 = (min(active_selected.y +
                              active_selected.height + padding,
                              frame_h))
                    target_crop = frame[y1:y2, x1:x2].copy()

                rec_profile: FaceMemory | None = None
                if recognition_name is not None:
                    rec_profile = (
                        detector.stored_faces.get(recognition_name))

                rendered = detector.draw(
                    frame, detections, self.selected, decision,
                    self.target_profile, rec_profile,
                    self.familiar_score if (
                        self.familiar_alert_countdown > 0
                        or recognition_name is not None) else 0.0)

                with self.frame_lock:
                    self.latest_frame = (rendered, preview,
                                         decision, len(detections),
                                         fps, target_crop,
                                         recognition_name,
                                         self.familiar_alert_countdown
                                         > 0, memory_data)

        except Exception as exc:
            self.publish(("error", f"Tracking stopped: {exc}"))
        finally:
            if stream is not None:
                stream.release()
            client.stop()
            self.publish(("stopped", self))

    # ── tracking helpers ─────────────────────────────────────────────────

    def _select_target(self, face, profile, detections, frame,
                       frame_w, frame_h):
        """Set the newly selected face as target."""
        self.selected = face
        self.target_profile = profile
        self.target_missed_frames = 0
        # Check if this matches any stored face
        detector = self.detector
        if detector is not None:
            rec_name, rec_face, rec_score = detector.match_stored_faces(
                detections, frame)
            if rec_name is not None:
                self.publish(("log",
                              f"Already known: {rec_name} "
                              f"({rec_score:.2f})"))

    def _update_target(self, face, detections, frame,
                       frame_w, frame_h):
        """Update target state from a confirmed detection."""
        self.selected = face
        if self.target_profile is not None:
            detector = self.detector
            if detector is not None:
                detector.learn_from_target(
                    self.target_profile, face,
                    frame, frame_w, frame_h)
            self.target_missed_frames = 0

    def _clear_all(self):
        """Release tracking selection and memory."""
        if self.detector is not None:
            self.detector._drop_target_embeddings()
        self.selected = None
        self.target_profile = None
        self.target_missed_frames = 0
        with self.selection_lock:
            self.selection_point = None
        self.familiar_name = None
        self.familiar_score = 0.0
        self.familiar_alert_countdown = 0


# ── desktop app ─────────────────────────────────────────────────────────────

class EspApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XIAO ESP32S3 Face Control")
        self.geometry("1180x780")
        self.minsize(960, 660)
        self.worker = None
        self.worker_settings = None
        self.disconnecting = False
        self.events = queue.Queue()
        self.photo = None
        self.preview_photo = None
        self.frame_size = (1, 1)
        self.display_rect = (0, 0, 1, 1)
        self.has_frame = False
        self.last_frame_time = 0.0
        self.detector = None
        self.detector_name = None
        self.loading = None
        self.status_tone = "idle"
        self.overlay_state = {}
        self._build_ui()
        self._begin_load(DEFAULT_MODEL, download=False)
        self.after(25, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind_all("<Escape>", lambda _event: self._stop_robot())

    # ── UI construction ──────────────────────────────────────────────────

    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        self.configure(bg=PANEL)
        base = tkfont.nametofont("TkDefaultFont")
        size = abs(base.cget("size")) or 10
        self.font_title = base.copy()
        self.font_title.configure(size=size + 5, weight="bold")
        self.font_banner = base.copy()
        self.font_banner.configure(size=size + 4, weight="bold")
        self.font_head = base.copy()
        self.font_head.configure(weight="bold")

        style.configure(".", background=PANEL, foreground=INK,
                        bordercolor=LINE)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Section.TLabel", font=self.font_head)
        style.configure("TSeparator", background=LINE)
        style.configure("TButton", padding=(10, 6),
                        background="#e2e7ec",
                        lightcolor="#e2e7ec",
                        darkcolor="#e2e7ec")
        style.map("TButton",
                  background=[("disabled", PANEL),
                              ("pressed", "#c9d1d8"),
                              ("active", "#d5dce2")],
                  foreground=[("disabled", "#9aa5ae")])
        style.configure("Accent.TButton", background=ACCENT,
                        foreground="#ffffff",
                        lightcolor=ACCENT, darkcolor=ACCENT,
                        bordercolor=ACCENT)
        style.map("Accent.TButton",
                  background=[("disabled", "#9ab8c6"),
                              ("pressed", "#084f6f"),
                              ("active", "#095c82")],
                  foreground=[("disabled", "#ffffff")])
        style.configure("Stop.TButton", background=STOP,
                        foreground="#ffffff",
                        padding=(10, 12), font=self.font_head,
                        lightcolor=STOP, darkcolor=STOP,
                        bordercolor=STOP)
        style.map("Stop.TButton",
                  background=[("pressed", "#8f1d1d"),
                              ("active", "#ad2222")])
        style.configure("TCombobox", fieldbackground=SURFACE,
                        background="#e2e7ec",
                        arrowcolor=INK, padding=4)
        style.map("TCombobox",
                  fieldbackground=[("readonly", SURFACE),
                                   ("disabled", PANEL)],
                  foreground=[("disabled", "#9aa5ae")])
        style.configure("Horizontal.TScale", background=PANEL,
                        troughcolor=LINE)
        style.configure("Horizontal.TProgressbar", troughcolor=LINE,
                        background=ACCENT, borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", SURFACE)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground",
                        ACCENT)

    def _build_ui(self):
        self._apply_theme()
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Left control panel ─────────────────────────────────────────
        controls = ttk.Frame(self, padding=(16, 14, 12, 12), width=310)
        controls.grid(row=0, column=0, sticky="ns")
        controls.pack_propagate(False)

        viewer = ttk.Frame(self, padding=(0, 12, 12, 12))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Face control",
                  font=self.font_title).pack(anchor="w")

        # ── Connection info ────────────────────────────────────────────
        self.stream_url = "http://192.168.4.1/stream"
        self.robot_host = "192.168.4.1"
        self.robot_port = 8080
        self._section(controls, "Robot")
        info = ttk.Frame(controls)
        info.pack(fill="x")
        for row, (name, value) in enumerate((("Stream", self.stream_url),
                                             ("Commands",
                                              f"{self.robot_host}"
                                              f":{self.robot_port}"))):
            ttk.Label(info, text=name,
                      style="Muted.TLabel").grid(
                row=row, column=0, sticky="w",
                padx=(0, 12))
            ttk.Label(info, text=value).grid(
                row=row, column=1, sticky="w")
        self.connect_button = ttk.Button(
            controls, text="Connect",
            style="Accent.TButton",
            command=self._toggle_connection)
        self.connect_button.pack(fill="x", pady=(10, 8))

        # ── Decision display card ──────────────────────────────────────
        card = tk.Frame(controls, bg=SURFACE,
                        highlightbackground=LINE,
                        highlightthickness=1,
                        padx=10, pady=8)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=SURFACE)
        head.pack(fill="x")
        self.decision_dot = tk.Label(head, text="●", bg=SURFACE)
        self.decision_dot.pack(side="left")
        self.decision = tk.Label(head, bg=SURFACE, fg=INK,
                                 font=self.font_banner)
        self.decision.pack(side="left", padx=(6, 0))
        self.action = tk.Label(card, bg=SURFACE, fg=MUTED,
                               anchor="w")
        self.action.pack(fill="x")
        self._show_decision("No target", "idle", "Stopped")

        ttk.Button(controls, text="Stop robot (Esc)",
                   style="Stop.TButton",
                   command=self._stop_robot).pack(fill="x")

        # ── Model selection ────────────────────────────────────────────
        self._section(controls, "Model")
        self.model = ttk.Combobox(controls,
                                  values=FACE_MODEL_OPTIONS,
                                  state="readonly")
        self.model.set(DEFAULT_MODEL)
        self.model.pack(fill="x", pady=(0, 6))
        row, self.model_dot, self.model_status = (
            self._status_line(controls, "Not loaded"))
        row.pack(anchor="w", pady=(0, 6))
        self.model_buttons = ttk.Frame(controls)
        self.model_buttons.pack(fill="x")
        self.load_model_button = ttk.Button(
            self.model_buttons, text="Load",
            command=self._load_model)
        self.load_model_button.pack(side="left",
                                    expand=True, fill="x",
                                    padx=(0, 3))
        self.unload_model_button = ttk.Button(
            self.model_buttons, text="Unload",
            command=self._unload_model)
        self.unload_model_button.pack(side="left",
                                      expand=True, fill="x",
                                      padx=(3, 0))
        self.download_bar = ttk.Progressbar(
            controls, maximum=1.0)

        # ── Target section (single-target memory heatmap) ────────────
        self._section(controls, "Target")
        # Confidence slider
        confidence_row = ttk.Frame(controls)
        confidence_row.pack(fill="x")
        ttk.Label(confidence_row, text="Confidence Threshold").pack(
            side="left")
        self.confidence_label = ttk.Label(
            confidence_row, text="0.40")
        self.confidence_label.pack(side="right")
        self.confidence = ttk.Scale(
            controls, from_=0.00, to=1.00, value=0.40,
            command=self._on_confidence)
        self.confidence.pack(fill="x", pady=(2, 6))
        # Target info card
        self.target_card = tk.Frame(controls,
                                    bg=SURFACE,
                                    highlightbackground=LINE,
                                    highlightthickness=1)
        self.target_card.pack(fill="x", pady=(0, 8))
        self.target_name_label = tk.Label(
            self.target_card,
            text="No target selected",
            bg=SURFACE, fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w")
        self.target_name_label.pack(anchor="w",
                                    padx=10, pady=(8, 2))
        # Heatmap canvas — 8 cols × up to 6 rows, 12px cells, 2px gaps
        # Width: 8*12 + 7*2 = 110; Height: 6*12 + 5*2 = 82
        self.heatmap_canvas = tk.Canvas(
            self.target_card,
            width=114, height=86,
            bg=SURFACE, highlightthickness=0)
        self.heatmap_canvas.pack(padx=10, pady=6)
        # Detail label below heatmap
        self.heatmap_detail = tk.Label(
            self.target_card,
            text="Samples: 0",
            bg=SURFACE, fg=MUTED,
            font=("TkFixedFont", 7))
        self.heatmap_detail.pack(anchor="w",
                                 padx=10, pady=(0, 8))

        # ── Main video canvas ──────────────────────────────────────────
        self.canvas = tk.Canvas(viewer, bg=STAGE,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._select_face)
        self.canvas.bind("<Configure>",
                         lambda _event:
                         self._update_stage_message())
        self.video_item = self.canvas.create_image(
            0, 0, anchor="nw", state="hidden")
        self.preview_border = (
            self.canvas.create_rectangle(
                0, 0, 0, 0, outline=PREVIEW, width=2,
                state="hidden"))
        self.preview_item = self.canvas.create_image(
            0, 0, anchor="sw", state="hidden")
        self.preview_label = self.canvas.create_text(
            0, 0, anchor="sw", fill=PREVIEW,
            text="Target", state="hidden")
        self.message = self._overlay("center",
                                     self.font_head,
                                     justify="center")
        self.hint = self._overlay("s", "TkDefaultFont")

        # ── Bottom status bar ──────────────────────────────────────────
        bar = ttk.Frame(viewer, padding=(2, 8, 2, 0))
        bar.grid(row=1, column=0, sticky="ew")
        row, self.status_dot, self.status = (
            self._status_line(bar, "Not connected"))
        row.pack(side="left")
        self.stats = ttk.Label(bar, style="Muted.TLabel")
        self.stats.pack(side="right")
        self._update_controls()

    # ── UI helpers ─────────────────────────────────────────────────────

    def _section(self, parent, title):
        ttk.Separator(parent).pack(fill="x", pady=(12, 0))
        ttk.Label(parent, text=title,
                  style="Section.TLabel").pack(
            anchor="w", pady=(8, 4))

    @staticmethod
    def _status_line(parent, text):
        row = ttk.Frame(parent)
        dot = ttk.Label(row, text="●",
                        foreground=TONE["idle"])
        dot.pack(side="left")
        label = ttk.Label(row, text=text)
        label.pack(side="left", padx=(6, 0))
        return row, dot, label

    def _overlay(self, anchor, font, **options):
        background = self.canvas.create_rectangle(
            0, 0, 0, 0, fill=STAGE, outline="",
            state="hidden")
        label = self.canvas.create_text(
            0, 0, anchor=anchor, fill="#d8e2e8",
            font=font, state="hidden", **options)
        return background, label

    def _show_text(self, overlay, text, x, y):
        background, label = overlay
        key = (text, round(x), round(y))
        if self.overlay_state.get(label) == key:
            return
        self.overlay_state[label] = key
        if not text:
            self.canvas.itemconfigure(background,
                                      state="hidden")
            self.canvas.itemconfigure(label,
                                      state="hidden")
            return
        self.canvas.itemconfigure(label, text=text,
                                  state="normal")
        self.canvas.coords(label, x, y)
        x1, y1, x2, y2 = self.canvas.bbox(label)
        self.canvas.coords(background,
                           x1 - 10, y1 - 6, x2 + 10,
                           y2 + 6)
        self.canvas.itemconfigure(background,
                                  state="normal")
        self.canvas.tag_raise(background)
        self.canvas.tag_raise(label)

    def _update_stage_message(self):
        if self.worker is None:
            text = ("Not connected\nJoin the robot's Wi-Fi, "
                    "then press Connect.")
        elif (not self.has_frame
              or time.monotonic() - self.last_frame_time >
              NO_VIDEO_SECONDS):
            text = "Waiting for video..."
        else:
            text = None
        self._show_text(
            self.message, text,
            self.canvas.winfo_width() / 2,
            self.canvas.winfo_height() / 2)

    def _reset_stage(self):
        self.has_frame = False
        self.canvas.itemconfigure(
            self.video_item, image="")
        self.photo = None
        for item in (self.video_item,
                     self.preview_border,
                     self.preview_item,
                     self.preview_label):
            self.canvas.itemconfigure(item,
                                      state="hidden")
        self._show_text(self.hint, None, 0, 0)

    def _show_decision(self, label, tone, action):
        self.decision.configure(text=label)
        self.decision_dot.configure(fg=TONE[tone])
        self.action.configure(text=action)

    def _set_status(self, text, tone="idle"):
        self.status_tone = tone
        self.status.configure(text=text)
        self.status_dot.configure(
            foreground=TONE[tone])

    def _set_model_status(self, text, tone):
        self.model_status.configure(text=text)
        self.model_dot.configure(foreground=TONE[tone])

    def _refresh_model_status(self):
        if self.loading:
            self._set_model_status(
                f"Loading {self.loading}...", "busy")
        elif self.detector is not None:
            self._set_model_status(
                f"Ready: {self.detector_name}", "ok")
        else:
            self._set_model_status("Not loaded", "idle")

    def _update_controls(self):
        idle = (self.worker is None
                and not self.loading)
        self.model.configure(
            state="readonly" if idle else "disabled")
        self.load_model_button.configure(
            state="normal" if idle else "disabled")
        self.unload_model_button.configure(
            state=("normal"
                   if idle and self.detector
                   else "disabled"))
        if self.worker is None:
            self.connect_button.configure(
                text="Connect",
                style="Accent.TButton",
                state="disabled"
                if self.loading else "normal")
        elif self.disconnecting:
            self.connect_button.configure(
                text="Disconnecting...",
                state="disabled")
        else:
            self.connect_button.configure(
                text="Disconnect",
                style="TButton",
                state="normal")
        self.canvas.configure(
            cursor="crosshair"
            if self.worker else "")

    def _log(self, message, tone="info"):
        # Print all log messages to stdout (terminal)
        print(f"[LOG] {message}")

    def _on_confidence(self, value):
        value = round(float(value), 2)
        self.confidence_label.configure(
            text=f"{value:.2f}")
        if self.worker_settings is not None:
            self.worker_settings["confidence"] = value

    # ── heatmap helpers ──────────────────────────────────────────────

    def _get_heatmap_color(self, similarity: float) -> str:
        """Return GitHub-style heatmap color for a similarity score."""
        if similarity >= 0.90:
            return "#216e39"
        elif similarity >= 0.75:
            return "#30a14e"
        elif similarity >= 0.50:
            return "#40c463"
        elif similarity >= 0.30:
            return "#9be9a8"
        else:
            return "#ebedf0"

    def _draw_heatmap(self, similarity_scores: list[float]):
        """Draw GitHub-style contribution grid on the heatmap canvas."""
        self.heatmap_canvas.delete("all")
        cell_size = 12
        gap = 2
        cols = 8
        max_cells = 48

        if not similarity_scores:
            # Show "no data" placeholder
            self.heatmap_canvas.create_text(
                57, 43, text="no data",
                fill=MUTED, font=("TkFixedFont", 8))
            return

        n = min(len(similarity_scores), max_cells)
        for i in range(n):
            col = i % cols
            row = i // cols
            x = col * (cell_size + gap)
            y = row * (cell_size + gap)
            score = similarity_scores[i]
            color = self._get_heatmap_color(score)
            self.heatmap_canvas.create_rectangle(
                x, y, x + cell_size, y + cell_size,
                fill=color, outline="",
                width=0)

    # ── model management ─────────────────────────────────────────────

    def _load_model(self):
        self._begin_load(self.model.get(), download=True)

    def _begin_load(self, model_name, download):
        if self.loading or self.worker is not None:
            return
        self.loading = model_name
        self._refresh_model_status()
        self._update_controls()
        self._log(f"Loading {model_name}...")
        confidence = round(
            float(self.confidence.get()), 2)

        def load():
            try:
                if (not os.path.exists(
                        find_model_path(model_name))):
                    if not download:
                        self.events.put((
                            "log",
                            f"{model_name} not found "
                            f"locally; press Load to "
                            f"download it"))
                        return
                    self.events.put((
                        "log",
                        f"Downloading {model_name} "
                        f"from GitHub..."))
                    download_model(model_name,
                                   progress=(
                                       lambda done:
                                       self.events.put(
                                           ("progress",
                                            done))))
                detector = FaceDetector(
                    model_name,
                    confidence=confidence)
                self.events.put((
                    "detector_ready", detector,
                    model_name))
                self.events.put(("log",
                                 f"YOLO ready: "
                                 f"{detector.model_path}"))
            except Exception as exc:
                self.events.put(("error",
                                 f"Model load failed: "
                                 f"{exc}"))
            finally:
                self.events.put(("load_finished",))

        threading.Thread(target=load,
                         daemon=True).start()

    def _unload_model(self):
        self.detector = None
        self.detector_name = None
        self._log("YOLO model unloaded")
        self._refresh_model_status()
        self._update_controls()

    # ── connection management ────────────────────────────────────────

    def _toggle_connection(self):
        if self.worker is None:
            self._connect()
        else:
            self.disconnecting = True
            self.worker.stop()
            self._log("Disconnecting")
            self._update_controls()

    def _connect(self):
        if self.worker is not None or self.loading:
            return
        model_name = self.model.get()
        if (self.detector is None
                or self.detector_name != model_name):
            self._set_status(
                "Load the selected model before connecting",
                "warn")
            self._log(
                f"Cannot connect: {model_name} is not "
                f"loaded locally", "warn")
            return
        self.worker_settings = {
            "url": self.stream_url,
            "host": self.robot_host,
            "port": self.robot_port,
            "model": model_name,
            "confidence": round(
                float(self.confidence.get()), 2),
        }
        self.disconnecting = False
        self.has_frame = False
        self.worker = VisionWorker(
            self.worker_settings, self.events)
        self.worker.start()
        self._update_controls()
        self._log("Starting connection")

    # ── face selection ───────────────────────────────────────────────

    def _select_face(self, event):
        if not self.worker or not self.worker.is_alive():
            return
        frame_w, frame_h = self.frame_size
        (image_x, image_y,
         image_width, image_height) = (
            self.display_rect)
        if (not (image_x <= event.x <= image_x + image_width
                 and image_y <= event.y
                 <= image_y + image_height)):
            return
        point = (int((event.x - image_x) *
                      frame_w / image_width),
                 int((event.y - image_y) *
                      frame_h / image_height))
        self.worker.select(point)

    # ── stop robot ───────────────────────────────────────────────────

    def _stop_robot(self):
        if self.worker:
            self.worker.clear_selection()
        # Send directly so stop doesn't wait on vision loop
        threading.Thread(
            target=self._send_stop,
            args=(self.worker is None,),
            daemon=True).start()
        self._log("Stop requested")
        self._log("Selection cleared")

    def _send_stop(self, log_success):
        client = RobotCommandClient(
            self.robot_host,
            self.robot_port,
            timeout=1.0)
        sent, detail = client.send("p", 0,
                                   force=True)
        if not sent:
            self.events.put((
                "warning",
                f"Stop failed: {detail}"))
        elif log_success:
            self.events.put(("log",
                             "Stop command sent"))

    def _handle_event(self, event):
        kind = event[0]
        if kind == "status":
            self._set_status(event[1], event[2])
            self._log(event[1])
        elif kind in {"warning", "error"}:
            tone = "warn" if kind == "warning" else "bad"
            self._set_status(event[1], tone)
            self._log(event[1], tone)
        elif kind == "log":
            self._log(event[1])
        elif kind == "detector_ready":
            self.detector, self.detector_name = (
                event[1], event[2])
            self._refresh_model_status()
            self._update_controls()
        elif kind == "progress":
            if not self.download_bar.winfo_ismapped():
                self.download_bar.pack(
                    fill="x", pady=(6, 0),
                    after=self.model_buttons)
            self.download_bar.configure(
                value=event[1])
            self._set_model_status(
                f"Downloading {self.loading}... "
                f"{event[1]:.0%}", "busy")
        elif kind == "load_finished":
            self.loading = None
            self.download_bar.pack_forget()
            self._refresh_model_status()
            self._update_controls()
        elif (kind == "stopped"
              and event[1] is self.worker):
            self.worker = None
            self.worker_settings = None
            self.disconnecting = False
            self.stats.configure(text="")
            if self.status_tone != "bad":
                self._set_status("Disconnected",
                                 "idle")
            self._log("Disconnected")
            self._reset_stage()
            self._show_decision("No target", "idle",
                                "Stopped")
            self._update_controls()

    def _poll_events(self):
        try:
            try:
                while True:
                    self._handle_event(
                        self.events.get_nowait())
            except queue.Empty:
                pass
            if self.worker is not None:
                frame = self.worker.take_frame()
                if frame is not None:
                    self._show_frame(*frame)
            self._update_stage_message()
        finally:
            self.after(25, self._poll_events)

    # ── frame rendering ──────────────────────────────────────────────

    def _show_frame(self, frame, preview, decision,
                    detection_count, fps, target_crop,
                    recognition_name=None,
                    show_familiar=False,
                    memory_data=None):
        self.has_frame = True
        self.last_frame_time = time.monotonic()
        frame_height, frame_width = frame.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_width / frame_width,
                    canvas_height / frame_height)
        width, height = (max(int(frame_width * scale), 1),
                         max(int(frame_height * scale), 1))
        if (width, height) != (frame_width, frame_height):
            frame = cv2.resize(
                frame, (width, height),
                interpolation=(cv2.INTER_AREA
                               if scale < 1
                               else cv2.INTER_LINEAR))
        image = Image.fromarray(
            cv2.cvtColor(frame,
                         cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        image_x = (canvas_width - width) // 2
        image_y = (canvas_height - height) // 2
        self.canvas.itemconfigure(
            self.video_item, image=self.photo,
            state="normal")
        self.canvas.coords(
            self.video_item, image_x, image_y)
        self.display_rect = (
            image_x, image_y, width, height)
        self.frame_size = (frame_width, frame_height)

        # Preview crop
        if target_crop is None:
            for item in (self.preview_item,
                         self.preview_border,
                         self.preview_label):
                self.canvas.itemconfigure(
                    item, state="hidden")
        else:
            target_image = Image.fromarray(
                cv2.cvtColor(target_crop,
                             cv2.COLOR_BGR2RGB))
            target_w, target_h = target_image.size
            target_scale = min(
                PREVIEW_WIDTH / target_w,
                120 / target_h, 1.0)
            target_size = (
                max(int(target_w * target_scale), 1),
                max(int(target_h * target_scale), 1))
            target_image = target_image.resize(
                target_size,
                Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(
                target_image)
            preview_w, preview_h = target_image.size
            self.canvas.itemconfigure(
                self.preview_item,
                image=self.preview_photo,
                state="normal")
            self.canvas.coords(
                self.preview_item,
                10, canvas_height - 12)
            self.canvas.coords(
                self.preview_border,
                8, canvas_height
                - preview_h - 28,
                preview_w + 12,
                canvas_height - 8)
            self.canvas.coords(
                self.preview_label,
                14, canvas_height
                - preview_h - 24)
            for item in (self.preview_border,
                         self.preview_label):
                self.canvas.itemconfigure(
                    item, state="normal")

        # Decision status
        label, tone = (
            STATE_TEXT.get(decision.label,
                           (decision.label.capitalize(),
                            "idle")))
        action = ACTION_TEXT[decision.command]
        if decision.speed:
            action += f", speed {decision.speed}"
        self._show_decision(label, tone, action)

        # Hint text
        if decision.label == "NO TARGET":
            hint = ("Click a face to follow it"
                    if detection_count
                    else "No faces in view")
        else:
            hint = None
        self._show_text(
            self.hint, hint,
            image_x + width / 2,
            image_y + height - 14)

        # Familiar face alert
        if show_familiar and recognition_name:
            alert_text = f"★ FAMILIAR: {recognition_name}"
            self._show_text(
                self.hint, alert_text,
                image_x + width / 2,
                image_y + 20)

        # Memory detail card update (heatmap + detail text)
        if memory_data:
            target_name = memory_data["name"]
            sim = memory_data["similarity"]
            sim_color = TONE["ok"] if sim >= 0.65 else TONE["warn"]
            self.target_name_label.configure(
                text=f"● {target_name}",
                fg=sim_color)
            # Update heatmap detail text
            score_list = memory_data.get("similarity_scores", [])
            if not score_list:
                score_list = [sim]
            min_sim = min(score_list)
            max_sim = max(score_list)
            self.heatmap_detail.configure(
                text=f"\nRange: {min_sim:.2f}–{max_sim:.2f}  "
                     f"\nMean: {sim:.2f}  "
                     f"\nSamples: {memory_data['samples']}  "
                     f"\nEnsemble: {memory_data['ensemble_size']}  "
                     f"\nC: {memory_data['confidence']:.2f}")
            # Draw heatmap from similarity scores computed by worker
            if score_list and memory_data['ensemble_size'] > 0:
                self._draw_heatmap(score_list)
            else:
                self.heatmap_canvas.delete("all")
                self.heatmap_canvas.create_text(
                    57, 43, text="no data",
                    fill=MUTED, font=("TkFixedFont", 8))
        else:
            self.target_name_label.configure(
                text="No target selected", fg=MUTED)
            self.heatmap_detail.configure(
                text="No target selected")
            self.heatmap_canvas.delete("all")
            self.heatmap_canvas.create_text(
                57, 43, text="no data",
                fill=MUTED, font=("TkFixedFont", 8))

        # Stats
        self.stats.configure(
            text=(f"{detection_count} "
                  f"face"
                  f"{'s' if detection_count != 1 else ''}  "
                  f"{fps:.0f} fps"))
        if self.status_tone not in {"bad", "warn"}:
            self._set_status("Live", "ok")

    # ── close ────────────────────────────────────────────────────────

    def _close(self):
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=2)
        self.destroy()


if __name__ == "__main__":
    EspApp().mainloop()
