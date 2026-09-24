"""YOLO face detection, memory-based recognition, and movement classification.

This module provides:
- YOLO-based face detection
- FaceDetection: lightweight per-session target representation
- FaceMemory: persistent, learnable memory with multi-sample ensemble
- FaceDetector with embedding-based matching, adaptive thresholds, and learning
- Movement classification into LEFT/RIGHT/CLOSE/FAR/GOOD/NO TARGET
"""

import os
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
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
        search_radius = max(search_radius, 30)

        near = []
        for face in detections:
            cx, cy = face.center
            dist = math.sqrt((cx - selected.center[0]) ** 2 + (cy - selected.center[1]) ** 2)
            if dist <= search_radius:
                near.append((dist, face))

        if near:
            near.sort(key=lambda x: x[0])
            return near[0][1]

        # Step 3: embedding-based fallback
        if detection is not None and frame is not None:
            return match_profile_face(detections, detection, frame)

        return None

    def match_stored_faces(self, detections: list[FaceBox], frame: np.ndarray
                           ) -> tuple[Optional[str], Optional[FaceBox], float]:
        """Check if any detected face matches a stored (learned) memory.

        Returns (name, face, cosine_similarity) or (None, None, 0.0).
        """
        best_name = None
        best_face = None
        best_sim = 0.0

        for name, memory in self.stored_faces.items():
            face, sim = match_stored_face(detections, memory, frame)
            if face is not None and sim > best_sim:
                best_name = name
                best_face = face
                best_sim = sim

        if best_name is not None:
            return best_name, best_face, best_sim
        return None, None, 0.0

    def add_face_memory(self, name: str, face: FaceBox, frame: np.ndarray,
                        frame_w: int, frame_h: int) -> FaceMemory:
        """Store a new memory entry."""
        memory = _create_face_memory(name, face, frame, frame_w, frame_h)
        self.stored_faces[name] = memory
        return memory

    def update_stored_face(self, name: str, face: FaceBox, frame: np.ndarray,
                           frame_w: int, frame_h: int):
        """Update a stored memory from a confirmed detection (alias for API compat)."""
        _update_face_memory(self.stored_faces[name], face, frame, frame_w, frame_h)

    def learn_from_target(self, detection: FaceDetection, face: FaceBox,
                          frame: np.ndarray, frame_w: int, frame_h: int):
        """Incrementally learn from a confirmed tracked face (alias for API compat)."""
        self.learn_from_detection(detection, face, frame, frame_w, frame_h)

    def select_face(self, detections: list[FaceBox], point: tuple[int, int],
                    frame: np.ndarray, frame_w: int, frame_h: int,
                    prompt_name: bool = True
                    ) -> tuple[Optional[FaceBox], Optional[FaceDetection], str, Optional[np.ndarray]]:
        """Select a face by click point.

        Returns (selected_face, detection, log_message, face_embedding).
        The embedding is pre-computed so callers can pass it to
        :meth:`_save_target` to avoid redundant ArcFace extraction.

        If prompt_name is True and the face is not already in memory, creates a new
        memory entry after the user confirms a name via dialog.
        """
        chosen = self.choose(detections, point)
        if chosen is None:
            return None, None, "No face found at the selected point", None
        detection = _create_face_detection(chosen, frame, frame_w, frame_h)

        # Check if this face matches any stored memory (auto-identify)
        face_emb = _create_face_embedding(frame, chosen)
        for stored_name, mem in self.stored_faces.items():
            if face_emb is not None and mem.cosine_similarity(face_emb) >= _memory_threshold(mem):
                return chosen, detection, f"Already known: {stored_name} (similarity {mem.cosine_similarity(face_emb):.2f})", face_emb

        return chosen, detection, "Face selected (not saved to memory)", face_emb

    def learn_from_detection(self, face_detection: FaceDetection, face: FaceBox,
                             frame: np.ndarray, frame_w: int, frame_h: int):
        """Incrementally learn from a confirmed tracked face."""
        _learn_from_detection(face_detection, face, frame, frame_w, frame_h,
                              learning_rate=DEFAULT_PROFILE_LEARNING_RATE)

    def cosine_similarity(self, detection: FaceDetection, face: FaceBox,
                          frame: np.ndarray) -> float:
        """Get the cosine similarity between detection and a face."""
        return detection_cosine_similarity(face, detection, frame)

    def classify(self, face: Optional[FaceBox], frame_w: int, frame_h: int) -> FaceDecision:
        return classify(face, frame_w, frame_h,
                        self.align_margin, self.distance_margin)

    def turn_speed(self, offset: float, frame_width: int) -> int:
        return turn_speed(offset, frame_width, frame_width * self.align_margin)

    def draw(self, frame: np.ndarray, detections, selected, decision,
             detection: Optional[FaceDetection] = None,
             face_memory: Optional[FaceMemory] = None,
             active_recognition_score: float = 0.0) -> np.ndarray:
        return draw(frame, detections, selected, decision, detection,
                    face_memory, active_recognition_score)

    def get_face_memory(self, name: str) -> Optional[FaceMemory]:
        """Retrieve a stored memory by name."""
        return self.stored_faces.get(name)

    def memory_embedding_dim(self) -> int:
        """Return the dimension of the stored embeddings."""
        if _FACE_EMBEDDER_READY and _face_embedder is not None:
            # Try to extract to get actual dimension
            test_crop = np.zeros((100, 100, 3), dtype=np.uint8)
            test_crop[20:80, 20:80] = [128, 128, 128]
            try:
                faces = _face_embedder.get(test_crop)
                if faces:
                    return len(faces[0].embedding)
            except Exception:
                pass
        return 512  # default for ArcFace

    def _drop_target_embeddings(self):
        """Clear all stored embeddings for the 'target' face memory."""
        target = self.stored_faces.get("target")
        if target is not None:
            target.best_embeddings.clear()
            target.mean_embedding = np.zeros(512, dtype=np.float32)
            target.sample_count = 0
            target.confidence = 0.0
            _save_memory_file("target", target)

    def memory_is_arcface(self) -> bool:
        """Return True if ArcFace embeddings are available."""
        return _FACE_EMBEDDER_READY

    def _save_target(self, face: FaceBox, frame: np.ndarray,
                         frame_w: int, frame_h: int,
                         detection: Optional[FaceDetection] = None,
                         embedding: Optional[np.ndarray] = None
                         ) -> Optional[FaceMemory]:
        """Save the selected face as the persistent target memory.

        If a ``target`` memory already exists, updates it via EMA learning.
        Otherwise creates a new entry.  Accepts a pre-computed embedding to
        avoid re-running the expensive ArcFace extraction.

        Returns the created :class:`FaceMemory` or ``None``.
        """
        name = "target"

        # --- drop old embeddings before saving new target ---
        self._drop_target_embeddings()

        # --- update existing target memory (Fix #1) ---
        if "target" in self.stored_faces:
            existing = self.stored_faces[name]
            face_emb = embedding if embedding is not None else _create_face_embedding(frame, face)
            if face_emb is not None:
                # Blend into existing memory so tracking improves over time
                existing.add_sample_embedding(face_emb)
                wr = face.width / frame_w
                hr = face.height / frame_h
                ar = face.width / max(face.height, 1)
                existing.mean_width_ratio = (existing.mean_width_ratio * (1 - 0.05)
                                             + wr * 0.05)
                existing.mean_height_ratio = (existing.mean_height_ratio * (1 - 0.05)
                                              + hr * 0.05)
                existing.mean_aspect_ratio = (existing.mean_aspect_ratio * (1 - 0.05)
                                              + ar * 0.05)
                existing.confidence = min(1.0, existing.confidence + 0.01)
                _save_memory_file(name, existing)
                return self.stored_faces[name]
            return None

        # --- new face — create memory ---
        if embedding is None:
            embedding = _create_face_embedding(frame, face)
        if embedding is None:
            return None
        wr = face.width / frame_w
        hr = face.height / frame_h
        ar = face.width / max(face.height, 1)
        now = datetime.now().isoformat()
        ts = int(now[:10].replace("-", ""))
        mem = FaceMemory(
            name=name,
            mean_embedding=embedding.copy(),
            mean_width_ratio=wr,
            mean_height_ratio=hr,
            mean_aspect_ratio=ar,
            sample_count=1,
            confidence=face.confidence,
            created=now,
            created_ts=ts,
            total_tracking_time_frames=0,
        )
        # Store the embedding in best_embeddings so heatmap has data
        mem.best_embeddings = [embedding.copy()]
        _save_memory_file(name, mem)
        self.stored_faces[name] = mem
        return mem


# ── module-level ensemble helper ────────────────────────────────────────────────

def ensemble_similarity(memory: FaceMemory, query=None, top_k: int = 3) -> float:
    """Compute ensemble similarity for a memory (module-level wrapper).

    Uses the stored mean embedding as the query. The ``query`` argument
    is deprecated and ignored — kept for API compatibility with callers
    that pass a float.

    Used by ``esp_app.py`` to display the current similarity metric.
    """
    return memory.ensemble_similarity(memory.mean_embedding, top_k=top_k)
