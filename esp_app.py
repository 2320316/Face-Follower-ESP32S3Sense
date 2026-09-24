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
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
import numpy as np

from espstream import MjpegStream, RobotCommandClient
from espyolo import (
    DEFAULT_MODEL,
    FACE_MODEL_OPTIONS,
    FaceDetection,
    FaceDetector,
    FaceMemory,
    download_model,
    ensemble_similarity,
    find_model_path,
)

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
            detection_count = 0
            memory_data = None

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
                                      f"({rec_score:.2f})"))
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
                            best_embeddings = [
                                np.asarray(e, dtype=np.float32).tolist()
                                for e in stored.best_embeddings
                            ]
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
