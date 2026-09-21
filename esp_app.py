"""Desktop control app for the XIAO ESP32S3 face-following robot."""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from espstream import MjpegStream, RobotCommandClient
from espyolo import (
    DEFAULT_MODEL,
    FACE_MODEL_OPTIONS,
    FaceDetector,
    download_model,
    find_model_path,
)

# Light control panel; dark surfaces for the live video and the log.
PANEL, SURFACE, STAGE = "#eef1f4", "#ffffff", "#111820"
INK, MUTED, LINE = "#1c2630", "#5b6770", "#cfd6dc"
ACCENT, STOP, PREVIEW = "#0b6e99", "#c62828", "#42d9ff"
TONE = {"idle": "#7b8794", "ok": "#1a8f4b", "busy": ACCENT, "warn": "#b7791f", "bad": STOP}
LOG_COLORS = {"warn": "#f0b429", "bad": "#ff7b72"}

TARGET_LOST_FRAMES = 45  # frames without the target before the selection is released
NO_VIDEO_SECONDS = 1.5
PREVIEW_WIDTH = 220
LOG_LINES = 500

# espyolo labels and firmware commands, in plain language.
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


class VisionWorker(threading.Thread):
    def __init__(self, settings, events):
        super().__init__(daemon=True)
        self.settings = settings
        self.events = events
        self.stop_event = threading.Event()
        self.stop_robot_event = threading.Event()
        self.clear_event = threading.Event()
        self.selection_lock = threading.Lock()
        self.selection_point = None
        self.selected = None
        self.target_profile = None
        self.target_missed_frames = 0
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def select(self, point):
        with self.selection_lock:
            self.selection_point = point

    def stop(self):
        self.stop_event.set()

    def stop_robot(self):
        self.stop_robot_event.set()

    def clear_selection(self):
        # The worker thread owns `selected`; the UI only raises a flag.
        self.clear_event.set()

    def publish(self, event):
        self.events.put(event)

    def take_frame(self):
        """Latest rendered frame, or None. Older frames are dropped, never queued."""
        with self.frame_lock:
            frame, self.latest_frame = self.latest_frame, None
        return frame

    def _load_detector(self):
        detector = self.settings.get("detector")
        try:
            if detector is None:
                model_path = find_model_path(self.settings["model"])
                if not os.path.exists(model_path):
                    self.publish(("error", f"Model not found locally: {self.settings['model']}; load it before connecting"))
                    return None
                self.publish(("log", f"Loading {self.settings['model']}..."))
                detector = FaceDetector(model_path, confidence=self.settings["confidence"])
                self.publish(("detector_ready", detector, self.settings["model"]))
            self.publish(("log", f"YOLO ready: {detector.model_path}"))
            return detector
        except Exception as exc:
            self.publish(("error", f"YOLO could not load: {exc}"))
            return None

    def run(self):
        stream = None
        client = RobotCommandClient(self.settings["host"], self.settings["port"])
        try:
            detector = self._load_detector()
            if detector is None:
                return

            retrying = False
            link_ok = True
            fps = 0.0
            last_frame_time = None
            while not self.stop_event.is_set():
                if self.stop_robot_event.is_set():
                    sent, detail = client.send("p", 0, force=True)
                    self.publish(("log", "Stop command sent" if sent else f"Stop failed: {detail}"))
                    self.stop_robot_event.clear()
                if self.clear_event.is_set():
                    self.clear_event.clear()
                    self.selected = None
                    self.target_profile = None
                    self.target_missed_frames = 0
                    with self.selection_lock:
                        self.selection_point = None

                if stream is None or not stream.is_opened():
                    if not retrying:
                        self.publish(("status", "Connecting to XIAO stream...", "busy"))
                    stream = MjpegStream(self.settings["url"])
                    if not stream.is_opened():
                        if not retrying:
                            self.publish(("status", "Stream unavailable; retrying", "warn"))
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
                    self.publish(("status", "Stream lost; target memory retained", "bad"))
                    continue

                now = time.monotonic()
                if last_frame_time is not None:
                    instant = 1.0 / max(now - last_frame_time, 1e-3)
                    fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
                last_frame_time = now

                frame = cv2.flip(frame, -1) # type: ignore
                preview = cv2.cvtColor(
                    cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_WIDTH * frame.shape[0] // frame.shape[1]),
                               interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                detector.confidence = self.settings["confidence"]  # slider applies live
                detections = detector.detect(frame)
                with self.selection_lock:
                    point = self.selection_point
                    self.selection_point = None
                if point is not None:
                    chosen = detector.choose(detections, point)
                    if chosen is None:
                        self.publish(("log", "No face found at the selected point"))
                    else:
                        self.selected = chosen
                        self.target_profile = detector.profile(
                            chosen, frame, frame.shape[1], frame.shape[0]
                        )
                        self.target_missed_frames = 0
                        self.publish(("log", "Face selected"))
                active_selected = None
                if self.selected is not None and detections:
                    active_selected = detector.match_selected(
                        detections, self.selected, self.target_profile, frame
                    )
                    if active_selected is None:
                        wide_search = True
                        active_selected = detector.match_profile(
                            detections, self.target_profile, frame
                        )
                    else:
                        wide_search = False
                    if active_selected is not None:
                        self.selected = active_selected
                        self.target_profile = detector.profile(
                            active_selected, frame, frame.shape[1], frame.shape[0]
                        )
                        self.target_missed_frames = 0
                        if wide_search:
                            self.publish(("log", "Target reacquired by wide search"))

                if self.selected is not None and active_selected is None:
                    self.target_missed_frames += 1
                    if self.target_missed_frames == TARGET_LOST_FRAMES:
                        self.publish(("log", "Target lost; memory retained for reacquisition"))

                decision = detector.classify(active_selected, frame.shape[1], frame.shape[0])
                if self.stop_robot_event.is_set():
                    continue  # the top of the loop sends the stop before any new movement
                sent, detail = client.send(decision.command, decision.speed)
                if sent != link_ok:
                    link_ok = sent
                    if sent:
                        self.publish(("status", "Robot link restored", "ok"))
                    else:
                        self.publish(("warning", f"Robot command failed: {detail}"))

                target_crop = None
                if active_selected is not None:
                    padding = max(active_selected.width, active_selected.height)
                    x1 = max(active_selected.x - padding, 0)
                    y1 = max(active_selected.y - padding, 0)
                    x2 = min(active_selected.x + active_selected.width + padding, frame.shape[1])
                    y2 = min(active_selected.y + active_selected.height + padding, frame.shape[0])
                    target_crop = frame[y1:y2, x1:x2].copy()
                rendered = detector.draw(
                    frame, detections, self.selected, decision, self.target_profile
                )
                with self.frame_lock:
                    self.latest_frame = (rendered, preview, decision, len(detections), fps, target_crop)
        except Exception as exc:
            self.publish(("error", f"Tracking stopped: {exc}"))
        finally:
            if stream is not None:
                stream.release()
            client.stop()
            self.publish(("stopped", self))


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
        self.target_photo = None
        self.target_window = None
        self.target_canvas = None
        self.frame_size = (1, 1)
        self.display_rect = (0, 0, 1, 1)
        self.has_frame = False
        self.last_frame_time = 0.0
        self.detector = None
        self.detector_name = None
        self.loading = None  # name of the model being loaded, or None
        self.status_tone = "idle"
        self.overlay_state = {}
        self._build_ui()
        self._begin_load(DEFAULT_MODEL, download=False)
        self.after(25, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind_all("<Escape>", lambda _event: self._stop_robot())

    def _apply_theme(self):
        # "clam" honors custom colors on every OS; the native macOS theme ignores them.
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

        style.configure(".", background=PANEL, foreground=INK, bordercolor=LINE)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Section.TLabel", font=self.font_head)
        style.configure("TSeparator", background=LINE)
        style.configure("TButton", padding=(10, 6), background="#e2e7ec",
                        lightcolor="#e2e7ec", darkcolor="#e2e7ec")
        style.map("TButton",
                  background=[("disabled", PANEL), ("pressed", "#c9d1d8"), ("active", "#d5dce2")],
                  foreground=[("disabled", "#9aa5ae")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                        lightcolor=ACCENT, darkcolor=ACCENT, bordercolor=ACCENT)
        style.map("Accent.TButton",
                  background=[("disabled", "#9ab8c6"), ("pressed", "#084f6f"), ("active", "#095c82")],
                  foreground=[("disabled", "#ffffff")])
        style.configure("Stop.TButton", background=STOP, foreground="#ffffff", padding=(10, 12),
                        font=self.font_head, lightcolor=STOP, darkcolor=STOP, bordercolor=STOP)
        style.map("Stop.TButton", background=[("pressed", "#8f1d1d"), ("active", "#ad2222")])
        style.configure("TCombobox", fieldbackground=SURFACE, background="#e2e7ec",
                        arrowcolor=INK, padding=4)
        style.map("TCombobox",
                  fieldbackground=[("readonly", SURFACE), ("disabled", PANEL)],
                  foreground=[("disabled", "#9aa5ae")])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=LINE)
        style.configure("Horizontal.TProgressbar", troughcolor=LINE, background=ACCENT, borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", SURFACE)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _build_ui(self):
        self._apply_theme()
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=(16, 14, 12, 12), width=310)
        controls.grid(row=0, column=0, sticky="ns")
        controls.pack_propagate(False)
        viewer = ttk.Frame(self, padding=(0, 12, 12, 12))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.rowconfigure(0, weight=1)
        viewer.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Face control", font=self.font_title).pack(anchor="w")

        self.stream_url = "http://192.168.4.1/stream"
        self.robot_host = "192.168.4.1"
        self.robot_port = 8080
        self._section(controls, "Robot")
        info = ttk.Frame(controls)
        info.pack(fill="x")
        for row, (name, value) in enumerate((("Stream", self.stream_url),
                                             ("Commands", f"{self.robot_host}:{self.robot_port}"))):
            ttk.Label(info, text=name, style="Muted.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 12))
            ttk.Label(info, text=value).grid(row=row, column=1, sticky="w")
        self.connect_button = ttk.Button(controls, text="Connect", style="Accent.TButton",
                                         command=self._toggle_connection)
        self.connect_button.pack(fill="x", pady=(10, 8))

        card = tk.Frame(controls, bg=SURFACE, highlightbackground=LINE, highlightthickness=1,
                        padx=10, pady=8)
        card.pack(fill="x", pady=(0, 8))
        head = tk.Frame(card, bg=SURFACE)
        head.pack(fill="x")
        self.decision_dot = tk.Label(head, text="●", bg=SURFACE)
        self.decision_dot.pack(side="left")
        self.decision = tk.Label(head, bg=SURFACE, fg=INK, font=self.font_banner)
        self.decision.pack(side="left", padx=(6, 0))
        self.action = tk.Label(card, bg=SURFACE, fg=MUTED, anchor="w")
        self.action.pack(fill="x")
        self._show_decision("No target", "idle", "Stopped")

        ttk.Button(controls, text="Stop robot (Esc)", style="Stop.TButton",command=self._stop_robot).pack(fill="x")

        self._section(controls, "Model")
        self.model = ttk.Combobox(controls, values=FACE_MODEL_OPTIONS, state="readonly")
        self.model.set(DEFAULT_MODEL)
        self.model.pack(fill="x", pady=(0, 6))
        row, self.model_dot, self.model_status = self._status_line(controls, "Not loaded")
        row.pack(anchor="w", pady=(0, 6))
        self.model_buttons = ttk.Frame(controls)
        self.model_buttons.pack(fill="x")
        self.load_model_button = ttk.Button(self.model_buttons, text="Load", command=self._load_model)
        self.load_model_button.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.unload_model_button = ttk.Button(self.model_buttons, text="Unload", command=self._unload_model)
        self.unload_model_button.pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.download_bar = ttk.Progressbar(controls, maximum=1.0)  # shown only while downloading

        self._section(controls, "Tracking")
        confidence_row = ttk.Frame(controls)
        confidence_row.pack(fill="x")
        ttk.Label(confidence_row, text="Confidence").pack(side="left")
        self.confidence_label = ttk.Label(confidence_row, text="0.40")
        self.confidence_label.pack(side="right")
        self.confidence = ttk.Scale(controls, from_=0.05, to=0.95, value=0.4, command=self._on_confidence)
        self.confidence.pack(fill="x", pady=(2, 8))
        ttk.Button(controls, text="Open target view", command=self._open_target_view).pack(fill="x", pady=(6, 0))

        self._section(controls, "Log")
        self.console = tk.Text(controls, width=1, height=4, state="disabled", bg=STAGE, fg="#d8e2e8",
                               relief="flat", highlightthickness=0, padx=8, pady=6, wrap="word",
                               font="TkFixedFont")
        self.console.pack(fill="both", expand=True)
        for tone, color in LOG_COLORS.items():
            self.console.tag_configure(tone, foreground=color)

        self.canvas = tk.Canvas(viewer, bg=STAGE, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._select_face)
        self.canvas.bind("<Configure>", lambda _event: self._update_stage_message())
        self.video_item = self.canvas.create_image(0, 0, anchor="nw", state="hidden")
        self.preview_border = self.canvas.create_rectangle(0, 0, 0, 0, outline=PREVIEW, width=2, state="hidden")
        self.preview_item = self.canvas.create_image(0, 0, anchor="sw", state="hidden")
        self.preview_label = self.canvas.create_text(0, 0, anchor="sw", fill=PREVIEW, text="Target",
                                                     state="hidden")
        self.message = self._overlay("center", self.font_head, justify="center")
        self.hint = self._overlay("s", "TkDefaultFont")

        bar = ttk.Frame(viewer, padding=(2, 8, 2, 0))
        bar.grid(row=1, column=0, sticky="ew")
        row, self.status_dot, self.status = self._status_line(bar, "Not connected")
        row.pack(side="left")
        self.stats = ttk.Label(bar, style="Muted.TLabel")
        self.stats.pack(side="right")
        self._update_controls()

    def _section(self, parent, title):
        ttk.Separator(parent).pack(fill="x", pady=(12, 0))
        ttk.Label(parent, text=title, style="Section.TLabel").pack(anchor="w", pady=(8, 4))

    @staticmethod
    def _status_line(parent, text):
        row = ttk.Frame(parent)
        dot = ttk.Label(row, text="●", foreground=TONE["idle"])
        dot.pack(side="left")
        label = ttk.Label(row, text=text)
        label.pack(side="left", padx=(6, 0))
        return row, dot, label

    def _overlay(self, anchor, font, **options):
        background = self.canvas.create_rectangle(0, 0, 0, 0, fill=STAGE, outline="", state="hidden")
        label = self.canvas.create_text(0, 0, anchor=anchor, fill="#d8e2e8", font=font,
                                        state="hidden", **options)
        return background, label

    def _show_text(self, overlay, text, x, y):
        background, label = overlay
        key = (text, round(x), round(y))
        if self.overlay_state.get(label) == key:
            return
        self.overlay_state[label] = key
        if not text:
            self.canvas.itemconfigure(background, state="hidden")
            self.canvas.itemconfigure(label, state="hidden")
            return
        self.canvas.itemconfigure(label, text=text, state="normal")
        self.canvas.coords(label, x, y)
        x1, y1, x2, y2 = self.canvas.bbox(label)
        self.canvas.coords(background, x1 - 10, y1 - 6, x2 + 10, y2 + 6)
        self.canvas.itemconfigure(background, state="normal")
        self.canvas.tag_raise(background)
        self.canvas.tag_raise(label)

    def _update_stage_message(self):
        if self.worker is None:
            text = "Not connected\nJoin the robot's Wi-Fi, then press Connect."
        elif not self.has_frame or time.monotonic() - self.last_frame_time > NO_VIDEO_SECONDS:
            text = "Waiting for video..."
        else:
            text = None
        self._show_text(self.message, text,
                        self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def _reset_stage(self):
        self.has_frame = False
        self.canvas.itemconfigure(self.video_item, image="")  # detach before the PhotoImage is freed
        self.photo = None
        for item in (self.video_item, self.preview_border, self.preview_item, self.preview_label):
            self.canvas.itemconfigure(item, state="hidden")
        self._show_text(self.hint, None, 0, 0)
        self._show_target_crop(None)

    def _show_decision(self, label, tone, action):
        self.decision.configure(text=label)
        self.decision_dot.configure(fg=TONE[tone])
        self.action.configure(text=action)

    def _set_status(self, text, tone="idle"):
        self.status_tone = tone
        self.status.configure(text=text)
        self.status_dot.configure(foreground=TONE[tone])

    def _set_model_status(self, text, tone):
        self.model_status.configure(text=text)
        self.model_dot.configure(foreground=TONE[tone])

    def _refresh_model_status(self):
        if self.loading:
            self._set_model_status(f"Loading {self.loading}...", "busy")
        elif self.detector is not None:
            self._set_model_status(f"Ready: {self.detector_name}", "ok")
        else:
            self._set_model_status("Not loaded", "idle")

    def _update_controls(self):
        idle = self.worker is None and not self.loading
        self.model.configure(state="readonly" if idle else "disabled")
        self.load_model_button.configure(state="normal" if idle else "disabled")
        self.unload_model_button.configure(state="normal" if idle and self.detector else "disabled")
        if self.worker is None:
            self.connect_button.configure(text="Connect", style="Accent.TButton",
                                          state="disabled" if self.loading else "normal")
        elif self.disconnecting:
            self.connect_button.configure(text="Disconnecting...", state="disabled")
        else:
            self.connect_button.configure(text="Disconnect", style="TButton", state="normal")
        self.canvas.configure(cursor="crosshair" if self.worker else "")

    def _log(self, message, tone="info"):
        self.console.configure(state="normal")
        self.console.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n", tone)
        lines = int(self.console.index("end-1c").split(".")[0])
        if lines > LOG_LINES:
            self.console.delete("1.0", f"{lines - LOG_LINES + 1}.0")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _on_confidence(self, value):
        value = round(float(value), 2)
        self.confidence_label.configure(text=f"{value:.2f}")
        if self.worker_settings is not None:
            self.worker_settings["confidence"] = value

    def _load_model(self):
        self._begin_load(self.model.get(), download=True)

    def _begin_load(self, model_name, download):
        if self.loading or self.worker is not None:
            return
        self.loading = model_name
        self._refresh_model_status()
        self._update_controls()
        self._log(f"Loading {model_name}...")
        confidence = round(float(self.confidence.get()), 2)

        def load():
            try:
                if not os.path.exists(find_model_path(model_name)):
                    if not download:
                        self.events.put(("log", f"{model_name} not found locally; press Load to download it"))
                        return
                    self.events.put(("log", f"Downloading {model_name} from GitHub..."))
                    download_model(model_name, progress=lambda done: self.events.put(("progress", done)))
                detector = FaceDetector(model_name, confidence=confidence)
                self.events.put(("detector_ready", detector, model_name))
                self.events.put(("log", f"YOLO ready: {detector.model_path}"))
            except Exception as exc:
                self.events.put(("error", f"Model load failed: {exc}"))
            finally:
                self.events.put(("load_finished",))

        threading.Thread(target=load, daemon=True).start()

    def _unload_model(self):
        self.detector = None
        self.detector_name = None
        self._log("YOLO model unloaded")
        self._refresh_model_status()
        self._update_controls()

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
        if self.detector is None or self.detector_name != model_name:
            self._set_status("Load the selected model before connecting", "warn")
            self._log(f"Cannot connect: {model_name} is not loaded locally", "warn")
            return
        self.worker_settings = {
            "url": self.stream_url,
            "host": self.robot_host,
            "port": self.robot_port,
            "model": model_name,
            "confidence": round(float(self.confidence.get()), 2),
            "detector": self.detector if self.detector_name == model_name else None,
        }
        self.disconnecting = False
        self.has_frame = False
        self.worker = VisionWorker(self.worker_settings, self.events)
        self.worker.start()
        self._update_controls()
        self._log("Starting connection")

    def _select_face(self, event):
        if not self.worker or not self.worker.is_alive():
            return
        frame_width, frame_height = self.frame_size
        image_x, image_y, image_width, image_height = self.display_rect
        if not (image_x <= event.x <= image_x + image_width and
                image_y <= event.y <= image_y + image_height):
            return
        point = (int((event.x - image_x) * frame_width / image_width),
                 int((event.y - image_y) * frame_height / image_height))
        self.worker.select(point)

    def _clear_selection(self):
        if self.worker:
            self.worker.clear_selection()
        self._log("Selection cleared")

    def _open_target_view(self):
        if self.target_window is not None and self.target_window.winfo_exists():
            self.target_window.lift()
            return
        self.target_window = tk.Toplevel(self)
        self.target_window.title("Target view")
        self.target_window.geometry("420x420")
        self.target_window.configure(bg=STAGE)
        self.target_canvas = tk.Canvas(self.target_window, bg=STAGE, highlightthickness=0)
        self.target_canvas.pack(fill="both", expand=True)
        self.target_window.protocol("WM_DELETE_WINDOW", self._close_target_view)

    def _close_target_view(self):
        if self.target_window is not None:
            self.target_window.destroy()
        self.target_window = None
        self.target_canvas = None
        self.target_photo = None

    def _stop_robot(self):
        if self.worker:
            self.worker.stop_robot()
            self.worker.clear_selection()
        # Sent directly as well, so the stop does not wait on the vision loop.
        threading.Thread(target=self._send_stop, args=(self.worker is None,), daemon=True).start()
        self._log("Stop requested")
        self._log("Selection cleared")


    def _send_stop(self, log_success):
        client = RobotCommandClient(self.robot_host, self.robot_port, timeout=1.0)
        sent, detail = client.send("p", 0, force=True)
        if not sent:
            self.events.put(("warning", f"Stop failed: {detail}"))
        elif log_success:
            self.events.put(("log", "Stop command sent"))

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
            self.detector, self.detector_name = event[1], event[2]
            self._refresh_model_status()
            self._update_controls()
        elif kind == "progress":
            if not self.download_bar.winfo_ismapped():
                self.download_bar.pack(fill="x", pady=(6, 0), after=self.model_buttons)
            self.download_bar.configure(value=event[1])
            self._set_model_status(f"Downloading {self.loading}... {event[1]:.0%}", "busy")
        elif kind == "load_finished":
            self.loading = None
            self.download_bar.pack_forget()
            self._refresh_model_status()
            self._update_controls()
        elif kind == "stopped" and event[1] is self.worker:
            self.worker = None
            self.worker_settings = None
            self.disconnecting = False
            self.stats.configure(text="")
            if self.status_tone != "bad":  # keep an error visible; it explains the disconnect
                self._set_status("Disconnected", "idle")
            self._log("Disconnected")
            self._reset_stage()
            self._show_decision("No target", "idle", "Stopped")
            self._update_controls()

    def _poll_events(self):
        try:
            try:
                while True:
                    self._handle_event(self.events.get_nowait())
            except queue.Empty:
                pass
            if self.worker is not None:
                frame = self.worker.take_frame()
                if frame is not None:
                    self._show_frame(*frame)
            self._update_stage_message()
        finally:
            self.after(25, self._poll_events)  # one failing handler must not stop the UI

    def _show_frame(self, frame, preview, decision, detection_count, fps, target_crop):
        self.has_frame = True
        self.last_frame_time = time.monotonic()
        frame_height, frame_width = frame.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        width, height = max(int(frame_width * scale), 1), max(int(frame_height * scale), 1)
        if (width, height) != (frame_width, frame_height):
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        image_x = (canvas_width - width) // 2
        image_y = (canvas_height - height) // 2
        self.canvas.itemconfigure(self.video_item, image=self.photo, state="normal")
        self.canvas.coords(self.video_item, image_x, image_y)
        self.display_rect = (image_x, image_y, width, height)
        self.frame_size = (frame_width, frame_height)

        if target_crop is None:
            self.canvas.itemconfigure(self.preview_item, state="hidden")
            self.canvas.itemconfigure(self.preview_border, state="hidden")
            self.canvas.itemconfigure(self.preview_label, state="hidden")
        else:
            target_image = Image.fromarray(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB))
            target_width, target_height = target_image.size
            target_scale = min(PREVIEW_WIDTH / target_width, 120 / target_height, 1.0)
            target_size = (max(int(target_width * target_scale), 1),
                           max(int(target_height * target_scale), 1))
            target_image = target_image.resize(target_size, Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(target_image)
            preview_width, preview_height = target_image.size
            self.canvas.itemconfigure(self.preview_item, image=self.preview_photo, state="normal")
            self.canvas.coords(self.preview_item, 10, canvas_height - 12)
            self.canvas.coords(self.preview_border, 8, canvas_height - preview_height - 28,
                               preview_width + 12, canvas_height - 8)
            self.canvas.coords(self.preview_label, 14, canvas_height - preview_height - 24)
            self.canvas.itemconfigure(self.preview_border, state="normal")
            self.canvas.itemconfigure(self.preview_label, state="normal")

        label, tone = STATE_TEXT.get(decision.label, (decision.label.capitalize(), "idle"))
        action = ACTION_TEXT[decision.command]
        if decision.speed:
            action += f", speed {decision.speed}"
        self._show_decision(label, tone, action)
        if decision.label == "NO TARGET":
            hint = "Click a face to follow it" if detection_count else "No faces in view"
        else:
            hint = None
        self._show_text(self.hint, hint, image_x + width / 2, image_y + height - 14)
        self.stats.configure(text=f"{detection_count} face{'s' if detection_count != 1 else ''}   {fps:.0f} fps")
        self._show_target_crop(target_crop)
        if self.status_tone != "bad" and self.status_tone != "warn":
            self._set_status("Live", "ok")

    def _show_target_crop(self, target_crop):
        if (self.target_canvas is None or self.target_window is None or
                not self.target_window.winfo_exists()):
            return
        if target_crop is None:
            self.target_canvas.delete("all")
            self.target_photo = None
            self.target_canvas.create_text(
                self.target_canvas.winfo_width() / 2,
                self.target_canvas.winfo_height() / 2,
                text="Target not visible", fill="#d8e2e8"
            )
            return
        canvas_width = max(self.target_canvas.winfo_width(), 1)
        canvas_height = max(self.target_canvas.winfo_height(), 1)
        crop_height, crop_width = target_crop.shape[:2]
        scale = min(canvas_width / crop_width, canvas_height / crop_height)
        width = max(int(crop_width * scale), 1)
        height = max(int(crop_height * scale), 1)
        image = cv2.resize(target_crop, (width, height), interpolation=cv2.INTER_AREA)
        self.target_photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        )
        self.target_canvas.delete("all")
        self.target_canvas.create_image(
            (canvas_width - width) // 2, (canvas_height - height) // 2,
            anchor="nw", image=self.target_photo
        )

    def _close(self):
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=2)
        self._close_target_view()
        self.destroy()


if __name__ == "__main__":
    EspApp().mainloop()