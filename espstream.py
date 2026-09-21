"""XIAO ESP32S3 stream and robot command transport."""

import socket
import urllib.error
import urllib.request

import cv2
import numpy as np


class MjpegStream:
    def __init__(self, url, connect_timeout=1.0, read_timeout=0.5):
        self.response = None
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
        if self.response is None:
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
        except (OSError, TimeoutError, urllib.error.URLError):
            self.release()
            return False, None

    def release(self):
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

    def send(self, command, speed=100, force=False):
        if command not in self.VALID_COMMANDS:
            raise ValueError(f"Unsupported robot command: {command}")

        speed = int(speed)
        if not force and command == self.last_command and speed == self.last_speed:
            return True, "duplicate suppressed"

        payload = f"{command}, {speed}\r\n".encode("utf-8")
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