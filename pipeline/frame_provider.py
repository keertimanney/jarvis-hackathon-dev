"""Shared camera capture thread — single owner of /dev/video0.

All modes (person tracking, gesture control, etc.) call get_frame()
instead of opening their own VideoCapture.
"""

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_WIDTH = 640


class FrameProvider:
    def __init__(self, camera_index=0, reachy_mini=None):
        """
        Args:
            camera_index: /dev/video<N> index (ignored if reachy_mini is set).
            reachy_mini: Optional ReachyMini instance — uses SDK camera instead of cv2.
        """
        self._camera_index = camera_index
        self._mini = reachy_mini
        self._cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="frame_provider")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self):
        """Return (frame_copy, frame_id) or (None, 0) if no frame yet."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0
            return self._latest_frame.copy(), self._frame_id

    def _grab_loop(self):
        use_sdk = self._mini is not None

        if use_sdk:
            logger.info("FrameProvider using Reachy SDK camera")
            for attempt in range(40):
                frame = self._mini.media.get_frame()
                if frame is not None:
                    logger.info("Reachy SDK camera ready (attempt %d)", attempt + 1)
                    break
                time.sleep(0.25)
            else:
                logger.error("Reachy SDK camera not available")
                self._running = False
                return
        else:
            dev_path = f"/dev/video{self._camera_index}"
            self._cap = cv2.VideoCapture(dev_path, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                logger.error("Cannot open camera %d", self._camera_index)
                self._running = False
                return
            logger.info("Camera %d opened", self._camera_index)

        while self._running:
            if use_sdk:
                frame = self._mini.media.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            # ZED stereo: take left half
            if frame.shape[1] > 2000:
                frame = frame[:, :frame.shape[1] // 2]

            # Downscale to TARGET_WIDTH for inference
            h, w = frame.shape[:2]
            if w > TARGET_WIDTH:
                scale = TARGET_WIDTH / w
                frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))

            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

            time.sleep(0.01)  # ~100fps cap, consumers run at their own rate

        logger.info("FrameProvider stopped")
