"""Central mode orchestrator — manages mutually exclusive robot body modes.

Modes:
  IDLE              — robot neutral, no active control loops
  PERSON_TRACKING   — camera + YOLOv8-pose re-ID, head/body follows person
  GESTURE_CONTROL   — hand gesture recognition → xdotool desktop actions
  DANCE             — beat detection + 4-beat dance motion loop
"""

import enum
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Person tracking constants (from scripts/person_follow.py)
# ---------------------------------------------------------------------------
MOTION_HZ = 30
H_FOV_DEG = 90.0
V_FOV_DEG = 60.0
HEAD_YAW_MAX = 20.0
BODY_YAW_MAX = 45.0
PITCH_MAX = 15.0
KP = 0.6
SMOOTH_ALPHA = 0.4
BODY_SMOOTH_ALPHA = 0.2
CENTER_RETURN_RATE = 0.05

# ---------------------------------------------------------------------------
# Dance constants (from scripts/dance_to_music.py)
# ---------------------------------------------------------------------------
SWAY_MAX = 20.0
BOP_TILT = 15.0
ANTENNA_MAX = 45.0
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
COOLDOWN_SEC = 0.25
MIN_NOISE_FLOOR = 0.01
SENSITIVITY = 1.6
LISTEN_SECONDS = 5.0
SILENCE_TIMEOUT = 4.0


class Mode(enum.Enum):
    IDLE = "idle"
    PERSON_TRACKING = "person_tracking"
    GESTURE_CONTROL = "gesture_control"
    DANCE = "dance"


class ModeManager:
    def __init__(self, frame_provider, reachy):
        """
        Args:
            frame_provider: FrameProvider instance (shared camera).
            reachy: ReachyBridge instance.
        """
        self._frame_provider = frame_provider
        self._reachy = reachy
        self._current_mode = Mode.IDLE
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []

        # Lazy-loaded components
        self._person_tracker = None
        self._gesture_recognizer = None
        self._action_mapper = None

    @property
    def current_mode(self):
        return self._current_mode.value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_person_tracking(self):
        with self._lock:
            if self._current_mode == Mode.PERSON_TRACKING:
                return "Person tracking is already active."
            self._stop_current_mode()
            self._ensure_person_tracker()
            self._person_tracker.reset()
            self._reachy.pause_wobble()
            self._current_mode = Mode.PERSON_TRACKING
            self._start_threads([
                ("track", self._person_tracking_loop),
                ("track_ctrl", self._person_control_loop),
            ])
            logger.info("Mode → PERSON_TRACKING")
            return "Person tracking started. Robot will follow the nearest person."

    def stop_person_tracking(self):
        with self._lock:
            if self._current_mode != Mode.PERSON_TRACKING:
                return "Person tracking is not active."
            self._stop_current_mode()
            logger.info("Mode → IDLE")
            return "Person tracking stopped."

    def start_gesture_control(self):
        with self._lock:
            if self._current_mode == Mode.GESTURE_CONTROL:
                return "Gesture control is already active."
            self._stop_current_mode()
            self._ensure_gesture_recognizer()
            self._reachy.pause_wobble()
            self._current_mode = Mode.GESTURE_CONTROL
            self._start_threads([
                ("gesture", self._gesture_loop),
            ])
            logger.info("Mode → GESTURE_CONTROL")
            return "Gesture control started. Hand gestures now control the desktop."

    def stop_gesture_control(self):
        with self._lock:
            if self._current_mode != Mode.GESTURE_CONTROL:
                return "Gesture control is not active."
            self._stop_current_mode()
            logger.info("Mode → IDLE")
            return "Gesture control stopped."

    def start_dance(self):
        with self._lock:
            if self._current_mode == Mode.DANCE:
                return "Dance mode is already active."
            self._stop_current_mode()
            self._reachy.pause_wobble()
            self._current_mode = Mode.DANCE
            self._dance_state = _DanceState()
            self._start_threads([
                ("dance_motion", self._dance_motion_loop),
                ("dance_audio", self._dance_audio_loop),
            ])
            logger.info("Mode → DANCE")
            return "Dance mode started. Listening for music beats..."

    def stop_dance(self):
        with self._lock:
            if self._current_mode != Mode.DANCE:
                return "Dance mode is not active."
            self._stop_current_mode()
            logger.info("Mode → IDLE")
            return "Dance mode stopped."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_current_mode(self):
        """Stop all running threads and return robot to neutral."""
        if self._current_mode == Mode.IDLE:
            return
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self._stop_event.clear()
        self._current_mode = Mode.IDLE
        self._reachy.resume_wobble()
        self._return_to_neutral()

    def _start_threads(self, targets):
        for name, fn in targets:
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def _return_to_neutral(self):
        """Slowly return head/body/antennas to neutral."""
        if not self._reachy.connected or not self._reachy.robot:
            return
        try:
            from reachy_mini.utils import create_head_pose
            self._reachy.robot.goto_target(
                head=create_head_pose(yaw=0, pitch=0, roll=0), duration=0.5
            )
            self._reachy.robot.goto_target(body_yaw=0.0, duration=0.5)
            self._reachy.robot.goto_target(
                antennas=np.deg2rad([0, 0]), duration=0.5
            )
        except Exception as e:
            logger.error("Return to neutral error: %s", e)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_person_tracker(self):
        if self._person_tracker is None:
            from pipeline.person_tracker import PersonTracker
            self._person_tracker = PersonTracker()
            self._person_tracker.load_models()
            logger.info("PersonTracker models loaded")

    def _ensure_gesture_recognizer(self):
        if self._gesture_recognizer is None:
            from pipeline.gestures import GestureRecognizer
            from pipeline.actions import ActionMapper
            self._gesture_recognizer = GestureRecognizer()
            self._gesture_recognizer.load_model()
            self._action_mapper = ActionMapper()
            logger.info("GestureRecognizer + ActionMapper loaded")

    # ------------------------------------------------------------------
    # Person Tracking loops
    # ------------------------------------------------------------------

    def _person_tracking_loop(self):
        """Camera → PersonTracker → shared follow state."""
        logger.info("Person tracking loop started")
        self._follow_target_center = None
        self._follow_frame_size = (640, 480)
        self._follow_tracker_state = "scanning"
        last_frame_id = -1
        last_enroll_time = 0.0

        while not self._stop_event.is_set():
            frame, frame_id = self._frame_provider.get_frame()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            result = self._person_tracker.process_frame(frame)

            # Auto-enroll: if scanning and we see people, enroll closest to center
            # Cooldown prevents re-enrolling every frame if tracker loses match
            now = time.time()
            if result["state"] == "scanning" and result["all_persons"] and (now - last_enroll_time) > 3.0:
                h, w = frame.shape[:2]
                cx = w / 2
                best_dist = float("inf")
                best_idx = -1
                for i, p in enumerate(result["all_persons"]):
                    px = p["bbox"][0] + p["bbox"][2] / 2
                    dist = abs(px - cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0 and result["all_persons"][best_idx].get("embedding") is not None:
                    self._person_tracker._enroll_target(
                        result["all_persons"][best_idx]["embedding"]
                    )
                    last_enroll_time = now
                    logger.info("Auto-enrolled closest person for tracking")

            self._follow_target_center = result["target_center"]
            self._follow_frame_size = frame.shape[1], frame.shape[0]
            self._follow_tracker_state = result["state"]

        logger.info("Person tracking loop stopped")

    def _person_control_loop(self):
        """30Hz head+body movement loop (ported from person_follow.py)."""
        logger.info("Person control loop started")
        mini = self._reachy.robot
        if mini is None:
            logger.warning("No robot connected — control loop is a no-op")
            while not self._stop_event.is_set():
                time.sleep(0.1)
            return

        from reachy_mini.utils import create_head_pose

        curr_head_yaw = 0.0
        curr_body_yaw = 0.0
        curr_pitch = 0.0

        while not self._stop_event.is_set():
            target = self._follow_target_center
            fw, fh = self._follow_frame_size
            tracker_state = self._follow_tracker_state

            if tracker_state == "tracking" and target is not None:
                tx, ty = target
                dx = (tx - fw / 2) / (fw / 2)
                dy = (ty - fh / 2) / (fh / 2)

                total_yaw = -dx * (H_FOV_DEG / 2) * KP
                target_pitch = -dy * (V_FOV_DEG / 2) * KP

                target_body_yaw = curr_body_yaw + total_yaw * 0.3
                target_body_yaw = max(-BODY_YAW_MAX, min(BODY_YAW_MAX, target_body_yaw))

                target_head_yaw = total_yaw - (target_body_yaw - curr_body_yaw)
                target_head_yaw = max(-HEAD_YAW_MAX, min(HEAD_YAW_MAX, target_head_yaw))

                target_pitch = max(-PITCH_MAX, min(PITCH_MAX, target_pitch))
            else:
                target_head_yaw = 0.0
                target_body_yaw = 0.0
                target_pitch = 0.0

            curr_head_yaw += (target_head_yaw - curr_head_yaw) * SMOOTH_ALPHA
            curr_body_yaw += (target_body_yaw - curr_body_yaw) * BODY_SMOOTH_ALPHA
            curr_pitch += (target_pitch - curr_pitch) * SMOOTH_ALPHA

            if tracker_state != "tracking":
                curr_head_yaw += (0.0 - curr_head_yaw) * CENTER_RETURN_RATE
                curr_body_yaw += (0.0 - curr_body_yaw) * CENTER_RETURN_RATE
                curr_pitch += (0.0 - curr_pitch) * CENTER_RETURN_RATE

            try:
                mini.set_target(head=create_head_pose(yaw=curr_head_yaw, pitch=curr_pitch))
                mini.set_target(body_yaw=np.deg2rad(curr_body_yaw))
            except Exception:
                pass

            time.sleep(1.0 / MOTION_HZ)

        logger.info("Person control loop stopped")

    # ------------------------------------------------------------------
    # Gesture Control loop
    # ------------------------------------------------------------------

    def _gesture_loop(self):
        """Camera → GestureRecognizer → ActionMapper → xdotool."""
        logger.info("Gesture control loop started")
        last_frame_id = -1

        while not self._stop_event.is_set():
            frame, frame_id = self._frame_provider.get_frame()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            result = self._gesture_recognizer.process_frame(frame)
            gesture = result.get("gesture", "no_hand")
            hand_pos = result.get("hand_position")
            motion = result.get("motion")

            if gesture not in ("no_hand", "none"):
                action = self._action_mapper.execute_gesture(gesture, hand_pos, motion)
                if action:
                    logger.info("Gesture action: %s (gesture=%s)", action, gesture)

        logger.info("Gesture control loop stopped")

    # ------------------------------------------------------------------
    # Dance loops
    # ------------------------------------------------------------------

    def _dance_motion_loop(self):
        """4-beat dance sequence (ported from dance_to_music.py)."""
        logger.info("Dance motion loop started")
        mini = self._reachy.robot
        create_head_pose = None
        if mini is not None:
            from reachy_mini.utils import create_head_pose

        curr_body_yaw = 0.0
        curr_head_yaw = 0.0
        curr_roll = 0.0
        curr_antennas = 0.0

        while not self._stop_event.is_set():
            now = time.time()
            ds = self._dance_state
            mode = ds.mode
            period = max(ds.beat_period, 0.25)
            dance_start = ds.dance_start_time

            target_body_yaw = 0.0
            target_head_yaw = 0.0
            target_roll = 0.0
            target_antennas = 0.0

            if mode == "dancing":
                elapsed = now - dance_start
                beat_index = int(elapsed / period)
                phase = (elapsed % period) / period
                ease = phase * phase * (3 - 2 * phase)
                bop_envelope = np.sin(phase * np.pi)

                cycle = beat_index % 4
                if cycle == 0:
                    target_body_yaw = -SWAY_MAX + (SWAY_MAX * 2) * ease
                    target_head_yaw = -target_body_yaw
                elif cycle == 1:
                    target_body_yaw = SWAY_MAX
                    target_head_yaw = -SWAY_MAX
                    target_roll = BOP_TILT * bop_envelope
                    target_antennas = ANTENNA_MAX * bop_envelope
                elif cycle == 2:
                    target_body_yaw = SWAY_MAX - (SWAY_MAX * 2) * ease
                    target_head_yaw = -target_body_yaw
                elif cycle == 3:
                    target_body_yaw = -SWAY_MAX
                    target_head_yaw = SWAY_MAX
                    target_roll = -BOP_TILT * bop_envelope
                    target_antennas = ANTENNA_MAX * bop_envelope

            curr_body_yaw += (target_body_yaw - curr_body_yaw) * 0.4
            curr_head_yaw += (target_head_yaw - curr_head_yaw) * 0.4
            curr_roll += (target_roll - curr_roll) * 0.4
            curr_antennas += (target_antennas - curr_antennas) * 0.5

            if mini is not None and create_head_pose is not None:
                try:
                    mini.set_target(
                        head=create_head_pose(pitch=0.0, roll=curr_roll, yaw=curr_head_yaw)
                    )
                    mini.set_target(body_yaw=np.deg2rad(curr_body_yaw))
                    mini.set_target(antennas=np.deg2rad([curr_antennas, curr_antennas]))
                except Exception:
                    pass

            time.sleep(1.0 / MOTION_HZ)

        logger.info("Dance motion loop stopped")

    def _dance_audio_loop(self):
        """Beat detection via sounddevice (ported from dance_to_music.py)."""
        logger.info("Dance audio loop started")
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed — dance mode unavailable")
            return

        ds = self._dance_state
        ds.mode = "idle"

        history_len = 43  # ~1 sec at 44100/1024
        energy_history = np.zeros(history_len)
        history_idx = 0
        last_audio_beat = 0.0

        def audio_callback(indata, frames, time_info, status):
            nonlocal history_idx, last_audio_beat

            if self._stop_event.is_set():
                raise sd.CallbackStop()

            energy = float(np.sqrt(np.mean(indata ** 2)))
            now = time.time()

            energy_history[history_idx] = energy
            history_idx = (history_idx + 1) % history_len

            # Silence timeout → back to idle
            if ds.mode == "dancing" and (now - ds.last_beat_time) > SILENCE_TIMEOUT:
                logger.info("Music stopped — dance idle")
                ds.mode = "idle"
                ds.beat_timestamps = []

            if energy < MIN_NOISE_FLOOR:
                return

            avg_energy = float(np.mean(energy_history))
            threshold = avg_energy * SENSITIVITY

            if energy > threshold and (now - last_audio_beat) > COOLDOWN_SEC:
                last_audio_beat = now

                if ds.mode == "idle":
                    ds.mode = "listening"
                    ds.listen_start_time = now
                    ds.beat_timestamps = [now]
                    logger.info("Heard music — listening for tempo...")

                elif ds.mode == "listening":
                    ds.beat_timestamps.append(now)
                    if (now - ds.listen_start_time) >= LISTEN_SECONDS:
                        if len(ds.beat_timestamps) >= 4:
                            intervals = np.diff(ds.beat_timestamps)
                            valid = [i for i in intervals if 0.25 <= i <= 2.0]
                            if valid:
                                ds.beat_period = float(np.median(valid))
                                ds.mode = "dancing"
                                ds.dance_start_time = now
                                bpm = 60.0 / ds.beat_period
                                logger.info("Tempo locked: ~%.0f BPM", bpm)
                            else:
                                ds.listen_start_time = now
                                ds.beat_timestamps = [now]
                        else:
                            ds.listen_start_time = now
                            ds.beat_timestamps = [now]

                elif ds.mode == "dancing":
                    expected = round((now - ds.dance_start_time) / ds.beat_period)
                    ds.dance_start_time = now - (expected * ds.beat_period)

                ds.last_beat_time = now
                ds.beat_count += 1

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
                callback=audio_callback,
            ):
                logger.info("Dance audio stream open — waiting for music")
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            logger.error("Dance audio error: %s", e)

        logger.info("Dance audio loop stopped")


class _DanceState:
    """Thread-safe-ish dance state (fields written by audio callback, read by motion loop)."""
    def __init__(self):
        self.mode = "idle"
        self.listen_start_time = 0.0
        self.beat_timestamps = []
        self.beat_period = 0.5
        self.dance_start_time = 0.0
        self.last_beat_time = 0.0
        self.beat_count = 0
