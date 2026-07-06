"""
Shared helpers for KMD real-robot policy clients.
"""

from __future__ import annotations

import asyncio
import collections
import json
import threading
import time

import cv2
import numpy as np

try:
    from openpi_client import websocket_client_policy  # noqa: F401

    POLICY_CLIENT_AVAILABLE = True
except ImportError:
    POLICY_CLIENT_AVAILABLE = False

IMG_SIZE = 256
DEFAULT_TASK_INSTRUCTION = "Pick up the target object and place it at the target position"


def ensure_hwc_uint8(img):
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        from PIL import Image

        img = np.array(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return img.astype(np.uint8)


def crop_by_ratio(img, crop_left, crop_right, crop_top, crop_bottom):
    img = np.asarray(img)
    if img.ndim < 2:
        return img
    h, w = img.shape[:2]
    x0 = int(round(w * float(crop_left)))
    x1 = int(round(w * (1.0 - float(crop_right))))
    y0 = int(round(h * float(crop_top)))
    y1 = int(round(h * (1.0 - float(crop_bottom))))
    x0 = max(0, min(x0, w - 1))
    x1 = max(1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


class WSJointStateClient:
    def __init__(self, uri: str, reconnect_sec: float = 1.0, send_rate_hz: float = 10.0):
        self.uri = uri
        self.reconnect_sec = reconnect_sec
        self.send_rate_hz = max(0.1, float(send_rate_hz))
        self._latest_joint_state = None
        self._pending_actions = collections.deque()
        self._last_send_ts = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_latest_joint_state(self):
        with self._lock:
            return self._latest_joint_state

    def update_joint_action(self, joint_actions):
        arr = np.asarray(joint_actions, dtype=np.float64).flatten()
        payload = {
            "type": "action",
            "stamp_ns": time.time_ns(),
            "joint_actions": [float(x) for x in arr],
        }
        with self._lock:
            self._pending_actions.append(payload)

    def _run(self):
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            print(f"❌ [ws] Error: {exc}")

    async def _run_async(self):
        try:
            import websockets
        except Exception as exc:
            print(f"⚠️ [ws] websockets import failed: {exc}")
            return

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.uri) as ws:
                    print(f"✅ [ws] Connected to {self.uri}")
                    while not self._stop_event.is_set():
                        now = time.time()
                        if (now - self._last_send_ts) >= (1.0 / self.send_rate_hz):
                            with self._lock:
                                action_payload = self._pending_actions.popleft() if self._pending_actions else None
                            if action_payload is not None:
                                await ws.send(json.dumps(action_payload))
                                self._last_send_ts = now

                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.02)
                        except asyncio.TimeoutError:
                            continue

                        payload = self._try_parse_json(msg)
                        if payload is None:
                            continue
                        js = self._extract_joint_state(payload)
                        if js is not None:
                            with self._lock:
                                self._latest_joint_state = js
            except Exception as exc:
                if not self._stop_event.is_set():
                    print(f"⚠️ [ws] Reconnecting: {exc}")
                    await asyncio.sleep(self.reconnect_sec)

    @staticmethod
    def _try_parse_json(msg):
        try:
            return json.loads(msg)
        except Exception:
            return None

    @staticmethod
    def _extract_joint_state(payload):
        if not isinstance(payload, dict):
            return None
        candidate = payload.get("joint_states", payload)
        if not isinstance(candidate, dict):
            return None
        raw_pos = candidate.get("position", candidate.get("positions", []))
        if not isinstance(raw_pos, list) or len(raw_pos) == 0:
            return None
        result = {"position": [float(x) for x in raw_pos]}

        def _opt_float(d: dict, key: str):
            if key not in d or d[key] is None:
                return None
            try:
                return float(d[key])
            except (TypeError, ValueError):
                return None

        for gkey in ("gripper_left", "gripper_right"):
            v = _opt_float(candidate, gkey)
            if v is None and candidate is not payload:
                v = _opt_float(payload, gkey)
            if v is not None:
                result[gkey] = v
        return result
