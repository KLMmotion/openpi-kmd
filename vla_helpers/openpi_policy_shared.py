"""
Shared helpers for KMD real-robot policy clients.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

try:
    from openpi_client import websocket_client_policy  # noqa: F401

    POLICY_CLIENT_AVAILABLE = True
except ImportError:
    POLICY_CLIENT_AVAILABLE = False

IMG_SIZE = 256
DEFAULT_TASK_INSTRUCTION = "Pick up the target object and place it at the target position"
_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


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


def decode_quad_image(quad_image: dict[str, Any] | None) -> bytes | None:
    if not quad_image or "data" not in quad_image:
        return None
    return base64.b64decode(quad_image["data"])


def split_quad_bgr_3cam(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 2x2 quad-tile image into the 3 cameras used by KMD checkpoints."""
    h, w = img_bgr.shape[:2]
    sub_h, sub_w = h // 2, w // 2
    return {
        "cam_high": img_bgr[sub_h:h, sub_w:w].copy(),
        "cam_left_wrist": img_bgr[0:sub_h, sub_w:w].copy(),
        "cam_right_wrist": img_bgr[sub_h:h, 0:sub_w].copy(),
    }


def quad_image_dict_to_cameras(quad_image: dict[str, Any] | None) -> dict[str, np.ndarray | None]:
    image_bytes = decode_quad_image(quad_image)
    if image_bytes is None:
        return {k: None for k in _CAMERA_KEYS}
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {k: None for k in _CAMERA_KEYS}
    return split_quad_bgr_3cam(img)


class VlaHostClient:
    """HTTP client for the vlahost server (GET /state, POST /action)."""

    def __init__(self, server_url: str, timeout_sec: float = 0.5):
        base = server_url.rstrip("/")
        self.state_url = f"{base}/state"
        self.action_url = f"{base}/action"
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._latest_state: dict[str, Any] | None = None

    def fetch_state(self) -> dict[str, Any]:
        resp = requests.get(self.state_url, timeout=self.timeout_sec)
        resp.raise_for_status()
        state = resp.json()
        with self._lock:
            self._latest_state = state
        return state

    def get_latest_joint_state(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_state is None:
                return None
            return self._extract_joint_state(self._latest_state)

    def post_joint_action(self, joint_actions: np.ndarray | list[float]) -> dict[str, Any]:
        arr = np.asarray(joint_actions, dtype=np.float64).flatten()
        payload = {"joint_actions": [float(x) for x in arr]}
        resp = requests.post(self.action_url, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_joint_state(state: dict[str, Any]) -> dict[str, Any] | None:
        candidate = state.get("joint_states")
        if not isinstance(candidate, dict):
            return None
        raw_pos = candidate.get("positions", candidate.get("position", []))
        if not isinstance(raw_pos, list) or len(raw_pos) == 0:
            return None
        result: dict[str, Any] = {"position": [float(x) for x in raw_pos]}

        def _opt_float(d: dict, key: str):
            if key not in d or d[key] is None:
                return None
            try:
                return float(d[key])
            except (TypeError, ValueError):
                return None

        for gkey in ("gripper_left", "gripper_right"):
            v = _opt_float(candidate, gkey)
            if v is not None:
                result[gkey] = v
        return result
