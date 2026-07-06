#!/usr/bin/env python3
"""vlahost client.

Runs on the machine hosting the VLA model (e.g. pi0). This module must have
NO ROS / rclpy dependency so it can run standalone on a machine that does not
have ROS installed - only `pip install -r client_requirements.txt` is needed.

Loops at --rate-hz: GETs the robot's latest state + quadcam image from the
vlahost server (running on the robot host), runs inference, and POSTs the
resulting action back.

Usage:
    python3 client.py --server-url http://<robot-host>:8000
    python3 client.py --show-images   # also pop up the 4 quadcam views with OpenCV
"""
import argparse
import base64
import time
from typing import Any, Dict, List, Optional

import requests

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Quad-tile composite layout (see quad_csi_webrtc.cpp): 2x2 grid of
# [top-left, top-right, bottom-left, bottom-right] = [cam1, cam2, cam3, cam4].
_QUAD_WINDOW_NAMES = ("cam1 (top-left)", "cam2 (top-right)", "cam3 (bottom-left)", "cam4 (bottom-right)")


def decode_quad_image(quad_image: Optional[Dict[str, Any]]) -> Optional[bytes]:
    if not quad_image or "data" not in quad_image:
        return None
    return base64.b64decode(quad_image["data"])


def split_quad_image(image_bytes: bytes) -> Optional[List["np.ndarray"]]:
    """Decode the quad-tile jpeg and split it into its 4 source camera frames."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    half_h, half_w = h // 2, w // 2
    return [
        frame[0:half_h, 0:half_w],
        frame[0:half_h, half_w:w],
        frame[half_h:h, 0:half_w],
        frame[half_h:h, half_w:w],
    ]


def show_quad_images(image_bytes: Optional[bytes]) -> None:
    if image_bytes is None or cv2 is None:
        return
    quadrants = split_quad_image(image_bytes)
    if quadrants is None:
        return
    for name, quadrant in zip(_QUAD_WINDOW_NAMES, quadrants):
        cv2.imshow(name, quadrant)
    cv2.waitKey(1)


def run_inference(state: Dict[str, Any]) -> Dict[str, Any]:
    """Plug the actual pi0 / VLA model call in here.

    `state` carries joint_states, eef_left/eef_right poses and the raw
    quadcam jpeg (via decode_quad_image(state.get("quad_image"))). Returning
    all-None fields is a safe no-op action for the server.
    """
    return {
        "eef_left": None,
        "eef_right": None,
        "gripper_left": None,
        "gripper_right": None,
    }


def run_loop(server_url: str, rate_hz: float, timeout_sec: float, show_images: bool) -> None:
    period = 1.0 / rate_hz if rate_hz > 0.0 else 0.1
    state_url = f"{server_url}/state"
    action_url = f"{server_url}/action"

    while True:
        start = time.monotonic()
        try:
            resp = requests.get(state_url, timeout=timeout_sec)
            resp.raise_for_status()
            state = resp.json()
        except requests.RequestException as exc:
            print(f"vlahost: failed to fetch state from {state_url}: {exc}")
            time.sleep(period)
            continue

        if show_images:
            show_quad_images(decode_quad_image(state.get("quad_image")))

        action = run_inference(state)

        try:
            resp = requests.post(action_url, json=action, timeout=timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"vlahost: failed to post action to {action_url}: {exc}")

        elapsed = time.monotonic() - start
        time.sleep(max(0.0, period - elapsed))


def main(argv=None):
    parser = argparse.ArgumentParser(description="vlahost client")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000",
        help="Base URL of the vlahost server running on the robot",
    )
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--timeout-sec", type=float, default=0.5)
    parser.add_argument(
        "--show-images", action="store_true",
        help="Pop up OpenCV windows showing the 4 quadcam views from each polled state",
    )
    args = parser.parse_args(argv)

    if args.show_images and cv2 is None:
        parser.error("--show-images requires opencv-python and numpy to be installed")

    try:
        run_loop(args.server_url, args.rate_hz, args.timeout_sec, args.show_images)
    except KeyboardInterrupt:
        pass
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
