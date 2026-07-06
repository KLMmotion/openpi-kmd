#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
except ImportError as exc:
    print("需要 ROS2 Python: rclpy, sensor_msgs", file=sys.stderr)
    raise SystemExit(1) from exc

from openpi_policy_shared import (
    DEFAULT_TASK_INSTRUCTION,
    IMG_SIZE,
    POLICY_CLIENT_AVAILABLE,
    WSJointStateClient,
    crop_by_ratio,
    ensure_hwc_uint8,
)

if POLICY_CLIENT_AVAILABLE:
    from openpi_client import websocket_client_policy

QOS_SENSOR = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)

_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_JOINT_LEFT_SLICE = slice(0, 7)
_JOINT_RIGHT_SLICE = slice(8, 15)
_GRIP_INDICES = (7, 15)


def state_ws_to_model_deg(state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy()
    out[_JOINT_LEFT_SLICE] = np.rad2deg(out[_JOINT_LEFT_SLICE])
    out[_JOINT_RIGHT_SLICE] = np.rad2deg(out[_JOINT_RIGHT_SLICE])
    return out


def action_model_deg_to_ws_rad(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[_JOINT_LEFT_SLICE] = np.deg2rad(out[_JOINT_LEFT_SLICE])
    out[_JOINT_RIGHT_SLICE] = np.deg2rad(out[_JOINT_RIGHT_SLICE])
    return out


def split_quad_bgr_3cam(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img_bgr.shape[:2]
    sub_h, sub_w = h // 2, w // 2
    return {
        "cam_high": img_bgr[sub_h:h, sub_w:w].copy(),
        "cam_left_wrist": img_bgr[0:sub_h, sub_w:w].copy(),
        "cam_right_wrist": img_bgr[sub_h:h, 0:sub_w].copy(),
    }


def build_state_vector(
    joint_state: dict | None,
    *,
    state_dim: int,
    ws_arm_position_dim: int,
    arm_joints_left: int,
    arm_joints_right: int,
    gripper_fallback: dict[str, float],
) -> np.ndarray:
    out = np.zeros(state_dim, dtype=np.float32)
    if joint_state is None:
        return out
    pos = joint_state.get("position")
    if not isinstance(pos, list) or len(pos) == 0:
        return out

    if ws_arm_position_dim == 0:
        n = min(state_dim, len(pos))
        out[:n] = np.asarray(pos[:n], dtype=np.float32)
        return out

    if ws_arm_position_dim < 0:
        if len(pos) >= state_dim:
            n = min(state_dim, len(pos))
            out[:n] = np.asarray(pos[:n], dtype=np.float32)
            return out
        expected_arm = arm_joints_left + arm_joints_right
        if len(pos) >= expected_arm:
            ws_arm_position_dim = expected_arm
        else:
            n = min(state_dim, len(pos))
            out[:n] = np.asarray(pos[:n], dtype=np.float32)
            return out

    left = np.asarray(pos[:arm_joints_left], dtype=np.float32)
    right = np.asarray(pos[arm_joints_left : arm_joints_left + arm_joints_right], dtype=np.float32)
    gl = float(joint_state.get("gripper_left", gripper_fallback.get("left", 0.0)))
    gr = float(joint_state.get("gripper_right", gripper_fallback.get("right", 0.0)))
    merged = np.concatenate([left, np.array([gl], dtype=np.float32), right, np.array([gr], dtype=np.float32)])
    n = min(state_dim, len(merged))
    out[:n] = merged[:n]
    return out


class RosQuadTileReceiver(Node):
    def __init__(self, topic: str):
        super().__init__("openpi_policy_ros2_kmd_joint")
        self.frame_lock = threading.Lock()
        self.latest_frames = {k: None for k in _CAMERA_KEYS}
        self.create_subscription(CompressedImage, topic, self._on_compressed, QOS_SENSOR)
        self.get_logger().info(f"订阅 CompressedImage: {topic} (KMD 三相机切分)")

    def _on_compressed(self, msg: CompressedImage) -> None:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warning("压缩图解码失败")
            return
        tiles = split_quad_bgr_3cam(img)
        with self.frame_lock:
            for key, arr in tiles.items():
                self.latest_frames[key] = arr


def start_ros_executor_in_thread(executor: MultiThreadedExecutor) -> threading.Thread:
    def _run():
        try:
            executor.spin()
        except Exception as exc:
            print(f"❌ [ros2 executor] {exc}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def parse_args():
    p = argparse.ArgumentParser(description="OpenPI ROS2 client for KMD 16D joint-space checkpoints")
    p.add_argument("--topic", default="/quad_tile/compressed")
    p.add_argument("--ws-host", default="192.168.15.123")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--disable-ws", action="store_true")
    p.add_argument("--ws-send-rate", type=float, default=10.0)
    p.add_argument("--task-prompt", type=str, default=DEFAULT_TASK_INSTRUCTION)
    p.add_argument("--replan-steps", type=int, default=6)
    p.add_argument("--action-rate", type=float, default=20.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--replan-wait-sec", type=float, default=0.5)
    p.add_argument("--robot-action-dim", type=int, default=16)
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--ws-arm-position-dim", type=int, default=-1)
    p.add_argument("--arm-joints-left", type=int, default=7)
    p.add_argument("--arm-joints-right", type=int, default=7)
    p.add_argument(
        "--ws-joints-in-radians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="机器人 WebSocket 上报/接收关节角默认按弧度处理",
    )
    p.add_argument("--crop-left", type=float, default=0.0)
    p.add_argument("--crop-right", type=float, default=0.0)
    p.add_argument("--crop-top", type=float, default=0.0)
    p.add_argument("--crop-bottom", type=float, default=0.0)
    p.add_argument("--wrist-crop-left", type=float, default=0.0)
    p.add_argument("--wrist-crop-right", type=float, default=0.0)
    p.add_argument("--wrist-crop-top", type=float, default=0.0)
    p.add_argument("--wrist-crop-bottom", type=float, default=0.0)
    p.add_argument("--cv-show-inputs", action="store_true")
    p.add_argument("--cv-show-wait-ms", type=int, default=1)
    p.add_argument("--cv-show-scale", type=float, default=1.0)
    p.add_argument("--print-input", action="store_true", default=True)
    p.add_argument("--print-output", action="store_true", default=True)
    return p.parse_args()


def make_preview_canvas(images: dict[str, np.ndarray], state_vec: np.ndarray) -> np.ndarray:
    high = images["cam_high"]
    left = images["cam_left_wrist"]
    right = images["cam_right_wrist"]
    canvas = np.hstack([high, left, right])
    st = ", ".join(f"{float(x):.2f}" for x in state_vec)
    cv2.putText(canvas, f"state[{len(state_vec)}]: {st}", (8, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    if args.ws_joints_in_radians:
        print("🔄 [units] WS joint_states rad -> model deg -> WS joint_actions rad")

    rclpy.init()
    receiver = RosQuadTileReceiver(args.topic)
    executor = MultiThreadedExecutor()
    executor.add_node(receiver)
    start_ros_executor_in_thread(executor)

    dump_session_dir = Path(__file__).resolve().parent / "obs_dumps" / time.strftime("%Y%m%d_%H%M%S")
    dump_session_dir.mkdir(parents=True, exist_ok=True)

    ws_client = None
    if not args.disable_ws:
        ws_client = WSJointStateClient(f"ws://{args.ws_host}:{args.ws_port}", send_rate_hz=args.ws_send_rate)
        ws_client.start()
        print(f"✅ WebSocket: ws://{args.ws_host}:{args.ws_port}")

    policy_client = None
    if POLICY_CLIENT_AVAILABLE:
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
        print(f"✅ Policy: {args.policy_host}:{args.policy_port}")
    else:
        print("⚠️ openpi_client 不可用")

    grip_fb: dict[str, float] = {"left": 0.0, "right": 0.0}
    action_plan: collections.deque = collections.deque()
    infer_count = 0
    wait_before_replan = False
    cv_show_state = {"enabled": bool(args.cv_show_inputs)}

    def dump_observation_inputs(step_idx, images_dict, state_vec, prompt):
        step_dir = dump_session_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in images_dict.items():
            if isinstance(arr, np.ndarray) and arr.size:
                cv2.imwrite(str(step_dir / f"{name}.png"), arr)
        np.save(step_dir / "observation_state.npy", state_vec.astype(np.float32))
        (step_dir / "prompt.txt").write_text(str(prompt), encoding="utf-8")

    def cv_show_observation_inputs(step_idx, images_dict, state_vec):
        if not cv_show_state["enabled"]:
            return
        try:
            canvas = make_preview_canvas(images_dict, state_vec)
            if args.cv_show_scale != 1.0:
                scale = float(args.cv_show_scale)
                h, w = canvas.shape[:2]
                canvas = cv2.resize(canvas, (max(1, int(w * scale)), max(1, int(h * scale))))
            cv2.imshow(f"kmd joint inputs (step {step_idx})", canvas)
            key = cv2.waitKey(args.cv_show_wait_ms) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"⚠️ [cv2.imshow] disabled: {exc}")
            cv_show_state["enabled"] = False

    def get_latest_observation_inputs():
        with receiver.frame_lock:
            frames = {k: receiver.latest_frames.get(k) for k in _CAMERA_KEYS}

        def _head(im):
            if im is None or not isinstance(im, np.ndarray) or im.size == 0:
                return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            cropped = crop_by_ratio(im, args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)
            return ensure_hwc_uint8(cropped)

        def _wrist(im, fallback):
            if im is None or not isinstance(im, np.ndarray) or im.size == 0:
                return fallback.copy()
            cropped = crop_by_ratio(
                im,
                args.wrist_crop_left,
                args.wrist_crop_right,
                args.wrist_crop_top,
                args.wrist_crop_bottom,
            )
            return ensure_hwc_uint8(cropped)

        cam_high = _head(frames.get("cam_high"))
        cam_left = _wrist(frames.get("cam_left_wrist"), cam_high)
        cam_right = _wrist(frames.get("cam_right_wrist"), cam_high)
        images = {
            "cam_high": cam_high,
            "cam_left_wrist": cam_left,
            "cam_right_wrist": cam_right,
        }

        js = ws_client.get_latest_joint_state() if ws_client else None
        state = build_state_vector(
            js,
            state_dim=args.state_dim,
            ws_arm_position_dim=args.ws_arm_position_dim,
            arm_joints_left=args.arm_joints_left,
            arm_joints_right=args.arm_joints_right,
            gripper_fallback=grip_fb,
        )
        if args.ws_joints_in_radians:
            state = state_ws_to_model_deg(state)
        return images, state

    try:
        while True:
            images, state_vec = get_latest_observation_inputs()

            if policy_client is not None and not action_plan:
                if wait_before_replan and args.replan_wait_sec > 0:
                    time.sleep(args.replan_wait_sec)
                    images, state_vec = get_latest_observation_inputs()
                    wait_before_replan = False

                observation = {"images": images, "state": state_vec, "prompt": args.task_prompt}
                cv_show_observation_inputs(infer_count, images, state_vec)
                dump_observation_inputs(infer_count, images, state_vec, args.task_prompt)

                if args.print_input:
                    img_info = ", ".join(f"{k}:{v.shape}" for k, v in images.items())
                    st = ", ".join(f"{float(x):.4f}" for x in state_vec)
                    print(f"📥 images => {img_info}")
                    print(f"📥 state[{args.state_dim}] => [{st}]")
                    print(f"📥 prompt => {args.task_prompt}")

                infer_count += 1
                try:
                    print("🔄 Requesting action chunk...")
                    t0 = time.perf_counter()
                    result = policy_client.infer(observation)
                    t_infer = time.perf_counter() - t0
                    action_chunk = result.get("actions")
                    if action_chunk is not None:
                        ac = np.asarray(action_chunk, dtype=np.float32)
                        n_step = min(args.replan_steps, ac.shape[0])
                        d = min(args.robot_action_dim, ac.shape[1])
                        if args.print_output:
                            first = ", ".join(f"{float(x):.4f}" for x in ac[0, :d])
                            print(f"📤 actions shape={ac.shape}")
                            print(f"📤 actions[0, :{d}] => [{first}]")
                        for i in range(n_step):
                            row = ac[i, :d].copy() * float(args.action_scale)
                            action_plan.append(row)
                        wait_before_replan = len(action_plan) > 0
                        print(f"✅ infer={t_infer:.3f}s | queued={len(action_plan)}")
                    else:
                        print(f"⚠️ no actions ({t_infer:.3f}s)")
                except Exception as exc:
                    print(f"❌ infer: {exc}")
                    import traceback

                    traceback.print_exc()

            if action_plan and ws_client is not None:
                q_model = action_plan.popleft()
                grip_fb["left"] = float(q_model[_GRIP_INDICES[0]])
                grip_fb["right"] = float(q_model[_GRIP_INDICES[1]])
                q_ws = action_model_deg_to_ws_rad(q_model) if args.ws_joints_in_radians else q_model
                ws_client.update_joint_action(q_ws)
                if args.print_output:
                    qvals = ", ".join(f"{float(x):.4f}" for x in np.asarray(q_ws).flatten())
                    print(f"📡 joint_actions => [{qvals}] | remaining={len(action_plan)}")

            time.sleep(1.0 / max(0.1, args.action_rate))

    except KeyboardInterrupt:
        print("\n🛑 shutdown")
    finally:
        if ws_client is not None:
            ws_client.stop()
        executor.shutdown()
        receiver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
