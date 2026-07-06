#!/usr/bin/env python3
"""vlahost server.

Runs on the robot machine (local host, requires ROS 2 / rclpy). Hosts a FastAPI app
that exposes the robot's joint feedback, end-effector poses and quadcam
composite image over HTTP, and applies actions posted back by the remote
vlahost client (e.g. running on the pi0 machine) onto the robot's control
topics.

Follows the same rclpy + FastAPI lifespan pattern as
UI_node/apex_backend/apex_backend/ros_state.py.

Usage:
    ros2 run vlahost vlahost_server --ros-args -p host:=0.0.0.0 -p port:=8000
"""
import argparse
import base64
import math
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import os

import numpy as np
import pinocchio as pin
import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from geometry_msgs.msg import PoseStamped
from marvin_msgs.msg import Jointfeedback
from pydantic import BaseModel
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Int16MultiArray

_DEBUG_PAGE: str | None = None


def _load_debug_page() -> str:
    path = os.path.join(os.path.dirname(__file__), "html", "debug.html")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"<h1>debug page not found</h1><pre>{exc}</pre>"


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return [qx, qy, qz, qw]


class StateResponse(BaseModel):
    stamp: Optional[int] = None
    joint_states: Optional[Dict[str, Any]] = None
    eef_left: Optional[Dict[str, Any]] = None
    eef_right: Optional[Dict[str, Any]] = None
    quad_image: Optional[Dict[str, Any]] = None  # {"format": "jpeg", "data": base64 str}


class ActionRequest(BaseModel):
    eef_left: Optional[List[float]] = None
    eef_right: Optional[List[float]] = None
    gripper_left: Optional[float] = None
    gripper_right: Optional[float] = None


class VlaHostNode(Node):
    def __init__(self):
        super().__init__('vlahost_server')

        self.declare_parameter('joint_states_topic', '/info/joint_feedback')
        self.declare_parameter('eef_left_topic', '/info/eef_left')
        self.declare_parameter('eef_right_topic', '/info/eef_right')
        self.declare_parameter('quad_image_topic', 'quad_tile/compressed')
        self.declare_parameter('target_pose_left_topic', '/control/target_poseL_model')
        self.declare_parameter('target_pose_right_topic', '/control/target_poseR_model')
        self.declare_parameter('gripper_left_topic', 'control/gripperValueL')
        self.declare_parameter('gripper_right_topic', 'control/gripperValueR')
        self.declare_parameter('eef_constraint_topic', 'control/eef_constraint')

        self.joint_state_sub = self.create_subscription(
            Jointfeedback,
            self.get_parameter('joint_states_topic').get_parameter_value().string_value,
            self.callback_joint_states,
            10,
        )
        self.subL = self.create_subscription(
            PoseStamped,
            self.get_parameter('eef_left_topic').get_parameter_value().string_value,
            self.callback_eef_L,
            10,
        )
        self.subR = self.create_subscription(
            PoseStamped,
            self.get_parameter('eef_right_topic').get_parameter_value().string_value,
            self.callback_eef_R,
            10,
        )
        self.quad_image_sub = self.create_subscription(
            CompressedImage,
            self.get_parameter('quad_image_topic').get_parameter_value().string_value,
            self.callback_quad_image,
            10,
        )

        self.target_constraint_pub = self.create_publisher(
            Int16MultiArray, self.get_parameter('eef_constraint_topic').get_parameter_value().string_value, 10)
        self.cmd_pub_L = self.create_publisher(
            PoseStamped, self.get_parameter('target_pose_left_topic').get_parameter_value().string_value, 10)
        self.cmd_pub_R = self.create_publisher(
            PoseStamped, self.get_parameter('target_pose_right_topic').get_parameter_value().string_value, 10)
        self.gripper_controller_pub_L = self.create_publisher(
            Float32, self.get_parameter('gripper_left_topic').get_parameter_value().string_value, 10)
        self.gripper_controller_pub_R = self.create_publisher(
            Float32, self.get_parameter('gripper_right_topic').get_parameter_value().string_value, 10)

        self._state_lock = threading.Lock()
        self.current_eef_pose_L = None   # pin.SE3
        self.current_eef_pose_R = None   # pin.SE3
        self.target_eef_pose_L = None    # pin.SE3
        self.target_eef_pose_R = None    # pin.SE3
        self.latest_joint_state = None
        self.latest_eef_left = None
        self.latest_eef_right = None
        self.latest_quad_image = None    # sensor_msgs/CompressedImage

        self.get_logger().info("vlahost server node started")

    # =====================================================
    # State subscription callbacks
    # =====================================================
    def callback_joint_states(self, msg: Jointfeedback):
        with self._state_lock:
            self.latest_joint_state = msg

    def callback_eef_L(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = pin.Quaternion(q.w, q.x, q.y, q.z)
        quat.normalize()
        pose = pin.SE3(quat.toRotationMatrix(), np.array([p.x, p.y, p.z]))
        with self._state_lock:
            self.latest_eef_left = msg
            self.current_eef_pose_L = pose
            if self.target_eef_pose_L is None:
                self.target_eef_pose_L = pose

    def callback_eef_R(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = pin.Quaternion(q.w, q.x, q.y, q.z)
        quat.normalize()
        pose = pin.SE3(quat.toRotationMatrix(), np.array([p.x, p.y, p.z]))
        with self._state_lock:
            self.latest_eef_right = msg
            self.current_eef_pose_R = pose
            if self.target_eef_pose_R is None:
                self.target_eef_pose_R = pose

    def callback_quad_image(self, msg: CompressedImage):
        with self._state_lock:
            self.latest_quad_image = msg

    # =====================================================
    # GET /state
    # =====================================================
    def snapshot(self) -> StateResponse:
        with self._state_lock:
            joint_state = self.latest_joint_state
            eef_left = self.latest_eef_left
            eef_right = self.latest_eef_right
            quad_image = self.latest_quad_image
        return StateResponse(
            stamp=self.get_clock().now().nanoseconds,
            joint_states=self._joint_state_to_dict(joint_state),
            eef_left=self._pose_to_dict(eef_left),
            eef_right=self._pose_to_dict(eef_right),
            quad_image=self._image_to_dict(quad_image),
        )

    def _pose_to_dict(self, msg: PoseStamped):
        if msg is None:
            return None
        p = msg.pose.position
        q = msg.pose.orientation
        return {
            "position": {"x": p.x, "y": p.y, "z": p.z},
            "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
        }

    def _joint_state_to_dict(self, msg: Jointfeedback):
        if msg is None:
            return None
        return {
            "positions": list(msg.positions),
            "velocities": list(msg.velocities),
            "efforts": list(msg.efforts),
            "est_joint_force": list(msg.est_joint_force),
        }

    def _image_to_dict(self, msg: CompressedImage):
        if msg is None:
            return None
        return {
            "format": msg.format,
            "data": base64.b64encode(bytes(msg.data)).decode('ascii'),
        }

    # =====================================================
    # POST /action - apply model action onto the robot control topics
    # =====================================================
    def _publish_target_pose(self, publisher, pose: pin.SE3):
        pos = pose.translation
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        quat = pin.Quaternion(pose.rotation)
        msg.pose.orientation.w = quat.w
        msg.pose.orientation.x = quat.x
        msg.pose.orientation.y = quat.y
        msg.pose.orientation.z = quat.z
        publisher.publish(msg)

    def apply_action(self, action: ActionRequest) -> Dict[str, Any]:
        with self._state_lock:
            have_current_pose = self.current_eef_pose_L is not None and self.current_eef_pose_R is not None
        if not have_current_pose:
            return {"success": False, "error": "eef poses not received yet"}

        if action.eef_left is not None:
            if len(action.eef_left) != 6:
                return {"success": False, "error": f"eef_left expected 6 values, got {len(action.eef_left)}"}
            p, q = action.eef_left[:3], action.eef_left[3:]
            quat = pin.Quaternion(np.array(euler_to_quaternion(*q)))
            quat.normalize()
            self.target_eef_pose_L = pin.SE3(quat.toRotationMatrix(), np.array(p))
            self._publish_target_pose(self.cmd_pub_L, self.target_eef_pose_L)

        if action.eef_right is not None:
            if len(action.eef_right) != 6:
                return {"success": False, "error": f"eef_right expected 6 values, got {len(action.eef_right)}"}
            p, q = action.eef_right[:3], action.eef_right[3:]
            quat = pin.Quaternion(np.array(euler_to_quaternion(*q)))
            quat.normalize()
            self.target_eef_pose_R = pin.SE3(quat.toRotationMatrix(), np.array(p))
            self._publish_target_pose(self.cmd_pub_R, self.target_eef_pose_R)

        constraint_msg = Int16MultiArray()
        constraint_msg.data = [100, 100, 100, 100, 100, 100]
        self.target_constraint_pub.publish(constraint_msg)

        if action.gripper_left is not None:
            gripper_msg = Float32()
            gripper_msg.data = action.gripper_left
            self.gripper_controller_pub_L.publish(gripper_msg)
        if action.gripper_right is not None:
            gripper_msg = Float32()
            gripper_msg.data = action.gripper_right
            self.gripper_controller_pub_R.publish(gripper_msg)

        return {"success": True}


class RosStateRuntime:
    def __init__(self) -> None:
        self.node: Optional[VlaHostNode] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> VlaHostNode:
        if not rclpy.ok():
            rclpy.init()
        self.node = VlaHostNode()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self.node

    def stop(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

    def _spin(self) -> None:
        if self.node is None:
            return
        try:
            rclpy.spin(self.node)
        except ExternalShutdownException:
            return


ros_runtime = RosStateRuntime()
vla_node: Optional[VlaHostNode] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global vla_node
    vla_node = ros_runtime.start()
    try:
        yield
    finally:
        ros_runtime.stop()


app = FastAPI(title="vlahost server", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def debug_page():
    global _DEBUG_PAGE
    if _DEBUG_PAGE is None:
        _DEBUG_PAGE = _load_debug_page()
    return _DEBUG_PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/state", response_model=StateResponse)
def get_state():
    return vla_node.snapshot()


@app.post("/action")
def post_action(action: ActionRequest):
    return vla_node.apply_action(action)


def main(argv=None):
    parser = argparse.ArgumentParser(description="vlahost server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
