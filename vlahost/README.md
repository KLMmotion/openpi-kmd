# vlahost

HTTP bridge between the robot (joint feedback, end-effector poses, quad camera) and a remote VLA inference client (e.g. pi0).

```
robot (local, with ROS)                    pi0 / GPU machine (no ROS)
┌───────────────────────┐  GET /state      ┌──────────────────────┐
│ vlahost_server        │ ◀─────────────── │ openpi HTTP client    │
│ - /info/joint_feedback│  state+image     │ - no rclpy dependency │
│ - /info/eef_left/right│ ───────────────▶ │ - calls OpenPI policy │
│ - quad_tile/compressed│                  │   over WebSocket      │
│ - publishes control/* │  POST /action    │                       │
│                       │ ◀─────────────── │                       │
└───────────────────────┘  action(json)     └──────────────────────┘
```

## server (robot machine, requires ROS 2 + FastAPI)

Subscribes to:

- `/info/joint_feedback` (`marvin_msgs/Jointfeedback`)
- `/info/eef_left`, `/info/eef_right` (`geometry_msgs/PoseStamped`)
- `quad_tile/compressed` (`sensor_msgs/CompressedImage`, jpeg from `gmsl_quadcam`)

Exposes via FastAPI:

- `GET /` — browser debug page with live quadcam preview and manual action form
- `GET /state` — latest joint_states + eef_left/right + quadcam jpeg (base64)
- `POST /action` — apply action to `/control/target_poseL_model`,
  `/control/target_poseR_model`, `control/gripperValueL/R`, `control/eef_constraint`
- `GET /health`

Build and run from this repository:

```bash
# Link vlahost into your ROS 2 workspace, then build
ln -s /path/to/openpi-kmd/vlahost ~/ros_ws/src/vlahost
cd ~/ros_ws
colcon build --packages-select vlahost
source install/setup.bash
ros2 launch vlahost vlahost_server.launch.py host:=0.0.0.0 port:=8000
```

## client (reference stub, no ROS)

The production OpenPI client for KMD joint checkpoints lives at
[`vla_helpers/openpi_client_policy_http_kmd_joint.py`](../vla_helpers/openpi_client_policy_http_kmd_joint.py).

`vlahost/client.py` is a minimal reference loop that polls `/state` and posts a no-op
action. Use it only for quick connectivity checks:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r vlahost/client_requirements.txt
python3 vlahost/vlahost/client.py --server-url http://<robot-host>:8000
```

## API format

`GET /state` response:

```json
{
  "stamp": 123456789,
  "joint_states": {"positions": [...14], "velocities": [...14], "efforts": [...14], "est_joint_force": [...14]},
  "eef_left": {"position": {"x":0,"y":0,"z":0}, "orientation": {"x":0,"y":0,"z":0,"w":1}},
  "eef_right": {"position": {...}, "orientation": {...}},
  "quad_image": {"format": "jpeg", "data": "<base64>"}
}
```

`POST /action` request body (EEF control):

```json
{
  "eef_left": [x, y, z, rx, ry, rz],
  "eef_right": [x, y, z, rx, ry, rz],
  "gripper_left": 0.0,
  "gripper_right": 0.0
}
```

For 16D joint-space OpenPI checkpoints, the inference client instead posts:

```json
{
  "joint_actions": [16 floats in radians]
}
```

Any field may be `null`; the server skips fields that are not provided.
