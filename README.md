# OpenPI KMD Real-Robot Inference Guide

This repository is prepared for one thing only: deploy a trained OpenPI checkpoint to a real robot with 16D joint-space control.

The supported KMD setup in this repo is:

- 3 camera inputs: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- 16D joint state: `[left_7_joints, left_gripper, right_7_joints, right_gripper]`
- 16D joint action with the same layout
- [vlahost](https://github.com/KLMmotion/vlahost) HTTP server on the robot (`GET /state`, `POST /action`)
- OpenPI policy WebSocket server on the GPU machine

## Files Added for KMD

The customer only needs these files:

- `scripts/serve_policy_kmd_joint.py`
- `vla_helpers/openpi_client_policy_http_kmd_joint.py`
- `vla_helpers/openpi_policy_shared.py`
- `src/openpi/policies/aloha7_policy.py`

## What You Need Before Running

You need all of the following:

1. A trained checkpoint directory containing `model.safetensors`
2. A checkpoint `assets/<asset_id>/norm_stats.json`
3. [vlahost](https://github.com/KLMmotion/vlahost) server running on the robot and exposing `GET /state` and `POST /action`
4. OpenPI policy server running on the GPU machine

Typical checkpoint example:

```text
checkpoints/pi05_0629_jointstate/0629_run1/20000
```

## Data Contract

This deployment path assumes:

- Model input `state` is 16D
- Joint angles inside the model are in `degrees`
- vlahost `joint_states.positions` is normally in `radians` (14 arm joints)
- The client converts:
  - `rad -> deg` before inference
  - `deg -> rad` before `POST /action` with `joint_actions`

Default quad-camera split (from vlahost `quad_image`):

- `top_right -> cam_left_wrist`
- `bottom_left -> cam_right_wrist`
- `bottom_right -> cam_high`

The top-left tile is ignored.

### vlahost HTTP interface

`GET /state` response (from vlahost server on the robot):

```json
{
  "stamp": 123456789,
  "joint_states": {"positions": [...14], "velocities": [...], "efforts": [...], "est_joint_force": [...]},
  "eef_left": {"position": {"x":0,"y":0,"z":0}, "orientation": {"x":0,"y":0,"z":0,"w":1}},
  "eef_right": {"position": {...}, "orientation": {...}},
  "quad_image": {"format": "jpeg", "data": "<base64>"}
}
```

`POST /action` request body for 16D joint-space checkpoints:

```json
{
  "joint_actions": [16 floats in radians]
}
```

## Step 1: Install Environment

From the repository root:

```bash
cd /home/wyz/openpi-vla-kmd/openpi-kmd
uv sync
```

On the robot machine, clone and build vlahost (requires ROS 2):

```bash
git clone https://github.com/KLMmotion/vlahost.git
ln -s /path/to/vlahost ~/ros_ws/src/vlahost
cd ~/ros_ws
colcon build --packages-select vlahost
source install/setup.bash
ros2 launch vlahost vlahost_server.launch.py host:=0.0.0.0 port:=8000
```

See [vlahost](https://github.com/KLMmotion/vlahost) for full server details.

The inference client runs on the GPU machine and does **not** require ROS 2.

## Step 2: Start the Policy Server

Run this on the GPU machine:

```bash
cd /home/wyz/openpi-vla-kmd/openpi-kmd

uv run scripts/serve_policy_kmd_joint.py \
  --checkpoint-dir /ABS/PATH/TO/CHECKPOINT_STEP \
  --repo-id 0629_jointstate \
  --asset-id 0629_jointstate \
  --port 8000
```

Notes:

- `--checkpoint-dir` must point to the exact step directory
- `model.safetensors` must exist in that directory
- `assets/<asset_id>/norm_stats.json` must exist under the same checkpoint directory
- The server listens on `0.0.0.0:8000`

If your training dataset name was not `0629_jointstate`, replace both:

- `--repo-id`
- `--asset-id`

with the dataset id used to store the checkpoint assets.

## Step 3: Start the Inference Client

Run this on the GPU machine (same host as the policy server, or any machine that can reach both the robot and the policy server):

```bash
cd /home/wyz/openpi-vla-kmd/openpi-kmd

uv run python vla_helpers/openpi_client_policy_http_kmd_joint.py \
  --robot-server-url http://<ROBOT_HOST>:8000 \
  --policy-host localhost \
  --policy-port 8000 \
  --task-prompt "pick and place"
```

## Most Important Client Arguments

- `--robot-server-url`: vlahost server base URL on the robot, e.g. `http://192.168.1.10:8000`
- `--robot-timeout-sec`: HTTP timeout for `/state` and `/action`
- `--policy-host`: OpenPI policy server IP
- `--policy-port`: OpenPI policy server port
- `--task-prompt`: language instruction sent to the model
- `--replan-steps`: how many actions to execute from each predicted chunk
- `--action-rate`: control loop rate in Hz
- `--disable-action-post`: inference only, do not POST actions to the robot
- `--no-joints-in-radians`: use this only if vlahost already sends joint angles in degrees

## Minimal Startup Order

1. Confirm the checkpoint path is correct
2. Confirm `model.safetensors` exists
3. Confirm checkpoint assets include `norm_stats.json`
4. Start [vlahost](https://github.com/KLMmotion/vlahost) on the robot (`ros2 launch vlahost vlahost_server.launch.py`)
5. Confirm `GET http://<ROBOT_HOST>:8000/health` returns OK
6. Start `scripts/serve_policy_kmd_joint.py` on the GPU machine
7. Start `vla_helpers/openpi_client_policy_http_kmd_joint.py`

## Troubleshooting

### Server starts but inference fails immediately

Check:

- checkpoint path is correct
- `asset_id` matches the folder name under `checkpoint/assets/`
- the checkpoint was trained with the same 3-camera 16D joint-space layout

### Client connects but gets no robot state

Check:

- vlahost is running and `GET /state` returns `joint_states.positions`
- `joint_states.positions` length matches your robot layout (typically 14 arm joints)
- if the server only sends 14 arm joints, keep `--ws-arm-position-dim -1` or set `--ws-arm-position-dim 14`

### Robot moves incorrectly

Check:

- rad/deg assumption is correct
- left/right arm order matches `[left, left_gripper, right, right_gripper]`
- the quad camera split matches the actual physical camera layout

### Camera images look wrong

Check:

- vlahost `quad_image` content is correct
- crop parameters if needed:
  - `--crop-left/right/top/bottom`
  - `--wrist-crop-left/right/top/bottom`

## Customer Summary

For customer deployment, the typical command pair is:

```bash
# GPU machine
uv run scripts/serve_policy_kmd_joint.py --checkpoint-dir /ABS/PATH/TO/CHECKPOINT_STEP --repo-id 0629_jointstate --asset-id 0629_jointstate
```

```bash
# GPU machine (polls robot vlahost, calls policy server)
uv run python vla_helpers/openpi_client_policy_http_kmd_joint.py --robot-server-url http://<ROBOT_HOST>:8000 --policy-host localhost --task-prompt "pick and place"
```
