# OpenPI KMD Real-Robot Inference Guide

This repository is prepared for one thing only: deploy a trained OpenPI checkpoint to a real robot with 16D joint-space control.

The supported KMD setup in this repo is:

- 3 camera inputs: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- 16D joint state: `[left_7_joints, left_gripper, right_7_joints, right_gripper]`
- 16D joint action with the same layout
- ROS2 image topic for the quad camera stream
- Robot WebSocket bridge for `joint_states` input and `joint_actions` output

## Files Added for KMD

The customer only needs these files:

- `scripts/serve_policy_kmd_joint.py`
- `vla_helpers/openpi_client_policy_ros2_kmd_joint.py`
- `vla_helpers/openpi_policy_shared.py`
- `src/openpi/policies/aloha7_policy.py`

## What You Need Before Running

You need all of the following:

1. A trained checkpoint directory containing `model.safetensors`
2. A checkpoint `assets/<asset_id>/norm_stats.json`
3. A ROS2 topic publishing `/quad_tile/compressed`
4. A robot WebSocket bridge publishing `joint_states` and accepting `joint_actions`

Typical checkpoint example:

```text
checkpoints/pi05_0629_jointstate/0629_run1/20000
```

## Data Contract

This deployment path assumes:

- Model input `state` is 16D
- Joint angles inside the model are in `degrees`
- Robot WebSocket `joint_states.position` is normally in `radians`
- The client converts:
  - `rad -> deg` before inference
  - `deg -> rad` before sending `joint_actions`

Default quad-camera split:

- `top_right -> cam_left_wrist`
- `bottom_left -> cam_right_wrist`
- `bottom_right -> cam_high`

The top-left tile is ignored.

## Step 1: Install Environment

From the repository root:

```bash
cd /home/wyz/openpi-vla-kmd/openpi-kmd
uv sync
```

For the robot-side client you also need ROS2 Python packages available in the current shell:

```bash
source /opt/ros/humble/setup.bash
```

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

## Step 3: Start the Robot Client

Run this on the robot machine:

```bash
cd /home/wyz/openpi-vla-kmd/openpi-kmd
source /opt/ros/humble/setup.bash

uv run python vla_helpers/openpi_client_policy_ros2_kmd_joint.py \
  --policy-host <POLICY_SERVER_IP> \
  --policy-port 8000 \
  --ws-host <ROBOT_WS_IP> \
  --ws-port 8765 \
  --task-prompt "pick and place"
```

## Most Important Client Arguments

- `--topic`: ROS2 image topic, default `/quad_tile/compressed`
- `--policy-host`: policy server IP
- `--policy-port`: policy server port
- `--ws-host`: robot WebSocket bridge IP
- `--ws-port`: robot WebSocket bridge port
- `--task-prompt`: language instruction sent to the model
- `--replan-steps`: how many actions to execute from each predicted chunk
- `--action-rate`: action send rate in Hz
- `--disable-ws`: inference only, do not send actions to the robot
- `--no-ws-joints-in-radians`: use this only if the robot WebSocket already sends joint angles in degrees

## Minimal Startup Order

1. Confirm the checkpoint path is correct
2. Confirm `model.safetensors` exists
3. Confirm checkpoint assets include `norm_stats.json`
4. Start `scripts/serve_policy_kmd_joint.py`
5. Confirm the robot ROS2 topic `/quad_tile/compressed` is active
6. Confirm the robot WebSocket bridge is reachable
7. Start `vla_helpers/openpi_client_policy_ros2_kmd_joint.py`

## Troubleshooting

### Server starts but inference fails immediately

Check:

- checkpoint path is correct
- `asset_id` matches the folder name under `checkpoint/assets/`
- the checkpoint was trained with the same 3-camera 16D joint-space layout

### Client connects but gets no robot state

Check:

- the WebSocket bridge is really sending `joint_states`
- `joint_states.position` length matches your robot layout
- if the bridge only sends 14 arm joints, keep `--ws-arm-position-dim -1` or set `--ws-arm-position-dim 14`

### Robot moves incorrectly

Check:

- rad/deg assumption is correct
- left/right arm order matches `[left, left_gripper, right, right_gripper]`
- the quad camera split matches the actual physical camera layout

### Camera images look wrong

Check:

- `/quad_tile/compressed` content is correct
- crop parameters if needed:
  - `--crop-left/right/top/bottom`
  - `--wrist-crop-left/right/top/bottom`

## Customer Summary

For customer deployment, the only command pair they normally need is:

```bash
uv run scripts/serve_policy_kmd_joint.py --checkpoint-dir /ABS/PATH/TO/CHECKPOINT_STEP --repo-id 0629_jointstate --asset-id 0629_jointstate
```

```bash
uv run python vla_helpers/openpi_client_policy_ros2_kmd_joint.py --policy-host <POLICY_SERVER_IP> --ws-host <ROBOT_WS_IP> --task-prompt "pick and place"
```
