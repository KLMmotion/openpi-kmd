#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket
import sys
from collections.abc import Sequence

import numpy as np
import tyro
from typing_extensions import override

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from openpi.models import pi0_config
from openpi.policies import aloha7_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi_client import base_policy as _base_policy


@dataclasses.dataclass(frozen=True)
class KMDJointDataConfig(_config.DataConfigFactory):
    """Three-camera, 16D joint-space LeRobot data config used by KMD checkpoints."""

    use_delta_joint_actions: bool = False
    default_prompt: str | None = None
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha7_policy.Aloha7Inputs()],
            outputs=[aloha7_policy.Aloha7Outputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def build_train_config(args: "Args") -> _config.TrainConfig:
    return _config.TrainConfig(
        name="kmd_joint_inference",
        exp_name="serve",
        model=pi0_config.Pi0Config(
            pi05=args.pi05,
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            discrete_state_input=args.discrete_state_input,
        ),
        data=KMDJointDataConfig(
            repo_id=args.repo_id,
            assets=_config.AssetsConfig(asset_id=args.asset_id or args.repo_id),
            base_config=_config.DataConfig(prompt_from_task=True),
            use_delta_joint_actions=args.use_delta_joint_actions,
            default_prompt=args.default_prompt,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        policy_metadata={"mode": "kmd_joint_real_robot"},
        wandb_enabled=False,
    )


DEFAULT_PROMPT = (
    "First, pick up the right cube with the right hand and place it on the blue cube in the middle. "
    "Then, pick up the left cube with the left hand and place it on the blue cube in the middle."
)


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    repo_id: str = "0629_jointstate"
    asset_id: str | None = None
    default_prompt: str | None = DEFAULT_PROMPT
    port: int = 8000
    record: bool = False
    robot_state_dim: int = 16
    action_dim: int = 16
    action_horizon: int = 10
    pi05: bool = True
    discrete_state_input: bool = True
    use_delta_joint_actions: bool = False


def create_policy(args: Args) -> _base_policy.BasePolicy:
    checkpoint_path = args.checkpoint_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    if not (checkpoint_path / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found under: {checkpoint_path}")

    train_config = build_train_config(args)
    logging.info("Loading checkpoint from: %s", checkpoint_path)
    logging.info(
        "Using repo_id=%s asset_id=%s action_dim=%d action_horizon=%d",
        args.repo_id,
        args.asset_id or args.repo_id,
        args.action_dim,
        args.action_horizon,
    )
    return _policy_config.create_trained_policy(
        train_config,
        str(checkpoint_path),
        default_prompt=args.default_prompt,
    )


def warmup_policy(policy: _base_policy.BasePolicy, *, default_prompt: str | None, robot_state_dim: int) -> None:
    logging.info("Warming up policy...")
    zero = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_observation = {
        "images": {
            "cam_high": zero,
            "cam_left_wrist": zero,
            "cam_right_wrist": zero,
        },
        "state": np.zeros(robot_state_dim, dtype=np.float32),
        "prompt": default_prompt or "",
    }
    try:
        out = policy.infer(dummy_observation)
        actions = out.get("actions")
        logging.info("Warmup OK. actions shape: %s", getattr(actions, "shape", actions))
    except Exception as exc:
        logging.warning("Warmup failed (non-fatal): %s", exc)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    policy = create_policy(args)
    policy_metadata = getattr(policy, "metadata", {})
    warmup_policy(policy, default_prompt=args.default_prompt, robot_state_dim=args.robot_state_dim)

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    logging.info("Policy server started (KMD joint-space)")
    logging.info("  Checkpoint: %s", args.checkpoint_dir)
    logging.info("  Repo id: %s", args.repo_id)
    logging.info("  Asset id: %s", args.asset_id or args.repo_id)
    logging.info("  Port: %s", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
