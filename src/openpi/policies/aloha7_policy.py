import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class Aloha7Inputs(transforms.DataTransformFn):
    """Map three-camera, 16D joint-space samples to OpenPI model inputs."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Unexpected camera names: {tuple(in_images)}")

        def convert_image(img: np.ndarray) -> np.ndarray:
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        base_image = convert_image(in_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = convert_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class Aloha7Outputs(transforms.DataTransformFn):
    """Keep the full 16D action output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}
