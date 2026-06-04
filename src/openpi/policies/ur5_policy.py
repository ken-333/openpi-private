"""Policy transforms for the UR5 pick-and-place dataset (Step 3).

Mirrors src/openpi/policies/libero_policy.py. Two jobs:
  - UR5Inputs : LeRobot/runtime dict  ->  model input format  (train + inference)
  - UR5Outputs: model action output   ->  UR5 action format   (inference only)

The input keys below (observation/state, observation/image, observation/wrist_image,
prompt) are what the model side expects. Your Step-4 training config must REPACK the
raw LeRobot feature names into these keys, e.g.:
    observation.state         -> observation/state
    observation.images.base   -> observation/image
    observation.images.wrist  -> observation/wrist_image
    task                      -> prompt
(See the RepackTransform in the libero TrainConfig for the pattern.)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Random example with the shapes the model receives (for smoke tests / shape checks)."""
    return {
        "observation/state": np.random.rand(7),  # 6 joints + gripper
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the item and place it in the plate",
    }


def _parse_image(image) -> np.ndarray:
    """LeRobot stores images float32 (C,H,W); the model wants uint8 (H,W,C). Normalize that here."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """LeRobot/runtime dict -> model input format. Used for training AND inference."""

    # Determines which model is used (set by the training config). Do not change.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # --- images: convert to uint8 (H,W,C) ---
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # --- build model inputs (do NOT rename these keys) ---
        # UR5 has ONE wrist camera, so map it to the left wrist slot and zero-pad the
        # right wrist (Pi0/Pi0.5 then masks it out via image_mask below).
        inputs = {
            # state is already 7-dim (joints+gripper) from collect_data.py; openpi pads
            # it up to the model action_dim downstream, so leave it as-is here.
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # no right wrist on UR5
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # masked (False) for Pi0/Pi0.5; only Pi0-FAST keeps padded images visible
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # actions: present only during training; pass through (output transform trims later)
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Model action output -> UR5 action format. Inference only."""

    def __call__(self, data: dict) -> dict:
        # Model emits action_dim (e.g. 32) actions; UR5 uses the first 7 (6 joints + gripper).
        return {"actions": np.asarray(data["actions"][:, :7])}
