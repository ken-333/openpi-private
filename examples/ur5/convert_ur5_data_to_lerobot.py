"""
Convert UR5 collected HDF5 episodes to the LeRobot v2.0 dataset format (for openpi training).

Example:
    uv run examples/ur5/convert_ur5_data_to_lerobot.py --raw-dir collected_data --repo-id <org>/ur5-pick-place

Each episode_*.hdf5 (written by collect_data.py) contains:
    observation/qpos          (N, 7)            float64   6 joints + gripper
    action                    (N, 7)            float64
    observation/images/base   (N, 480, 640, 3)  uint8 RGB
    observation/images/wrist  (N, 480, 640, 3)  uint8 RGB
    task                      ()                str       language instruction
    timestamp / image_age     (N,)              float64   (recorded but not needed here)

Adapted from examples/aloha_real/convert_aloha_data_to_lerobot.py to match this schema.
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

# 6 UR5 joints + gripper, matching the qpos/action column order in collect_data.py
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3", "gripper"] # note that the gripper is included as a "joint" here, even though it's not a physical joint of the UR5 arm, because it's part of the action/state space and we want to include it in the dataset features
CAMERAS = ["base", "wrist"]
IMAGE_SHAPE = (3, 480, 640)   # CHW, as LeRobot declares image features


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    fps: int,
    robot_type: str = "ur5",
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": IMAGE_SHAPE,
            "names": ["channels", "height", "width"],
        }

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=(mode == "video"),   # image mode → store frames directly (no video codec)
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(ep_path: Path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["observation/qpos"][:]).float()
        action = torch.from_numpy(ep["action"][:]).float()
        # images are already stored HWC uint8 RGB by collect_data.py — no conversion needed
        imgs_per_cam = {cam: ep[f"observation/images/{cam}"][:] for cam in CAMERAS}
        raw_task = ep["task"][()]
        task = raw_task.decode() if isinstance(raw_task, bytes) else str(raw_task)
    return imgs_per_cam, state, action, task


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        imgs_per_cam, state, action, task = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,   # language instruction — per-frame in this lerobot API
            }
            for cam, imgs in imgs_per_cam.items():
                frame[f"observation.images.{cam}"] = imgs[i]   # HWC uint8 RGB
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def convert_ur5(
    raw_dir: Path = Path("collected_data"),
    repo_id: str = "ur5/pick_place",
    fps: int = 25,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {raw_dir}")

    dataset = create_empty_dataset(repo_id, fps=fps, mode=mode, dataset_config=dataset_config)
    dataset = populate_dataset(dataset, hdf5_files, episodes=episodes)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert_ur5)
