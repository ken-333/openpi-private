"""
Verify a collected HDF5 episode.

Usage:
    python verify_episode.py path/to/episode_0000.hdf5

Outputs (into debug_output/ next to the HDF5 file):
    <name>_video.mp4   — base + wrist side-by-side with frame/joint overlay
    <name>_joints.png  — joint angle curves over time
"""

import argparse
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3", "Gripper"]
FPS = 30


def load_episode(path):
    with h5py.File(path, "r") as f:
        qpos   = f["observation/qpos"][:]        # (N, 7)
        base   = f["observation/images/base"][:]  # (N, H, W, 3) RGB
        wrist  = f["observation/images/wrist"][:] # (N, H, W, 3) RGB
        task   = f["task"][()].decode() if isinstance(f["task"][()], bytes) else str(f["task"][()])
    return qpos, base, wrist, task


def print_stats(path, qpos, base, task):
    n = len(qpos)
    h, w = base.shape[1:3]
    print(f"\n--- Episode: {path.name} ---")
    print(f"  Task    : {task}")
    print(f"  Frames  : {n}  ({n / FPS:.1f}s at {FPS}Hz)")
    print(f"  Image   : {w}×{h}")
    print(f"  qpos min: {np.degrees(qpos.min(axis=0)).round(1)} deg")
    print(f"  qpos max: {np.degrees(qpos.max(axis=0)).round(1)} deg")
    print()


def make_video(out_path, base_imgs, wrist_imgs, qpos):
    n, h, w, _ = base_imgs.shape
    frame_w = w * 2   # side by side
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (frame_w, h),
    )

    for i in range(n):
        left  = cv2.cvtColor(base_imgs[i],  cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(wrist_imgs[i], cv2.COLOR_RGB2BGR)

        # labels
        cv2.putText(left,  "BASE",  (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(right, "WRIST", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # frame counter
        cv2.putText(left, f"frame {i:04d}/{n}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # joint angles overlay (degrees) on right panel
        for j, name in enumerate(JOINT_NAMES):
            val = np.degrees(qpos[i, j]) if j < 6 else qpos[i, j]
            unit = "deg" if j < 6 else ""
            text = f"{name}: {val:+.1f}{unit}"
            cv2.putText(right, text, (10, 24 + (j + 1) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 100), 1)

        frame = np.concatenate([left, right], axis=1)
        writer.write(frame)

    writer.release()
    print(f"  Video   → {out_path}")


def make_joint_plot(out_path, qpos):
    n = len(qpos)
    t = np.arange(n) / FPS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(out_path.stem.replace("_joints", ""))

    # joint angles (degrees)
    for j in range(6):
        ax1.plot(t, np.degrees(qpos[:, j]), label=JOINT_NAMES[j])
    ax1.set_ylabel("Joint angle (deg)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # gripper (0–1)
    ax2.plot(t, qpos[:, 6], color="tab:orange", label="Gripper")
    ax2.set_ylabel("Gripper (0=open, 1=closed)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Joints  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path, help="Path to episode HDF5 file")
    args = parser.parse_args()

    path = args.hdf5.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    out_dir = path.parent / "debug_output"
    out_dir.mkdir(exist_ok=True)
    stem = path.stem

    qpos, base, wrist, task = load_episode(path)
    print_stats(path, qpos, base, task)
    make_video(out_dir / f"{stem}_video.mp4", base, wrist, qpos)
    make_joint_plot(out_dir / f"{stem}_joints.png", qpos)
    print("\nDone.")


if __name__ == "__main__":
    main()