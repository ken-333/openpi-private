"""
Verify all collected HDF5 episodes in a directory.

Usage:
    python verify_episode.py [data_dir]      # default data_dir = collected_data

Processes every episode_*.hdf5 in data_dir (sequentially, one ~1GB episode
in memory at a time) and writes outputs under data_dir/verification/:
    verification/videos/<episode>.mp4   — base + wrist side-by-side with frame/joint overlay
    verification/joints/<episode>.png   — joint angle curves over time
"""

import argparse
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3", "Gripper"]
FPS = 20


def load_episode(path):
    with h5py.File(path, "r") as f:
        qpos   = f["observation/qpos"][:]        # (N, 7)
        base   = f["observation/images/base"][:]  # (N, H, W, 3) RGB
        wrist  = f["observation/images/wrist"][:] # (N, H, W, 3) RGB
        task   = f["task"][()].decode() if isinstance(f["task"][()], bytes) else str(f["task"][()])
        # timestamp/image_age only exist in episodes recorded after the timing patch
        timestamps = f["timestamp"][:] if "timestamp" in f else None
        image_ages = f["image_age"][:] if "image_age" in f else None
    return qpos, base, wrist, task, timestamps, image_ages


def effective_fps(timestamps):
    """Real frame rate from recorded timestamps; falls back to FPS for old episodes."""
    if timestamps is not None and len(timestamps) > 1:
        mean_dt = np.diff(timestamps).mean()
        if mean_dt > 0:
            return 1.0 / mean_dt
    return FPS


def time_axis(qpos, timestamps):
    """Seconds-from-start for each frame (real timestamps if present, else uniform FPS)."""
    if timestamps is not None and len(timestamps) > 1:
        return timestamps - timestamps[0]
    return np.arange(len(qpos)) / FPS


def timing_report(timestamps, image_ages):
    """Print rate / jitter / image-vs-state lag so you can judge temporal quality."""
    if timestamps is None or len(timestamps) < 2:
        print("  timing  : no timestamps (episode recorded before timing patch)")
        return
    dt = np.diff(timestamps)
    rate = 1.0 / dt.mean()
    print(f"  rate    : {rate:5.1f} Hz   dt mean {dt.mean()*1e3:5.1f} ms"
          f"  std {dt.std()*1e3:4.1f} ms  max {dt.max()*1e3:5.1f} ms")
    if dt.std() > 0.5 * dt.mean():
        print("  ⚠ high timing jitter — frame spacing is uneven (loop stalls?)")
    if image_ages is not None and len(image_ages):
        print(f"  img lag : mean {image_ages.mean()*1e3:5.1f} ms  max {image_ages.max()*1e3:5.1f} ms"
              f"  (camera frame age vs joint read)")
        if image_ages.max() > 2 * dt.mean():
            print("  ⚠ image lag exceeds 2 frame intervals — image/state may be misaligned")


def print_stats(path, qpos, base, task, timestamps, image_ages):
    n = len(qpos)
    h, w = base.shape[1:3]
    fps = effective_fps(timestamps)
    t = time_axis(qpos, timestamps)
    src = "measured" if timestamps is not None else f"assumed {FPS}Hz"
    print(f"\n--- Episode: {path.name} ---")
    print(f"  Task    : {task}")
    print(f"  Frames  : {n}  ({t[-1]:.1f}s, {fps:.1f}Hz {src})")
    print(f"  Image   : {w}×{h}")
    print(f"  joints min: {np.degrees(qpos[:, :6].min(axis=0)).round(1)} deg")
    print(f"  joints max: {np.degrees(qpos[:, :6].max(axis=0)).round(1)} deg")
    print(f"  gripper   : {qpos[:, 6].min():.2f} – {qpos[:, 6].max():.2f}")
    timing_report(timestamps, image_ages)
    print()


def make_video(out_path, base_imgs, wrist_imgs, qpos, fps):
    n, h, w, _ = base_imgs.shape
    frame_w = w * 2   # side by side
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,   # real recording rate → video plays at true speed
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


def make_joint_plot(out_path, qpos, t):
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
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("collected_data"),
                        help="Directory containing episode_*.hdf5 (default: collected_data)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    episodes = sorted(data_dir.glob("episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")

    video_dir  = data_dir / "verification" / "videos"
    joints_dir = data_dir / "verification" / "joints"
    video_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(episodes)} episode(s) in {data_dir}")
    failed = []
    for path in episodes:
        stem = path.stem  # e.g. "episode_0000" → keeps episode id in every output filename
        try:
            qpos, base, wrist, task, timestamps, image_ages = load_episode(path)
            print_stats(path, qpos, base, task, timestamps, image_ages)
            make_video(video_dir / f"{stem}.mp4", base, wrist, qpos, effective_fps(timestamps))
            make_joint_plot(joints_dir / f"{stem}.png", qpos, time_axis(qpos, timestamps))
        except Exception as e:
            print(f"  ERROR processing {path.name}: {e}")
            failed.append(path.name)

    print(f"\nDone. {len(episodes) - len(failed)}/{len(episodes)} episode(s) succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()