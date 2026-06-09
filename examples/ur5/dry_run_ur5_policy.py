"""
Dry-run the UR5 policy with real sensors and a remote policy server.

This script is intentionally read-only for the robot:
  - reads UR5 joints through RTDE receive
  - reads base/wrist cameras
  - sends observations to the policy server
  - prints returned actions

It does not send robot or gripper commands.

Example:
    python examples/ur5/dry_run_ur5_policy.py \
      --host 100.112.30.96 \
      --prompt "pick up the banana and place it in the blue plate"
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy
import rtde_receive


ROBOT_IP = "192.168.0.101"
BASE_CAMERA_INDEX = 22
WRIST_CAMERA_INDEX = 4


def open_camera(index: int, name: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name} camera at index {index}")
    return cap


def read_rgb(cap: cv2.VideoCapture, name: str) -> np.ndarray:
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read from {name} camera")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_observation(
    robot_r: rtde_receive.RTDEReceiveInterface,
    base_cap: cv2.VideoCapture,
    wrist_cap: cv2.VideoCapture,
    prompt: str,
    gripper_position: float,
) -> dict:
    joints = np.asarray(robot_r.getActualQ(), dtype=np.float32)
    gripper = np.asarray([gripper_position], dtype=np.float32)
    state = np.concatenate([joints, gripper], axis=0)

    return {
        "observation/state": state,
        "observation/image": read_rgb(base_cap, "base"),
        "observation/wrist_image": read_rgb(wrist_cap, "wrist"),
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5 real-sensor policy dry-run; no robot motion.")
    parser.add_argument("--host", required=True, help="Policy server host, e.g. GPU machine Tailscale IP.")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
    parser.add_argument("--prompt", required=True, help="Language instruction to send to the policy.")
    parser.add_argument("--robot-ip", default=ROBOT_IP, help="UR5 robot IP.")
    parser.add_argument("--base-camera", type=int, default=BASE_CAMERA_INDEX, help="Base camera index.")
    parser.add_argument("--wrist-camera", type=int, default=WRIST_CAMERA_INDEX, help="Wrist camera index.")
    parser.add_argument(
        "--gripper-position",
        type=float,
        default=0.0,
        help="Manual gripper state in [0, 1] for dry-run. 0=open, 1=closed.",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Number of policy calls to make.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between policy calls.")
    args = parser.parse_args()

    gripper_position = float(np.clip(args.gripper_position, 0.0, 1.0))

    print(f"Connecting to policy server at {args.host}:{args.port}...")
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Policy server connected.")
    print("Server metadata:", policy.get_server_metadata())

    robot_r = base_cap = wrist_cap = None
    try:
        print(f"Connecting to UR5 receive interface at {args.robot_ip}...")
        robot_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        print("UR5 receive connected.")

        print(f"Opening base camera {args.base_camera} and wrist camera {args.wrist_camera}...")
        base_cap = open_camera(args.base_camera, "base")
        wrist_cap = open_camera(args.wrist_camera, "wrist")
        print("Cameras opened.")

        for i in range(args.iterations):
            obs = build_observation(robot_r, base_cap, wrist_cap, args.prompt, gripper_position)

            print(f"\nDry-run inference {i + 1}/{args.iterations}")
            print("state:", np.array2string(obs["observation/state"], precision=4))
            print("base image:", obs["observation/image"].shape, obs["observation/image"].dtype)
            print("wrist image:", obs["observation/wrist_image"].shape, obs["observation/wrist_image"].dtype)

            start = time.time()
            out = policy.infer(obs)
            elapsed = time.time() - start
            actions = np.asarray(out["actions"])

            print("actions shape:", actions.shape)
            print("first action:", np.array2string(actions[0], precision=5))
            print("policy_timing:", out.get("policy_timing"))
            print("server_timing:", out.get("server_timing"))
            print(f"client round trip: {elapsed * 1000:.1f} ms")

            if i + 1 < args.iterations:
                time.sleep(args.sleep)
    finally:
        if robot_r is not None:
            robot_r.disconnect()
        if base_cap is not None:
            base_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()


if __name__ == "__main__":
    main()
