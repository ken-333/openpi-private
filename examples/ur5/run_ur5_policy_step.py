"""
Run the UR5 policy one confirmed step at a time.

This is the first real-motion test after dry_run_ur5_policy.py.
It is intentionally conservative:
  - asks for Enter before every motion
  - executes only actions[0] from each predicted chunk
  - clips per-joint motion to a small delta
  - uses low moveJ speed/acceleration
  - prints gripper output but does not command the gripper by default

Example:
    python examples/ur5/run_ur5_policy_step.py \
      --host 100.112.30.96 \
      --prompt "pick up the banana and place it in the blue plate"
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from openpi_client import websocket_client_policy
import rtde_control
import rtde_receive

from dry_run_ur5_policy import BASE_CAMERA_INDEX
from dry_run_ur5_policy import ROBOT_IP
from dry_run_ur5_policy import WRIST_CAMERA_INDEX
from dry_run_ur5_policy import build_observation
from dry_run_ur5_policy import open_camera


SAFE_SPEED = 0.08
SAFE_ACCEL = 0.08
MAX_JOINT_DELTA = 0.03


def clipped_joint_target(current_joints: np.ndarray, predicted_joints: np.ndarray, max_delta: float) -> np.ndarray:
    delta = predicted_joints - current_joints
    clipped_delta = np.clip(delta, -max_delta, max_delta) 
    return current_joints + clipped_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5 policy step runner with manual confirmation.")
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
        help="Manual gripper state in [0, 1]. 0=open, 1=closed.",
    )
    parser.add_argument("--steps", type=int, default=1, help="Maximum number of confirmed robot steps.")
    parser.add_argument("--speed", type=float, default=SAFE_SPEED, help="moveJ speed in rad/s.")
    parser.add_argument("--accel", type=float, default=SAFE_ACCEL, help="moveJ acceleration in rad/s^2.")
    parser.add_argument("--max-joint-delta", type=float, default=MAX_JOINT_DELTA, help="Max joint delta per step in rad.")
    args = parser.parse_args()

    gripper_position = float(np.clip(args.gripper_position, 0.0, 1.0))

    print(f"Connecting to policy server at {args.host}:{args.port}...")
    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("Policy server connected.")

    robot_c = robot_r = base_cap = wrist_cap = None
    try:
        print(f"Connecting to UR5 receive/control at {args.robot_ip}...")
        robot_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        robot_c = rtde_control.RTDEControlInterface(args.robot_ip)
        print("UR5 connected.")

        print(f"Opening base camera {args.base_camera} and wrist camera {args.wrist_camera}...")
        base_cap = open_camera(args.base_camera, "base")
        wrist_cap = open_camera(args.wrist_camera, "wrist")
        print("Cameras opened.")

        for step_idx in range(args.steps):
            obs = build_observation(robot_r, base_cap, wrist_cap, args.prompt, gripper_position)
            current_joints = np.asarray(obs["observation/state"][:6], dtype=np.float64)

            print(f"\nStep {step_idx + 1}/{args.steps}")
            print("current joints:", np.array2string(current_joints, precision=5))

            start = time.time()
            out = policy.infer(obs)
            elapsed = time.time() - start
            actions = np.asarray(out["actions"], dtype=np.float64)

            predicted = actions[0]
            predicted_joints = predicted[:6]
            predicted_gripper = float(np.clip(predicted[6], 0.0, 1.0))
            safe_joints = clipped_joint_target(current_joints, predicted_joints, args.max_joint_delta)

            print("raw first action:", np.array2string(predicted, precision=5))
            print("safe joint target:", np.array2string(safe_joints, precision=5))
            print("joint delta:", np.array2string(safe_joints - current_joints, precision=5))
            print(f"gripper prediction clipped to [0, 1]: {predicted_gripper:.4f} (not executed)")
            print("server_timing:", out.get("server_timing"))
            print(f"client round trip: {elapsed * 1000:.1f} ms")

            answer = input("Press Enter to execute this safe joint step, or type q then Enter to quit: ").strip()
            if answer.lower() == "q":
                print("Stopped before executing step.")
                break

            robot_c.moveJ(safe_joints.tolist(), args.speed, args.accel)
            print("Step executed.")
    finally:
        if robot_c is not None:
            try:
                robot_c.stopScript()
            except Exception as e:
                print(f"stopScript failed: {e}")
            try:
                robot_c.disconnect()
            except Exception as e:
                print(f"control disconnect failed: {e}")
        if robot_r is not None:
            try:
                robot_r.disconnect()
            except Exception as e:
                print(f"receive disconnect failed: {e}")
        if base_cap is not None:
            base_cap.release()
        if wrist_cap is not None:
            wrist_cap.release()


if __name__ == "__main__":
    main()
