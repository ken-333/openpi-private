"""
UR5 closed-loop inference using a remote OpenPI policy server.

This follows the structure of the UR3 TCP-velocity inference example, but adapts
the robot interface to this UR5 dataset:
  - observation/state is [6 joints, gripper]
  - policy action is [6 absolute joint targets, gripper]
  - arm execution uses servoJ, not speedL

Safety defaults:
  - dry-run by default; pass --execute-arm to move the robot
  - gripper dry-run by default; pass --execute-gripper to command it
  - q quits, p pauses/resumes, h moves home when arm execution is enabled
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import socket
import threading
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy
import rtde_control
import rtde_receive

from go_home import HOME_JOINTS_DEG


ROBOT_IP = "192.168.0.101"
BASE_CAMERA_INDEX = 22
WRIST_CAMERA_INDEX = 4
GRIPPER_PORT = 63352

CONTROL_HZ = 20
CHUNK_STEPS = 5
MAX_JOINT_DELTA = 0.03
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
GRIPPER_THRESHOLD = 0.5
GRIPPER_FREEZE_TIME = 0.8
HOME_SPEED = 0.3
HOME_ACCEL = 0.2


class CameraThread(threading.Thread):
    def __init__(self, base_index: int, wrist_index: int):
        super().__init__(daemon=True)
        self.base_index = base_index
        self.wrist_index = wrist_index
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.base_rgb = None
        self.wrist_rgb = None
        self.base_bgr = None
        self.wrist_bgr = None
        self.base_ts = 0.0
        self.wrist_ts = 0.0

    def run(self) -> None:
        base = cv2.VideoCapture(self.base_index)
        wrist = cv2.VideoCapture(self.wrist_index)
        for cap in (base, wrist):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not base.isOpened():
            raise RuntimeError(f"Could not open base camera {self.base_index}")
        if not wrist.isOpened():
            raise RuntimeError(f"Could not open wrist camera {self.wrist_index}")
        try:
            while not self.stop_event.is_set():
                ret_b, bgr_b = base.read()
                ret_w, bgr_w = wrist.read()
                now = time.time()
                if not ret_b or not ret_w:
                    time.sleep(0.01)
                    continue
                rgb_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)
                rgb_w = cv2.cvtColor(bgr_w, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.base_bgr = bgr_b
                    self.wrist_bgr = bgr_w
                    self.base_rgb = rgb_b
                    self.wrist_rgb = rgb_w
                    self.base_ts = now
                    self.wrist_ts = now
        finally:
            base.release()
            wrist.release()

    def latest(self):
        with self.lock:
            if self.base_rgb is None or self.wrist_rgb is None:
                return None
            return {
                "base_rgb": self.base_rgb.copy(),
                "wrist_rgb": self.wrist_rgb.copy(),
                "base_bgr": self.base_bgr.copy(),
                "wrist_bgr": self.wrist_bgr.copy(),
                "base_ts": self.base_ts,
                "wrist_ts": self.wrist_ts,
            }

    def stop(self) -> None:
        self.stop_event.set()


class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = False
        self.paused = True
        self.request_home = False
        self.mode = "PAUSED"
        self.step = 0
        self.chunk_idx = 0
        self.infer_ms = 0.0
        self.loop_ms = 0.0
        self.last_joints = None
        self.last_delta = None
        self.last_action = None
        self.last_gripper = None
        self.gripper_state = None
        self.status = "starting"

    def update(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.__dict__, lock=None)


def _gripper_cmd(sock, cmd: str) -> str:
    sock.sendall((cmd + "\n").encode())
    return sock.recv(1024).decode().strip()


def connect_gripper(robot_ip: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect((robot_ip, GRIPPER_PORT))
    _gripper_cmd(sock, "SET ACT 1")
    _gripper_cmd(sock, "SET GTO 1")
    while _gripper_cmd(sock, "GET STA") != "STA 3":
        time.sleep(0.1)
    return sock


def set_gripper(sock, position: float) -> None:
    pos = int(np.clip(position, 0.0, 1.0) * 255)
    _gripper_cmd(sock, "SET SPE 255")
    _gripper_cmd(sock, "SET FOR 150")
    _gripper_cmd(sock, f"SET POS {pos}")
    _gripper_cmd(sock, "SET GTO 1")


def gripper_target(raw_value: float, threshold: float) -> float:
    return 1.0 if raw_value >= threshold else 0.0


def clip_joint_target(current: np.ndarray, target: np.ndarray, max_delta: float) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(target[:6], dtype=np.float64) - current
    clipped = np.clip(delta, -max_delta, max_delta)
    return current + clipped, clipped


def move_home(robot_c) -> None:
    home = [math.radians(j) for j in HOME_JOINTS_DEG]
    robot_c.moveJ(home, HOME_SPEED, HOME_ACCEL)


def build_observation(robot_r, frames: dict, prompt: str, gripper_position: float) -> dict:
    joints = np.asarray(robot_r.getActualQ(), dtype=np.float32)
    gripper = np.asarray([gripper_position], dtype=np.float32)
    return {
        "observation/state": np.concatenate([joints, gripper], axis=0),
        "observation/image": frames["base_rgb"],
        "observation/wrist_image": frames["wrist_rgb"],
        "prompt": prompt,
    }


def control_loop(args, cam: CameraThread, state: RuntimeState) -> None:
    policy = robot_r = robot_c = gripper = None
    action_buffer = deque()
    gripper_position = gripper_target(float(np.clip(args.gripper_position, 0.0, 1.0)), args.gripper_threshold)
    gripper_lock_until = 0.0
    dt = 1.0 / args.control_hz

    try:
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        robot_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        robot_c = rtde_control.RTDEControlInterface(args.robot_ip) if args.execute_arm else None
        if args.execute_gripper:
            gripper = connect_gripper(args.robot_ip)

        state.update(status="ready", mode="PAUSED")
        while not state.snapshot()["stop"]:
            loop_start = time.perf_counter()
            snap = state.snapshot()

            if snap["request_home"]:
                state.update(request_home=False, paused=True, mode="HOME", status="moving home")
                action_buffer.clear()
                if robot_c is not None:
                    move_home(robot_c)
                state.update(mode="PAUSED", status="home done")
                continue

            if snap["paused"]:
                state.update(mode="PAUSED")
                time.sleep(0.05)
                continue

            if robot_r.getRobotMode() != 7:
                action_buffer.clear()
                state.update(status="robot not ready", mode="WAIT")
                time.sleep(0.5)
                continue

            frames = cam.latest()
            if frames is None:
                state.update(status="waiting for cameras", mode="WAIT")
                time.sleep(0.02)
                continue

            if not action_buffer:
                obs = build_observation(robot_r, frames, args.prompt, gripper_position)
                t0 = time.perf_counter()
                out = policy.infer(obs)
                infer_ms = (time.perf_counter() - t0) * 1000
                actions = np.asarray(out["actions"], dtype=np.float64)
                for action in actions[: args.chunk_steps]:
                    action_buffer.append(action)
                state.update(infer_ms=infer_ms, chunk_idx=0)

            action = action_buffer.popleft()
            current = np.asarray(robot_r.getActualQ(), dtype=np.float64)
            target, delta = clip_joint_target(current, action[:6], args.max_joint_delta)
            raw_gripper = float(np.clip(action[6], 0.0, 1.0))
            predicted_gripper = gripper_target(raw_gripper, args.gripper_threshold)

            now = time.perf_counter()
            gripper_locked = now < gripper_lock_until

            if args.execute_arm and robot_c is not None and not gripper_locked:
                robot_c.servoJ(target.tolist(), 0.0, 0.0, dt, args.servo_lookahead, args.servo_gain)

            if args.execute_gripper and gripper is not None and not gripper_locked:
                if predicted_gripper != gripper_position:
                    set_gripper(gripper, predicted_gripper)
                    gripper_position = predicted_gripper
                    gripper_lock_until = now + args.gripper_freeze_time
                    action_buffer.clear()

            state.update(
                mode="RUN" if args.execute_arm else "DRY",
                status="ok" if not gripper_locked else "gripper freeze",
                step=state.snapshot()["step"] + 1,
                chunk_idx=args.chunk_steps - len(action_buffer),
                loop_ms=(time.perf_counter() - loop_start) * 1000,
                last_joints=current,
                last_delta=delta,
                last_action=action,
                last_gripper=raw_gripper,
                gripper_state=gripper_position,
            )

            sleep_t = dt - (time.perf_counter() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)
    except Exception as exc:
        state.update(status=f"ERROR: {exc}", paused=True, mode="ERROR")
        raise
    finally:
        if robot_c is not None:
            try:
                robot_c.servoStop()
            except Exception:
                pass
            try:
                robot_c.stopScript()
            except Exception:
                pass
            robot_c.disconnect()
        if robot_r is not None:
            robot_r.disconnect()
        if gripper is not None:
            gripper.close()


def draw_overlay(frame, title: str, snap: dict, prompt: str) -> np.ndarray:
    out = frame.copy()
    lines = [
        title,
        f"mode={snap['mode']} status={snap['status']} step={snap['step']} chunk={snap['chunk_idx']}",
        f"infer={snap['infer_ms']:.1f}ms loop={snap['loop_ms']:.1f}ms",
        prompt[:80],
        "keys: p pause/resume | h home | q quit",
    ]
    if snap["last_delta"] is not None:
        lines.append("delta=" + np.array2string(np.asarray(snap["last_delta"]), precision=4))
    if snap["last_gripper"] is not None:
        lines.append(f"gripper_raw={snap['last_gripper']:.3f} committed={snap['gripper_state']}")
    y = 25
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        y += 24
    return out


def display_loop(cam: CameraThread, state: RuntimeState, prompt: str) -> None:
    while True:
        snap = state.snapshot()
        if snap["stop"]:
            break
        frames = cam.latest()
        if frames is not None:
            base = draw_overlay(frames["base_bgr"], "base", snap, prompt)
            wrist = draw_overlay(frames["wrist_bgr"], "wrist", snap, prompt)
            cv2.imshow("ur5 base", base)
            cv2.imshow("ur5 wrist", wrist)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            state.update(stop=True)
            break
        if key == ord("p"):
            paused = not state.snapshot()["paused"]
            state.update(paused=paused, mode="PAUSED" if paused else "RUN")
        if key == ord("h"):
            state.update(request_home=True)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5 servoJ policy inference with camera display.")
    parser.add_argument("--host", required=True, help="Policy server host.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--base-camera", type=int, default=BASE_CAMERA_INDEX)
    parser.add_argument("--wrist-camera", type=int, default=WRIST_CAMERA_INDEX)
    parser.add_argument("--gripper-position", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    parser.add_argument("--max-joint-delta", type=float, default=MAX_JOINT_DELTA)
    parser.add_argument("--servo-lookahead", type=float, default=SERVO_LOOKAHEAD)
    parser.add_argument("--servo-gain", type=int, default=SERVO_GAIN)
    parser.add_argument("--gripper-threshold", type=float, default=GRIPPER_THRESHOLD)
    parser.add_argument("--gripper-freeze-time", type=float, default=GRIPPER_FREEZE_TIME)
    parser.add_argument("--execute-arm", action="store_true")
    parser.add_argument("--execute-gripper", action="store_true")
    args = parser.parse_args()

    cam = CameraThread(args.base_camera, args.wrist_camera)
    state = RuntimeState()
    cam.start()
    worker = threading.Thread(target=control_loop, args=(args, cam, state), daemon=True)
    worker.start()

    print("Camera windows will open. Press p to start/pause, q to quit, h to home.")
    try:
        display_loop(cam, state, args.prompt)
    finally:
        state.update(stop=True)
        worker.join(timeout=3)
        cam.stop()
        cam.join(timeout=1)


if __name__ == "__main__":
    main()
