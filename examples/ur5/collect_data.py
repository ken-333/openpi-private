"""
SpaceMouse + UR5 data collection script.
Task: pick and place with language conditioning.

Usage:
    python collect_data.py <task_queue.json>

Controls:
    SpaceMouse axes  → UR5 TCP velocity
    Button LEFT      → gripper close
    Button RIGHT     → gripper open
    Keyboard 's'     → start / stop recording
    Keyboard 'y'     → (REVIEW) save episode
    Keyboard 'n'     → (REVIEW) discard episode
    Keyboard 'q'     → quit
"""

import rtde_control
import rtde_receive
import cv2
import h5py # 用于保存数据的 HDF5 文件格式库，适合存储大量结构化数据，如图像和关节状态等。
import numpy as np
import time
import threading
import ctypes
from collections import defaultdict
from threading import Thread, Event
from pathlib import Path
import socket
import os
import json
import sys
from datetime import datetime

# spnav 0.9 (PyPI) is Python-2 only: at import it binds PyCObject_AsVoidPtr
# (removed in CPython 3.3), used only by the unused X11 path. Stub it for py3.
try:
    ctypes.pythonapi.PyCObject_AsVoidPtr
except AttributeError:
    class _PyCObjectStub:
        restype = None
        argtypes = None
    ctypes.pythonapi.PyCObject_AsVoidPtr = _PyCObjectStub()

from spnav import (
    spnav_open, spnav_poll_event, spnav_close,
    SpnavMotionEvent, SpnavButtonEvent,
)


ROBOT_IP = "192.168.0.101"             # change to your robot's IP
CONTROL_HZ = 150                       # motion command rate (speedL) — high = smooth
RECORD_HZ = 25                         # preview + episode recording rate (matches achievable loop rate)
SAVE_DIR = Path("./collected_data")    # directory to save episodes
SPACEMOUSE_DEADZONE = 0.3             # threshold for noise cancellation
GRIPPER_PORT = 63352

SCALE_FACTOR = 0.1     # velocity scale (same as teleop_ur5_spnav.py)
SPNAV_MAX_VALUE = 300  # 300 wired SpaceMouse, 500 wireless


class Spacemouse(Thread):
    """Verbatim from teleop_ur5_spnav.py: threaded spnav reader with the
    tx_zup_spnav frame transform + constructor deadzone."""

    def __init__(self, max_value=SPNAV_MAX_VALUE, deadzone=(0, 0, 0, 0, 0, 0), dtype=np.float32):
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()
        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=dtype)

    def get_motion_state(self):
        me = self.motion_event
        state = np.array(me.translation + me.rotation,
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        tf_state = tf_state * SCALE_FACTOR
        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1 / 200)
        finally:
            spnav_close()

# 4: def connect_robot(ip)  创建一个到机器人的网络连接，并封装成对象
#   创建并返回 rtde_control 和 rtde_receive 两个接口对象
def connect_robot(ip):
    robot_control = rtde_control.RTDEControlInterface(ip)
    robot_receive = rtde_receive.RTDEReceiveInterface(ip)
    return robot_control, robot_receive


# 5: def connect_cameras()
#   打开 base 和 wrist 两个摄像头 (cv2.VideoCapture)
#   返回一个 dict: {"base": cap0, "wrist": cap1}
def connect_cameras():
    camera = {
        "base": cv2.VideoCapture(22),   # RealSense color stream (YUYV)
        "wrist": cv2.VideoCapture(10)  # RealSense color stream (YUYV)
    }                                  #为何用dict？因为有两个摄像头，使用dict可以更清晰地管理它们，并通过键名访问对应的摄像头对象。
    for cap in camera.values():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only newest frame → less lag
    if not camera["base"].isOpened():
        print("Error: Could not open base camera.")
        exit(1)
    if not camera["wrist"].isOpened():
        print("Error: Could not open wrist camera.")
        exit(1)
    return camera


# 6: def read_cameras(cameras)
#   从两个摄像头各读一帧
#   返回 dict: {"base": img, "wrist": img}
def read_cameras(cameras):
    frames = {}
    for cam_name, cap in cameras.items():
        ret, frame = cap.read()
        if not ret: # ret 是一个布尔值，表示是否成功读取到帧
            print(f"Error: Could not read from {cam_name} camera.")
            exit(1)
        else:
            frames[cam_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frames


class CameraReader:
    """Grabs the newest frame from each camera in a background thread, so the
    control loop never blocks on cap.read(). Control loop just reads the buffer."""

    def __init__(self, cameras):
        self.cameras = cameras              # {"base": cap, "wrist": cap}
        self.lock = threading.Lock()        # protects self.latest (shared buffer)
        self.latest = {}                    # newest RGB frame per camera
        self.latest_ts = {}                 # capture time (time.time()) per camera
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        # Runs on its own thread: keep grabbing frames as fast as the cameras allow.
        while self.running:
            for name, cap in self.cameras.items():
                ret, frame = cap.read()                  # blocking, but only HERE
                if ret:
                    ts = time.time()                     # stamp at capture, for sync auditing
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.latest[name] = rgb
                        self.latest_ts[name] = ts

    def start(self):
        self.running = True
        self.thread.start()
        # wait until every camera has produced at least one frame
        while True:
            with self.lock:
                if self.latest.keys() == self.cameras.keys():
                    break
            time.sleep(0.01)
        return self

    def read(self):
        # Instant: hand back the most recent frames + their capture times (no camera I/O here).
        with self.lock:
            return dict(self.latest), dict(self.latest_ts)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


def _gripper_cmd(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    return sock.recv(1024).decode().strip()


#7: def connect_gripper()
#   连接夹爪 (Robotiq URCap text protocol over socket, port 63352)
def connect_gripper():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect((ROBOT_IP, GRIPPER_PORT))

    _gripper_cmd(sock, "SET ACT 1")   # activate
    _gripper_cmd(sock, "SET GTO 1")   # act on position commands

    while _gripper_cmd(sock, "GET STA") != "STA 3":   # wait until activated
        time.sleep(0.1)

    print("Gripper connected and activated")
    return sock
    

# TODO 8: def set_gripper(position)
#   控制夹爪开合 (先留空, 写 pass)
def set_gripper(gripper, position):
    position = np.clip(position, 0.0, 1.0)
    pos = int(position * 255)
    _gripper_cmd(gripper, "SET SPE 255")
    _gripper_cmd(gripper, "SET FOR 150")
    _gripper_cmd(gripper, f"SET POS {pos}")
    _gripper_cmd(gripper, "SET GTO 1")
    

# 9: class EpisodeRecorder
#   __init__: 初始化空列表 self.frames
#   add_frame(joints, gripper, images): 把这一帧数据 append 到 self.frames
#   save(path, task): 把 self.frames 写成 HDF5 文件
#   discard(): 清空 self.frames
#   __len__: 返回帧数
class EpisodeRecorder:
    def __init__(self):
        self.joints = []
        self.gripper = []
        self.images = []
        self.timestamps = []   # sample time (joint-read time) per frame, seconds
        self.image_ages = []   # how stale the camera frames were vs the joint read, seconds

    def add_frame(self, joints, gripper, images, timestamp, image_age):
        self.joints.append(joints)
        self.gripper.append(gripper)
        self.images.append(images)
        self.timestamps.append(timestamp)
        self.image_ages.append(image_age)


    def save(self, path, task):
        if len(self.images) < 2:
            print("Episode too short (need at least 2 frames), discarding.")
            self.discard()
            return False
        joints = np.array(self.joints)
        gripper = np.array(self.gripper).reshape(-1,1)
        qpos = np.concatenate([joints, gripper], axis=1) # 将关节状态和夹爪状态合并成一个数组

        action = qpos[1:] # 动作序列，等于状态序列向前移动一帧，因为动作是从当前状态到下一状态的变化    训练 config 里有 DeltaActions transform，它会在训练时自动转成 delta
        qpos = qpos[:-1] # 状态序列，去掉最后一帧，因为它没有对应的动作

        # timestamps/ages align with the STATE samples (qpos), so drop the last like qpos
        timestamps = np.array(self.timestamps)[:-1]
        image_ages = np.array(self.image_ages)[:-1]

        base_images = []
        wrist_images = []
        for frame in self.images:
            base_images.append(frame["base"])
            wrist_images.append(frame["wrist"])
        base_images = np.array(base_images)[:-1] # 去掉最后一帧，因为它没有对应的动作
        wrist_images = np.array(wrist_images)[:-1]

        with h5py.File(path, 'w') as f:
            f.create_dataset('task', data=task) # 创建一个名为 'task' 的数据集，存储任务指令
            f.create_dataset('action', data=action)
            f.create_dataset('observation/qpos', data=qpos)
            f.create_dataset('observation/images/base', data = base_images )
            f.create_dataset('observation/images/wrist', data = wrist_images )
            f.create_dataset('timestamp', data=timestamps)   # per-frame sample time (s)
            f.create_dataset('image_age', data=image_ages)   # image-vs-state lag (s)
        return True

    def discard(self):
        self.joints = []
        self.gripper = []
        self.images = []
        self.timestamps = []
        self.image_ages = []

    def __len__(self):  #调用 len() 时返回帧数
        return len(self.images) # 以帧数为长度


def load_task_queue(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["session_name"], data["tasks"], data.get("sampling_strategy", "sequential")


def scan_existing_episodes(save_dir, tasks):
    prompt_to_idx = {t["prompt"]: i for i, t in enumerate(tasks)}
    task_counts = {i: 0 for i in range(len(tasks))}
    next_episode_id = 0
    files = sorted(Path(save_dir).glob("episode_*.hdf5"))
    for f in files:
        ep_num = int(f.stem.split("_")[1])
        next_episode_id = max(next_episode_id, ep_num + 1)
        with h5py.File(f, "r") as hf:
            raw = hf["task"][()]
            prompt = raw.decode() if isinstance(raw, bytes) else str(raw)
        if prompt in prompt_to_idx:
            task_counts[prompt_to_idx[prompt]] += 1
    return task_counts, next_episode_id


def load_progress(save_dir, tasks):
    manifest_path = Path(save_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                entries = json.load(f)
            task_counts = {i: 0 for i in range(len(tasks))}
            manifest_next_id = 0
            for entry in entries:
                if entry.get("status") == "saved":
                    idx = entry["task_index"]
                    if idx in task_counts:
                        task_counts[idx] += 1
                    ep_num = int(entry["episode_id"].split("_")[1])
                    manifest_next_id = max(manifest_next_id, ep_num + 1)
            # cross-check: 以磁盘实际 HDF5 文件的最大编号为准，防止崩溃后覆盖已有文件
            _, hdf5_next_id = scan_existing_episodes(save_dir, tasks)
            next_episode_id = max(manifest_next_id, hdf5_next_id)
            if next_episode_id > manifest_next_id:
                print(f"Warning: HDF5 files beyond manifest detected up to "
                      f"episode_{next_episode_id - 1:04d}. Those IDs will be skipped.")
            return task_counts, next_episode_id
        except Exception:
            print("Manifest damaged, rebuilding from HDF5 files...")
    else:
        print("No manifest found, scanning existing episodes...")
    return scan_existing_episodes(save_dir, tasks)


def write_manifest_entry(save_dir, entry):
    manifest_path = Path(save_dir) / "manifest.json"
    entries = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            entries = json.load(f)
    entries.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)


def overlay_info(frame_bgr, state, prompt, task_count, target, episode_id):
    frame = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = {"IDLE": (200, 200, 200), "RECORDING": (0, 0, 255), "REVIEW": (0, 165, 255)}.get(state, (255, 255, 255))
    hint = {"IDLE": "S: start", "RECORDING": "S: stop", "REVIEW": "Y: save  N: discard"}.get(state, "")
    lines = [
        prompt[:55] + ("..." if len(prompt) > 55 else ""),
        f"Progress: {task_count} / {target}   Episode: {episode_id:04d}   [{state}]",
        hint,
    ]
    if state == "REVIEW":
        lines.append("Save this episode? [Y/N]")
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), font, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), font, 0.6, color, 1)
        y += 28
    return frame


# 10: def run_collection_loop(robot_c, robot_r, cameras, recorder)
#   主循环:
#     - 读 SpaceMouse
#     - apply_deadzone
#     - 乘 scale → 速度向量
#     - 发给机器人 (robot_c.speedl)
#     - 读关节角 (robot_r.getActualQ)
#     - 读摄像头
#     - recorder.add_frame(...)
#     - 处理键盘/按钮输入 (save / discard / quit)
#     - 控制频率 (time.sleep)
def run_collection_loop(robot_c, robot_r, cam_reader, gripper, recorder, tasks, task_counts, next_episode_id, save_dir, sampling_strategy="sequential"):
    sm = Spacemouse(deadzone=SPACEMOUSE_DEADZONE)
    sm.start()
    try:
        acceleration = 0.5
        speed_time = 0.1
        dt = 1.0 / CONTROL_HZ
        record_every = max(1, round(CONTROL_HZ / RECORD_HZ))

        state = "IDLE"
        current_task_idx = 0
        episode_id = next_episode_id
        t_record_start = 0.0

        # skip already-complete tasks on startup
        while current_task_idx < len(tasks) and \
              task_counts.get(current_task_idx, 0) >= tasks[current_task_idx]["target_episodes"]:
            current_task_idx += 1
        if current_task_idx >= len(tasks):
            print("All tasks complete!")
            return

        task = tasks[current_task_idx]
        print(f"\nCurrent task [{current_task_idx}]: {task['prompt']}")
        print(f"Progress: {task_counts.get(current_task_idx, 0)} / {task['target_episodes']}")
        print("Press S to start recording, Q to quit.\n")

        gripper_state = 0.0
        prev_buttons = [False, False]
        loop_count = 0

        while True:
            # motion path: skip speedL during REVIEW to hold robot still
            if state != "REVIEW":
                velocity = sm.get_motion_state_transformed()
                robot_c.speedL(velocity.tolist(), acceleration, speed_time)

            b0, b1 = sm.is_button_pressed(0), sm.is_button_pressed(1)
            if state != "REVIEW":
                if b0 and not prev_buttons[0]:
                    set_gripper(gripper, 1.0)
                    gripper_state = 1.0
                if b1 and not prev_buttons[1]:
                    set_gripper(gripper, 0.0)
                    gripper_state = 0.0
            prev_buttons = [b0, b1]

            if loop_count % record_every == 0:
                t_sample = time.time()
                joints_state = robot_r.getActualQ()
                frames, frame_ts = cam_reader.read()

                base_bgr = cv2.cvtColor(frames["base"], cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(frames["wrist"], cv2.COLOR_RGB2BGR)
                cv2.imshow("base", overlay_info(base_bgr, state, task["prompt"],
                                                task_counts.get(current_task_idx, 0),
                                                task["target_episodes"], episode_id))
                cv2.imshow("wrist", overlay_info(wrist_bgr, state, task["prompt"],
                                                 task_counts.get(current_task_idx, 0),
                                                 task["target_episodes"], episode_id))
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Quitting data collection.")
                    break
                elif state == "IDLE":
                    if key == ord('s'):
                        state = "RECORDING"
                        t_record_start = time.time()
                        print(f"Recording episode {episode_id:04d} | {task['prompt']}")
                elif state == "RECORDING":
                    if key == ord('s'):
                        state = "REVIEW"
                        robot_c.speedStop()
                        print(f"\nSave this episode? [Y/N]  ({len(recorder)} frames)")
                elif state == "REVIEW":
                    if key == ord('y'):
                        path = save_dir / f"episode_{episode_id:04d}.hdf5"
                        num_frames = max(0, len(recorder) - 1)
                        duration = time.time() - t_record_start
                        if not recorder.save(path, task["prompt"]):
                            state = "IDLE"
                            continue
                        write_manifest_entry(save_dir, {
                            "episode_id": f"episode_{episode_id:04d}",
                            "task_index": current_task_idx,
                            "prompt": task["prompt"],
                            "object": task.get("object", ""),
                            "plate_color": task.get("plate_color", ""),
                            "num_frames": num_frames,
                            "duration_s": round(duration, 2),
                            "saved_at": datetime.now().isoformat(),
                            "status": "saved",
                        })
                        task_counts[current_task_idx] = task_counts.get(current_task_idx, 0) + 1
                        episode_id += 1
                        recorder.discard()
                        print(f"Saved → episode_{episode_id - 1:04d}.hdf5  "
                              f"({task_counts[current_task_idx]} / {task['target_episodes']})")

                        # advance to next incomplete task
                        n = len(tasks)
                        if sampling_strategy == "round_robin":
                            next_idx = (current_task_idx + 1) % n
                            for _ in range(n):
                                if task_counts.get(next_idx, 0) < tasks[next_idx]["target_episodes"]:
                                    break
                                next_idx = (next_idx + 1) % n
                            else:
                                print("\nAll tasks complete!")
                                break
                            current_task_idx = next_idx
                        else:
                            while current_task_idx < n and \
                                  task_counts.get(current_task_idx, 0) >= tasks[current_task_idx]["target_episodes"]:
                                current_task_idx += 1
                            if current_task_idx >= n:
                                print("\nAll tasks complete!")
                                break

                        task = tasks[current_task_idx]
                        print(f"\nNext task [{current_task_idx}]: {task['prompt']}")
                        print(f"Progress: {task_counts.get(current_task_idx, 0)} / {task['target_episodes']}")
                        state = "IDLE"
                    elif key == ord('n'):
                        recorder.discard()
                        print("Episode discarded. Retrying current task.")
                        state = "IDLE"

                if state == "RECORDING":
                    image_age = t_sample - min(frame_ts.values())
                    recorder.add_frame(joints_state, gripper_state, frames, t_sample, image_age)

            loop_count += 1
            time.sleep(dt)
    finally:
        sm.stop()


def main():
    if len(sys.argv) < 2: # 检查是否提供了命令行参数，如果没有，则打印使用说明并退出程序。
        print("Usage: python collect_data.py <task_queue.json>")
        sys.exit(1)

    json_path = sys.argv[1]  # 从命令行参数获取任务队列的 JSON 文件路径
    session_name, tasks, sampling_strategy = load_task_queue(json_path)
    os.makedirs(SAVE_DIR, exist_ok=True) 
    task_counts, next_episode_id = load_progress(SAVE_DIR, tasks)

    print(f"\nSession: {session_name}")
    for i, t in enumerate(tasks): # 打印每个任务的进度，显示已完成的 episode 数量和目标 episode 数量
        count = task_counts.get(i, 0)
        status = "DONE" if count >= t["target_episodes"] else f"{count}/{t['target_episodes']}" 
        print(f"  [{i}] {t['prompt']}  —  {status}") # 显示任务索引、提示语和完成状态（DONE 或 进度计数）

    if all(task_counts.get(i, 0) >= t["target_episodes"] for i, t in enumerate(tasks)):
        print("\nAll tasks complete. Nothing to collect.")
        return

    robot_control = robot_receive = cameras = cam_reader = gripper = None
    try:
        print("\nConnecting to robot...")
        robot_control, robot_receive = connect_robot(ROBOT_IP)
        print("Robot connected.")
        print("Connecting to cameras...")
        cameras = connect_cameras()
        cam_reader = CameraReader(cameras).start()
        print("Cameras connected.")
        print("Connecting to gripper...")
        gripper = connect_gripper()
        print("All devices ready. Starting collection loop.")
        recorder = EpisodeRecorder()
        run_collection_loop(robot_control, robot_receive, cam_reader, gripper, recorder,
                            tasks, task_counts, next_episode_id, SAVE_DIR, sampling_strategy)
    finally:
        if robot_control is not None:
            try:
                robot_control.speedStop()
            except Exception as e:
                print(f"speedStop failed: {e}")
            try:
                robot_control.stopScript()
            except Exception as e:
                print(f"stopScript failed: {e}")
            try:
                robot_control.disconnect()
            except Exception as e:
                print(f"control disconnect failed: {e}")
        if robot_receive is not None:
            try:
                robot_receive.disconnect()
            except Exception as e:
                print(f"receive disconnect failed: {e}")
        if gripper is not None:
            try:
                gripper.close()
            except Exception as e:
                print(f"gripper disconnect failed: {e}")
        if cam_reader is not None:
            cam_reader.stop()
        if cameras is not None:
            for name, cap in cameras.items():
                try:
                    cap.release()
                except Exception as e:
                    print(f"camera {name} release failed: {e}")
            
    cv2.destroyAllWindows()

    print("All connections closed. Goodbye!")

if __name__ == "__main__":
    main()  