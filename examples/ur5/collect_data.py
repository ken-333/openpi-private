"""
SpaceMouse + UR5 data collection script.
Task: pick and place with language conditioning.

Controls:
    SpaceMouse axes  → UR5 TCP velocity
    Button LEFT      → gripper close
    Button RIGHT     → gripper open
    Keyboard 's'     → save episode
    Keyboard 'd'     → discard episode
    Keyboard 'q'     → quit
"""

import pyspacemouse 
import rtde_control 
import rtde_receive
import cv2
import h5py # 用于保存数据的 HDF5 文件格式库，适合存储大量结构化数据，如图像和关节状态等。
import numpy as np
import time
import threading
from pathlib import Path
import socket
import os


ROBOT_IP = "192.168.0.101"             # change to your robot's IP
CONTROL_HZ = 150                       # motion command rate (speedL) — high = smooth
RECORD_HZ = 30                         # preview + episode recording rate (~camera fps)
SAVE_DIR = Path("./collected_data")    # directory to save episodes
SPACEMOUSE_DEADZONE = 0.05             # threshold for noise cancellation
GRIPPER_PORT = 63352

# Motion control mapping (tuned in teleop_ur5.py)
AXIS_SIGN = {
    "x": +1, "y": +1, "z": +1,
    "roll": +0.3, "pitch": +0.3, "yaw": +2,
}
TRANS_SCALE = 0.1   # translation: m/s at full SpaceMouse deflection
ROT_SCALE   = 0.1   # rotation:    rad/s at full SpaceMouse deflection

# 3: def apply_deadzone(state, threshold)
#   输入: SpaceMouse 原始 6D 向量, 阈值
#   输出: 小于阈值的分量清零后的向量
def apply_deadzone(state, threshold):
    state = np.array(state)
    state[np.abs(state) < threshold] =0 # 将小于阈值的分量清零
    return state


def apply_dominant_axis(raw):
    # 只保留绝对值最大的那个轴，其余归零，模拟 3DxWare 主轴锁定
    idx = np.argmax(np.abs(raw))
    result = np.zeros(6)
    result[idx] = raw[idx]
    return result


def map_to_velocity(state):
    raw = np.array([
        AXIS_SIGN["x"]     * state.x,
        AXIS_SIGN["y"]     * state.y,
        AXIS_SIGN["z"]     * state.z,
        AXIS_SIGN["roll"]  * state.roll,
        AXIS_SIGN["pitch"] * state.pitch,
        AXIS_SIGN["yaw"]   * state.yaw,
    ])
    raw = apply_deadzone(raw, SPACEMOUSE_DEADZONE)
    raw[:3] *= TRANS_SCALE
    raw[3:] *= ROT_SCALE
    raw = apply_dominant_axis(raw)
    return raw

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
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        # Runs on its own thread: keep grabbing frames as fast as the cameras allow.
        while self.running:
            for name, cap in self.cameras.items():
                ret, frame = cap.read()                  # blocking, but only HERE
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.latest[name] = rgb

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
        # Instant: hand back the most recent frames (no camera I/O here).
        with self.lock:
            return dict(self.latest)

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
    
    def add_frame(self, joints, gripper, images):
        self.joints.append(joints)
        self.gripper.append(gripper)
        self.images.append(images)

    
    def save(self, path, task):
        if len(self.images) < 2:
            print("Episode too short (need at least 2 frames), discarding.")
            self.discard()
            return
        joints = np.array(self.joints)
        gripper = np.array(self.gripper).reshape(-1,1)
        qpos = np.concatenate([joints, gripper], axis=1) # 将关节状态和夹爪状态合并成一个数组
        
        action = qpos[1:] # 动作序列，等于状态序列向前移动一帧，因为动作是从当前状态到下一状态的变化    训练 config 里有 DeltaActions transform，它会在训练时自动转成 delta
        qpos = qpos[:-1] # 状态序列，去掉最后一帧，因为它没有对应的动作

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

    def discard(self):
        self.joints = []
        self.gripper = []
        self.images = []
    
    def __len__(self):  #调用 len() 时返回帧数
        return len(self.images) # 以帧数为长度

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
def run_collection_loop(robot_c, robot_r, cam_reader, gripper, recorder, task_instruction):
    with pyspacemouse.open() as device:
        acceleration = 0.5                  # 加速度 m/s^2 (在 teleop_ur5.py 中调好)
        speed_time = 0.1                    # speedL 持续时间
        dt = 1.0 / CONTROL_HZ               # 控制循环节流
        record_every = max(1, round(CONTROL_HZ / RECORD_HZ))  # 每 N 圈记录/预览一次
        episode_number = 0
        is_recording = False
        gripper_state = 0.0
        prev_buttons = [False, False]   # edge detection: fire gripper once per press
        loop_count = 0
        while True:
            # ---- motion path: runs EVERY loop at CONTROL_HZ (keep it lean!) ----
            state = device.read()
            velocity = map_to_velocity(state)
            robot_c.speedL(velocity.tolist(), acceleration, speed_time)

            # gripper: only send on the press transition (cheap, fine every loop)
            if state.buttons[0] and not prev_buttons[0]:      # LEFT → close
                set_gripper(gripper, 1.0)
                gripper_state = 1.0
            if state.buttons[1] and not prev_buttons[1]:      # RIGHT → open
                set_gripper(gripper, 0.0)
                gripper_state = 0.0
            prev_buttons = list(state.buttons[:2])

            # ---- preview + keys + recording: throttled to RECORD_HZ ----
            if loop_count % record_every == 0:
                joints_state = robot_r.getActualQ()
                frames = cam_reader.read()
                cv2.imshow("base", cv2.cvtColor(frames["base"], cv2.COLOR_RGB2BGR))
                cv2.imshow("wrist", cv2.cvtColor(frames["wrist"], cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if not is_recording:
                        print(f"Started recording episode {episode_number:04d}.")
                        is_recording = True
                    else:
                        path = SAVE_DIR / f"episode_{episode_number:04d}.hdf5"
                        recorder.save(path, task_instruction)
                        print(f"Episode saved to {path}.")
                        episode_number += 1
                        is_recording = False
                        recorder.discard()
                elif key == ord('d'):
                    recorder.discard()
                    is_recording = False
                    print(f"Episode {episode_number:04d} discarded.")
                elif key == ord('q'):
                    print("Quitting data collection.")
                    break

                if is_recording:
                    recorder.add_frame(joints_state, gripper_state, frames)

            loop_count += 1
            time.sleep(dt)


#11: def main()
#   - 创建 SAVE_DIR
#   - 让用户输入 task_instruction (input())
#   - connect_robot, connect_cameras, connect_gripper
#   - 创建 EpisodeRecorder
#   - 调用 run_collection_loop
#   - 断开连接
def main():
    os.makedirs(SAVE_DIR, exist_ok=True) 
    task_instruction = input("Enter task instruction for this episode: ")
    print("Connecting to robot...")
    robot_control, robot_receive = connect_robot(ROBOT_IP)
    print("Robot connected.")
    print("Connecting to cameras...")
    cameras = connect_cameras()
    cam_reader = CameraReader(cameras).start()   # start background capture thread
    print("Cameras connected.")
    print("Connecting to gripper...")
    gripper = connect_gripper()
    print("All devices ready. Starting collection loop.")
    recorder = EpisodeRecorder()
    run_collection_loop(robot_control, robot_receive, cam_reader, gripper, recorder, task_instruction)

    try:
        robot_control.speedStop()
    except Exception as e:
        print(f"speedStop failed: {e}")

    try:
        robot_control.stopScript()  # 停止机器人当前的运动脚本，确保它不再执行任何命令
    except Exception as e:
        print(f"stopScript failed: {e}")

    try:
        robot_control.disconnect()
    except Exception as e:
        print(f"control disconnect failed: {e}")

    try:
        robot_receive.disconnect()
    except Exception as e:
        print(f"receive disconnect failed: {e}")

    try:
        gripper.close()
    except Exception as e:
        print(f"gripper disconnect failed: {e}")

    cam_reader.stop()   # stop the background thread before releasing the cameras

    for name, cap in cameras.items():
        try:
            cap.release()
        except Exception as e:
            print(f"camera {name} release failed: {e}")
            
    cv2.destroyAllWindows()

    print("All connections closed. Goodbye!")

if __name__ == "__main__":
    main()  