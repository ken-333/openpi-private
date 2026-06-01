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
from pathlib import Path 
import os


ROBOT_IP = "192.168.0.101"             # change to your robot's IP
CONTROL_HZ = 20                        # control loop frequency
SAVE_DIR = Path("./collected_data")    # directory to save episodes
SPACEMOUSE_SCALE = 0.1                 # scale factor for SpaceMouse input to velocity
SPACEMOUSE_DEADZONE = 0.05             # threshold for noise cancellation

# 3: def apply_deadzone(state, threshold)
#   输入: SpaceMouse 原始 6D 向量, 阈值
#   输出: 小于阈值的分量清零后的向量
def apply_deadzone(state, threshold):
    state = np.array(state)
    state[np.abs(state) < threshold] =0 # 将小于阈值的分量清零
    return state

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
        "base": cv2.VideoCapture(0),  # 根据实际情况调整摄像头索引
        "wrist": cv2.VideoCapture(1)  
    }                                 #为何用dict？因为有两个摄像头，使用dict可以更清晰地管理它们，并通过键名访问对应的摄像头对象。
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
            frames[cam_name] = frame
    return frames
        

# TODO 7: def connect_gripper()
#   连接夹爪 (先留空, 写 pass, 后面再填)
def connect_gripper():
    pass

# TODO 8: def set_gripper(position)
#   控制夹爪开合 (先留空, 写 pass)
def set_gripper(position):
    pass

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
        joints = np.array(self.joints)
        gripper = np.array(self.gripper)
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
def run_collection_loop(robot_c, robot_r, cameras, recorder, task_instruction):
    with pyspacemouse.open() as device:
        acceleration = 0.5                  # 设置一个适中的加速度值，单位 m/s^2
        dt = 1.0 / CONTROL_HZ               # 这个速度持续时间，单位秒（每帧的时间间隔）
        episode_number = 0
        is_recording = False
        while True:
            # 1. read SpaceMouse input
            state = device.read() # 偏移量是相对于上次读取的增量，初始状态为0，持续按一个方向会不断累积增大，松开后会回到0。它反映了用户对 SpaceMouse 的操作力度和方向。
        
            # 2. apply deadzone and scale
            velocity = apply_deadzone([state.x, state.y, state.z, state.rx, state.ry, state.rz], SPACEMOUSE_DEADZONE) * SPACEMOUSE_SCALE

            # 3. send velocity command to robot
            robot_c.speedl(velocity, acceleration, dt)

            # 4. read joint states
            joints_state = robot_r.getActualQ()
            gripper_state = 0      # TODO 7 写完后改成实际夹爪状态

            # 5. read camera frames
            frames = read_cameras(cameras)

            # 7. check for keyboard/button input (save&start 's' /discard 'd' /quit 'q' )
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if not is_recording:
                    print(f"Started recording episode {episode_number:04d}.")
                    is_recording = True
                else:
                    path = SAVE_DIR / f"episode_{episode_number:04d}.hdf5"
                    recorder.save(path, task_instruction) # TODO 11 让用户输入 task_instruction
                    print(f"Episode saved to {path}.")
                    episode_number += 1
                    recorder.discard() # 保存后清空当前记录器，准备下一集
                    is_recording = False
            elif key == ord('d'):
                recorder.discard()
                print(f"Episode {episode_number:04d} discarded.")
            elif key == ord('q'):
                print("Quitting data collection.")
                break
           
            # 6. add frame to recorder
            if is_recording:
                recorder.add_frame(joints_state, gripper_state, frames)

            # 8. control loop frequency
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
    robot_control, robot_receive = connect_robot(ROBOT_IP)
    cameras = connect_cameras()
    connect_gripper() # TODO 7 写完后取消注释
    recorder = EpisodeRecorder()
    run_collection_loop(robot_control, robot_receive, cameras, recorder, task_instruction)

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

    for name, cap in cameras.items():
        try:
            cap.release()
        except Exception as e:
            print(f"camera {name} release failed: {e}")

    print("All connections closed. Goodbye!")

if __name__ == "__main__":
    main()  