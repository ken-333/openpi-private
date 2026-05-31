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
import h5py
import numpy as np
import time
from pathlib import Path 


ROBOT_IP = "192.168.1.100"             # change to your robot's IP
CONTROL_HZ = 20                        # control loop frequency
SAVE_DIR = Path("./collected_data")    # directory to save episodes
SPACEMOUSE_SCALE = 0.1                 # scale factor for SpaceMouse input to velocity
SPACEMOUSE_DEADZONE = 0.05             # threshold for noise cancellation

# TODO 3: def apply_deadzone(state, threshold)
#   输入: SpaceMouse 原始 6D 向量, 阈值
#   输出: 小于阈值的分量清零后的向量


# TODO 4: def connect_robot(ip)
#   创建并返回 rtde_control 和 rtde_receive 两个接口对象


# TODO 5: def connect_cameras()
#   打开 base 和 wrist 两个摄像头 (cv2.VideoCapture)
#   返回一个 dict: {"base": cap0, "wrist": cap1}


# TODO 6: def read_cameras(cameras)
#   从两个摄像头各读一帧
#   返回 dict: {"base": img, "wrist": img}


# TODO 7: def connect_gripper()
#   连接夹爪 (先留空, 写 pass, 后面再填)


# TODO 8: def set_gripper(position)
#   控制夹爪开合 (先留空, 写 pass)


# TODO 9: class EpisodeRecorder
#   __init__: 初始化空列表 self.frames
#   add_frame(joints, gripper, images): 把这一帧数据 append 到 self.frames
#   save(path, task): 把 self.frames 写成 HDF5 文件
#   discard(): 清空 self.frames
#   __len__: 返回帧数


# TODO 10: def run_collection_loop(robot_c, robot_r, cameras, recorder)
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


# TODO 11: def main()
#   - 创建 SAVE_DIR
#   - 让用户输入 task instruction (input())
#   - connect_robot, connect_cameras, connect_gripper
#   - 创建 EpisodeRecorder
#   - 调用 run_collection_loop
#   - 断开连接


if __name__ == "__main__":
    main()  # TODO 11 写完后取消注释