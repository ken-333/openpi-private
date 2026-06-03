"""
SpaceMouse → UR5 teleoperation (motion-control tuning only).

No cameras, no recording — just drive the robot (and gripper) so you can
fine-tune the SpaceMouse→TCP velocity mapping. Once it feels right, copy
map_to_velocity() (signs/order/scale) back into collect_data.py.

Controls:
    SpaceMouse axes  → UR5 TCP velocity
    Button LEFT      → gripper close
    Button RIGHT     → gripper open
    Ctrl+C           → stop robot and quit
"""

import pyspacemouse
import rtde_control
import rtde_receive
import numpy as np
import socket
import time


ROBOT_IP = "192.168.0.101"
CONTROL_HZ = 125
SPACEMOUSE_DEADZONE = 0.05
GRIPPER_PORT = 63352   # Robotiq URCap text protocol over socket

# SpaceMouse → robot base frame coordinate transform
# Converts SpaceMouse axes to UR5 base frame via rotation matrix (from openpi teleop.py)
# If an axis moves the wrong way after testing, flip the sign of the corresponding row.
TX_ZUP_SPNAV = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=np.float32)

TRANS_SCALE = 0.1   # translation: m/s at full SpaceMouse deflection
ROT_SCALE   = 0.1   # rotation:    rad/s at full SpaceMouse deflection

ACCELERATION = 0.5  # m/s^2
PRINT_POSE = True   # print TCP pose while moving (helps verify direction)
PRINT_EVERY = 50    # print once every N loops (~2 Hz at 100 Hz loop)
# ------------------------------------------------------------------------------


def apply_deadzone(state, threshold):
    state = np.array(state)
    state[np.abs(state) < threshold] = 0
    return state


def connect_robot(ip):
    robot_control = rtde_control.RTDEControlInterface(ip)
    robot_receive = rtde_receive.RTDEReceiveInterface(ip)
    return robot_control, robot_receive


def _gripper_cmd(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    return sock.recv(1024).decode().strip()


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


def set_gripper(gripper, position):
    position = np.clip(position, 0.0, 1.0)
    pos = int(position * 255)
    _gripper_cmd(gripper, "SET SPE 255")
    _gripper_cmd(gripper, "SET FOR 150")
    _gripper_cmd(gripper, f"SET POS {pos}")
    _gripper_cmd(gripper, "SET GTO 1")


def map_to_velocity(state):
    raw_trans = np.array([state.x, state.y, state.z])
    raw_rot   = np.array([state.roll, state.pitch, state.yaw])

    trans = TX_ZUP_SPNAV @ raw_trans * TRANS_SCALE
    rot   = TX_ZUP_SPNAV @ raw_rot   * ROT_SCALE

    velocity = np.concatenate([trans, rot])
    return apply_deadzone(velocity, SPACEMOUSE_DEADZONE)


def teleop(robot_c, robot_r, gripper):
    dt = 1.0 / CONTROL_HZ
    speed_time = 0.1   # 0 = non-blocking: speedL returns immediately, loop runs at full rate
    loop_count = 0
    prev_buttons = [False, False]   # edge detection: fire gripper once per press
    with pyspacemouse.open() as device:
        print("Teleop running. Move the SpaceMouse. Ctrl+C to stop.")
        while True:
            state = device.read()
            velocity = map_to_velocity(state)
            robot_c.speedL(velocity.tolist(), ACCELERATION, speed_time)

            # Gripper: only send on the press transition (not every held frame),
            # so we don't flood the gripper socket and stall the loop.
            if state.buttons[0] and not prev_buttons[0]:      # LEFT → close
                set_gripper(gripper, 1.0)
            if state.buttons[1] and not prev_buttons[1]:      # RIGHT → open
                set_gripper(gripper, 0.0)
            prev_buttons = list(state.buttons[:2])

            # Throttled so the RTDE pose read + terminal I/O don't add jitter
            # to every control loop.
            if PRINT_POSE and loop_count % PRINT_EVERY == 0:
                pose = robot_r.getActualTCPPose()  # [x,y,z,rx,ry,rz] base frame
                pos = [round(p, 4) for p in pose[:3]]      # meters
                rot = [round(p, 4) for p in pose[3:]]      # rotation vector (rad)
                cmd = [round(v, 3) for v in velocity]      # commanded TCP velocity
                print(f"cmd: {cmd}   pos(m): {pos}   rot(rad): {rot}")

            loop_count += 1
            time.sleep(dt)


def main():
    print("Connecting to robot...")
    robot_control, robot_receive = connect_robot(ROBOT_IP)
    print("Connecting to gripper...")
    gripper = connect_gripper()
    print("Robot connected. Starting teleop.")
    try:
        teleop(robot_control, robot_receive, gripper)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
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
        try:
            robot_receive.disconnect()
        except Exception as e:
            print(f"receive disconnect failed: {e}")
        try:
            gripper.close()
        except Exception as e:
            print(f"gripper disconnect failed: {e}")
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    main()
