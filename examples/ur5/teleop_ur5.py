"""
SpaceMouse → UR5 teleoperation (motion-control tuning only).

No cameras, no gripper, no recording — just drive the robot so you can
fine-tune the SpaceMouse→TCP velocity mapping. Once it feels right, copy
map_to_velocity() (signs/order/scale) back into collect_data.py.

Controls:
    SpaceMouse axes  → UR5 TCP velocity
    Ctrl+C           → stop robot and quit
"""

import pyspacemouse
import rtde_control
import rtde_receive
import numpy as np
import time


ROBOT_IP = "192.168.0.101"
CONTROL_HZ = 20
SPACEMOUSE_DEADZONE = 0.05

# ---- Motion control tuning (this is what you fine-tune) ----------------------
# speedL expects TCP velocity in the robot BASE frame: [vx, vy, vz, rx, ry, rz]
# SpaceMouse gives:                                      x,  y,  z,  roll, pitch, yaw
# Flip a sign (+1/-1) if an axis moves the wrong way; swap the assignments in
# map_to_velocity() if two axes are interchanged.
AXIS_SIGN = {
    "x": +1, "y": +1, "z": +1,
    "roll": +1, "pitch": +1, "yaw": +1,
}
TRANS_SCALE = 0.1   # translation: m/s at full SpaceMouse deflection
ROT_SCALE   = 0.1   # rotation:    rad/s at full SpaceMouse deflection

ACCELERATION = 2.0  # m/s^2
PRINT_POSE = True   # print TCP pose while moving (helps verify direction)
# ------------------------------------------------------------------------------


def apply_deadzone(state, threshold):
    state = np.array(state)
    state[np.abs(state) < threshold] = 0
    return state


def connect_robot(ip):
    robot_control = rtde_control.RTDEControlInterface(ip)
    robot_receive = rtde_receive.RTDEReceiveInterface(ip)
    return robot_control, robot_receive


def map_to_velocity(state):
    # TODO(tune): adjust the axis assignments + AXIS_SIGN until the robot
    #             moves the same way your hand pushes the SpaceMouse.
    raw = np.array([
        AXIS_SIGN["x"]     * state.x,
        AXIS_SIGN["y"]     * state.y,
        AXIS_SIGN["z"]     * state.z,
        AXIS_SIGN["roll"]  * state.roll,
        AXIS_SIGN["pitch"] * state.pitch,
        AXIS_SIGN["yaw"]   * state.yaw,
    ])
    raw = apply_deadzone(raw, SPACEMOUSE_DEADZONE)
    raw[:3] *= TRANS_SCALE   # translation components
    raw[3:] *= ROT_SCALE     # rotation components
    return raw


def teleop(robot_c, robot_r):
    dt = 1.0 / CONTROL_HZ
    speed_time = 0.0   # 0 = non-blocking: speedL returns immediately, loop runs at full rate
    with pyspacemouse.open() as device:
        print("Teleop running. Move the SpaceMouse. Ctrl+C to stop.")
        while True:
            state = device.read()
            velocity = map_to_velocity(state)
            robot_c.speedL(velocity.tolist(), ACCELERATION, speed_time)

            if PRINT_POSE:
                pose = robot_r.getActualTCPPose()  # [x,y,z,rx,ry,rz] base frame
                pos = [round(p, 4) for p in pose[:3]]      # meters
                rot = [round(p, 4) for p in pose[3:]]      # rotation vector (rad)
                cmd = [round(v, 3) for v in velocity]      # commanded TCP velocity
                print(f"cmd: {cmd}   pos(m): {pos}   rot(rad): {rot}")

            time.sleep(dt)


def main():
    print("Connecting to robot...")
    robot_control, robot_receive = connect_robot(ROBOT_IP)
    print("Robot connected. Starting teleop.")
    try:
        teleop(robot_control, robot_receive)
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
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    main()
