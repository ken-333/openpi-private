"""
SpaceMouse → UR5 teleoperation — spnav backend.

Uses the exact Spacemouse class from examples/ur5/teleop.py (threaded spnav
reader with the tx_zup_spnav frame transform + constructor deadzone), wrapped in
the same control/gripper/cleanup logic as teleop_ur5.py.

Requirements (system-level, Linux):
    sudo apt install spacenavd libspnav-dev
    pip install spnav
    sudo systemctl start spacenavd

Controls:
    SpaceMouse axes  → UR5 TCP velocity (frame handled by Spacemouse class)
    Button 0         → gripper close
    Button 1         → gripper open
    Ctrl+C           → stop robot and quit
"""

import ctypes
import numpy as np
import socket
import time
from collections import defaultdict
from threading import Thread, Event

import rtde_control
import rtde_receive

# The PyPI `spnav` 0.9 binding is Python-2 only: at import it binds
# PyCObject_AsVoidPtr (removed in CPython 3.3), only used by the unused X11 path.
# Stub it so the import succeeds on Python 3.
try:
    ctypes.pythonapi.PyCObject_AsVoidPtr
except AttributeError:
    class _PyCObjectStub:
        restype = None
        argtypes = None
    ctypes.pythonapi.PyCObject_AsVoidPtr = _PyCObjectStub()

try:
    from spnav import (
        spnav_open, spnav_poll_event, spnav_close,
        SpnavMotionEvent, SpnavButtonEvent,
    )
except ImportError as e:
    raise ImportError(
        "spnav not available. Install the daemon + binding:\n"
        "  sudo apt install spacenavd libspnav-dev\n"
        "  pip install spnav\n"
        "  sudo systemctl start spacenavd"
    ) from e


ROBOT_IP = "192.168.0.101"
CONTROL_HZ = 125
GRIPPER_PORT = 63352        # Robotiq URCap text protocol over socket
SCALE_FACTOR = 0.1          # velocity scale (same as teleop.py)
ACCELERATION = 0.5          # m/s^2
SPACEMOUSE_DEADZONE = 0.2   # normalized [0,1]; below this an axis stays at 0
SPNAV_MAX_VALUE = 300       # 300 wired SpaceMouse, 500 wireless
PRINT_EVERY = 50            # throttle TCP-velocity prints


class Spacemouse(Thread):
    """Verbatim from examples/ur5/teleop.py (the `else:code` typo fixed)."""

    def __init__(self, max_value=SPNAV_MAX_VALUE, deadzone=(0, 0, 0, 0, 0, 0), dtype=np.float32):
        """Continuously listen to 3DConnexion SpaceMouse events and keep latest state.

        max_value: 300 for wired SpaceMouse, 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this stays at 0
        """
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
        """Return TCP velocity in the right-handed base frame."""
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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


def teleop(robot_c, robot_r, gripper, sm):
    dt = 1.0 / CONTROL_HZ
    speed_time = 0.1
    loop_count = 0
    prev_buttons = [False, False]
    print("Teleop running. Move the SpaceMouse. Ctrl+C to stop.")
    while True:
        if robot_r.getRobotMode() != 7:
            print("Robot is not ready (mode != 7).")
            time.sleep(1)
            continue

        velocity = sm.get_motion_state_transformed()
        robot_c.speedL(velocity.tolist(), ACCELERATION, speed_time)

        # Gripper: fire once on the press transition, not every held frame.
        b0, b1 = sm.is_button_pressed(0), sm.is_button_pressed(1)
        if b0 and not prev_buttons[0]:      # button 0 → close
            set_gripper(gripper, 1.0)
        if b1 and not prev_buttons[1]:      # button 1 → open
            set_gripper(gripper, 0.0)
        prev_buttons = [b0, b1]

        if loop_count % PRINT_EVERY == 0:
            cmd = [round(v, 3) for v in velocity]
            print(f"cmd: {cmd}")

        loop_count += 1
        time.sleep(dt)


def main():
    sm = Spacemouse(deadzone=SPACEMOUSE_DEADZONE)
    sm.start()
    print("Connecting to robot...")
    robot_control, robot_receive = connect_robot(ROBOT_IP)
    print("Connecting to gripper...")
    gripper = connect_gripper()
    print("Robot connected. Starting teleop.")
    try:
        teleop(robot_control, robot_receive, gripper, sm)
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
        try:
            sm.stop()
        except Exception as e:
            print(f"spacemouse stop failed: {e}")
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    main()
