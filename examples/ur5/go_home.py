"""
Move UR5 back to the home joint position.

Run this whenever you need to reset the arm before/after a session:
    python go_home.py

Edit HOME_JOINTS (degrees) if your home pose changes.
"""

import math
import rtde_control

ROBOT_IP = "192.168.0.101"

# Joint angles in degrees — read from the UR teach pendant screenshot.
# Order: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
HOME_JOINTS_DEG = [-6.65, -84.31, 86.39, -91.85, -88.30, 171.55]

SPEED = 0.5   # rad/s  — reduce if motion feels too fast
ACCEL = 0.3   # rad/s²


def main():
    home_rad = [math.radians(j) for j in HOME_JOINTS_DEG]

    print("Connecting to robot...")
    rc = rtde_control.RTDEControlInterface(ROBOT_IP)
    print(f"Moving to home: {HOME_JOINTS_DEG} deg")
    rc.moveJ(home_rad, SPEED, ACCEL)
    print("Done.")
    rc.disconnect()


if __name__ == "__main__":
    main()
