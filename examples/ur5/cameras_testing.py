## i need to test the cameras in the ur5 environment which one is wrist camera and which one is eye in hand camera
#just output all the cameras shoot and see which one is which

# also i need to test the best fps for the cameras to work properly

# NOTE: there are 4 RealSense cameras here; each exposes 6 /dev/video* nodes
# (color + depth + IR). Only the YUYV color node is useful for previewing.
# The color streams live on indices 4, 10, 16, 22.

import cv2
import time
import threading

class Grabber(threading.Thread):
    """Continuously read the newest frame from one camera in its own thread.

    Keeping reads off the main loop lets the 4 cameras read in parallel instead
    of serially, so the display stays smooth.
    """
    def __init__(self, index, width, height):
        super().__init__(daemon=True)
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame, less lag
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.fps = 0.0
        self.running = True

    def run(self):
        last = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            now = time.time()
            inst = 1.0 / (now - last) if now > last else 0.0
            last = now
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self.frame = frame

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        self.cap.release()


def preview_cameras(indices, width=640, height=480):
    """Show all color cameras at once so you can label base vs wrist.

    Each camera is grabbed in a background thread; the main loop only displays
    + polls keys, so Esc/q quit immediately. Opened at modest resolution so the
    4 RealSense streams don't saturate USB.
    """
    grabbers = [Grabber(i, width, height) for i in indices]
    for g in grabbers:
        g.start()

    print("Showing all cameras. Press Ctrl+C in the terminal to quit.")
    try:
        while True:
            for g in grabbers:
                if g.frame is not None:
                    frame = g.frame.copy()
                    cv2.putText(frame, f"cam {g.index}: {g.fps:.1f} fps", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow(f"cam {g.index}", frame)
            cv2.waitKey(30)   # needed so the windows actually render
    except KeyboardInterrupt:
        print("\nCtrl+C — closing cameras.")
    finally:
        for g in grabbers:
            g.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # The 4 RealSense color streams, opened directly like collect_data.py.
    cams = [4, 10, 16, 22]
    print("Color camera indices:", cams)

    # Show all cameras with live FPS; press Ctrl+C once to close + exit.
    preview_cameras(cams)