"""
Hand tracker using OpenCV + MediaPipe Hand Landmarker.

Shows your webcam, draws the hand skeleton, and draws a trail
behind your index fingertip (the future "blade" for the slicing game).

Install:  pip install mediapipe opencv-python
Run:      python hand_tracker.py      (press q to quit)
"""

import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

INDEX_FINGERTIP = 8          # landmark id for the tip of the index finger
TRAIL_LENGTH = 15            # how many past fingertip points to keep
SMOOTHING = 0.5              # 0 = no smoothing, closer to 1 = smoother but laggier

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    # Convert normalized (0-1) coordinates to pixel coordinates
    points = []
    for lm in landmarks:
        points.append((int(lm.x * width), int(lm.y * height)))

    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 200, 0), -1)

    return points


def draw_trail(frame, trail):
    for i in range(1, len(trail)):
        thickness = int(2 + 8 * i / len(trail))   # newer points are thicker
        cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), thickness)


def main():
    download_model_if_missing()
    landmarker = create_hand_landmarker()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam.")
        return

    # Ask for a small frame size and a high frame rate. The camera may not
    # support these exactly; it will pick the closest mode it has.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 60)
    print("Camera mode: "
          f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f" @ {camera.get(cv2.CAP_PROP_FPS):.0f} FPS")

    trail = deque(maxlen=TRAIL_LENGTH)
    smoothed_tip = None
    start_time = time.time()
    last_frame_time = time.time()
    fps = 0.0

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # Mirror the image so moving right feels like moving right
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs an increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)
        infer_start = time.time()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        infer_ms = (time.time() - infer_start) * 1000

        if result.hand_landmarks:
            points = draw_hand(frame, result.hand_landmarks[0])
            tip = points[INDEX_FINGERTIP]

            # Simple smoothing to reduce jitter
            if smoothed_tip is None:
                smoothed_tip = tip
            else:
                x = int(SMOOTHING * smoothed_tip[0] + (1 - SMOOTHING) * tip[0])
                y = int(SMOOTHING * smoothed_tip[1] + (1 - SMOOTHING) * tip[1])
                smoothed_tip = (x, y)

            trail.append(smoothed_tip)
            cv2.putText(frame, f"Tip: {smoothed_tip}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Hand lost: clear the trail so it doesn't jump when the hand returns
            trail.clear()
            smoothed_tip = None

        draw_trail(frame, list(trail))

        # FPS counter, averaged over recent frames so the number doesn't flicker
        now = time.time()
        instant_fps = 1 / max(now - last_frame_time, 1e-6)
        fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
        last_frame_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}  (tracking {infer_ms:.0f} ms)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Hand Tracker (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()