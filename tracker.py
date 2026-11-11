"""
Two-player body and hand tracker using OpenCV + MediaPipe.

Tracks up to two people at once. For each player it draws the body
skeleton, both hands, and a trail behind each index fingertip (the future
"blades" for the slicing game).

Player 1 is whoever stands on the left of the screen, Player 2 on the right.
Each hand is given to the player whose body wrist it is closest to.

Install:  pip install mediapipe opencv-python
Run:      python tracker.py      (press q to quit)
"""

import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/"
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = MODEL_BASE_URL + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
# "lite" is the fastest of MediaPipe's three body models (lite / full / heavy)
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL = MODEL_BASE_URL + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

MAX_PLAYERS = 2
MAX_HANDS = 2 * MAX_PLAYERS
SIDES = ("Left", "Right")

# Landmark ids
HAND_WRIST = 0
INDEX_FINGERTIP = 8
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_WRISTS = (15, 16)

TRAIL_LENGTH = 15            # how many past fingertip points to keep
SMOOTHING = 0.5              # 0 = no smoothing, closer to 1 = smoother but laggier
# How close (as a fraction of the frame) a hand's wrist must be to a body's
# wrist to count as that person's hand
WRIST_MATCH_DISTANCE = 0.15

# One colour per player, in OpenCV's BGR order
PLAYER_COLORS = [
    (255, 160, 0),    # Player 1: blue
    (0, 160, 255),    # Player 2: orange
]

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS


def download_model_if_missing(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} (one time)...")
        urllib.request.urlretrieve(url, path)


def create_landmarkers():
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=MAX_PLAYERS,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return hand_landmarker, pose_landmarker


def open_camera():
    """Open the webcam, or return None if it can't be opened."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam.")
        return None

    # Ask for a small frame size and a high frame rate. The camera may not
    # support these exactly; it will pick the closest mode it has.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 60)
    print("Camera mode: "
          f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f" @ {camera.get(cv2.CAP_PROP_FPS):.0f} FPS")
    return camera


def to_pixels(landmarks, width, height):
    """Convert normalized (0-1) landmark coordinates to pixel coordinates."""
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


def draw_skeleton(frame, points, connections, line_color, dot_color, dot_size):
    for connection in connections:
        cv2.line(frame, points[connection.start], points[connection.end], line_color, 2)
    for point in points:
        cv2.circle(frame, point, dot_size, dot_color, -1)


def draw_trail(frame, trail, color):
    for i in range(1, len(trail)):
        thickness = int(2 + 8 * i / len(trail))   # newer points are thicker
        cv2.line(frame, trail[i - 1], trail[i], color, thickness)


def body_center_x(pose):
    return (pose[POSE_LEFT_SHOULDER].x + pose[POSE_RIGHT_SHOULDER].x) / 2


def assign_players(poses):
    """Return {player_index: pose_landmarks}, numbering players left to right.

    With only one person in view, they count as Player 1 on the left half of
    the screen and Player 2 on the right half, so numbers don't swap when the
    second person walks in.
    """
    poses = sorted(poses, key=body_center_x)[:MAX_PLAYERS]
    if len(poses) == 1:
        return {0 if body_center_x(poses[0]) < 0.5 else 1: poses[0]}
    return dict(enumerate(poses))


def assign_hands(hand_result, players):
    """Return {(player_index, side): hand_landmarks}, side being "Left"/"Right"."""
    hands = list(zip(hand_result.hand_landmarks, hand_result.handedness))
    owner = {}   # hand index -> player index

    # 1. Match hands to body wrists, closest pairs first. Each body wrist
    #    can only hold one hand.
    pairs = []
    for h, (hand, _) in enumerate(hands):
        for p, pose in players.items():
            for wrist in POSE_WRISTS:
                distance = math.hypot(hand[HAND_WRIST].x - pose[wrist].x,
                                      hand[HAND_WRIST].y - pose[wrist].y)
                pairs.append((distance, h, p, wrist))
    used_wrists = set()
    for distance, h, p, wrist in sorted(pairs):
        if distance > WRIST_MATCH_DISTANCE:
            break
        if h in owner or (p, wrist) in used_wrists:
            continue
        owner[h] = p
        used_wrists.add((p, wrist))

    # 2. Hands with no matching body wrist (e.g. the body wasn't detected)
    #    go to the nearest body, or by screen half if no body is visible.
    for h, (hand, _) in enumerate(hands):
        if h in owner:
            continue
        x = hand[HAND_WRIST].x
        if players:
            p = min(players, key=lambda p: abs(body_center_x(players[p]) - x))
        else:
            p = 0 if x < 0.5 else 1
        if list(owner.values()).count(p) < 2:
            owner[h] = p

    # 3. Label each player's hands left/right. MediaPipe assumes a mirrored
    #    (selfie) image, which is what we give it, so its labels match the
    #    player's real hands. If it labels both of one player's hands the
    #    same, the second one gets the other label.
    result = {}
    for h, p in owner.items():
        hand, handedness = hands[h]
        side = handedness[0].category_name
        if (p, side) in result:
            side = "Right" if side == "Left" else "Left"
        result[(p, side)] = hand
    return result


def main():
    download_model_if_missing(HAND_MODEL_PATH, HAND_MODEL_URL)
    download_model_if_missing(POSE_MODEL_PATH, POSE_MODEL_URL)
    hand_landmarker, pose_landmarker = create_landmarkers()

    camera = open_camera()
    if camera is None:
        return

    # One trail and one smoothed fingertip per hand, keyed by (player, side)
    hand_keys = [(p, side) for p in range(MAX_PLAYERS) for side in SIDES]
    trails = {key: deque(maxlen=TRAIL_LENGTH) for key in hand_keys}
    smoothed_tips = {key: None for key in hand_keys}

    start_time = time.time()
    last_frame_time = time.time()
    fps = 0.0

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # Mirror the image so moving right feels like moving right
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs an increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)
        t0 = time.time()
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        t1 = time.time()
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        t2 = time.time()
        hand_ms, body_ms = (t1 - t0) * 1000, (t2 - t1) * 1000

        # Bodies
        players = assign_players(pose_result.pose_landmarks)
        for p, pose in players.items():
            color = PLAYER_COLORS[p]
            points = to_pixels(pose, width, height)
            draw_skeleton(frame, points, POSE_CONNECTIONS, color, color, 3)
            nose_x, nose_y = points[POSE_NOSE]
            cv2.putText(frame, f"P{p + 1}", (nose_x - 15, max(nose_y - 40, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Hands
        hands = assign_hands(hand_result, players)
        for key, hand in hands.items():
            points = to_pixels(hand, width, height)
            draw_skeleton(frame, points, HAND_CONNECTIONS, (255, 255, 255), (0, 200, 0), 4)
            tip = points[INDEX_FINGERTIP]

            # Simple smoothing to reduce jitter
            previous = smoothed_tips[key]
            if previous is None:
                smoothed_tips[key] = tip
            else:
                x = int(SMOOTHING * previous[0] + (1 - SMOOTHING) * tip[0])
                y = int(SMOOTHING * previous[1] + (1 - SMOOTHING) * tip[1])
                smoothed_tips[key] = (x, y)
            trails[key].append(smoothed_tips[key])

        for key in hand_keys:
            if key not in hands:
                # Hand lost: clear its trail so it doesn't jump when the hand returns
                trails[key].clear()
                smoothed_tips[key] = None
            else:
                draw_trail(frame, list(trails[key]), PLAYER_COLORS[key[0]])

        # FPS counter, averaged over recent frames so the number doesn't flicker
        now = time.time()
        instant_fps = 1 / max(now - last_frame_time, 1e-6)
        fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
        last_frame_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}  (hands {hand_ms:.0f} ms, bodies {body_ms:.0f} ms)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # What each player has in view
        for p in range(MAX_PLAYERS):
            body = "body" if p in players else "no body"
            hand_count = sum(1 for side in SIDES if (p, side) in hands)
            cv2.putText(frame, f"P{p + 1}: {body}, {hand_count} hand(s)", (10, 60 + 25 * p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLAYER_COLORS[p], 2)

        cv2.imshow("Two-Player Tracker (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()


if __name__ == "__main__":
    main()
