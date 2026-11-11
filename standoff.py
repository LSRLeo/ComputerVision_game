"""
Cowboy Standoff: a two-player quick-draw game played with your body.

How a round works:
  1. Lobby     - both players step into view and each makes an "OK" sign
                 (thumb and index finger touching, other fingers up).
  2. Holster   - both players lower both arms to their sides.
  3. Countdown - a random 3-10 second countdown runs on screen.
                 Raising your arm before it ends is a false start: you lose.
  4. Draw!     - the first player to raise an arm out to 90 degrees
                 (pointing sideways, level with the shoulder) wins.

Player 1 stands on the left of the screen, Player 2 on the right. Stand far
enough back that your whole upper body, including your hands at your sides,
is in view.

Files:
  standoff.py   this file: the main loop
  mechanics.py  rules, tuning values and the round state machine
  ui.py         text, lobby screen, HUD and camera preview
  animation.py  desert scene, cowboys and gunshot effects
  audio.py      countdown ticks, draw bell, gunshot and false-start buzzer
  tracker.py    webcam and MediaPipe body/hand tracking

Run:  python standoff.py     (q = quit, r = back to the lobby)
"""

import time

import cv2
import mediapipe as mp

import animation
import audio
import mechanics as M
import tracker as T
import ui

CANVAS_WIDTH = 960
WINDOW_NAME = "Cowboy Standoff (q = quit, r = lobby)"


def main():
    T.download_model_if_missing(T.HAND_MODEL_PATH, T.HAND_MODEL_URL)
    T.download_model_if_missing(T.POSE_MODEL_PATH, T.POSE_MODEL_URL)
    hand_landmarker, pose_landmarker = T.create_landmarkers()
    camera = T.open_camera()
    if camera is None:
        return

    game = M.Standoff()
    sounds = audio.SoundEffects()
    scene = animation.Scene()
    start_time = time.time()

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        now = time.time()

        frame = cv2.flip(frame, 1)   # mirror, so moving right feels like moving right
        height, width = frame.shape[:2]
        aspect = width / height
        size = (CANVAS_WIDTH, int(CANVAS_WIDTH / aspect))

        # Track bodies and hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((now - start_time) * 1000)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        players = T.assign_players(pose_result.pose_landmarks)
        hands = T.assign_hands(hand_result, players)

        # Game mechanics, then any sounds they trigger
        game.update(now, players, hands, aspect)
        sounds.update(game, now)

        # Draw
        if game.state == M.LOBBY:
            screen = ui.render_lobby(game, frame, players, hands, now, size)
        else:
            screen = scene.render(size, game, players, now)
            ui.draw_hud(screen, game, players, now, aspect)
            ui.draw_camera_preview(screen, frame, players, hands, aspect)
        cv2.imshow(WINDOW_NAME, screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            game.reset_to_lobby()

    camera.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()
    sounds.close()


if __name__ == "__main__":
    main()
