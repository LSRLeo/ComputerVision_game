# Cowboy Standoff

A two-player quick-draw game you play with your body. A webcam tracks both players. When the countdown ends, whoever raises their arm first wins.

Built with Python, [OpenCV](https://opencv.org/) and [MediaPipe](https://ai.google.dev/edge/mediapipe) body and hand tracking.

## How to play

1. **Get ready.** Both players stand side by side in front of the webcam. Player 1 stands on the left of the screen and Player 2 on the right. Each player makes an **OK sign** (thumb and index finger touching, other fingers up) and holds it until their status shows **READY!**
2. **Holster.** Both players lower their arms to their sides and stay still for a moment.
3. **Countdown.** A random countdown of 3 to 10 seconds runs on screen, with a ticking clock and a voice counting down. Raise your arm before it ends and you lose.
4. **Draw!** When the countdown ends, the first player to raise an arm out to the side, level with the shoulder, shoots and wins. The screen shows your reaction time.
5. **Play again.** Both players make the OK sign again.

**Tips**
- Stand far enough back that your whole upper body is in view, including your hands when your arms hang down.
- Good lighting makes tracking more reliable and keeps the frame rate up.

## Setup

You need Python 3.10 or newer and a webcam.

```bash
pip install mediapipe opencv-python numpy sounddevice
```

## Running

```bash
python3 standoff.py
```

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Go back to the lobby |

On the first run the game downloads MediaPipe's body and hand tracking models. It also records the countdown voice lines (macOS only). Both are saved locally and reused on later runs.

### macOS camera permission

If you see `Could not open webcam`, allow camera access for the app you're running Python from:

1. Open **System Settings → Privacy & Security → Camera**.
2. Turn on **Visual Studio Code** or **Terminal**.
3. Restart that app completely.

## Project structure

| File | What it does |
|---|---|
| `standoff.py` | Main loop: camera → tracking → game rules → drawing and sound |
| `mechanics.py` | Game rules, round stages, pose and gesture checks, and all tuning values |
| `ui.py` | Lobby screen, on-screen text, countdown, results and the camera preview |
| `animation.py` | Desert scene, the cowboys that follow each player's movements, and gunshot effects |
| `audio.py` | Countdown ticks, voice-over, bell, gunshot and false-start buzzer |
| `tracker.py` | Webcam setup and MediaPipe tracking for two bodies and four hands. Run it on its own (`python3 tracker.py`) to see the raw tracking. |

## Tuning

The main settings are constants at the top of each file:

- **`mechanics.py`:**
  - how long to hold the OK sign
  - how far down your arms must be to count as holstered
  - the countdown range
  - the false-start and draw angles
- **`animation.py`:** how much the cowboys' movement is smoothed (`MIN_CUTOFF`, `BETA`)
- **`audio.py`:**
  - the volume
  - which voice reads the countdown (`VOICE`; list voices with `say -v '?'`)
  - the words it says

## Notes

- The voice-over uses the macOS `say` command. On other systems the game still runs, with ticks but no voice.
- The frame rate is limited by your webcam, usually 24–30 FPS on a laptop camera. Tracking itself takes about 20 ms per frame.
