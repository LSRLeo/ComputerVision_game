"""
Sound effects for Cowboy Standoff.

All sounds are generated in code when the game starts, so there are no
audio files to ship. Playback goes through one always-open audio stream
with a small buffer, so a sound starts within a few milliseconds of being
triggered. That matters most for the "DRAW!" bell, which players react to.

Sounds:
  tick       every second of the countdown (higher pitch for the last 3)
  voice      a spoken "Ten, nine, ... one" with each number, then "Draw!"
  draw bell  the moment the countdown ends
  gunshot    when someone wins by shooting
  buzzer     when someone draws too early

The voice uses macOS's built-in text-to-speech (the `say` command). The
spoken lines are made once and saved in voice_cache/, so later starts are
instant. On a computer without `say` the game runs with ticks only.
"""

import math
import os
import shutil
import subprocess
import threading
import wave

import numpy as np
import sounddevice as sd

import mechanics as M

SAMPLE_RATE = 44100
BLOCK_SIZE = 256         # samples per audio callback; smaller = less delay (256 is about 6 ms)
VOLUME = 0.8             # master volume for sound effects, 0 to 1
MIN_FIRST_TICK = 0.5     # skip the first countdown tick if its second is shorter than this

# Voice-over
VOICE = "Ralph"          # macOS voice; list them with `say -v '?'`. Falls back to the system voice.
VOICE_RATE = 200         # words per minute
VOICE_VOLUME = 1.0
VOICE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_cache")
NUMBER_WORDS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
DRAW_WORD = "Draw!"


# --- Sound generation -------------------------------------------------------------

def _time(seconds):
    return np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE


def _tick(pitch):
    """Short wooden 'tock', like a clock."""
    t = _time(0.06)
    return np.sin(2 * math.pi * pitch * t) * np.exp(-t * 70)


def _bell():
    """Bright bell 'ding' with a few overtones that rings out."""
    t = _time(1.2)
    tone = sum(amp * np.sin(2 * math.pi * freq * t)
               for freq, amp in ((880, 1.0), (1760, 0.5), (2640, 0.3), (3520, 0.15)))
    return tone / 1.95 * np.exp(-t * 3.5)


def _gunshot():
    """Sharp crack of noise followed by a low boom."""
    t = _time(0.7)
    rng = np.random.default_rng(7)
    crack = rng.uniform(-1, 1, t.size) * np.exp(-t * 25)
    crack = np.convolve(crack, np.ones(4) / 4, mode="same")   # soften the hiss a little
    boom = np.sin(2 * math.pi * 55 * t) * np.exp(-t * 6)
    return np.clip(crack * 1.2 + boom * 0.8, -1, 1)


def _buzzer():
    """Harsh low buzz for a false start."""
    t = _time(0.6)
    square = np.sign(np.sin(2 * math.pi * 140 * t))
    fade = np.minimum(1, (0.6 - t) * 20)        # avoid a click at the end
    return square * 0.5 * fade


SOUND_MAKERS = {
    "tick": lambda: _tick(900),
    "tick_high": lambda: _tick(1400),
    "draw": _bell,
    "bang": _gunshot,
    "buzzer": _buzzer,
}


# --- Voice-over -------------------------------------------------------------------------

def _installed_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def _trim_silence(wave_data, threshold=0.02, margin=0.005):
    """Cut the quiet lead-in and tail so each word starts the instant it's played."""
    loud = np.nonzero(np.abs(wave_data) > threshold)[0]
    if loud.size == 0:
        return wave_data
    pad = int(margin * SAMPLE_RATE)
    return wave_data[max(loud[0] - pad, 0):loud[-1] + pad]


def _read_wav(path):
    with wave.open(path, "rb") as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768


def make_voice_lines():
    """Return {sound name: waveform} for the spoken countdown, or {} without `say`."""
    if shutil.which("say") is None:
        return {}
    voice = VOICE if VOICE in _installed_voices() else None
    lines = {f"voice_{n}": word for n, word in enumerate(NUMBER_WORDS, start=1)}
    lines["voice_draw"] = DRAW_WORD

    os.makedirs(VOICE_CACHE_DIR, exist_ok=True)
    paths, jobs = {}, []
    for name, text in lines.items():
        path = os.path.join(VOICE_CACHE_DIR, f"{voice or 'default'}_{VOICE_RATE}_{name}.wav")
        paths[name] = path
        if not os.path.exists(path):
            command = ["say", "-o", path, "--file-format=WAVE",
                       f"--data-format=LEI16@{SAMPLE_RATE}", "-r", str(VOICE_RATE)]
            if voice:
                command += ["-v", voice]
            jobs.append((path, subprocess.Popen(command + [text])))   # all lines in parallel
    if jobs:
        print("Recording countdown voice (one time)...")
    for path, job in jobs:
        if job.wait() != 0 and os.path.exists(path):
            os.remove(path)

    return {name: _trim_silence(_read_wav(path)) * VOICE_VOLUME
            for name, path in paths.items() if os.path.exists(path)}


# --- Mixer -----------------------------------------------------------------------------

class Mixer:
    """Plays any number of sounds at once over a single open audio stream."""

    def __init__(self, extra_sounds=None):
        self.sounds = {name: (make() * VOLUME).astype(np.float32)
                       for name, make in SOUND_MAKERS.items()}
        for name, wave_data in (extra_sounds or {}).items():
            self.sounds[name] = np.asarray(wave_data, dtype=np.float32)
        self.playing = []           # [sound array, position] for each sound in progress
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                      blocksize=BLOCK_SIZE, latency="low",
                                      callback=self._callback)
        self.stream.start()

    def play(self, name):
        if name not in self.sounds:      # e.g. a voice line that couldn't be recorded
            return
        with self.lock:
            self.playing.append([self.sounds[name], 0])

    def _callback(self, out, frames, _time_info, _status):
        # Runs on the audio thread: add up every playing sound into this block
        out.fill(0)
        with self.lock:
            still_playing = []
            for item in self.playing:
                sound, pos = item
                chunk = sound[pos:pos + frames]
                out[:len(chunk), 0] += chunk
                item[1] = pos + frames
                if item[1] < len(sound):
                    still_playing.append(item)
            self.playing = still_playing
        np.clip(out, -1, 1, out=out)

    def close(self):
        self.stream.stop()
        self.stream.close()


# --- Playing sounds at the right moments -------------------------------------------------

class SoundEffects:
    """Watches the game each frame and plays a sound when something happens.

    If no audio device is available the game still runs, just silently.
    """

    def __init__(self):
        try:
            voice = make_voice_lines()
        except Exception as error:      # text-to-speech failed: carry on without the voice
            print(f"Countdown voice disabled: {error}")
            voice = {}
        try:
            self.mixer = Mixer(voice)
        except Exception as error:      # no output device, audio system busy, etc.
            print(f"Sound disabled: {error}")
            self.mixer = None
        self.last_state = None
        self.last_number = None

    def play(self, name):
        if self.mixer:
            self.mixer.play(name)

    def _count(self, number):
        """Tick and say the number now on screen."""
        self.play("tick_high" if number <= 3 else "tick")
        self.play(f"voice_{number}")

    def update(self, game, now):
        state_changed = game.state != self.last_state

        if game.state == M.COUNTDOWN:
            # Tick each time the number on screen changes
            remaining = game.countdown_end - now
            number = max(math.ceil(remaining), 1)
            if state_changed:
                # The random length usually starts with a partial second (e.g. 6.4 s).
                # If it's very short, stay quiet, or two numbers land almost together.
                first_second = remaining - math.floor(remaining) or 1
                if first_second >= MIN_FIRST_TICK:
                    self._count(number)
            elif number != self.last_number:
                self._count(number)
            self.last_number = number

        elif state_changed and game.state == M.DRAW:
            self.play("draw")
            self.play("voice_draw")

        elif state_changed and game.state == M.RESULT:
            if game.outcome == "false_start":
                self.play("buzzer")
            elif game.outcome in ("shot", "tie"):
                self.play("bang")

        self.last_state = game.state

    def close(self):
        if self.mixer:
            self.mixer.close()
