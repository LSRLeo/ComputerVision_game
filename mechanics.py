"""
Game mechanics for Cowboy Standoff: the rules, tuning values, pose and
gesture checks, and the round state machine. Nothing in here draws anything.
"""

import math
import random

import tracker as T

# --- Tuning -------------------------------------------------------------------
OK_HOLD_SECONDS = 0.5        # how long the OK sign must be held to count as ready
HOLSTER_ANGLE = 45           # arms within this many degrees of straight down count as holstered
HOLSTER_HOLD_SECONDS = 0.8   # both players must stay holstered this long before the countdown
HOLSTER_GRACE_SECONDS = 0.5  # brief slips out of position (fidgeting, tracking jitter) shorter
                             # than this don't reset the hold
COUNTDOWN_RANGE = (3, 10)    # countdown length in seconds, picked at random
EARLY_DRAW_ANGLE = 50        # raising an arm past this during the countdown is a false start
EARLY_DRAW_FRAMES = 2        # ...for this many frames in a row (ignores one-frame tracking glitches)
DRAW_ANGLE = 80              # an arm at least this high counts as drawn (90 = level with the shoulder)
DRAW_TIMEOUT = 5.0           # seconds after "DRAW!" before the round ends with nobody shooting
RESULT_MIN_SECONDS = 2.0     # how long the result shows before players can ready up again
PLAYER_LOST_GRACE = 1.0      # seconds a player can drop out of tracking before counting as gone
MIN_WRIST_VISIBILITY = 0.5   # ignore arms whose wrist MediaPipe isn't confident about

# Body landmark ids: (shoulder, elbow, wrist) for each arm
ARMS = ((11, 13, 15), (12, 14, 16))

# Round states
LOBBY, HOLSTER, COUNTDOWN, DRAW, RESULT = "lobby", "holster", "countdown", "draw", "result"


# --- Pose and gesture checks -------------------------------------------------------

def _visible(landmark):
    v = getattr(landmark, "visibility", None)
    return v is None or v >= MIN_WRIST_VISIBILITY


def arm_angles(pose, aspect):
    """Return [(angle, arm_index)] for each confidently tracked arm.

    The angle is in degrees from straight down: 0 = hanging at your side,
    90 = pointing sideways, 180 = straight up. It's measured from shoulder to
    wrist. `aspect` (frame width / height) corrects for the frame not being
    square.
    """
    angles = []
    for i, (shoulder, _, wrist) in enumerate(ARMS):
        if not _visible(pose[wrist]):
            continue
        dx = (pose[wrist].x - pose[shoulder].x) * aspect
        dy = pose[wrist].y - pose[shoulder].y
        angles.append((math.degrees(math.atan2(abs(dx), dy)), i))
    return angles


def highest_arm(pose, aspect):
    """Return (angle, arm_index) of the higher arm, or (0, None) if neither is tracked."""
    return max(arm_angles(pose, aspect), default=(0.0, None))


def _angle_from_down(pose, a, b, aspect):
    dx = (pose[b].x - pose[a].x) * aspect
    dy = pose[b].y - pose[a].y
    return math.degrees(math.atan2(abs(dx), dy))


def arm_is_down(pose, arm, aspect):
    """Is this arm hanging roughly at the player's side?

    Uses the wrist when it's visible. If the wrist is out of frame or hidden
    (common when hands hang below the camera's view), falls back to the
    elbow. If neither is visible, the arm is assumed to be down, since a
    raised arm would be in view.
    """
    shoulder, elbow, wrist = ARMS[arm]
    if _visible(pose[wrist]):
        return _angle_from_down(pose, shoulder, wrist, aspect) < HOLSTER_ANGLE
    if _visible(pose[elbow]):
        return _angle_from_down(pose, shoulder, elbow, aspect) < HOLSTER_ANGLE
    return True


def arms_holstered(pose, aspect):
    return all(arm_is_down(pose, arm, aspect) for arm in range(len(ARMS)))


def is_ok_sign(hand, aspect):
    """Thumb tip touching index tip, with middle, ring and pinky fingers extended."""
    def dist(a, b):
        return math.hypot((hand[a].x - hand[b].x) * aspect, hand[a].y - hand[b].y)

    hand_size = dist(0, 9)            # wrist to base of middle finger
    if hand_size == 0:
        return False
    pinch = dist(4, 8) < 0.35 * hand_size
    # A finger is extended when its tip is clearly further from the wrist than its middle joint
    others_up = all(dist(0, tip) > 1.15 * dist(0, joint)
                    for tip, joint in ((12, 10), (16, 14), (20, 18)))
    return pinch and others_up


# --- Round state machine -------------------------------------------------------------

class Standoff:
    """Tracks which stage the round is in and decides who wins.

    Call update() once per camera frame with the tracked players and hands.
    """

    def __init__(self):
        self.last_seen = [-math.inf, -math.inf]
        self.reset_to_lobby()

    def reset_to_lobby(self, message=""):
        self.state = LOBBY
        self.message = message
        self._reset_ready()

    def present(self, p, now):
        return now - self.last_seen[p] <= PLAYER_LOST_GRACE

    def _reset_ready(self):
        self.ready = [False, False]
        self.ok_since = [None, None]

    def _start_holster(self, message=""):
        self.state = HOLSTER
        self.message = message
        self.holster_since = None      # when both players first got into position
        self.holster_last_ok = None    # the last frame both were in position

    def _finish(self, now, outcome, winner, message, arm=None):
        self.state = RESULT
        self.result_time = now
        self.outcome = outcome    # "shot", "false_start", "tie" or "timeout"
        self.winner = winner      # 0, 1, or None for no winner
        self.winner_arm = arm     # which arm fired, for the muzzle flash
        self.message = message
        self._reset_ready()

    def _update_ready(self, now, hands, aspect):
        for p in range(2):
            if not self.present(p, now):
                self.ready[p] = False
                self.ok_since[p] = None
                continue
            if self.ready[p]:
                continue
            ok = any(is_ok_sign(hands[(p, side)], aspect)
                     for side in T.SIDES if (p, side) in hands)
            if not ok:
                self.ok_since[p] = None
            elif self.ok_since[p] is None:
                self.ok_since[p] = now
            elif now - self.ok_since[p] >= OK_HOLD_SECONDS:
                self.ready[p] = True

    def update(self, now, players, hands, aspect):
        for p in players:
            self.last_seen[p] = now
        both_present = self.present(0, now) and self.present(1, now)

        if self.state == LOBBY:
            self._update_ready(now, hands, aspect)
            if all(self.ready):
                self._start_holster()

        elif self.state == HOLSTER:
            if not both_present:
                self.reset_to_lobby("A player left. Both players step back in.")
                return
            # A player whose body drops out of tracking for a frame keeps their
            # last known position instead of breaking the hold
            in_position = all(p not in players or arms_holstered(players[p], aspect)
                              for p in range(2))
            if in_position:
                self.holster_last_ok = now
                if self.holster_since is None:
                    self.holster_since = now
                elif now - self.holster_since >= HOLSTER_HOLD_SECONDS:
                    self.state = COUNTDOWN
                    self.message = ""
                    self.countdown_end = now + random.uniform(*COUNTDOWN_RANGE)
                    self.early_frames = [0, 0]
            elif (self.holster_since is not None
                  and now - self.holster_last_ok > HOLSTER_GRACE_SECONDS):
                # Only reset once they've been out of position for a while
                self.holster_since = None

        elif self.state == COUNTDOWN:
            if not both_present:
                self._start_holster("A player went out of view. Countdown cancelled.")
                return
            for p in range(2):
                raised = p in players and highest_arm(players[p], aspect)[0] > EARLY_DRAW_ANGLE
                self.early_frames[p] = self.early_frames[p] + 1 if raised else 0
            early = [p for p in range(2) if self.early_frames[p] >= EARLY_DRAW_FRAMES]
            if len(early) == 2:
                self._finish(now, "false_start", None, "Both players drew too early!")
            elif early:
                p = early[0]
                self._finish(now, "false_start", 1 - p, f"Player {p + 1} drew too early!")
            elif now >= self.countdown_end:
                self.state = DRAW
                self.draw_time = now

        elif self.state == DRAW:
            shooters = []
            for p, pose in players.items():
                angle, arm = highest_arm(pose, aspect)
                if angle >= DRAW_ANGLE:
                    shooters.append((p, arm))
            if len(shooters) == 2:
                self._finish(now, "tie", None, "Both fired at the same moment. It's a tie!")
            elif shooters:
                p, arm = shooters[0]
                reaction_ms = (now - self.draw_time) * 1000
                self._finish(now, "shot", p, f"Reaction time: {reaction_ms:.0f} ms", arm)
            elif now - self.draw_time > DRAW_TIMEOUT:
                self._finish(now, "timeout", None, "Nobody drew. Too slow, partners.")

        elif self.state == RESULT:
            if now - self.result_time >= RESULT_MIN_SECONDS:
                self._update_ready(now, hands, aspect)
                if all(self.ready):
                    self._start_holster()
