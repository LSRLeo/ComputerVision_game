"""
Animation for Cowboy Standoff: the desert scene, the cowboys that move with
each player's tracked body, their guns, and the gunshot effects.

Body tracking jitters a little from frame to frame, so every landmark goes
through a One Euro filter before it's drawn. The filter smooths heavily when
a joint is nearly still (no shaking) and hardly at all when it moves fast
(no lag on a quick draw). Only the drawing is smoothed: the game rules in
mechanics.py still use the raw tracking, so nobody's draw is slowed down.
"""

import math

import cv2
import numpy as np

import mechanics as M
import tracker as T
import ui

# Body landmark ids
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
HAND_POINTS = ((15, 17, 19), (16, 18, 20))     # (wrist, pinky, index) for each arm in M.ARMS

# Smoothing (One Euro filter). Lower MIN_CUTOFF = steadier when still;
# higher BETA = less lag when moving fast.
MIN_CUTOFF = 0.5
BETA = 25.0
SMOOTH_RESET_SECONDS = 0.5     # start fresh if a player was missing this long

FLASH_SECONDS = 0.15           # how long the screen flashes white after a shot

# Colours (BGR)
SKIN = (120, 160, 205)
DENIM = (125, 85, 55)
BOOT = (35, 55, 85)
BELT = (30, 45, 70)
BUCKLE = (60, 200, 240)
HAT = (35, 70, 120)
HAT_BAND = (25, 35, 50)
GUN = (55, 55, 60)
EYE = (35, 30, 30)
SHADOW = (70, 125, 170)
CACTUS_GREEN = (60, 120, 50)


# --- Smoothing -------------------------------------------------------------------

class OneEuroFilter:
    """Smooths an array of 2D points over time (one filter per player)."""

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.x is None or t - self.t > SMOOTH_RESET_SECONDS:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        # How fast each point is moving (itself smoothed, so noise doesn't count as speed)
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        speed = np.linalg.norm(self.dx, axis=1, keepdims=True)
        # Faster points get a higher cutoff, i.e. less smoothing
        self.x = self.x + self._alpha(self.min_cutoff + self.beta * speed, dt) * (x - self.x)
        self.t = t
        return self.x


# --- Background -------------------------------------------------------------------

def make_desert(width, height):
    """Sunset sky, sun, sandy ground and a couple of cacti. Built once and reused."""
    img = np.zeros((height, width, 3), np.uint8)
    horizon = int(height * 0.62)
    top, bottom = np.array([120, 90, 200]), np.array([120, 190, 255])   # purple to orange
    for y in range(horizon):
        t = y / horizon
        img[y, :] = (1 - t) * top + t * bottom
    cv2.circle(img, (width // 2, horizon), int(height * 0.12), (150, 230, 255), -1)
    img[horizon:, :] = (90, 160, 210)                                   # sand
    cv2.line(img, (0, horizon), (width, horizon), (70, 130, 180), 3)
    for x, h in ((int(width * 0.08), 0.22), (int(width * 0.9), 0.17)):
        top_y = horizon - int(height * h)
        cv2.rectangle(img, (x - 10, top_y), (x + 10, horizon + 10), CACTUS_GREEN, -1)
        cv2.rectangle(img, (x - 38, top_y + 40), (x - 22, top_y + 90), CACTUS_GREEN, -1)
        cv2.rectangle(img, (x - 38, top_y + 80), (x - 10, top_y + 95), CACTUS_GREEN, -1)
        cv2.rectangle(img, (x + 22, top_y + 25), (x + 38, top_y + 70), CACTUS_GREEN, -1)
        cv2.rectangle(img, (x + 10, top_y + 60), (x + 38, top_y + 75), CACTUS_GREEN, -1)
    return img


# --- Drawing helpers ----------------------------------------------------------------

def _ipt(p):
    return int(round(p[0])), int(round(p[1]))


def _limb(img, a, b, width_a, width_b, color):
    """A tapered limb from a to b with rounded joints, like a capsule."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = np.linalg.norm(d)
    if length >= 1:
        perp = np.array([-d[1], d[0]]) / length
        quad = np.array([a + perp * width_a / 2, b + perp * width_b / 2,
                         b - perp * width_b / 2, a - perp * width_a / 2])
        cv2.fillConvexPoly(img, np.round(quad).astype(np.int32), color, cv2.LINE_AA)
    cv2.circle(img, _ipt(a), max(int(width_a / 2), 1), color, -1, cv2.LINE_AA)
    cv2.circle(img, _ipt(b), max(int(width_b / 2), 1), color, -1, cv2.LINE_AA)


def _gray(color):
    """Faded grey version of a colour, for the loser."""
    b, g, r = color
    lum = 0.11 * b + 0.59 * g + 0.3 * r
    v = int(lum * 0.5 + 120 * 0.5)
    return (v, v, v)


def _darker(color, f=0.6):
    return tuple(int(c * f) for c in color)


# --- The cowboy -------------------------------------------------------------------------

def draw_cowboy(img, pts, shirt, dead=False, muzzle_arm=None):
    """Draw a cowboy from 33 body points in pixel coordinates (a (33, 2) array).

    shirt: the player's colour. dead: draw faded with X eyes.
    muzzle_arm: index into mechanics.ARMS of the arm that just fired, if any.
    """
    paint = _gray if dead else (lambda c: c)
    skin, denim, boot, shirt_c = paint(SKIN), paint(DENIM), paint(BOOT), paint(shirt)

    s = max(np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER]), 30)   # body scale: shoulder width
    height = img.shape[0]

    # Shadow on the ground under the feet, if the feet are on screen
    feet_y = max(pts[L_ANKLE][1], pts[R_ANKLE][1])
    if feet_y < height:
        feet_x = (pts[L_ANKLE][0] + pts[R_ANKLE][0]) / 2
        cv2.ellipse(img, _ipt((feet_x, feet_y + 0.12 * s)), (int(0.9 * s), int(0.13 * s)),
                    0, 0, 360, SHADOW, -1, cv2.LINE_AA)

    # Legs and boots
    for hip, knee, ankle, heel, toe in ((L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_TOE),
                                        (R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_TOE)):
        _limb(img, pts[hip], pts[knee], 0.36 * s, 0.27 * s, denim)
        _limb(img, pts[knee], pts[ankle], 0.27 * s, 0.22 * s, denim)
        _limb(img, pts[heel], pts[toe], 0.2 * s, 0.15 * s, boot)
        _limb(img, pts[ankle], pts[heel], 0.22 * s, 0.2 * s, boot)

    # Torso: shirt from shoulders to hips, a little wider than the joints
    shoulder_dir = (pts[R_SHOULDER] - pts[L_SHOULDER]) / s
    hip_width = max(np.linalg.norm(pts[R_HIP] - pts[L_HIP]), 1)
    hip_dir = (pts[R_HIP] - pts[L_HIP]) / hip_width
    torso = np.array([pts[L_SHOULDER] - shoulder_dir * 0.1 * s,
                      pts[R_SHOULDER] + shoulder_dir * 0.1 * s,
                      pts[R_HIP] + hip_dir * 0.12 * s,
                      pts[L_HIP] - hip_dir * 0.12 * s])
    cv2.fillPoly(img, [np.round(torso).astype(np.int32)], shirt_c, cv2.LINE_AA)

    # Belt and buckle
    hips_mid = (pts[L_HIP] + pts[R_HIP]) / 2
    _limb(img, pts[L_HIP] - hip_dir * 0.1 * s, pts[R_HIP] + hip_dir * 0.1 * s,
          0.12 * s, 0.12 * s, paint(BELT))
    cv2.circle(img, _ipt(hips_mid), int(0.08 * s), paint(BUCKLE), -1, cv2.LINE_AA)

    # Head position and tilt come from the ears
    ear_l, ear_r = pts[LEFT_EAR], pts[RIGHT_EAR]
    if ear_l[0] > ear_r[0]:
        ear_l, ear_r = ear_r, ear_l                     # make "along" point screen-right
    head = (ear_l + ear_r) / 2
    ear_dist = np.linalg.norm(ear_r - ear_l)
    along = (ear_r - ear_l) / ear_dist if ear_dist > 1 else np.array([1.0, 0.0])
    up = np.array([along[1], -along[0]])
    tilt = math.degrees(math.atan2(along[1], along[0]))
    r = max(0.27 * s, 0.6 * ear_dist)                   # head radius

    # Neck and bandana
    shoulders_mid = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2
    _limb(img, shoulders_mid, head, 0.35 * r, 0.35 * r, skin)
    chin = head - up * 1.0 * r
    bandana = np.array([chin + along * 0.65 * r, chin - along * 0.65 * r, chin - up * 0.7 * r])
    cv2.fillConvexPoly(img, np.round(bandana).astype(np.int32), _darker(shirt_c), cv2.LINE_AA)

    # Arms (sleeves), hands and guns
    for i, (shoulder, elbow, wrist) in enumerate(M.ARMS):
        _limb(img, pts[shoulder], pts[elbow], 0.27 * s, 0.21 * s, shirt_c)
        _limb(img, pts[elbow], pts[wrist], 0.21 * s, 0.16 * s, shirt_c)

        _, pinky, index = HAND_POINTS[i]
        hand = (pts[wrist] + pts[pinky] + pts[index]) / 3
        forearm = hand - pts[elbow]
        direction = forearm / max(np.linalg.norm(forearm), 1)
        muzzle = hand + direction * 0.55 * s
        _limb(img, hand, muzzle, 0.1 * s, 0.08 * s, paint(GUN))               # barrel
        cv2.circle(img, _ipt(hand + direction * 0.12 * s), int(0.08 * s),
                   paint(GUN), -1, cv2.LINE_AA)                                 # cylinder
        cv2.circle(img, _ipt(hand), int(0.1 * s), skin, -1, cv2.LINE_AA)        # hand
        if i == muzzle_arm:
            draw_muzzle_flash(img, muzzle, r)

    # Head and face
    cv2.ellipse(img, _ipt(head), (int(r * 0.88), int(r)), tilt, 0, 360, skin, -1, cv2.LINE_AA)
    for eye in (pts[LEFT_EYE], pts[RIGHT_EYE]):
        if dead:
            d = 0.1 * r
            cv2.line(img, _ipt(eye - d), _ipt(eye + d), EYE, 2, cv2.LINE_AA)
            cv2.line(img, _ipt(eye + (d, -d)), _ipt(eye - (d, -d)), EYE, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, _ipt(eye), max(int(0.08 * r), 2), EYE, -1, cv2.LINE_AA)

    # Hat: tilts with the head. Crown with a dent on top, wide brim, dark band.
    def local(u, v):          # point in the head's own frame (u across, v up), in head radii
        return head + along * u * r + up * v * r
    crown = np.array([local(-0.95, 0.75), local(-0.8, 1.75), local(0, 1.55),
                      local(0.8, 1.75), local(0.95, 0.75)])
    cv2.fillPoly(img, [np.round(crown).astype(np.int32)], paint(HAT), cv2.LINE_AA)
    _limb(img, local(-0.93, 0.92), local(0.93, 0.92), 0.18 * r, 0.18 * r, paint(HAT_BAND))
    cv2.ellipse(img, _ipt(local(0, 0.75)), (int(1.65 * r), int(0.26 * r)), tilt, 0, 360,
                paint(HAT), -1, cv2.LINE_AA)


def draw_muzzle_flash(img, tip, size):
    cv2.circle(img, _ipt(tip), int(size * 1.2), (80, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(img, _ipt(tip), int(size * 0.6), ui.WHITE, -1, cv2.LINE_AA)
    ui.put_text(img, "BANG!", tip[0], tip[1] - size * 1.5, 1.4, ui.GOLD, 3)


def flash_screen(img, strength=0.7):
    img[:] = cv2.addWeighted(img, 1 - strength, np.full_like(img, 255), strength, 0)


# --- Scene -----------------------------------------------------------------------------

class Scene:
    """Draws the game world: the desert plus a smoothed cowboy for each player."""

    def __init__(self):
        self.background = None
        self.smoothers = {}        # player index -> OneEuroFilter

    def render(self, size, game, players, now):
        width, height = size
        if self.background is None or self.background.shape[1::-1] != size:
            self.background = make_desert(width, height)
        img = self.background.copy()

        result = game.state == M.RESULT
        shot = result and game.winner is not None and game.winner_arm is not None

        for p, pose in players.items():
            raw = np.array([[lm.x, lm.y] for lm in pose])
            smooth = self.smoothers.setdefault(p, OneEuroFilter())(raw, now)
            # Pixel coordinates; clip far off-screen points so drawing stays sane
            pts = np.clip(smooth * (width, height), -2 * width, 3 * width)

            dead = result and game.winner is not None and p != game.winner
            muzzle = game.winner_arm if shot and p == game.winner else None
            draw_cowboy(img, pts, T.PLAYER_COLORS[p], dead=dead, muzzle_arm=muzzle)

        if shot and now - game.result_time < FLASH_SECONDS:
            flash_screen(img)
        return img
