"""
User interface for Cowboy Standoff: colours, text, the lobby screen, the
in-game HUD (instructions, countdown, results) and the live camera preview.
"""

import math

import cv2
import numpy as np

import mechanics as M
import tracker as T

# Colours (OpenCV uses BGR order)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (40, 40, 230)
GREEN = (80, 230, 80)
GOLD = (60, 200, 255)

TITLE = "COWBOY STANDOFF"


def put_text(img, text, center_x, y, scale, color, thickness=2):
    """Draw text centred on center_x, with a dark outline so it reads on any background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    # Keep the text inside the screen even if center_x is near an edge
    x = min(max(center_x - w / 2, 5), img.shape[1] - w - 5)
    origin = (int(x), int(y))
    cv2.putText(img, text, origin, font, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def player_x(img, p):
    """Horizontal centre of each player's half of the screen."""
    return img.shape[1] * (0.25 if p == 0 else 0.75)


def draw_tracking(img, players, hands):
    """Draw raw tracking skeletons: bodies in player colours, hands in white."""
    height, width = img.shape[:2]
    for p, pose in players.items():
        color = T.PLAYER_COLORS[p]
        T.draw_skeleton(img, T.to_pixels(pose, width, height), T.POSE_CONNECTIONS, color, color, 2)
    for hand in hands.values():
        T.draw_skeleton(img, T.to_pixels(hand, width, height), T.HAND_CONNECTIONS, WHITE, (0, 200, 0), 2)


def draw_ready_status(img, game, now):
    """Per-player "make an OK sign" / "READY!" line along the bottom."""
    for p in range(2):
        if not game.present(p, now):
            status, color = "step into view", WHITE
        elif game.ready[p]:
            status, color = "READY!", GREEN
        elif game.ok_since[p] is not None:
            status, color = "hold it...", GOLD
        else:
            status, color = "make an OK sign", WHITE
        put_text(img, f"P{p + 1}: {status}", player_x(img, p), img.shape[0] - 75, 0.8, color)


def render_lobby(game, frame, players, hands, now, size):
    """Full-screen camera view with tracking and ready status."""
    width, height = size
    img = cv2.resize(frame, size)
    draw_tracking(img, players, hands)
    img[:90] = (img[:90] * 0.4).astype(np.uint8)    # darken the title bar
    put_text(img, TITLE, width / 2, 60, 1.6, GOLD, 3)
    if game.message:
        put_text(img, game.message, width / 2, 130, 0.7, RED)
    put_text(img, "Both players: make an OK sign to get ready", width / 2, height - 30, 0.75, WHITE)
    draw_ready_status(img, game, now)
    return img


def draw_hud(img, game, players, now, aspect):
    """Instructions, countdown and results drawn over the game scene."""
    width, height = img.shape[1], img.shape[0]

    if game.state == M.HOLSTER:
        put_text(img, "Lower both hands to your holsters", width / 2, 60, 1.0, WHITE)
        for p in range(2):
            ok = p in players and M.arms_holstered(players[p], aspect)
            put_text(img, f"P{p + 1}: " + ("holstered" if ok else "arms down"),
                     player_x(img, p), height - 75, 0.8, GREEN if ok else WHITE)
        if game.holster_since is not None:
            progress = min((now - game.holster_since) / M.HOLSTER_HOLD_SECONDS, 1)
            left = width // 2 - 150
            cv2.rectangle(img, (left, 125), (left + int(300 * progress), 140), GOLD, -1)
            cv2.rectangle(img, (left, 125), (left + 300, 140), WHITE, 2)

    elif game.state == M.COUNTDOWN:
        put_text(img, "Steady... don't draw yet", width / 2, 60, 1.0, WHITE)
        remaining = max(math.ceil(game.countdown_end - now), 1)
        put_text(img, str(remaining), width / 2, height * 0.35, 5.0, WHITE, 8)

    elif game.state == M.DRAW:
        put_text(img, "DRAW!", width / 2, height * 0.35, 5.0, RED, 10)

    elif game.state == M.RESULT:
        if game.winner is None:
            headline, color = "NO WINNER", WHITE
        else:
            headline, color = f"PLAYER {game.winner + 1} WINS!", T.PLAYER_COLORS[game.winner]
        # Dark banner down on the ground, below shoulder height, so it doesn't
        # cover the gunshot
        top = int(height * 0.62)
        img[top:top + 145] = (img[top:top + 145] * 0.35).astype(np.uint8)
        put_text(img, headline, width / 2, top + 70, 2.2, color, 5)
        put_text(img, game.message, width / 2, top + 125, 0.9, WHITE)
        if now - game.result_time >= M.RESULT_MIN_SECONDS:
            put_text(img, "Both make an OK sign to play again", width / 2, height - 30, 0.75, WHITE)
            draw_ready_status(img, game, now)

    if game.message and game.state in (M.HOLSTER, M.COUNTDOWN):
        put_text(img, game.message, width / 2, 175, 0.7, RED)


def draw_camera_preview(img, frame, players, hands, aspect, preview_width=180):
    """Small live camera view in the top-right corner so players can check they're in frame."""
    width = img.shape[1]
    preview_height = int(preview_width / aspect)
    preview = cv2.resize(frame, (preview_width, preview_height))
    draw_tracking(preview, players, hands)
    x0, y0 = width - preview_width - 10, 10
    img[y0:y0 + preview_height, x0:x0 + preview_width] = preview
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + preview_width + 1, y0 + preview_height + 1), WHITE, 2)
