import cv2
import mediapipe as mp
import time
import math
import os
import sys
import random
import numpy as np

# System Audio Support
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _devices = AudioUtilities.GetSpeakers()
    if hasattr(_devices, 'Activate'):
        _interface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _system_volume = cast(_interface, POINTER(IAudioEndpointVolume))
    else:
        _system_volume = _devices.EndpointVolume
except Exception:
    _system_volume = None


class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositionVol(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox


    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


    def fingersUp(self, myHand=None):
        if myHand is not None:
            if isinstance(myHand, dict) and "lmList" in myHand:
                raw_lm = myHand["lmList"]
            else:
                raw_lm = myHand
        else:
            raw_lm = getattr(self, "lmList", [])

        if not raw_lm or len(raw_lm) < 21:
            return [0, 0, 0, 0, 0]

        # Extract (x, y) for each landmark 0..20
        coords = []
        for pt in raw_lm:
            if len(pt) == 3 and isinstance(pt[0], int) and len(raw_lm) > 0 and raw_lm[0][0] == 0:
                # Format: [id, x, y]
                coords.append((pt[1], pt[2]))
            else:
                # Format: [x, y, ...]
                coords.append((pt[0], pt[1]))

        fingers = []

        # Landmark references:
        # 0: Wrist, 2: Thumb MCP, 3: Thumb IP, 4: Thumb TIP
        # 5: Index MCP, 9: Middle MCP, 17: Pinky MCP
        thumb_tip = coords[4]
        thumb_ip = coords[3]
        thumb_mcp = coords[2]
        idx_mcp = coords[5]
        mid_mcp = coords[9]
        pky_mcp = coords[17]
        wrist = coords[0]

        # Orientation check: Is thumb on the left or right side of the palm?
        # If Index MCP is to the left of Pinky MCP in the image (idx_mcp[0] < pky_mcp[0]),
        # thumb is on the LEFT. Otherwise, thumb is on the RIGHT.
        if idx_mcp[0] < pky_mcp[0]:
            # Thumb is on the left -> OPEN when extended to the left (tip_x < ip_x)
            lateral_open = (thumb_tip[0] < thumb_ip[0])
        else:
            # Thumb is on the right -> OPEN when extended to the right (tip_x > ip_x)
            lateral_open = (thumb_tip[0] > thumb_ip[0])

        # Distances: Tip to Pinky MCP vs IP to Pinky MCP
        d_tip_pky = math.hypot(thumb_tip[0] - pky_mcp[0], thumb_tip[1] - pky_mcp[1])
        d_ip_pky = math.hypot(thumb_ip[0] - pky_mcp[0], thumb_ip[1] - pky_mcp[1])
        d_mcp_pky = math.hypot(thumb_mcp[0] - pky_mcp[0], thumb_mcp[1] - pky_mcp[1])

        # Distance to Middle MCP (checking if folded onto palm/knuckles in a fist)
        d_tip_mid = math.hypot(thumb_tip[0] - mid_mcp[0], thumb_tip[1] - mid_mcp[1])
        palm_size = math.hypot(wrist[0] - mid_mcp[0], wrist[1] - mid_mcp[1])
        if palm_size == 0:
            palm_size = 1.0

        is_tucked = (d_tip_mid < 0.50 * palm_size) or (d_tip_pky < 0.90 * d_mcp_pky)

        # Vertical thumb up (thumbs up gesture: tip above IP above MCP and extended from palm)
        vertical_up = (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]) and (d_tip_mid > 0.58 * palm_size)

        if (lateral_open and d_tip_pky > d_ip_pky and not is_tucked) or vertical_up:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        # Finger is open when tip is higher (smaller y) than PIP joint (id - 2)
        for id in range(1, 5):
            tip_id = self.tipIds[id]
            pip_id = tip_id - 2
            if coords[tip_id][1] < coords[pip_id][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


# =============================================================================
# ALL-IN-ONE HAND GESTURE RECOGNITION APPLICATION
# =============================================================================
class AllInOneApp:
    def __init__(self):
        self.mode = 0  # 0: All-In-One, 1: Tracking, 2: Fingers, 3: Volume, 4: RPS, 5: Snake, 6: Keyboard, 7: Presentation
        self.modes_info = [
            (0, "ALL-IN-ONE"),
            (1, "TRACKING"),
            (2, "FINGERS"),
            (3, "VOLUME"),
            (4, "RPS GAME"),
            (5, "SNAKE"),
            (6, "KEYBOARD"),
            (7, "SLIDES"),
        ]

        # Audio Volume state
        self.vol_per = 50
        self.vol_locked = False

        # RPS Battle state
        self.rps_status = "READY"  # READY, COUNTING, RESULT
        self.rps_timer_start = 0
        self.rps_player_move = None
        self.rps_ai_move = None
        self.rps_result_text = "Press 's' or Peace to Start!"
        self.rps_scores = [0, 0]  # [AI, Player]
        self.rps_result_time = 0
        self.rps_history = []

        # Snake Game state
        self.snake_points = []
        self.snake_lengths = []
        self.snake_cur_len = 0
        self.snake_allowed_len = 160
        self.snake_prev_head = (0, 0)
        self.snake_food = (400, 300)
        self.snake_score = 0
        self.snake_game_over = False

        # Virtual Keyboard state
        self.typed_text = ""
        self.last_key_press_time = 0
        self.keyboard_layout = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
            ["SPACE", "DEL", "CLEAR"]
        ]

        # Presentation state
        self.slide_idx = 0
        self.slides = []
        self.slide_cooldown = 0
        self.annotations = [[]]
        self.annotation_start = False
        self.load_slides()

        # UI & Touch state
        self.mode_hover_cooldown = 0
        self.clicked_mode = None

    def load_slides(self):
        folder = "Presentation"
        if os.path.exists(folder):
            files = sorted(os.listdir(folder), key=len)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(folder, f))
                    if img is not None:
                        self.slides.append(img)

    def draw_alpha_rect(self, img, x, y, w, h, color, alpha=0.65):
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 > x1 and y2 > y1:
            sub_img = img[y1:y2, x1:x2]
            overlay = np.full(sub_img.shape, color, dtype=np.uint8)
            cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0, sub_img)

    def draw_nav_bar(self, img, w, h, lmList=None, fingers=None):
        # Top banner background
        self.draw_alpha_rect(img, 0, 0, w, 52, (20, 22, 30), alpha=0.88)
        cv2.line(img, (0, 52), (w, 52), (60, 65, 80), 1)

        btn_w = w // 8
        for i, (m_id, m_name) in enumerate(self.modes_info):
            x1 = i * btn_w + 3
            x2 = (i + 1) * btn_w - 3
            y1 = 6
            y2 = 46

            is_active = (self.mode == m_id)

            # Air touch check with index fingertip (lmList[8])
            is_hovered = False
            if lmList and len(lmList) > 8:
                ix, iy = lmList[8][1], lmList[8][2]
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    is_hovered = True
                    # If middle finger is also raised or pinch, switch mode!
                    if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                        if time.time() - self.mode_hover_cooldown > 0.6:
                            self.mode = m_id
                            self.mode_hover_cooldown = time.time()

            if is_active:
                cv2.rectangle(img, (x1, y1), (x2, y2), (40, 180, 50), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                text_col = (255, 255, 255)
            elif is_hovered:
                cv2.rectangle(img, (x1, y1), (x2, y2), (180, 80, 220), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                text_col = (255, 255, 255)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (36, 40, 52), cv2.FILLED)
                cv2.rectangle(img, (x1, y1), (x2, y2), (65, 70, 85), 1)
                text_col = (190, 195, 205)

            font_scale = 0.42 if w < 1000 else 0.52
            cv2.putText(img, m_name, (x1 + 6, y1 + 25), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, text_col, 1, cv2.LINE_AA)

    def draw_bottom_bar(self, img, w, h):
        self.draw_alpha_rect(img, 0, h - 34, w, 34, (15, 17, 24), alpha=0.90)
        cv2.line(img, (0, h - 34), (w, h - 34), (60, 65, 80), 1)
        help_text = "Press [0-7] or Click Tabs to switch | 's' RPS Round | 'r' Restart Snake | 'q' Exit"
        cv2.putText(img, help_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 225), 1, cv2.LINE_AA)

    def classify_gesture(self, fingers):
        if not fingers or len(fingers) < 5:
            return "No Hand"
        t, i, m, r, p = fingers
        total = sum(fingers)

        if total == 0:
            return "ROCK ✊"
        elif total == 5:
            return "PAPER / PALM ✋"
        elif i == 1 and m == 1 and r == 0 and p == 0:
            return "SCISSORS ✌️"
        elif i == 1 and m == 0 and r == 0 and p == 0:
            return "POINTING ☝️"
        elif t == 0 and i == 0 and m == 0 and r == 0 and p == 1:
            return "PINKY 🤙"
        elif t == 1 and i == 0 and m == 0 and r == 0 and p == 0:
            return "THUMB UP 👍"
        elif (t == 1 or t == 0) and i == 1 and m == 0 and r == 0 and p == 1:
            return "ROCK ON 🤘"
        else:
            return f"{total} FINGERS"

    # -------------------------------------------------------------------------
    # MODE 0: ALL-IN-ONE UNIFIED
    # -------------------------------------------------------------------------
    def run_mode_unified(self, img, lmList, bbox, fingers, w, h):
        gesture_name = self.classify_gesture(fingers) if fingers else "Searching Hand..."
        total_fingers = fingers.count(1) if fingers else 0

        # Gesture Card Top-Left
        self.draw_alpha_rect(img, 20, 65, 340, 105, (20, 24, 35), alpha=0.82)
        cv2.rectangle(img, (20, 65), (360, 170), (79, 70, 229), 2)
        cv2.putText(img, "UNIFIED GESTURE HUD", (32, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 168, 185), 1, cv2.LINE_AA)
        cv2.putText(img, gesture_name, (32, 120), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)

        # Individual finger status indicators
        f_names = ["T", "I", "M", "R", "P"]
        for idx, fn in enumerate(f_names):
            bx = 32 + idx * 42
            by = 132
            is_up = fingers[idx] if fingers else 0
            bg_col = (40, 180, 50) if is_up else (50, 55, 68)
            cv2.rectangle(img, (bx, by), (bx + 34, by + 26), bg_col, cv2.FILLED)
            cv2.putText(img, fn, (bx + 11, by + 19), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(img, f"Total: {total_fingers}", (260, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Unified Volume Controller (Thumb-Index Pinch)
        if lmList and len(lmList) > 8:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pinch_dist = math.hypot(x2 - x1, y2 - y1)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED)

            vol_calc = np.interp(pinch_dist, [30, 200], [0, 100])
            vol_calc = 5 * round(vol_calc / 5)

            # Pinky folded down locks and applies master volume
            if fingers and fingers[4] == 0:
                self.vol_per = int(vol_calc)
                self.vol_locked = True
                if _system_volume:
                    try:
                        _system_volume.SetMasterVolumeLevelScalar(self.vol_per / 100.0, None)
                    except Exception:
                        pass
                cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
                status_vol = f"VOL LOCKED: {self.vol_per}%"
                bar_color = (0, 255, 120)
            else:
                self.vol_locked = False
                status_vol = f"Vol Adjust: {int(vol_calc)}%"
                bar_color = (255, 120, 0)

            # Volume Meter Left
            self.draw_alpha_rect(img, 25, 190, 45, 240, (18, 20, 30), alpha=0.8)
            cv2.rectangle(img, (25, 190), (70, 430), (70, 75, 90), 1)
            vol_bar_h = int(np.interp(self.vol_per if self.vol_locked else vol_calc, [0, 100], [428, 192]))
            cv2.rectangle(img, (27, vol_bar_h), (68, 428), bar_color, cv2.FILLED)
            cv2.putText(img, f"{self.vol_per}%", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(img, status_vol, (80, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 2)

    # -------------------------------------------------------------------------
    # MODE 1: HAND LANDMARK TRACKING & MESH INSPECTOR
    # -------------------------------------------------------------------------
    def run_mode_tracking(self, img, lmList, bbox, fingers, w, h):
        if bbox:
            cv2.rectangle(img, (bbox[0] - 15, bbox[1] - 15),
                          (bbox[2] + 15, bbox[3] + 15), (0, 255, 255), 2)
            cv2.putText(img, f"Hand: {bbox[2] - bbox[0]}x{bbox[3] - bbox[1]}",
                        (bbox[0] - 15, bbox[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Telemetry panel on right side
        self.draw_alpha_rect(img, w - 240, 65, 220, 210, (18, 22, 32), alpha=0.85)
        cv2.rectangle(img, (w - 240, 65), (w - 20, 275), (0, 220, 255), 1)
        cv2.putText(img, "LANDMARK TELEMETRY", (w - 225, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        pts_to_show = [(0, "Wrist"), (4, "Thumb"), (8, "Index"), (12, "Middle"), (16, "Ring"), (20, "Pinky")]
        for idx, (pid, pname) in enumerate(pts_to_show):
            coord_str = f"{pname}: None"
            if lmList and len(lmList) > pid:
                coord_str = f"{pname}: ({lmList[pid][1]}, {lmList[pid][2]})"
            cv2.putText(img, coord_str, (w - 225, 115 + idx * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 225, 235), 1)

    # -------------------------------------------------------------------------
    # MODE 2: FINGER COUNTER HUD
    # -------------------------------------------------------------------------
    def run_mode_fingers(self, img, lmList, bbox, fingers, w, h):
        total = fingers.count(1) if fingers else 0

        # Big Digital Counter Box
        self.draw_alpha_rect(img, 30, 80, 190, 210, (15, 20, 30), alpha=0.88)
        cv2.rectangle(img, (30, 80), (220, 290), (0, 200, 100), 3)
        cv2.putText(img, "FINGER COUNT", (45, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 205), 1)
        cv2.putText(img, str(total), (85, 230), cv2.FONT_HERSHEY_DUPLEX, 4.2, (0, 255, 120), 8)

        # Status of each finger
        f_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.draw_alpha_rect(img, 240, 80, 220, 190, (18, 22, 32), alpha=0.85)
        cv2.rectangle(img, (240, 80), (460, 270), (70, 75, 90), 1)
        cv2.putText(img, "FINGER STATES", (255, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        for idx, fl in enumerate(f_labels):
            is_open = fingers[idx] if fingers else 0
            col = (0, 255, 120) if is_open else (120, 125, 140)
            status = "EXTENDED" if is_open else "CLOSED"
            cv2.putText(img, f"{fl}: {status}", (255, 135 + idx * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # MODE 3: MASTER VOLUME CONTROLLER
    # -------------------------------------------------------------------------
    def run_mode_volume(self, img, lmList, bbox, fingers, w, h):
        if not lmList or len(lmList) < 9:
            cv2.putText(img, "Show hand to control master volume", (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            return

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 12, (0, 255, 255), cv2.FILLED)

        vol_calc = np.interp(length, [35, 220], [0, 100])
        vol_calc = 5 * round(vol_calc / 5)

        # Pinky fold down locks and applies master volume
        if fingers and fingers[4] == 0:
            self.vol_per = int(vol_calc)
            self.vol_locked = True
            if _system_volume:
                try:
                    _system_volume.SetMasterVolumeLevelScalar(self.vol_per / 100.0, None)
                except Exception:
                    pass
            cv2.circle(img, (cx, cy), 16, (0, 255, 0), cv2.FILLED)
            tag = f"VOLUME SET: {self.vol_per}% [LOCKED]"
            tag_color = (0, 255, 120)
        else:
            self.vol_locked = False
            tag = f"VOLUME: {int(vol_calc)}% (Fold Pinky to Apply)"
            tag_color = (0, 220, 255)

        # Vertical Volume Gauge
        self.draw_alpha_rect(img, 45, 120, 55, 340, (15, 18, 28), alpha=0.88)
        cv2.rectangle(img, (45, 120), (100, 460), (70, 75, 90), 2)
        target_vol = self.vol_per if self.vol_locked else vol_calc
        vol_bar_y = int(np.interp(target_vol, [0, 100], [458, 122]))
        cv2.rectangle(img, (47, vol_bar_y), (98, 458), tag_color, cv2.FILLED)

        cv2.putText(img, f"{int(target_vol)}%", (40, 490), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, tag, (120, 160), cv2.FONT_HERSHEY_DUPLEX, 0.9, tag_color, 2)

    # -------------------------------------------------------------------------
    # MODE 4: ROCK PAPER SCISSORS GAME
    # -------------------------------------------------------------------------
    def run_mode_rps(self, img, lmList, bbox, fingers, w, h):
        # Classify current gesture
        p_move = None
        move_name = "None"
        if fingers:
            if fingers == [0, 0, 0, 0, 0] or sum(fingers) == 0:
                p_move = 1
                move_name = "ROCK ✊"
            elif sum(fingers) >= 4:
                p_move = 2
                move_name = "PAPER ✋"
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                p_move = 3
                move_name = "SCISSORS ✌️"

        # Top Live Gesture Badge
        self.draw_alpha_rect(img, w - 280, 65, 260, 55, (20, 24, 35), alpha=0.85)
        cv2.rectangle(img, (w - 280, 65), (w - 20, 120), (0, 220, 255), 1)
        cv2.putText(img, f"LIVE: {move_name}", (w - 265, 102), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 230, 255), 2)

        # Score Banner
        self.draw_alpha_rect(img, 30, 65, 300, 55, (20, 24, 35), alpha=0.85)
        cv2.rectangle(img, (30, 65), (330, 120), (70, 75, 90), 1)
        score_text = f"YOU: {self.rps_scores[1]}  |  AI: {self.rps_scores[0]}"
        cv2.putText(img, score_text, (45, 102), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 120), 2)

        now = time.time()
        if self.rps_status == "READY":
            self.draw_alpha_rect(img, w // 2 - 270, h // 2 - 50, 540, 100, (15, 18, 28), alpha=0.9)
            cv2.rectangle(img, (w // 2 - 270, h // 2 - 50), (w // 2 + 270, h // 2 + 50), (0, 220, 255), 2)
            cv2.putText(img, "Press 's' or show Scissors to Start!", (w // 2 - 240, h // 2 + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        elif self.rps_status == "COUNTING":
            if p_move is not None:
                self.rps_history.append(p_move)

            elapsed = now - self.rps_timer_start
            rem = 2.2 - elapsed
            if rem > 0:
                count_val = max(1, int(math.ceil(rem)))
                self.draw_alpha_rect(img, w // 2 - 160, h // 2 - 80, 320, 160, (15, 18, 28), alpha=0.92)
                cv2.rectangle(img, (w // 2 - 160, h // 2 - 80), (w // 2 + 160, h // 2 + 80), (0, 220, 255), 3)
                cv2.putText(img, f"SHOW IN: {count_val}", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 220, 255), 3)
                cv2.putText(img, "HOLD YOUR MOVE!", (w // 2 - 110, h // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Round complete!
                final_move = p_move
                if final_move is None and self.rps_history:
                    final_move = max(set(self.rps_history), key=self.rps_history.count)

                self.rps_status = "RESULT"
                self.rps_player_move = final_move
                self.rps_ai_move = random.randint(1, 3)
                self.rps_result_time = now

                # Evaluate winner
                if self.rps_player_move is None:
                    self.rps_result_text = "No hand detected! Try again"
                elif self.rps_player_move == self.rps_ai_move:
                    self.rps_result_text = "IT'S A TIE!"
                elif (self.rps_player_move == 1 and self.rps_ai_move == 3) or \
                     (self.rps_player_move == 2 and self.rps_ai_move == 1) or \
                     (self.rps_player_move == 3 and self.rps_ai_move == 2):
                    self.rps_scores[1] += 1
                    self.rps_result_text = "YOU WIN! 🎉"
                else:
                    self.rps_scores[0] += 1
                    self.rps_result_text = "AI WINS! 🤖"

        elif self.rps_status == "RESULT":
            if now - self.rps_result_time > 2.8:
                self.rps_status = "READY"
                return

            self.draw_alpha_rect(img, w // 2 - 280, h // 2 - 90, 560, 180, (15, 18, 28), alpha=0.92)
            b_col = (0, 255, 120) if "WIN" in self.rps_result_text else ((0, 100, 255) if "AI" in self.rps_result_text else (0, 220, 255))
            cv2.rectangle(img, (w // 2 - 280, h // 2 - 90), (w // 2 + 280, h // 2 + 90), b_col, 3)

            move_names = {1: "ROCK ✊", 2: "PAPER ✋", 3: "SCISSORS ✌️"}
            p_str = move_names.get(self.rps_player_move, "None")
            a_str = move_names.get(self.rps_ai_move, "None")

            cv2.putText(img, self.rps_result_text, (w // 2 - 160, h // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(img, f"You: {p_str}   vs   AI: {a_str}", (w // 2 - 230, h // 2 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 230, 255), 2)

    # -------------------------------------------------------------------------
    # MODE 5: SNAKE ARCADE GAME
    # -------------------------------------------------------------------------
    def run_mode_snake(self, img, lmList, bbox, fingers, w, h):
        if self.snake_game_over:
            self.draw_alpha_rect(img, w // 2 - 250, h // 2 - 80, 500, 160, (15, 18, 28), alpha=0.92)
            cv2.rectangle(img, (w // 2 - 250, h // 2 - 80), (w // 2 + 250, h // 2 + 80), (0, 0, 255), 3)
            cv2.putText(img, "GAME OVER!", (w // 2 - 140, h // 2 - 15), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 3)
            cv2.putText(img, f"Final Score: {self.snake_score}  (Press 'r' to Restart)",
                        (w // 2 - 210, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            return

        if lmList and len(lmList) > 8:
            cx, cy = lmList[8][1], lmList[8][2]
            px, py = self.snake_prev_head

            if px == 0 and py == 0:
                self.snake_prev_head = (cx, cy)
                return

            self.snake_points.append([cx, cy])
            d = math.hypot(cx - px, cy - py)
            self.snake_lengths.append(d)
            self.snake_cur_len += d
            self.snake_prev_head = (cx, cy)

            # Reduce snake length to allowed length
            if self.snake_cur_len > self.snake_allowed_len:
                for i, length in enumerate(self.snake_lengths):
                    self.snake_cur_len -= length
                    self.snake_lengths.pop(i)
                    self.snake_points.pop(i)
                    if self.snake_cur_len < self.snake_allowed_len:
                        break

            # Eat Food check
            fx, fy = self.snake_food
            if math.hypot(cx - fx, cy - fy) < 45:
                self.snake_food = (random.randint(120, max(200, w - 120)), random.randint(100, max(180, h - 100)))
                self.snake_allowed_len += 45
                self.snake_score += 1

            # Self collision check
            if len(self.snake_points) > 12:
                pts = np.array(self.snake_points[:-8], np.int32).reshape((-1, 1, 2))
                dist_tail = cv2.pointPolygonTest(pts, (cx, cy), True)
                if -0.2 <= dist_tail <= 0.2:
                    self.snake_game_over = True

        # Draw Snake
        if self.snake_points:
            for i in range(1, len(self.snake_points)):
                cv2.line(img, tuple(self.snake_points[i - 1]), tuple(self.snake_points[i]), (0, 0, 255), 18)
            cv2.circle(img, tuple(self.snake_points[-1]), 18, (0, 255, 0), cv2.FILLED)

        # Draw Food Donut
        fx, fy = self.snake_food
        cv2.circle(img, (fx, fy), 20, (0, 215, 255), cv2.FILLED)
        cv2.circle(img, (fx, fy), 8, (255, 255, 255), cv2.FILLED)
        cv2.putText(img, "FOOD", (fx - 18, fy + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

        # Draw Score Badge
        self.draw_alpha_rect(img, 30, 65, 200, 50, (20, 24, 35), alpha=0.85)
        cv2.rectangle(img, (30, 65), (230, 115), (0, 255, 120), 1)
        cv2.putText(img, f"SCORE: {self.snake_score}", (45, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 120), 2)

    # -------------------------------------------------------------------------
    # MODE 6: VIRTUAL AIR KEYBOARD
    # -------------------------------------------------------------------------
    def run_mode_keyboard(self, img, lmList, bbox, fingers, w, h):
        key_w = 64
        key_h = 56
        spacing = 8
        start_y = 80
        total_grid_w = 10 * (key_w + spacing) - spacing
        start_x = (w - total_grid_w) // 2

        ix, iy = (lmList[8][1], lmList[8][2]) if (lmList and len(lmList) > 8) else (-1, -1)

        # Draw Rows
        for r_idx, row in enumerate(self.keyboard_layout):
            curr_y = start_y + r_idx * (key_h + spacing)

            if r_idx < 3:
                curr_x = start_x + (r_idx * 18)
                for k in row:
                    kw = key_w
                    is_hov = (curr_x <= ix <= curr_x + kw and curr_y <= iy <= curr_y + key_h)
                    bg_col = (0, 220, 100) if is_hov else (180, 50, 180)

                    cv2.rectangle(img, (curr_x, curr_y), (curr_x + kw, curr_y + key_h), bg_col, cv2.FILLED)
                    cv2.rectangle(img, (curr_x, curr_y), (curr_x + kw, curr_y + key_h), (255, 255, 255), 1)
                    cv2.putText(img, k, (curr_x + 18, curr_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

                    # Pinch Click Check (Landmark 8 and 12)
                    if is_hov and lmList and len(lmList) > 12:
                        pinch_click = math.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
                        if pinch_click < 32 and (time.time() - self.last_key_press_time) > 0.35:
                            self.last_key_press_time = time.time()
                            self.typed_text += k

                    curr_x += kw + spacing
            else:
                # Row 4: SPACE, DEL, CLEAR
                special_keys = [("SPACE", 220), ("DEL", 95), ("CLEAR", 115)]
                curr_x = start_x + 80
                for sk_name, sk_w in special_keys:
                    is_hov = (curr_x <= ix <= curr_x + sk_w and curr_y <= iy <= curr_y + key_h)
                    bg_col = (0, 200, 220) if is_hov else (50, 55, 75)

                    cv2.rectangle(img, (curr_x, curr_y), (curr_x + sk_w, curr_y + key_h), bg_col, cv2.FILLED)
                    cv2.rectangle(img, (curr_x, curr_y), (curr_x + sk_w, curr_y + key_h), (255, 255, 255), 1)
                    cv2.putText(img, sk_name, (curr_x + 15, curr_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

                    if is_hov and lmList and len(lmList) > 12:
                        pinch_click = math.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
                        if pinch_click < 32 and (time.time() - self.last_key_press_time) > 0.35:
                            self.last_key_press_time = time.time()
                            if sk_name == "SPACE":
                                self.typed_text += " "
                            elif sk_name == "DEL":
                                self.typed_text = self.typed_text[:-1]
                            elif sk_name == "CLEAR":
                                self.typed_text = ""

                    curr_x += sk_w + spacing

        # Live Typed Text Output Box
        self.draw_alpha_rect(img, start_x, start_y + 4 * (key_h + spacing) + 10, total_grid_w, 65, (15, 18, 28), alpha=0.92)
        cv2.rectangle(img, (start_x, start_y + 4 * (key_h + spacing) + 10),
                      (start_x + total_grid_w, start_y + 4 * (key_h + spacing) + 75), (0, 220, 255), 2)
        cv2.putText(img, f"TYPED: {self.typed_text}_", (start_x + 20, start_y + 4 * (key_h + spacing) + 52),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # -------------------------------------------------------------------------
    # MODE 7: PRESENTATION SLIDES CONTROLLER
    # -------------------------------------------------------------------------
    def run_mode_presentation(self, img, lmList, bbox, fingers, w, h):
        num_slides = max(1, len(self.slides))

        # Slide switching gestures (with cooldown debounce)
        if fingers:
            if time.time() - self.slide_cooldown > 0.7:
                # Thumb only: Previous slide
                if fingers == [1, 0, 0, 0, 0]:
                    if self.slide_idx > 0:
                        self.slide_idx -= 1
                        self.slide_cooldown = time.time()
                        self.annotations = [[]]
                # Pinky only: Next slide
                elif fingers == [0, 0, 0, 0, 1]:
                    if self.slide_idx < num_slides - 1:
                        self.slide_idx += 1
                        self.slide_cooldown = time.time()
                        self.annotations = [[]]

        # In-Air Whiteboard Drawing
        if lmList and len(lmList) > 8 and fingers:
            ix, iy = lmList[8][1], lmList[8][2]
            # Index only: Draw
            if fingers == [0, 1, 0, 0, 0]:
                cv2.circle(img, (ix, iy), 10, (0, 0, 255), cv2.FILLED)
                if not self.annotation_start:
                    self.annotation_start = True
                    self.annotations.append([])
                self.annotations[-1].append((ix, iy))
            # 3 or more fingers up: Clear whiteboard drawings
            elif fingers.count(1) >= 3:
                self.annotations = [[]]
                self.annotation_start = False
            else:
                self.annotation_start = False

        # Draw annotations
        for line in self.annotations:
            for j in range(1, len(line)):
                cv2.line(img, line[j - 1], line[j], (0, 0, 255), 6)

        # Slide Preview Picture-In-Picture (PiP)
        if self.slides and self.slide_idx < len(self.slides):
            curr_slide = self.slides[self.slide_idx]
            pip_w, pip_h = 340, 210
            pip_resized = cv2.resize(curr_slide, (pip_w, pip_h))
            img[70:70 + pip_h, w - pip_w - 25:w - 25] = pip_resized
            cv2.rectangle(img, (w - pip_w - 25, 70), (w - 25, 70 + pip_h), (0, 220, 255), 2)

        # Slide Info Card
        self.draw_alpha_rect(img, 30, 70, 360, 110, (15, 18, 28), alpha=0.88)
        cv2.rectangle(img, (30, 70), (390, 180), (0, 220, 255), 2)
        cv2.putText(img, f"SLIDE {self.slide_idx + 1} OF {num_slides}", (45, 105),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "Thumb: Prev Slide  |  Pinky: Next Slide", (45, 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1)
        cv2.putText(img, "Index Up: Draw Ink  |  3 Fingers: Clear Canvas", (45, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 205, 215), 1)

    # -------------------------------------------------------------------------
    # MAIN EXECUTION LOOP
    # -------------------------------------------------------------------------
    def run(self):
        print("\n=======================================================")
        print("  HAND GESTURE RECOGNITION - ALL-IN-ONE ENGINE ACTIVE")
        print("=======================================================")
        print("Features available in this single execution:")
        print("  [0] Unified All-In-One HUD")
        print("  [1] Hand Tracking & Mesh Telemetry")
        print("  [2] Finger Counter (0-5)")
        print("  [3] Master Volume Controller")
        print("  [4] Rock-Paper-Scissors AI Battle")
        print("  [5] Snake Arcade Game")
        print("  [6] Virtual Air Keyboard")
        print("  [7] Presentation Controller")
        print("Press '0'-'7' or Click Top Tabs to switch modes.")
        print("Press 'q' or ESC to Exit.\n")

        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        detector = HandDetector(detectionCon=0.75, maxHands=1)
        p_time = time.time()

        window_name = "Hand Gesture Recognition - All-in-One Engine"
        cv2.namedWindow(window_name)

        # Mouse click callback for top buttons
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 5 <= y <= 48:
                    btn_w = param["width"] // 8
                    idx = int(x // btn_w)
                    if 0 <= idx <= 7:
                        self.mode = idx

        params = {"width": 1280}
        cv2.setMouseCallback(window_name, on_mouse, params)

        while True:
            success, img = cap.read()
            if not success or img is None:
                continue

            # Mirror for natural interaction
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            params["width"] = w

            # Detect hand and landmarks
            img = detector.findHands(img, draw=(self.mode in [0, 1, 2]))
            lmList, bbox = detector.findPositionVol(img, draw=False)
            fingers = detector.fingersUp() if lmList else [0, 0, 0, 0, 0]

            # Route to active mode
            if self.mode == 0:
                self.run_mode_unified(img, lmList, bbox, fingers, w, h)
            elif self.mode == 1:
                self.run_mode_tracking(img, lmList, bbox, fingers, w, h)
            elif self.mode == 2:
                self.run_mode_fingers(img, lmList, bbox, fingers, w, h)
            elif self.mode == 3:
                self.run_mode_volume(img, lmList, bbox, fingers, w, h)
            elif self.mode == 4:
                self.run_mode_rps(img, lmList, bbox, fingers, w, h)
            elif self.mode == 5:
                self.run_mode_snake(img, lmList, bbox, fingers, w, h)
            elif self.mode == 6:
                self.run_mode_keyboard(img, lmList, bbox, fingers, w, h)
            elif self.mode == 7:
                self.run_mode_presentation(img, lmList, bbox, fingers, w, h)

            # Draw Navigation & Bottom HUD
            self.draw_nav_bar(img, w, h, lmList, fingers)
            self.draw_bottom_bar(img, w, h)

            # FPS counter
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            cv2.putText(img, f"FPS: {int(fps)}", (w - 110, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)

            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif ord('0') <= key <= ord('7'):
                self.mode = key - ord('0')
            elif key == ord('s') and self.mode == 4:
                self.rps_status = "COUNTING"
                self.rps_timer_start = time.time()
                self.rps_history = []
            elif key == ord('r') and self.mode == 5:
                self.snake_points = []
                self.snake_lengths = []
                self.snake_cur_len = 0
                self.snake_allowed_len = 160
                self.snake_prev_head = (0, 0)
                self.snake_score = 0
                self.snake_game_over = False

        cap.release()
        cv2.destroyAllWindows()


def main():
    app = AllInOneApp()
    app.run()


if __name__ == "__main__":
    main()