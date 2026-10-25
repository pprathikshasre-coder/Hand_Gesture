import os
import sys
import time
import math
import random
import threading
import subprocess
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import psutil

# MediaPipe & Vision modules
import HandTrackingModule as htm
import cvzone

# Master Audio
try:
    from pycaw.pycaw import AudioUtilities
    devices = AudioUtilities.GetSpeakers()
    if hasattr(devices, 'Activate'):
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        system_volume = cast(interface, POINTER(IAudioEndpointVolume))
    else:
        system_volume = devices.EndpointVolume
except Exception as e:
    print(f"[Audio Init Warning]: {e}")
    system_volume = None

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
class State:
    def __init__(self):
        self.mode = "tracking"  # tracking, fingers, volume, rps, snake, keyboard, presentation
        self.mirror = True
        self.is_paused = False
        self.fps = 0.0
        self.detected_gesture = "Detecting..."
        self.hands_count = 0
        self.fingers_state = [0, 0, 0, 0, 0]
        self.fingers_count = 0
        self.current_volume = 50
        
        # RPS State
        self.rps_status = "Ready"  # Ready, Counting, Result
        self.rps_timer_start = 0
        self.rps_player_move = None
        self.rps_ai_move = None
        self.rps_result_text = "Press 'Start Round' to Play!"
        self.rps_scores = [0, 0]  # [AI, Player]
        
        # Snake State
        self.snake_points = []
        self.snake_lengths = []
        self.snake_cur_len = 0
        self.snake_allowed_len = 150
        self.snake_prev_head = (0, 0)
        self.snake_food = (320, 240)
        self.snake_score = 0
        self.snake_game_over = False
        
        # Keyboard State
        self.typed_text = ""
        self.last_key_press_time = 0
        
        # Presentation State
        self.slide_index = 0
        self.slide_btn_pressed = False
        self.slide_btn_counter = 0

state = State()

# Load Assets
donut_img = cv2.imread("Resources/Donut.png", cv2.IMREAD_UNCHANGED)
rps_images = {
    1: cv2.imread("Resources/1.png", cv2.IMREAD_UNCHANGED),
    2: cv2.imread("Resources/2.png", cv2.IMREAD_UNCHANGED),
    3: cv2.imread("Resources/3.png", cv2.IMREAD_UNCHANGED)
}

slides_list = []
if os.path.exists("Presentation"):
    slides_list = sorted([os.path.join("Presentation", f) for f in os.listdir("Presentation") if f.endswith(('.png', '.jpg'))], key=len)


# -----------------------------------------------------------------------------
# CAMERA WORKER & VISION ENGINE
# -----------------------------------------------------------------------------
class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = htm.HandDetector(detectionCon=0.75, trackCon=0.75)
        self.p_time = time.time()
        self.lock = threading.Lock()
        
        # Init keyboard keys
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M", "SPACE", "DEL"]
        ]
        self.key_buttons = []
        start_y = 60
        for r, row in enumerate(self.keys):
            for c, k in enumerate(row):
                w = 110 if k in ["SPACE", "DEL"] else 55
                x = 40 + c * 65
                y = start_y + r * 65
                self.key_buttons.append({"rect": (x, y, w, 55), "text": k})

    def get_frame(self):
        with self.lock:
            if not self.cap.isOpened() or state.is_paused:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera Paused or Closed", (420, 360),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)
                ret, jpeg = cv2.imencode('.jpg', blank)
                return jpeg.tobytes()

            success, frame = self.cap.read()
            if not success or frame is None:
                blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', blank)
                return jpeg.tobytes()

            if state.mirror:
                frame = cv2.flip(frame, 1)

            # Process with Hand Detector
            frame = self.detector.findHands(frame, draw=(state.mode == "tracking"))
            lm_list, bbox = self.detector.findPositionVol(frame, draw=False)

            # Compute FPS
            c_time = time.time()
            state.fps = round(1.0 / (c_time - self.p_time), 1) if (c_time - self.p_time) > 0 else 0.0
            self.p_time = c_time

            if lm_list:
                state.hands_count = 1
                fingers = self.detector.fingersUp()
                state.fingers_state = fingers
                state.fingers_count = fingers.count(1)

                # Process by active mode
                if state.mode == "tracking":
                    self.process_tracking(frame, lm_list, fingers)
                elif state.mode == "fingers":
                    self.process_fingers(frame, lm_list, fingers)
                elif state.mode == "volume":
                    self.process_volume(frame, lm_list, fingers)
                elif state.mode == "rps":
                    self.process_rps(frame, lm_list, fingers)
                elif state.mode == "snake":
                    self.process_snake(frame, lm_list, fingers)
                elif state.mode == "keyboard":
                    self.process_keyboard(frame, lm_list, fingers)
                elif state.mode == "presentation":
                    self.process_presentation(frame, lm_list, fingers)
            else:
                state.hands_count = 0
                state.fingers_count = 0
                state.detected_gesture = "No Hand Detected"

            # Overlay HUD top banner
            self.draw_hud_banner(frame)

            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return jpeg.tobytes()

    def draw_hud_banner(self, frame):
        # Subtle glassmorphism top header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 50), (20, 20, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Mode Badge
        mode_str = f"MODE: {state.mode.upper()}"
        cv2.putText(frame, mode_str, (25, 33), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 230, 255), 2)

        # FPS Badge
        fps_str = f"FPS: {int(state.fps)}"
        cv2.putText(frame, fps_str, (320, 33), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 255, 120), 2)

        # Gesture Badge
        cv2.putText(frame, f"GESTURE: {state.detected_gesture}", (500, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 0), 2)

        # Hands Badge
        h_str = f"HANDS: {state.hands_count}"
        cv2.putText(frame, h_str, (1120, 33), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 255), 2)

    # 1. TRACKING
    def process_tracking(self, frame, lm_list, fingers):
        state.detected_gesture = f"{state.fingers_count} Fingers Raised"

    # 2. FINGERS
    def process_fingers(self, frame, lm_list, fingers):
        count = state.fingers_count
        state.detected_gesture = f"Count: {count}"
        # Large graphical counter card on camera
        cv2.rectangle(frame, (40, 80), (220, 260), (0, 180, 60), cv2.FILLED)
        cv2.rectangle(frame, (40, 80), (220, 260), (255, 255, 255), 3)
        cv2.putText(frame, str(count), (90, 220), cv2.FONT_HERSHEY_DUPLEX, 4.5, (255, 255, 255), 8)
        cv2.putText(frame, "FINGERS", (65, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 3. VOLUME
    def process_volume(self, frame, lm_list, fingers):
        length, frame, line_info = self.detector.findDistance(4, 8, frame)
        vol_per = np.interp(length, [30, 220], [0, 100])
        vol_per = 5 * round(vol_per / 5)
        
        # If pinky is down, lock and apply volume
        if not fingers[4]:
            if system_volume:
                try:
                    system_volume.SetMasterVolumeLevelScalar(vol_per / 100.0, None)
                except Exception:
                    pass
            state.current_volume = int(vol_per)
            cv2.circle(frame, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            state.detected_gesture = f"Volume {int(vol_per)}% [APPLIED]"
            col = (0, 255, 0)
        else:
            state.detected_gesture = f"Volume {int(vol_per)}% (Pinch)"
            col = (255, 120, 0)

        # On-screen visual volume meter
        vol_bar_y = int(np.interp(vol_per, [0, 100], [420, 150]))
        cv2.rectangle(frame, (50, 150), (90, 420), col, 3)
        cv2.rectangle(frame, (50, vol_bar_y), (90, 420), col, cv2.FILLED)
        cv2.putText(frame, f"{int(vol_per)} %", (45, 460), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)
        cv2.putText(frame, "Pinky Down: LOCK", (45, 495), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 4. ROCK PAPER SCISSORS
    def process_rps(self, frame, lm_list, fingers):
        # Identify player move
        player_move = None
        if fingers == [0, 0, 0, 0, 0]:
            player_move = 1  # Rock
            state.detected_gesture = "Rock ✊"
        elif fingers == [1, 1, 1, 1, 1]:
            player_move = 2  # Paper
            state.detected_gesture = "Paper ✋"
        elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            player_move = 3  # Scissors
            state.detected_gesture = "Scissors ✌️"
        else:
            state.detected_gesture = "Unknown Pose"

        if state.rps_status == "Counting":
            elapsed = time.time() - state.rps_timer_start
            remaining = 3 - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f"SHOW IN: {remaining}", (440, 360),
                            cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 220, 255), 6)
            else:
                # Countdown finished, resolve round!
                state.rps_status = "Result"
                state.rps_player_move = player_move
                state.rps_ai_move = random.randint(1, 3)

                # Determine Winner
                if state.rps_player_move is None:
                    state.rps_result_text = "Round Incomplete: No gesture detected!"
                elif state.rps_player_move == state.rps_ai_move:
                    state.rps_result_text = "It's a TIE!"
                elif (state.rps_player_move == 1 and state.rps_ai_move == 3) or \
                     (state.rps_player_move == 2 and state.rps_ai_move == 1) or \
                     (state.rps_player_move == 3 and state.rps_ai_move == 2):
                    state.rps_scores[1] += 1
                    state.rps_result_text = "YOU WIN! 🎉"
                else:
                    state.rps_scores[0] += 1
                    state.rps_result_text = "AI WINS! 🤖"

        elif state.rps_status == "Result":
            # Display results on camera
            cv2.rectangle(frame, (320, 120), (960, 260), (30, 30, 45), cv2.FILLED)
            cv2.rectangle(frame, (320, 120), (960, 260), (0, 220, 255), 3)
            cv2.putText(frame, state.rps_result_text, (360, 180),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)

            move_names = {1: "ROCK ✊", 2: "PAPER ✋", 3: "SCISSORS ✌️"}
            p_name = move_names.get(state.rps_player_move, "None")
            a_name = move_names.get(state.rps_ai_move, "None")
            cv2.putText(frame, f"You: {p_name}  |  AI: {a_name}", (360, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)

    # 5. SNAKE GAME
    def process_snake(self, frame, lm_list, fingers):
        cx, cy = lm_list[8][1], lm_list[8][2]
        state.detected_gesture = f"Index Head ({cx}, {cy})"

        if state.snake_game_over:
            cv2.rectangle(frame, (350, 260), (930, 420), (0, 0, 180), cv2.FILLED)
            cv2.putText(frame, "GAME OVER", (430, 330), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4)
            cv2.putText(frame, f"Score: {state.snake_score} | Press 'Restart Snake' to play again", (380, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return

        px, py = state.snake_prev_head
        if px == 0 and py == 0:
            state.snake_prev_head = (cx, cy)
            return

        state.snake_points.append([cx, cy])
        dist = math.hypot(cx - px, cy - py)
        state.snake_lengths.append(dist)
        state.snake_cur_len += dist
        state.snake_prev_head = (cx, cy)

        # Shrink tail
        if state.snake_cur_len > state.snake_allowed_len:
            for i, length in enumerate(state.snake_lengths):
                state.snake_cur_len -= length
                state.snake_lengths.pop(i)
                state.snake_points.pop(i)
                if state.snake_cur_len < state.snake_allowed_len:
                    break

        # Check Food
        fx, fy = state.snake_food
        if math.hypot(cx - fx, cy - fy) < 45:
            state.snake_food = (random.randint(150, 1100), random.randint(120, 600))
            state.snake_allowed_len += 40
            state.snake_score += 1

        # Draw Snake Line
        if state.snake_points:
            for i in range(1, len(state.snake_points)):
                cv2.line(frame, tuple(state.snake_points[i - 1]), tuple(state.snake_points[i]), (0, 0, 255), 18)
            cv2.circle(frame, tuple(state.snake_points[-1]), 18, (0, 255, 0), cv2.FILLED)

        # Draw Food
        cv2.circle(frame, state.snake_food, 24, (0, 215, 255), cv2.FILLED)
        cv2.putText(frame, "FOOD", (state.snake_food[0] - 22, state.snake_food[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw Score
        cv2.putText(frame, f"SCORE: {state.snake_score}", (50, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

        # Self collision
        if len(state.snake_points) > 10:
            pts = np.array(state.snake_points[:-5], np.int32).reshape((-1, 1, 2))
            min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -0.2 <= min_dist <= 0.2:
                state.snake_game_over = True

    # 6. VIRTUAL KEYBOARD
    def process_keyboard(self, frame, lm_list, fingers):
        state.detected_gesture = "Virtual Air Typing"

        # Draw keys
        for btn in self.key_buttons:
            x, y, w, h = btn["rect"]
            text = btn["text"]

            # Check if index finger (8) hovers inside key
            ix, iy = lm_list[8][1], lm_list[8][2]
            hovered = (x < ix < x + w and y < iy < y + h)

            col = (0, 220, 100) if hovered else (180, 50, 180)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, cv2.FILLED)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, text, (x + 12, y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            # Check pinch click between 8 and 12
            if hovered:
                length, _, _ = self.detector.findDistance(8, 12, frame, draw=False)
                if length < 32 and (time.time() - state.last_key_press_time) > 0.4:
                    state.last_key_press_time = time.time()
                    if text == "SPACE":
                        state.typed_text += " "
                    elif text == "DEL":
                        state.typed_text = state.typed_text[:-1]
                    else:
                        state.typed_text += text

        # Draw Typed Text Box
        cv2.rectangle(frame, (40, 280), (1240, 360), (30, 30, 40), cv2.FILLED)
        cv2.rectangle(frame, (40, 280), (1240, 360), (0, 220, 255), 2)
        cv2.putText(frame, f"TYPED: {state.typed_text}_", (60, 335),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # 7. PRESENTATION
    def process_presentation(self, frame, lm_list, fingers):
        state.detected_gesture = "Presentation Navigator"
        num_slides = max(1, len(slides_list))

        if not state.slide_btn_pressed:
            # Thumb only: Prev Slide
            if fingers == [1, 0, 0, 0, 0]:
                if state.slide_index > 0:
                    state.slide_index -= 1
                    state.slide_btn_pressed = True
            # Pinky only: Next Slide
            elif fingers == [0, 0, 0, 0, 1]:
                if state.slide_index < num_slides - 1:
                    state.slide_index += 1
                    state.slide_btn_pressed = True
        else:
            state.slide_btn_counter += 1
            if state.slide_btn_counter > 20:
                state.slide_btn_counter = 0
                state.slide_btn_pressed = False

        # Draw slide indicator on camera
        cv2.rectangle(frame, (40, 80), (460, 180), (25, 25, 35), cv2.FILLED)
        cv2.rectangle(frame, (40, 80), (460, 180), (0, 220, 255), 2)
        cv2.putText(frame, f"SLIDE {state.slide_index + 1} / {num_slides}", (60, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, "Thumb: Prev | Pinky: Next", (60, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 1)


# Global Camera Singleton
camera = VideoCamera()


def gen(cam):
    while True:
        frame = cam.get_frame()
        time.sleep(0.02)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -----------------------------------------------------------------------------
# FLASK HTTP ROUTES & REST API
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
    except Exception:
        cpu, ram = 0, 0

    return jsonify({
        "mode": state.mode,
        "fps": state.fps,
        "mirror": state.mirror,
        "is_paused": state.is_paused,
        "detected_gesture": state.detected_gesture,
        "hands_count": state.hands_count,
        "fingers_count": state.fingers_count,
        "fingers_state": state.fingers_state,
        "volume": state.current_volume,
        "rps_status": state.rps_status,
        "rps_result": state.rps_result_text,
        "rps_scores": state.rps_scores,
        "snake_score": state.snake_score,
        "snake_game_over": state.snake_game_over,
        "typed_text": state.typed_text,
        "slide_index": state.slide_index + 1,
        "total_slides": max(1, len(slides_list)),
        "cpu": cpu,
        "ram": ram
    })


@app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    data = request.get_json() or {}
    new_mode = data.get("mode", "tracking")
    if new_mode in ["tracking", "fingers", "volume", "rps", "snake", "keyboard", "presentation"]:
        state.mode = new_mode
    return jsonify({"success": True, "mode": state.mode})


@app.route('/api/toggle_mirror', methods=['POST'])
def api_toggle_mirror():
    state.mirror = not state.mirror
    return jsonify({"success": True, "mirror": state.mirror})


@app.route('/api/toggle_pause', methods=['POST'])
def api_toggle_pause():
    state.is_paused = not state.is_paused
    return jsonify({"success": True, "is_paused": state.is_paused})


@app.route('/api/rps_start', methods=['POST'])
def api_rps_start():
    state.rps_status = "Counting"
    state.rps_timer_start = time.time()
    state.rps_player_move = None
    state.rps_ai_move = None
    state.rps_result_text = "Show your move!"
    return jsonify({"success": True})


@app.route('/api/rps_reset', methods=['POST'])
def api_rps_reset():
    state.rps_scores = [0, 0]
    state.rps_status = "Ready"
    state.rps_result_text = "Scores reset! Press Start Round."
    return jsonify({"success": True})


@app.route('/api/snake_reset', methods=['POST'])
def api_snake_reset():
    state.snake_points = []
    state.snake_lengths = []
    state.snake_cur_len = 0
    state.snake_allowed_len = 150
    state.snake_prev_head = (0, 0)
    state.snake_score = 0
    state.snake_game_over = False
    return jsonify({"success": True})


@app.route('/api/keyboard_clear', methods=['POST'])
def api_keyboard_clear():
    state.typed_text = ""
    return jsonify({"success": True})


@app.route('/api/launch_standalone', methods=['POST'])
def api_launch_standalone():
    data = request.get_json() or {}
    script = data.get("script")
    allowed = ["SnakeGame.py", "RockPaperScissorsGame.py", "VirtualKeyboard.py",
               "PresentationControl.py", "VolumeControl.py", "FingerCounter.py", "MenuScreen.py"]
    if script in allowed:
        # Pause web camera briefly so standalone can acquire Camera 0
        state.is_paused = True
        
        def run_proc():
            try:
                subprocess.run([sys.executable, script])
            finally:
                state.is_paused = False

        threading.Thread(target=run_proc, daemon=True).start()
        return jsonify({"success": True, "script": script})
    return jsonify({"success": False, "error": "Invalid script"}), 400


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("\n=======================================================")
    print(" [OK] HAND GESTURE RECOGNITION - WEB APPLICATION ACTIVE")
    print(" [OK] URL: http://localhost:5000  (or http://127.0.0.1:5000)")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
