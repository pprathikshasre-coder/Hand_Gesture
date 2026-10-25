import cv2
import sys
import subprocess
from HandTrackingModule import HandDetector

# Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

menu_items = [
    ("1. Track Hands", "Sample.py"),
    ("2. Count Fingers", "FingerCounter.py"),
    ("3. Volume Control", "VolumeControl.py"),
    ("4. Rock Paper Scissors", "RockPaperScissorsGame.py"),
    ("5. Snake Game", "SnakeGame.py"),
    ("6. Virtual Keyboard", "VirtualKeyboard.py"),
    ("7. Presentation Control", "PresentationControl.py")
]

keys = [item[0] for item in menu_items]
script_map = {i + 1: item[1] for i, item in enumerate(menu_items)}

class Button:
    def __init__(self, pos, text, size=[460, 55]):
        self.pos = pos
        self.size = size
        self.text = text


def create_buttons(h_screen=720):
    buttons = []
    start_y = 70
    spacing = 65
    for i, key in enumerate(keys):
        buttons.append(Button([200, start_y + spacing * i], key))
    return buttons


buttonList = create_buttons()
option = 0


def drawAll(img, buttonList, selected_option=0, hovered_idx=-1):
    overlay = img.copy()
    for idx, button in enumerate(buttonList):
        x, y = button.pos
        w, h = button.size
        btn_num = idx + 1

        if btn_num == selected_option:
            color = (0, 180, 0)      # Green for active selected
        elif idx == hovered_idx:
            color = (180, 0, 180)    # Magenta for hovered
        else:
            color = (232, 144, 24)   # Blue/orange default

        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, button.text, (x + 15, y + h - 18),
                    cv2.FONT_HERSHEY_PLAIN, 2.2, (255, 255, 255), 2)

    # Instructions box at the bottom
    cv2.rectangle(img, (20, 560), (1260, 700), (40, 40, 40), cv2.FILLED)
    cv2.rectangle(img, (20, 560), (1260, 700), (0, 200, 255), 2)
    cv2.putText(img, "GESTURE CONTROLS:", (40, 590),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "- Hover index finger over button & Middle Finger UP: Select Option", (40, 620),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(img, "- Pinky Finger UP: Launch selected mode | Press 'q' on keyboard to Exit", (40, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(img, f"Selected Option: {selected_option} ({keys[selected_option - 1] if 1 <= selected_option <= len(keys) else 'None'})",
                (40, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if selected_option > 0 else (180, 180, 180), 2)

    return img


def launch_module(module_file):
    global cap, option
    print(f"[Menu] Launching {module_file}...")
    cap.release()
    cv2.destroyAllWindows()

    subprocess.run([sys.executable, module_file])

    print("[Menu] Resuming Menu Screen...")
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    option = 0


while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPositionVol(img, draw=False)

    hovered_idx = -1
    if lmList:
        fingers = detector.fingersUp()
        
        # Check hover & middle finger select
        for idx, button in enumerate(buttonList):
            x, y = button.pos
            w, h = button.size
            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                hovered_idx = idx
                if fingers[2]:  # Middle finger up to select
                    option = idx + 1
                    break

        # Pinky up to launch selected option
        if fingers[4] and option in script_map:
            launch_module(script_map[option])
            continue

    img = drawAll(img, buttonList, selected_option=option, hovered_idx=hovered_idx)

    cv2.imshow("Gesture Menu", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif ord('1') <= key <= ord(str(len(menu_items))):
        num = key - ord('0')
        if num in script_map:
            launch_module(script_map[num])

cap.release()
cv2.destroyAllWindows()
