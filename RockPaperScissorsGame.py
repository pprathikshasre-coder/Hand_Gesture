import random
import math
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False
imgAI = None
scores = [0, 0]  # [AI, Player]
bgBase = cv2.imread("Resources/BG.png")
roundDuration = 2.0  # Snappy, responsive 2s countdown
recentMoves = []

def get_rps_move(hand):
    if not hand or "lmList" not in hand or len(hand["lmList"]) < 21:
        return None, "No Hand"
    lm = hand["lmList"]
    coords = [(p[0], p[1]) for p in lm]

    thumb_tip = coords[4]
    thumb_ip = coords[3]
    thumb_mcp = coords[2]
    idx_mcp = coords[5]
    mid_mcp = coords[9]
    pky_mcp = coords[17]
    wrist = coords[0]

    # Thumb orientation: is thumb on left or right side of palm?
    if idx_mcp[0] < pky_mcp[0]:
        lat_open = (thumb_tip[0] < thumb_ip[0])
    else:
        lat_open = (thumb_tip[0] > thumb_ip[0])

    d_tip_pky = math.hypot(thumb_tip[0] - pky_mcp[0], thumb_tip[1] - pky_mcp[1])
    d_ip_pky = math.hypot(thumb_ip[0] - pky_mcp[0], thumb_ip[1] - pky_mcp[1])
    d_mcp_pky = math.hypot(thumb_mcp[0] - pky_mcp[0], thumb_mcp[1] - pky_mcp[1])
    d_tip_mid = math.hypot(thumb_tip[0] - mid_mcp[0], thumb_tip[1] - mid_mcp[1])
    palm_size = math.hypot(wrist[0] - mid_mcp[0], wrist[1] - mid_mcp[1]) or 1.0

    is_tucked = (d_tip_mid < 0.50 * palm_size) or (d_tip_pky < 0.90 * d_mcp_pky)
    vert_up = (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]) and (d_tip_mid > 0.58 * palm_size)

    thumb_up = 1 if ((lat_open and d_tip_pky > d_ip_pky and not is_tucked) or vert_up) else 0

    f_up = [thumb_up]
    for tip_id in [8, 12, 16, 20]:
        pip_id = tip_id - 2
        f_up.append(1 if coords[tip_id][1] < coords[pip_id][1] else 0)

    thumb, index, middle, ring, pinky = f_up
    total = sum(f_up)

    # 1. Scissors (Index & Middle up, Ring & Pinky down)
    if index == 1 and middle == 1 and ring == 0 and pinky == 0:
        return 3, "Scissors"
    # 2. Paper (4 or 5 fingers up)
    if total >= 4 or (index == 1 and middle == 1 and ring == 1 and pinky == 1):
        return 2, "Paper"
    # 3. Rock (Fist - all 4 fingers down)
    if index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return 1, "Rock"
    if total <= 1 and ring == 0 and pinky == 0:
        return 1, "Rock"

    return None, "Detecting..."

while True:
    imgBG = bgBase.copy()
    success, img = cap.read()
    if not success:
        continue

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    # Find Hands
    hands, img = detector.findHands(imgScaled)  # with draw

    currentMove, moveName = None, "Show Move"
    if hands:
        currentMove, moveName = get_rps_move(hands[0])
        # Live HUD on top right
        cv2.putText(imgBG, f"Live: {moveName}", (810, 190), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 120), 3)

    if startGame:
        if stateResult is False:
            elapsed = time.time() - initialTime
            remaining = roundDuration - elapsed

            if currentMove is not None:
                recentMoves.append(currentMove)

            countdown_num = max(1, int(math.ceil(remaining)))
            cv2.putText(imgBG, str(countdown_num), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (0, 220, 255), 6)

            if remaining <= 0:
                stateResult = True

                # Pick final move with fallback to rolling history
                playerMove = currentMove
                if playerMove is None and recentMoves:
                    playerMove = max(set(recentMoves), key=recentMoves.count)

                randomNumber = random.randint(1, 3)
                imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)

                if playerMove is not None:
                    # Player Wins
                    if (playerMove == 1 and randomNumber == 3) or \
                            (playerMove == 2 and randomNumber == 1) or \
                            (playerMove == 3 and randomNumber == 2):
                        scores[1] += 1
                    # AI Wins
                    elif (playerMove == 3 and randomNumber == 1) or \
                            (playerMove == 1 and randomNumber == 2) or \
                            (playerMove == 2 and randomNumber == 3):
                        scores[0] += 1

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult and imgAI is not None:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("BG", imgBG)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
        recentMoves = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()