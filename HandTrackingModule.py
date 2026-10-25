import cv2
import mediapipe as mp
import time
import math


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


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            continue
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()