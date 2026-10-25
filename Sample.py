import mediapipe as mp
import time
import HandTrackingModule as htm
import cv2

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    if not success:
        continue
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

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