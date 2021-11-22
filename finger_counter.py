import cv2 as cv
import mediapipe as mp
import hand_tracking_module as htm
import time
import os

cap = cv.VideoCapture(0)
pTime = 0

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgpath in myList:
    image = cv.imread(f"{folderPath}/{imgpath}")
    img = cv.resize(image, (100, 200), interpolation=cv.INTER_AREA)
    overlayList.append(img)

detector = htm.handDetector()

tipIds = [4, 8, 12, 16, 20]

while True:
    suc, vid = cap.read()
    vid = detector.findHands(vid)
    lmList = detector.findPosition(vid)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalfingers = fingers.count(1)
        #print(totalfingers)


        h, w, c = overlayList[totalfingers-1].shape
        vid[0:h, 0:w] = overlayList[totalfingers-1]

        cv.rectangle(vid, (0, 225), (150, 425), (0, 255, 0), cv.FILLED)
        cv.putText(vid, str(totalfingers), (25, 375), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(vid, f'FPS:{int(fps)}', (420, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv.imshow("VIDEO", vid)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break