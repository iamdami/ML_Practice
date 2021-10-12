import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
CTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #img를 RGB로 변환
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks :
        for handLms in results.multi_hand_landmarks :
            for id, lm in enumerate(handLms.landmark) :
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                if int(id%4) == 0 and id != 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            #img에 특징점을 표시 해주는 코드 HAND_CONNECTIONS는 특징점끼리 연결해줌
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #img에 fps 표시
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "fps : " + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
