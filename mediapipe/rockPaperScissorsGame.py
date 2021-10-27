import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)  # 비디오 입력

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

rps_gesture = { 0:"rock", 5:"paper", 9:"scissors" }

pTime = 0
CTime = 0

file = np.genfromtxt("gesture_train.csv", delimiter=",") #이미 학습된 각도값 추출
angle = file[:,:-1].astype(np.float32) #각도값 추출
label = file[:, -1].astype(np.float32) #gesture_train.csv의 레이블(gesture_train의 맨 마지막 데이터, 제스처 종류 의미)
knn = cv2.ml.KNearest_create() #cv2에 knn 모델 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label) #추출한 데이터로 knn 모델 학습

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #img를 RGB로 변환
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if result.multi_hand_landmarks :
        for handLms in result.multi_hand_landmarks :
            #0으로 초기화 된 joint 배열 생성
            joint = np.zeros((21, 3))
            for id, lm in enumerate(handLms.landmark) :
                #각 특징점 위치를 joint 배열에 저장
                joint[id] = [lm.x, lm.y, lm.z]


            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1 #v2의 벡터값을 v1의 벡터값으로 빼서 나온 벡터값을 v에 저장
            v = v / np.linalg.norm(v, axis=1) [:, np.newaxis] #v의 벡터값을 1로 노멀라이즈


            """ 
            np.einsum() 통해 nt->n 벡터의 내적 구함
            0과 1의 벡터값으로 내적을 구한다고 가정하면 (0의 벡터 * 1의 벡터 * cosθ)가 됨
            이때 line45에서 v의 벡터값을 1로 표준화 했으므로 (1 * 1 * cosθ = cosθ)이 됨
            여기서 cosθ를 arccos() 이용하면 두 벡터 각도인 θ가 나오게 됨
            """
            angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32) #float32 데이터 타입의 numpy 배열 data 생성
            success, results, neighbours, dist = knn.findNearest(data, 3) #data와 k에 3을 인수로 주고 knn
            idx = int(results[0][0])
            # print(idx) #idx에 results(레이블값 들어감)

            if idx in rps_gesture.keys() :
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(handLms.landmark[0].x * img.shape[1]),
                                                                     int(handLms.landmark[0].y * img.shape[0] + 20)),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #img에 fps 표시
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "fps : " + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
