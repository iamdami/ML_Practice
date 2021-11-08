import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

rps_gesture = {0: "rock", 5: "paper", 9: "scissors"}

angle_data = np.empty((0, 3), np.float32)

pTime = 0
CTime = 0

file = np.genfromtxt("gesture_train.csv", delimiter=",")  # 이미 학습된 각도값 추출
angle = file[:, :-1].astype(np.float32)  # 각도값 추출
label = file[:, -1].astype(np.float32)  # gesture_train.csv에 라벨(gesture_train의 맨 마지막 데이터, 제스처의 종류를 의미)
knn = cv2.ml.KNearest_create()  # cv2에 knn 모델 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # 추출한 데이터로 knn 모델 학습


def Who_Is_Winner(Players):
    who = 0
    if Players[0]['gesture'] == Players[1]['gesture']:
        return '    Tie'
    elif Players[0]['gesture'] == "rock":
        if Players[1]['gesture'] == "paper":
            who = 1
        elif Players[1]['gesture'] == "scissors":
            who = 0
    elif Players[0]['gesture'] == "paper":
        if Players[1]['gesture'] == "scissors":
            who = 1
        elif Players[1]['gesture'] == "rock":
            who = 0
    elif Players[0]['gesture'] == "scissors":
        if Players[1]['gesture'] == "rock":
            who = 1
        elif Players[1]['gesture'] == "paper":
            who = 0

    return Players[who]['player_name'] + " Win!!"


while True:
    Hand_count = 0
    Players = []
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # img를 RGB로 변환
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # 0으로 초기화 된 joint 배열 생성
            joint = np.zeros((21, 3))
            for id, lm in enumerate(handLms.landmark):
                # 각각의 특징점의 위치를 joint 배열에 저장
                joint[id] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1  # v2의 벡터값을 v1의 벡터값으로 빼서 나온 벡터값을 v에 저장
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # v의 벡터값을 1로 노멀라이즈
            # print(v / np.linalg.norm(v, axis=1) [:, np.newaxis])

            """ 
            np.einsum()을 통해 nt->n 벡터의 내적을 구한다.
            만약 0과 1의 벡터값으로 내적을 구한다고 가정한다면 (0의 벡터 * 1의 벡터 * cosθ)가 된다.
            이때 45줄에서 v의 벡터값을 1로 표준화 했으므로 (1 * 1 * cosθ = cosθ)이 된다.
            여기서 cosθ를 arccos() 함수를 이용하면 두 벡터의 각도인 θ가 나오게 된다.
            """
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

            angle = np.degrees(angle)  # radian to degree

            for i in range(len(angle)):
                print(str(i) + ": " + str(angle[i]))
                
            
            data = np.array([angle], dtype=np.float32)  # float32 데이터 타입의 numpy 배열 data 생성
            success, results, neighbours, dist = knn.findNearest(data, 3)  # datadhk k에 3을 인수로 주고 knn 모델
            idx = int(results[0][0])
            # print(idx) #idx에 results(라벨값이 들어감)

            if idx in rps_gesture.keys():
                Hand_count += 1
                Players.append({'player_name': "Player" + str(Hand_count), 'gesture': rps_gesture[idx]})
                cv2.putText(img, text="Player" + str(Hand_count), org=(
                    int(handLms.landmark[0].x * img.shape[1]), int(handLms.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            if Hand_count == 2:
                cv2.putText(img, text=Who_Is_Winner(Players), org=(int(img.shape[1]/2 - 50), 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                # print(Who_Is_Winner(Players))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # print(str(img.shape[1]) + ", " + str(img.shape[0])) 640, 480

    # img에 fps 표시
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps : " + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
