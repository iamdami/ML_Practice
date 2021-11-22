import cv2
import mediapipe as mp
import numpy as np
import time, os

cap = cv2.VideoCapture(0)

actions = ["yes", "hate", "really_hate", "don't"]
seq_length = 30
secs_for_actions = 60  # 손 동작 녹화 시간

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)  # max_num_faces의 값으로 인식되는 얼굴 수 설정 가능!
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=[0, 255, 0])  # 특징점의 연결된 선의 두께, 점 반지름, 색깔 설정

joint = []

created_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
os.makedirs("dataset", exist_ok=True)


while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        success, img = cap.read()
        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_actions:
            can_recognize_face = False
            can_recognize_hands = False
            d = []
            success, img = cap.read()

            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)

            if result.multi_hand_landmarks is not None:
                can_recognize_hands = True
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1  # v2의 벡터값을 v1의 벡터값으로 빼서 나온 벡터값을 v에 저장
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # v의 벡터값을 1로 노멀라이즈

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                    angle = np.degrees(angle)  # radian to degree
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.reshape(angle_label, (15,))

                    d = np.append(d, np.concatenate([joint.flatten(), angle_label]))

                    mpDraw.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS)

            result = faceMesh.process(imgRGB)

            NFC_Point_Data = np.array([])
            if result.multi_face_landmarks is not None:
                can_recognize_face = True
                for faceLms in result.multi_face_landmarks:
                    i = 0
                    NFC_point = np.zeros((7, 4))  # 1:Nose, 10:Forehead, 152:Chin 합쳐서 NFC_point
                    for id, lm in enumerate(faceLms.landmark):
                        if id == 1 or id == 10 or id == 152 or id == 323 or id == 93 or id == 136 or id == 365:
                            NFC_point[i] = [lm.x, lm.y, lm.z, lm.visibility]
                            i += 1
                    if can_recognize_hands:
                        for i in range(0, len(NFC_point)):
                            v3 = NFC_point[[i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i], :3]
                            v4 = joint[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                            v = v4 - v3
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            angle = np.arccos(np.einsum('nt,nt->n',
                                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19], :],
                                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20], :]))

                            angle = np.degrees(angle)
                            float32_angle = np.array([angle], dtype=np.float32)
                            NFC_Point_Data = np.append(NFC_Point_Data, float32_angle)

                        NFC_Point_Data = np.append(NFC_Point_Data, idx)
                        d = np.append(d, np.concatenate([NFC_point.flatten(), NFC_Point_Data]))

                        data.append(d)
                        #  print(len(data), len((data[0])))
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            cv2.putText(img, "Time : " + str(int(secs_for_actions - (time.time() - start_time))),
                        (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        #  ex: come (752, 100)
        # come (722, 30, 100)

        data = np.array(data)
        print(action, data.shape)
        """
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)
        print(f'dataset 폴더에 raw_{action}_{created_time}이름으로 저장됨')"""

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        """
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
        print(f'dataset 폴더에 seq_{action}_{created_time}이름으로 저장됨')"""
    break
