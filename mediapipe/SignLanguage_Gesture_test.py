import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ["yes", "hate", "really_hate", "don't"]
seq_length = 30

model = load_model('models/model.h5')

cap = cv2.VideoCapture(0)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)  # max_num_faces의 값으로 인식되는 얼굴 수 설정 가능
drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[0, 255, 0])  # 특징점의 연결된 선의 두께, 점 반지름, 색깔 설정

joint = []
seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    can_recognize_face = False
    can_recognize_hands = False
    d = []

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        can_recognize_hands = True
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

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
                                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20],
                                                        :]))

                    angle = np.degrees(angle)
                    float32_angle = np.array([angle], dtype=np.float32)
                    NFC_Point_Data = np.append(NFC_Point_Data, float32_angle)

                d = np.append(d, np.concatenate([NFC_point.flatten(), NFC_Point_Data]))

                seq.append(d)
                print(f'seq.shape : {len(seq), len(seq[0])}')

                mp_drawing.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                print(i_pred)
                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                cv2.putText(img, f'{this_action.upper()}',
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                            )

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
