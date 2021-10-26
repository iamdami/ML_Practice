import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1) #max_num_faces의 값으로 인식되는 얼굴 수를 설정가능!
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=[0, 255, 0])#특징점의 연결된 선의 두께, 점 반지름, 색깔 설정

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1) #카메라 좌우반전

    #mediapiepe는 RGB 이미지를 이용하지만 img가 BGR이라 RGB변환을 해준다
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks :
        for faceLms in results.multi_face_landmarks :
            #img에 특징점을 표시하는 코드 ※FACEMESH.CONTOURS는 특징점끼리 연결시켜 주는 코드
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)


        for id, lm in enumerate(faceLms.landmark) :
            #print(lm) # 특징점 위치
            h, w, c = img.shape # 이미지의 가로, 세로, 채널값 추출
            x, y = int(lm.x*w), int(lm.y*h)
            #print(id, x, y) #각각의 특징점의 id와 위치


    #이미지에 fps 표시
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps : " + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
