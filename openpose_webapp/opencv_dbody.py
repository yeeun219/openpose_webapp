from django.conf import settings
import numpy as np
import cv2
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# 각 파일 path
protoFile = "./openpose_webapp/pose_deploy_linevec.prototxt"
weightsFile = "./openpose_webapp/pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
angle_list=[]

def opencv_dbody(path):
    image = cv2.imread(path, 1)
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    imageHeight, imageWidth, _ = image.shape

    # network에 넣기위해 전처리
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]


    # 키포인트 검출시 이미지에 그려줌
    points = []
    prob_arry = []
    for j in range(0, 15):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, j, :, :]

        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H
        temp = [x, y, prob]
        prob_arry.append(temp)
        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
    # 왼쪽으로 섰는지 오른쪽으로 섰는지 구분 prob 값을 0으로 만들어 if 문을 통과하지 못하게
    right = True
    if prob_arry[11][2] > prob_arry[8][2]:  # LElbow가 RElbow보다 클 경우
        prob_arry[2][2] = 0;
        prob_arry[3][2] = 0;
        prob_arry[4][2] = 0;
        prob_arry[8][2] = 0;
        prob_arry[9][2] = 0;
        prob_arry[10][2] = 0;
        right = False
    else:
        prob_arry[5][2] = 0
        prob_arry[6][2] = 0
        prob_arry[7][2] = 0
        prob_arry[11][2] = 0
        prob_arry[12][2] = 0
        prob_arry[13][2] = 0
    k = 0
    for x, y, prob in prob_arry:
        if prob > 0.05:
            #print(int(x))
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=5,
                       lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(image, "{}".format(k), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)
        k = k + 1
    cv2.imwrite(path, image)