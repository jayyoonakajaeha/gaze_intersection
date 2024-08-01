import json
import os

def bound(input_file,frame_width,frame_height):
    # JSON 파일 읽기
    with open(input_file, 'r') as f:
        data = json.load(f)

    bounding=[]

    # 눈 좌표 가져오기 및 정규화
    for person in data['people']:
        # 좌표 가져오기
        # 1번과 2번 keypoints가 각각 왼쪽 눈과 오른쪽 눈의 x, y 좌표임
        keypoints = person['pose_keypoints_2d']
        max_y = keypoints[3*1+1]
        min_y = keypoints[3*15+1]
        if keypoints[3*16+1]>max_y:
            max_y = keypoints[3*16+1]
        min_y-=80
        
        xs=[]

        left_shoulder = keypoints[3*5]
        right_shoulder = keypoints[3*2]
        nose = keypoints[0]
        left_ear = keypoints[3*18]
        right_ear = keypoints[3*17]
        left_eye = keypoints[3*16]
        right_eye = keypoints[3*15]
        if left_shoulder!=0: xs.append(left_shoulder)
        if right_shoulder!=0: xs.append(right_shoulder)
        if nose!=0: xs.append(nose)
        if left_ear!=0: xs.append(left_ear)
        if right_ear!=0: xs.append(right_ear)
        if left_eye!=0: xs.append(left_eye)
        if right_eye!=0: xs.append(right_eye)
        
        min_x = min(xs)
        max_x = max(xs)

        # 좌표 정규화
        normalized_min_x = min_x / frame_width
        normalized_max_x = max_x / frame_width
        normalized_min_y = min_y / frame_height
        normalized_max_y = max_y / frame_height

        # 새로운 키 추가
        bounding.append({
            'min_y': normalized_min_y,
            'max_y': normalized_max_y,
            'min_x': normalized_min_x,
            'max_x': normalized_max_x
        })

    # 결과를 새로운 JSON 파일로 저장
    return bounding

# 사용 예시
input_json_path = '../../../openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/output/2_000000000000_keypoints.json'
frame_width = 1920  # 예시로 이미지의 너비를 설정
frame_height = 1080  # 예시로 이미지의 높이를 설정

