import json
import os

def normalize_coordinates(input_file, frame_width, frame_height):
    # JSON 파일 읽기
    with open(input_file, 'r') as f:
        data = json.load(f)

    norm_eyes=[]

    # 눈 좌표 가져오기 및 정규화
    for person in data['people']:
        # 좌표 가져오기
        # 1번과 2번 keypoints가 각각 왼쪽 눈과 오른쪽 눈의 x, y 좌표임
        keypoints = person['pose_keypoints_2d']
        left_eye_x, left_eye_y = keypoints[3*16:3*16+2]
        right_eye_x, right_eye_y = keypoints[3*15:3*15+2]

        # 좌표 정규화
        normalized_left_eye_x = left_eye_x / frame_width
        normalized_left_eye_y = left_eye_y / frame_height
        normalized_right_eye_x = right_eye_x / frame_width
        normalized_right_eye_y = right_eye_y / frame_height

        norm_eye_x = normalized_left_eye_x
        norm_eye_y = normalized_left_eye_y
        if norm_eye_x==0.0:
            norm_eye_x = normalized_right_eye_x
            norm_eye_y = normalized_right_eye_y
    

        # 새로운 키 추가
        norm_eyes.append({
            'norm_eye_x':norm_eye_x,
            'norm_eye_y':norm_eye_y
        })

    # 결과를 새로운 JSON 파일로 저장
    return norm_eyes

# 사용 예시
input_json_path = '../../../openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/output/3_000000000000_keypoints.json'
frame_width = 1920  # 예시로 이미지의 너비를 설정
frame_height = 1080  # 예시로 이미지의 높이를 설정

#normalize_coordinates(input_json_path, frame_width, frame_height)
