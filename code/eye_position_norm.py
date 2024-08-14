import json
import os

def normalize_coordinates(input_file, frame_width, frame_height):
    # JSON 파일 읽기
    with open(input_file, 'r') as f:
        data = json.load(f)

    norm_eyes=[]

    # 눈 좌표 가져오기 및 정규화
    for person in data['people']:
        keypoints = person['pose_keypoints_2d']
        left_eye_x, left_eye_y = keypoints[3*16:3*16+2]
        right_eye_x, right_eye_y = keypoints[3*15:3*15+2]

        # 좌표 정규화
        normalized_left_eye_x = left_eye_x / frame_width
        normalized_left_eye_y = left_eye_y / frame_height
        normalized_right_eye_x = right_eye_x / frame_width
        normalized_right_eye_y = right_eye_y / frame_height

        #if normalized_left_eye_x!=0 and normalized_right_eye_x!=0:
        #    norm_eye_x = (normalized_left_eye_x + normalized_right_eye_x)/2.0
        #    norm_eye_y = (normalized_left_eye_y+normalized_right_eye_y)/2.0
        if True:
            norm_eye_x = normalized_left_eye_x
            norm_eye_y = normalized_left_eye_y
            if norm_eye_y<normalized_right_eye_y:
                norm_eye_x = normalized_right_eye_x
                norm_eye_y = normalized_right_eye_y
    

        # 새로운 키 추가
        if norm_eye_x!=0 and norm_eye_y!=0:
            norm_eyes.append({
                'norm_eye_x':norm_eye_x,
                'norm_eye_y':norm_eye_y
            })
    # norm_eyes 리스트를 제자리에서 정렬
    norm_eyes.sort(key=lambda eye: eye['norm_eye_x'])
  

    return norm_eyes

# 사용 예시
input_json_path = '../../../openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/output/3_000000000000_keypoints.json'
frame_width = 1920  # 예시로 이미지의 너비를 설정
frame_height = 1080  # 예시로 이미지의 높이를 설정

#normalize_coordinates(input_json_path, frame_width, frame_height)
