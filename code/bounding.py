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
        # 1번 목 15 16 양눈
        keypoints = person['pose_keypoints_2d']

        lear_y = keypoints[3*18+1]
        rear_y = keypoints[3*17+1]
        leye_y = keypoints[3*16+1]
        reye_y = keypoints[3*15+1]
        nose_y = keypoints[1]
        neck_y = keypoints[3*1+1]

        max_y = 0
        min_y = frame_height

        if neck_y>max_y: max_y = neck_y
        if nose_y>max_y: max_y = nose_y
        if rear_y>max_y: max_y = rear_y
        if lear_y>max_y: max_y = lear_y
        if leye_y>max_y: max_y = leye_y
        if reye_y>max_y: max_y = reye_y


        if lear_y!=0 and min_y>lear_y: min_y = lear_y 
        if rear_y!=0 and min_y>rear_y: min_y = rear_y
        if leye_y!=0 and min_y>leye_y: min_y = leye_y
        if reye_y!=0 and min_y>reye_y: min_y = reye_y
        if nose_y!=0 and min_y>nose_y: min_y = nose_y
        if neck_y!=0 and min_y>neck_y: min_y = neck_y 

        if min_y==frame_height: min_y=0
        if min_y!=0: min_y-=80

        xs=[]

        left_shoulder = keypoints[3*5]
        right_shoulder = keypoints[3*2]
        nose = keypoints[0]
        left_ear = keypoints[3*18]
        right_ear = keypoints[3*17]
        left_eye = keypoints[3*16]
        right_eye = keypoints[3*15]
        
        

        min_x = frame_width
        max_x = 0

        if left_shoulder!=0 and min_x>left_shoulder: min_x = left_shoulder 
        if right_shoulder!=0 and min_x>right_shoulder: min_x = right_shoulder
        if nose!=0 and min_x>nose: min_x=nose
        if left_ear!=0 and min_x>left_ear: min_x=left_ear
        if right_ear!=0 and min_x>right_ear: min_x = right_ear
        if left_eye!=0 and min_x>left_eye: min_x = left_eye 
        if right_eye!=0 and min_x>right_eye: min_x = right_eye

        if left_shoulder!=0 and max_x<left_shoulder: max_x = left_shoulder 
        if right_shoulder!=0 and max_x<right_shoulder: max_x = right_shoulder
        if nose!=0 and max_x<nose: max_x=nose
        if left_ear!=0 and max_x<left_ear: max_x=left_ear
        if right_ear!=0 and max_x<right_ear: max_x = right_ear
        if left_eye!=0 and max_x<left_eye: max_x = left_eye 
        if right_eye!=0 and max_x<right_eye: max_x = right_eye 

        if min_x==frame_width: min_x=0

        '''
        if left_shoulder!=0: xs.append(left_shoulder)
        if right_shoulder!=0: xs.append(right_shoulder)
        if nose!=0: xs.append(nose)
        if left_ear!=0: xs.append(left_ear)
        if right_ear!=0: xs.append(right_ear)
        if left_eye!=0: xs.append(left_eye)
        if right_eye!=0: xs.append(right_eye)
        
        min_x = min(xs)
        max_x = max(xs)
        '''
        if min_x!=0:
            if min_x==left_shoulder or min_x==right_shoulder:
                min_x+=20
        if max_x!=0:
            if max_x==left_shoulder or max_x==right_shoulder:
                max_x-=20
        if max_y!=0 and max_y==neck_y: max_y-=30

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

    
    bounding.sort(key=lambda x: x['min_x'])


    return bounding

# 사용 예시
input_json_path = '../../../openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/output/2_000000000000_keypoints.json'
frame_width = 1920  # 예시로 이미지의 너비를 설정
frame_height = 1080  # 예시로 이미지의 높이를 설정

