import cv2
import os
import argparse

def extract_all_frames(video_path, output_dir):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 파일이 열리지 않으면 에러 발생
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    while True:
        # 비디오에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 저장
        frame_filename = os.path.join(output_dir, f'frame_{count:05d}.png')
        cv2.imwrite(frame_filename, frame)
        print(f'Saved: {frame_filename}')
        
        count += 1
    
    # 비디오 캡처 객체 해제
    cap.release()
    print('Done extracting frames.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='Path to the video file', required=True)
    parser.add_argument('--output_dir', type=str, help='Directory to save extracted frames', required=True)
    args = parser.parse_args()

    extract_all_frames(args.video_path, args.output_dir)
