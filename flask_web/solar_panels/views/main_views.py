from flask import Blueprint, render_template, Response
import torch
from ultralytics import YOLOv10
import cv2
import imutils
import base64

# YOLOv10 모델 로드 (ultralytics 라이브러리 사용)
model = YOLOv10('./solar_panels/models/best.pt')

# 카테고리 이름 설정
category_names = ['bird_drop', 'cracked', 'dusty', 'panel']

# 비디오 파일 경로
video_path = './solar_panels/static/solar_panel_fault_video.mp4'

def process_frame(frame):
    frame = imutils.resize(frame, width=350)
    
    # YOLOv10을 사용하여 프레임 예측
    results = model(frame)

    # 탐지된 객체 가져오기
    detections = results[0].boxes.data

    for detection in detections:
            # 박스 좌표와 신뢰도, 클래스 ID 가져오기
            xyxy = detection[:4].cpu().numpy().astype(int)
            conf = detection[4].item()
            cls = int(detection[5].item())
            
            # 클래스 라벨 설정
            label = category_names[cls]
            
            if conf > 0.30:  # 신뢰도가 30% 이상인 경우에만 표시
                # 박스 그리기
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # 라벨 및 신뢰도 표시
                cv2.putText(frame, f'{label} {conf * 100:.2f}%', (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def generate_frames(video_path):
    camera = cv2.VideoCapture(video_path)

    if not camera.isOpened():
        return "Error: Could not open video file."

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break

        # YOLO 모델로 프레임 처리
        frame = process_frame(frame)

        # 이미지 인코딩 : OpenCV 이미지(frame)를 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # 클라이언트에게 프레임 전송(MJPEG 스트림으로 응답)
        # MJPEG 스트림을 사용하는 이유: HTML <img> 태그와의 호환성, 동영상 스트리밍 처리 효율성
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

def get_result_frame(video_path):
    camera = cv2.VideoCapture(video_path)

    if not camera.isOpened():
        return "Error: Could not open video file."
    
    grabbed, frame = camera.read()
    if not grabbed:
        return "End of video file reached."

    # yolo 결과 프레임 반환
    return process_frame(frame)

bp = Blueprint('main', __name__, url_prefix='/', template_folder='templates')    # 블루프린트 객체

@bp.route('/')  # 해당 주소 접속 시 함수 호출(데코레이터)
def index():
    return 'Homepage'

@bp.route('/hello') # 예를 들어, http://127.0.0.1:5000/hello 접속 시 이 함수가 호출된다.
def hello_pybo():
    return 'Hello, World! via bp'

@bp.route('/model_frame', methods=['GET', 'POST'])
def frame_inference():  # http://127.0.0.1:5000/model_frame 접속 시
    results = get_result_frame(video_path=video_path)
    if type(results) == str:    # Error 발생 시
        return f'{results}'
    else:
        # 이미지 인코딩 : OpenCV 이미지(frame)를 JPEG로 인코딩하고 Base64로 변환
        # Base64를 사용하는 이유 : 디스크에 저장할 필요 없이 서버 메모리에서 처리된 이미지를 HTML로 전달 가능
        _, buffer = cv2.imencode('.jpg', results)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('show_result_frame.html', frame=frame_base64) # HTML 템플릿 렌더링(이미지를 넘겨 줌)

@bp.route('/video_feed')
def video_feed():
    # 스트리밍 Response 생성
    return Response(generate_frames(video_path=video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/model_video', methods=['GET', 'POST'])  # http://127.0.0.1:5000/model_video 접속 시
def video_inference():
    return render_template('show_result_video.html')    # 이 html 파일에서 video_feed 함수를 
