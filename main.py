import cv2
import pygame
import math
import mediapipe as mp
# Import trực tiếp để vượt qua lỗi cấu trúc thư mục của Windows
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# Khởi tạo hệ thống âm thanh
pygame.mixer.init()

# Sử dụng module mp_face_mesh vừa import trực tiếp
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_points):
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    h = math.dist(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)

def trigger_alarm():
    if not pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.load('alarm.mp3')
            pygame.mixer.music.play()
        except pygame.error:
            # Nếu không tìm thấy file, in ra cảnh báo thay vì tắt phần mềm
            print("LỖI: Không tìm thấy file alarm.mp3 trong thư mục!")

cap = cv2.VideoCapture(0)
FRAME_COUNTER = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            left_eye = []
            for index in LEFT_EYE_INDICES:
                pt = face_landmarks.landmark[index]
                left_eye.append((int(pt.x * w), int(pt.y * h)))
            
            ear = calculate_ear(left_eye)
            
            if ear < 0.22:
                FRAME_COUNTER += 1
                if FRAME_COUNTER >= 45:
                    trigger_alarm()
                    cv2.putText(frame, "CANH BAO NGU GAT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                FRAME_COUNTER = 0
            
            cv2.putText(frame, f"EAR: {ear:.2f} | Frames: {FRAME_COUNTER}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Driver Drowsiness Detection - Edge AI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()