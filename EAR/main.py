"""
Hệ thống giám sát trạng thái lái xe:
1. Nhận diện trạng thái Mắt (Mở, Lờ đờ, Nhắm).
2. Phát hiện Giấc ngủ trắng (Microsleep) và Ngủ gật sâu (Deep Sleep).
3. Cảnh báo đa cấp độ (Âm thanh đa tầng + HUD trực quan).
"""

import os
import sys
import time
import threading
import urllib.request
import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Âm thanh không đồng bộ trên Windows
if sys.platform.startswith("win"):
    import winsound
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đường dẫn model MediaPipe Face Landmarker
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Dang tai model MediaPipe tu Google...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Tai model thanh cong!")

class SoundAlertManager:
    """Quản lý phát âm thanh cảnh báo đa tầng không gây chặn luồng video."""
    def __init__(self):
        self.current_alert = None
        self.is_running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_alert(self, alert_level):
        """
        alert_level:
        None: Không có cảnh báo
        'sluggish': Cấp 1 - Mắt lờ đờ (bíp ngắn ngắt quãng)
        'microsleep': Cấp 2 - Giấc ngủ trắng (bíp dồn dập giật mình)
        'deepsleep': Cấp 3 - Ngủ gật sâu (hú liên tục)
        """
        with self.lock:
            self.current_alert = alert_level

    def _worker(self):
        while self.is_running:
            alert = self.current_alert
            if sys.platform.startswith("win"):
                try:
                    if alert == "sluggish":
                        winsound.Beep(900, 150)
                        time.sleep(0.4)
                    elif alert == "microsleep":
                        winsound.Beep(1400, 200)
                        time.sleep(0.15)
                        winsound.Beep(1400, 200)
                        time.sleep(0.3)
                    elif alert == "deepsleep":
                        winsound.Beep(2000, 350)
                        time.sleep(0.05)
                    else:
                        time.sleep(0.08)
                except Exception:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def stop(self):
        self.is_running = False

def calculate_ear(eye_points):
    """
    Tính tỉ lệ EAR cho 6 điểm mốc của mắt:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p2_p6 = dist.euclidean(eye_points[1], eye_points[5])
    p3_p5 = dist.euclidean(eye_points[2], eye_points[4])
    p1_p4 = dist.euclidean(eye_points[0], eye_points[3])
    
    if p1_p4 == 0:
        return 0.0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# 6 điểm mốc cho mỗi mắt từ MediaPipe Face Mesh
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

def draw_hud(frame, ear_avg, ear_closed_thresh, ear_sluggish_thresh, 
             state_name, alert_level, blink_count, microsleep_count, 
             closed_time, sluggish_time, fps, calibrating, calib_progress):
    h, w, _ = frame.shape
    
    # Tạo bảng overlay bán trong suốt ở trên cùng
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Màu sắc và viền cảnh báo theo cấp độ
    if alert_level == "deepsleep":
        status_color = (0, 0, 255) # Đỏ rực
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 8)
    elif alert_level == "microsleep":
        status_color = (0, 69, 255) # Cam đậm
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 69, 255), 5)
    elif alert_level == "sluggish":
        status_color = (0, 215, 255) # Vàng cam
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 215, 255), 4)
    else:
        status_color = (0, 255, 0) # Xanh lá an toàn

    # 1. Trạng thái người lái
    cv2.putText(frame, f"TRANG THAI: {state_name}", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

    # 2. Các thông số đo đạc EAR
    cv2.putText(frame, f"EAR: {ear_avg:.3f} | Nham: <{ear_closed_thresh:.2f} | Lo do: [{ear_closed_thresh:.2f}-{ear_sluggish_thresh:.2f}]", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    # 3. Thống kê chớp mắt, giấc ngủ trắng & thời gian
    timer_info = f"Thoi gian nham: {closed_time:.1f}s" if closed_time > 0 else f"Thoi gian lo do: {sluggish_time:.1f}s" if sluggish_time > 0 else ""
    cv2.putText(frame, f"Chop mat: {blink_count} | Giac ngu trang: {microsleep_count} | {timer_info}",
                (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 4. FPS ở góc phải
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 2)

    # 5. Thanh đo EAR 3 vùng màu (Tricolor Meter Bar)
    bar_x, bar_y, bar_w, bar_h = w - 260, 52, 240, 22
    # Nền xám
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    # Vẽ các vùng ngưỡng trên thanh
    max_ear_display = 0.40
    closed_px = int((ear_closed_thresh / max_ear_display) * bar_w)
    sluggish_px = int((ear_sluggish_thresh / max_ear_display) * bar_w)
    current_px = int(np.clip(ear_avg / max_ear_display, 0, 1.0) * bar_w)

    # Tô màu theo giá trị EAR hiện tại
    if ear_avg < ear_closed_thresh:
        fill_color = (0, 0, 255) # Đỏ
    elif ear_avg <= ear_sluggish_thresh:
        fill_color = (0, 215, 255) # Vàng
    else:
        fill_color = (0, 255, 0) # Xanh lá

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + current_px, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), 1)

    # Vạch ranh giới ngưỡng
    cv2.line(frame, (bar_x + closed_px, bar_y - 3), (bar_x + closed_px, bar_y + bar_h + 3), (0, 0, 255), 2)
    cv2.line(frame, (bar_x + sluggish_px, bar_y - 3), (bar_x + sluggish_px, bar_y + bar_h + 3), (0, 215, 255), 2)
    cv2.putText(frame, "Thanh do EAR (Do-Vang-Xanh)", (bar_x, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

    # 6. Giao diện hiệu chuẩn
    if calibrating:
        calib_w = int(calib_progress * 320)
        cv2.rectangle(frame, (w // 2 - 160, h - 85), (w // 2 + 160, h - 55), (40, 40, 40), -1)
        cv2.rectangle(frame, (w // 2 - 160, h - 85), (w // 2 - 160 + calib_w, h - 55), (255, 190, 0), -1)
        cv2.rectangle(frame, (w // 2 - 160, h - 85), (w // 2 + 160, h - 55), (255, 255, 255), 2)
        cv2.putText(frame, "Dang do co dia mat mo... Nhin thang camera", 
                    (w // 2 - 230, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Phím tắt
    cv2.putText(frame, "[C]: Hieu chuan | [R]: Reset dem | [Q]: Thoat", 
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

def main():
    ensure_model_exists()
    
    # Khởi tạo MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    sound_mgr = SoundAlertManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LOI] Khong the mo webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # CẤU HÌNH CÁC NGƯỠNG ĐA TẦNG
    ear_closed_thresh = 0.20     # Nhỏ hơn ngưỡng này là nhắm hẳn
    ear_sluggish_thresh = 0.25   # Nằm giữa 0.20 và 0.25 là mắt lờ đờ, sụp mí

    # CẤU HÌNH THỜI GIAN THEO DÕI (Tính bằng giây để độc lập với FPS)
    TIME_SLUGGISH_ALERT = 2.0     # Lờ đờ liên tục >= 2.0s -> Báo động Cấp 1
    TIME_MICROSLEEP_MIN = 0.35    # Nhắm mắt >= 0.35s -> Tính là Giấc ngủ trắng (Cấp 2)
    TIME_DEEP_SLEEP_MIN = 1.20    # Nhắm mắt >= 1.20s -> Tính là Ngủ gật sâu (Cấp 3)
    TIME_BLINK_MIN = 0.08         # Thời gian chớp mắt tối thiểu

    # Biến trạng thái
    closed_start_time = None
    sluggish_start_time = None
    is_in_microsleep = False

    blink_count = 0
    microsleep_count = 0
    smoothed_ear = 0.28
    ema_alpha = 0.45

    # Biến hiệu chuẩn cá nhân hóa
    calibrating = False
    calib_start_time = 0
    calib_samples = []

    prev_time = time.time()
    fps = 0.0

    print("\n" + "="*55)
    print("HỆ THỐNG GIÁM SÁT TRẠNG THÁI LÁI XE NÂNG CAO")
    print("- Phát hiện Mắt lờ đờ (Sluggish Eyes)")
    print("- Phát hiện Giấc ngủ trắng (Microsleep)")
    print("- Phát hiện Ngủ gật nguy hiểm (Deep Sleep)")
    print("="*55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # Tính FPS
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-6))
        prev_time = now

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = detector.detect(mp_image)

        state_name = "BINH THUONG"
        alert_level = None
        calib_progress = 0.0
        closed_duration = 0.0
        sluggish_duration = 0.0

        if res.face_landmarks:
            landmarks = res.face_landmarks[0]
            pts = np.array([(int(p.x * w), int(p.y * h)) for p in landmarks])

            left_eye_pts = pts[LEFT_EYE_IDXS]
            right_eye_pts = pts[RIGHT_EYE_IDXS]

            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            raw_ear = (left_ear + right_ear) / 2.0

            smoothed_ear = ema_alpha * raw_ear + (1.0 - ema_alpha) * smoothed_ear

            # ----------------- CHẾ ĐỘ HIỆU CHUẨN -----------------
            if calibrating:
                elapsed = now - calib_start_time
                calib_progress = min(elapsed / 3.0, 1.0)
                calib_samples.append(raw_ear)
                if elapsed >= 3.0:
                    calibrating = False
                    if calib_samples:
                        baseline = np.median(calib_samples)
                        ear_closed_thresh = round(float(baseline * 0.65), 3)
                        ear_sluggish_thresh = round(float(baseline * 0.82), 3)
                        print(f"[Hiệu chuẩn xong] Baseline: {baseline:.3f} | Nguong nham: {ear_closed_thresh:.3f} | Nguong lo do: {ear_sluggish_thresh:.3f}")
            else:
                # ----------------- LOGIC PHÂN TẦNG TRẠNG THÁI -----------------
                # TRƯỜNG HỢP A: MẮT NHẮM HOÀN TOÀN (EAR < ear_closed_thresh)
                if smoothed_ear < ear_closed_thresh:
                    sluggish_start_time = None # Reset đếm lờ đờ

                    if closed_start_time is None:
                        closed_start_time = now
                    closed_duration = now - closed_start_time

                    if closed_duration >= TIME_DEEP_SLEEP_MIN:
                        state_name = "NGUY HIEM: NGU GAT SAU!"
                        alert_level = "deepsleep"
                    elif closed_duration >= TIME_MICROSLEEP_MIN:
                        state_name = "CANH BAO: GIAC NGU TRANG!"
                        alert_level = "microsleep"
                        if not is_in_microsleep:
                            microsleep_count += 1
                            is_in_microsleep = True
                    else:
                        state_name = "DANG NHAM MAT"

                # TRƯỜNG HỢP B: MẮT LỜ ĐỜ / SỤP MÍ (ear_closed_thresh <= EAR <= ear_sluggish_thresh)
                elif smoothed_ear <= ear_sluggish_thresh:
                    # Nếu vừa từ trạng thái nhắm mắt mở ra
                    if closed_start_time is not None:
                        duration = now - closed_start_time
                        if TIME_BLINK_MIN <= duration < TIME_MICROSLEEP_MIN:
                            blink_count += 1
                        closed_start_time = None
                        is_in_microsleep = False

                    if sluggish_start_time is None:
                        sluggish_start_time = now
                    sluggish_duration = now - sluggish_start_time

                    if sluggish_duration >= TIME_SLUGGISH_ALERT:
                        state_name = "CANH BAO: MAT LO DO / SUP MI!"
                        alert_level = "sluggish"
                    else:
                        state_name = "DANG SUP MI (THEO DOI)"

                # TRƯỜNG HỢP C: MẮT MỞ BÌNH THƯỜNG (EAR > ear_sluggish_thresh)
                else:
                    if closed_start_time is not None:
                        duration = now - closed_start_time
                        if TIME_BLINK_MIN <= duration < TIME_MICROSLEEP_MIN:
                            blink_count += 1
                        closed_start_time = None
                        is_in_microsleep = False

                    sluggish_start_time = None
                    state_name = "MAT MO BINH THUONG"

            # Phát âm thanh cảnh báo tương ứng
            sound_mgr.set_alert(alert_level)

            # Vẽ viền mắt trực quan
            if smoothed_ear < ear_closed_thresh:
                eye_color = (0, 0, 255) # Đỏ
            elif smoothed_ear <= ear_sluggish_thresh:
                eye_color = (0, 215, 255) # Vàng
            else:
                eye_color = (0, 255, 0) # Xanh

            cv2.polylines(frame, [left_eye_pts], isClosed=True, color=eye_color, thickness=2)
            cv2.polylines(frame, [right_eye_pts], isClosed=True, color=eye_color, thickness=2)

            for pt in np.vstack((left_eye_pts, right_eye_pts)):
                cv2.circle(frame, tuple(pt), 2, (255, 255, 255), -1)

            draw_hud(frame, smoothed_ear, ear_closed_thresh, ear_sluggish_thresh,
                     state_name, alert_level, blink_count, microsleep_count,
                     closed_duration, sluggish_duration, fps, calibrating, calib_progress)
        else:
            sound_mgr.set_alert(None)
            cv2.putText(frame, "KHONG PHAT HIEN KHUON MAT", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Driver Drowsiness & Microsleep Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            blink_count = 0
            microsleep_count = 0
            closed_start_time = None
            sluggish_start_time = None
            print("[INFO] Da reset tat ca bo dem ve 0.")
        elif key == ord('c'):
            print("[INFO] Bat dau hieu chuan 3s... Hay mo mat tu nhien!")
            calibrating = True
            calib_start_time = time.time()
            calib_samples = []

    sound_mgr.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Da dung he thong.")

if __name__ == "__main__":
    main()
