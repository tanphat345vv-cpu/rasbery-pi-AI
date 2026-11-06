"""
Hệ thống giám sát trạng thái người lái xe (Driver Drowsiness Detection)
File: main.py (Controller chính)
Mục đích: 
  - Đọc cấu hình từ config/settings.yaml
  - Khởi tạo các Class EyeDetector, AlarmManager từ thư mục core/
  - Mở luồng video camera, vòng lặp xử lý từng khung hình theo chuẩn thời gian thực
  - Đánh giá trạng thái buồn ngủ đa mức độ (Bình thường / Lờ đờ / Ngủ gật)
  - Kích hoạt cảnh báo âm thanh và ghi nhận sự kiện vi phạm
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Any, Dict

import cv2
import yaml

from core.eye_detector import EyeDetector
from core.alarm_manager import AlarmManager


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Đọc cấu hình hệ thống từ file YAML, dự phòng cấu hình mặc định nếu gặp lỗi.
    """
    default_config: Dict[str, Any] = {
        "ear": {
            "threshold": 0.22,
            "consecutive_frames_drowsy": 20,
            "consecutive_frames_sleep": 45,
        },
        "mediapipe": {
            "max_num_faces": 1,
            "refine_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "camera": {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "flip_horizontal": True,
        },
        "alarm": {
            "sound_path": "assets/alarms/alarm.mp3",
            "volume": 1.0,
            "cooldown_seconds": 3.0,
        },
        "system": {
            "window_name": "Driver Drowsiness Detection - Edge AI (RPi 4)",
            "log_events": True,
            "save_captured_frames": True,
            "log_dir": "data/logs",
            "capture_dir": "data/captured_events",
        },
    }

    if not os.path.exists(config_path):
        print(f"[CẢNH BÁO] Không tìm thấy file {config_path}. Đang sử dụng cấu hình mặc định.")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            # Merge cấu hình người dùng vào default để tránh thiếu key
            for section, values in user_config.items():
                if section in default_config and isinstance(values, dict):
                    default_config[section].update(values)
                else:
                    default_config[section] = values
            return default_config
    except Exception as e:
        print(f"[LỖI] Lỗi khi đọc file {config_path}: {e}. Sử dụng cấu hình mặc định.")
        return default_config


def setup_logger(log_dir: str) -> logging.Logger:
    """Thiết lập hệ thống ghi log sự kiện."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("DriverMonitoring")
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, f"events_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger


def main() -> None:
    # 1. Đọc file cấu hình
    config = load_config("config/settings.yaml")

    ear_cfg = config["ear"]
    mp_cfg = config["mediapipe"]
    cam_cfg = config["camera"]
    alarm_cfg = config["alarm"]
    sys_cfg = config["system"]

    # Thiết lập thư mục lưu trữ & logger
    logger = setup_logger(sys_cfg.get("log_dir", "data/logs"))
    capture_dir = sys_cfg.get("capture_dir", "data/captured_events")
    os.makedirs(capture_dir, exist_ok=True)

    print("=" * 60)
    print(" HỆ THỐNG GIÁM SÁT TRẠNG THÁI NGƯỜI LÁI XE (DRIVER DROWSINESS)")
    print(f" - Ngưỡng EAR: {ear_cfg['threshold']}")
    print(f" - Frame lờ đờ: {ear_cfg['consecutive_frames_drowsy']}")
    print(f" - Frame ngủ gật: {ear_cfg['consecutive_frames_sleep']}")
    print(f" - File âm thanh: {alarm_cfg['sound_path']}")
    print("=" * 60)

    # 2. Khởi tạo các Class từ core/
    detector = EyeDetector(
        max_num_faces=mp_cfg.get("max_num_faces", 1),
        refine_landmarks=mp_cfg.get("refine_landmarks", True),
        min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
    )

    alarm_manager = AlarmManager(
        sound_path=alarm_cfg.get("sound_path", "assets/alarms/alarm.mp3"),
        volume=float(alarm_cfg.get("volume", 1.0)),
    )

    # 3. Mở luồng VideoCapture
    device_idx = cam_cfg.get("device_index", 0)
    cap = cv2.VideoCapture(device_idx)

    # Cấu hình phần cứng camera (Đặc biệt hiệu quả với camera Raspberry Pi / USB webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

    if not cap.isOpened():
        print(f"[LỖI NGHIÊM TRỌNG] Không thể kết nối tới Camera ID {device_idx}!")
        detector.close()
        alarm_manager.cleanup()
        sys.exit(1)

    # Biến trạng thái runtime
    frame_counter: int = 0
    is_sleeping: bool = False
    alarm_cooldown = float(alarm_cfg.get("cooldown_seconds", 3.0))

    # Biến tính FPS thực tế
    prev_time = time.time()
    fps = 0.0

    window_name = sys_cfg.get("window_name", "Driver Monitoring System")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Không nhận được khung hình từ camera. Đang thử lại...")
                break

            # Lật gương khung hình nếu được cấu hình
            if cam_cfg.get("flip_horizontal", True):
                frame = cv2.flip(frame, 1)

            # 4. Xử lý khung hình qua EyeDetector
            detection_res = detector.process(frame)

            # Tính toán FPS
            current_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time + 1e-6))
            prev_time = current_time

            # 5. Phân tích trạng thái và kích hoạt cảnh báo
            if detection_res["face_detected"]:
                avg_ear = detection_res["avg_ear"]
                left_ear = detection_res["left_ear"]
                right_ear = detection_res["right_ear"]
                left_pts = detection_res["left_eye_points"]
                right_pts = detection_res["right_eye_points"]

                # Vẽ viền mắt trực quan hóa
                detector.draw_eye_contours(frame, left_pts, color=(0, 255, 0), thickness=1)
                detector.draw_eye_contours(frame, right_pts, color=(0, 255, 0), thickness=1)

                # Định nghĩa ngưỡng EAR chuẩn theo báo cáo Tuần 11
                EAR_CLOSE = 0.20
                EAR_OPEN = 0.25

                # TRƯỜNG HỢP 1: Mắt nhắm hẳn (Giấc ngủ trắng & Ngủ gật sâu)
                if left_ear < EAR_CLOSE and right_ear < EAR_CLOSE:
                    frame_counter += 1

                    if frame_counter >= 40:
                        # Mức 3: Ngủ gật sâu (> 40 frames)
                        alarm_manager.play(cooldown=alarm_cooldown)
                        cv2.rectangle(frame, (10, 10), (630, 80), (0, 0, 200), -1)
                        cv2.putText(frame, "CANH BAO: NGU GAT SAU!", (30, 55),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                        if not is_sleeping:
                            is_sleeping = True
                            logger.warning(f"Ngủ gật sâu! Frames: {frame_counter}")

                    elif frame_counter >= 15:
                        # Mức 2: Giấc ngủ trắng (15 -> 39 frames)
                        cv2.rectangle(frame, (10, 10), (630, 70), (0, 140, 255), -1)
                        cv2.putText(frame, "NGUY HIEM: GIAC NGU TRANG!", (30, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

                # TRƯỜNG HỢP 2: Mắt lờ đờ / Buồn ngủ (Mí mắt sụp xuống nửa chừng)
                elif EAR_CLOSE <= avg_ear <= EAR_OPEN:
                    frame_counter += 1 
                    
                    if frame_counter >= 20:
                        # Chỉ hiện nhắc nhở nếu lờ đờ liên tục trong khoảng thời gian nhất định
                        cv2.rectangle(frame, (10, 10), (630, 70), (0, 255, 255), -1)
                        cv2.putText(frame, "NHAC NHO: MAT LO DO / MAT TAP TRUNG", (30, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

                # TRƯỜNG HỢP 3: Mắt mở to tỉnh táo
                else:
                    frame_counter = 0
                    if is_sleeping:
                        logger.info("Tài xế đã tỉnh lại.")
                        is_sleeping = False
                        alarm_manager.stop()

                # Hiển thị thông số OSD lên màn hình
                color_ear = (0, 255, 0) if avg_ear >= EAR_OPEN else (0, 165, 255) if avg_ear >= EAR_CLOSE else (0, 0, 255)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_ear, 2)
                cv2.putText(frame, f"Frame: {frame_counter}", (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            else:
                # Không nhận diện được khuôn mặt
                frame_counter = 0
                cv2.putText(
                    frame,
                    "KHONG PHAT HIEN KHUON MAT",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 140, 255),
                    2,
                )

            # Hiển thị FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (530, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # 6. Render cửa sổ OpenCV
            cv2.imshow(window_name, frame)

            # Phím tắt điều khiển: 'q' để thoát, 'r' để reset counter thủ công
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[THÔNG BÁO] Người dùng yêu cầu thoát chương trình.")
                break
            elif key == ord("r"):
                frame_counter = 0
                alarm_manager.stop()
                is_sleeping = False
                print("[THÔNG BÁO] Đã đặt lại trạng thái giám sát.")

    except KeyboardInterrupt:
        print("\n[THÔNG BÁO] Nhận tín hiệu ngắt (Ctrl+C). Đang tắt hệ thống...")
    finally:
        # 7. Giải phóng toàn bộ tài nguyên một cách an toàn
        print("[HỆ THỐNG] Đang dọn dẹp tài nguyên...")
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        alarm_manager.cleanup()
        print("[HỆ THỐNG] Đã thoát thành công.")


if __name__ == "__main__":
    main()
