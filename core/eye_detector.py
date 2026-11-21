"""
Module: eye_detector.py
Muc dich: Dong goi toan bo logic khoi tao MediaPipe Face Mesh / OpenCV Fallback,
           trich xuat diem moc khuon mat va tinh toan ti le co mat EAR (Eye Aspect Ratio).
Toi uu hoa: Tuong thich 100% voi Raspberry Pi 4 (tu dong nhan biet CPU thieu tap lenh AES
           de chuyen sang OpenCV Cascade Fallback, chong loi Illegal instruction).
"""

import os
import math
import platform
import urllib.request
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


def _is_raspberry_pi_or_no_aes() -> bool:
    """
    Kiem tra phan cung co phai la Raspberry Pi hoac CPU khong ho tro tap lenh AES hay khong.
    Tren Raspberry Pi 4 (Cortex-A72), phan cung khong co tap lenh AES (ARMv8 Cryptography),
    goi MediaPipe Tasks API tren Python 3.13 se bi crash he thong do loi:
    'FATAL ERROR: This binary was compiled with aes enabled (go/sigill-fail-fast) - Illegal instruction'.
    """
    # 1. Kiem tra bien moi truong uu tien
    if os.environ.get("USE_OPENCV", "0") == "1":
        return True

    try:
        machine = platform.machine().lower()
        # 2. Neu la kien truc aarch64 hoac arm
        if "aarch64" in machine or "arm" in machine:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                # Neu khong co tap lenh aes trong Features
                if "aes" not in content:
                    return True
            else:
                return True
    except Exception:
        pass

    return False


class EyeDetector:
    """
    Class xu ly phat hien mat va tinh toan chi so EAR theo chuan OOP.
    Ho tro co che Hardware Fault-Tolerance:
      - Neu phan cung co AES: Su dung Google MediaPipe Tasks API / Solutions.
      - Neu phan cung khong co AES (Raspberry Pi 4): Tu dong chuyen sang OpenCV Engine.
    """

    LEFT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._face_mesh = None
        self._detector = None
        self.use_opencv = False
        self.use_tasks_api = False

        # Neu la Raspberry Pi 4 / CPU thieu AES: Chuyen ngay sang OpenCV Fallback
        if _is_raspberry_pi_or_no_aes():
            print("[EyeDetector] Phat hien phan cung Raspberry Pi / CPU thieu tap lenh AES.")
            print("[EyeDetector] Tu dong kich hoat OpenCV Fallback Engine (chong Illegal instruction, toi uu 30-40 FPS)...")
            self._init_opencv_detector()
            return

        # Neu may co AES (nhu PC Windows/Linux x86, Pi 5): Su dung MediaPipe
        try:
            import mediapipe as mp
            self.use_tasks_api = not hasattr(mp, "solutions")

            if self.use_tasks_api:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base_dir, "face_landmarker.task")
                if not os.path.exists(model_path):
                    print("[EyeDetector] Dang tai face_landmarker.task...")
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                    urllib.request.urlretrieve(url, model_path)
                    print("[EyeDetector] Tai model thanh cong!")

                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._detector = vision.FaceLandmarker.create_from_options(options)
                print("[EyeDetector] Khoi tao MediaPipe Tasks API thanh cong.")
            else:
                mp_face_mesh = mp.solutions.face_mesh
                self._face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                print("[EyeDetector] Khoi tao MediaPipe Solutions FaceMesh thanh cong.")
        except Exception as e:
            print(f"[EyeDetector - CANH BAO] Khong the khoi tao MediaPipe: {e}")
            print("[EyeDetector] Tu dong chuyen sang OpenCV Fallback Engine...")
            self._init_opencv_detector()

    def _init_opencv_detector(self) -> None:
        self.use_opencv = True
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "assets", "models")
        os.makedirs(models_dir, exist_ok=True)

        face_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(models_dir, "haarcascade_eye.xml")

        if not os.path.exists(face_path) or os.path.getsize(face_path) < 1000:
            print("[EyeDetector] Dang tai mo hinh khuon mat OpenCV...")
            url_face = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url_face, face_path)

        if not os.path.exists(eye_path) or os.path.getsize(eye_path) < 1000:
            print("[EyeDetector] Dang tai mo hinh mat OpenCV...")
            url_eye = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
            urllib.request.urlretrieve(url_eye, eye_path)

        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_path)
        self._last_left_box = None
        self._last_right_box = None
        print("[EyeDetector] OpenCV Fallback Engine da san sang (FPS 30-40, toi uu cho RPi 4)!")

    @staticmethod
    def calculate_ear(eye_points: List[Tuple[int, int]]) -> float:
        if len(eye_points) < 6:
            return 0.0

        v1 = math.dist(eye_points[1], eye_points[5])
        v2 = math.dist(eye_points[2], eye_points[4])
        h = math.dist(eye_points[0], eye_points[3])

        if h == 0.0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def _extract_landmark_points(
        self,
        landmarks: Any,
        indices: List[int],
        width: int,
        height: int,
    ) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        for idx in indices:
            lm = landmarks[idx] if isinstance(landmarks, list) else landmarks.landmark[idx]
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))
        return points

    def _process_opencv(self, frame: np.ndarray) -> Dict[str, Any]:
        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result_data: Dict[str, Any] = {
            "face_detected": False,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "avg_ear": 0.0,
            "left_eye_points": [],
            "right_eye_points": [],
        }

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(120, 120)
        )

        if len(faces) == 0:
            self._last_left_box = None
            self._last_right_box = None
            return result_data

        face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = face
        result_data["face_detected"] = True

        eye_roi_y = fy + int(fh * 0.20)
        eye_roi_h = int(fh * 0.38)
        
        rx1 = fx + int(fw * 0.12)
        rw = int(fw * 0.36)
        right_roi_gray = gray[eye_roi_y : eye_roi_y + eye_roi_h, rx1 : rx1 + rw]

        lx1 = fx + int(fw * 0.52)
        lw = int(fw * 0.36)
        left_roi_gray = gray[eye_roi_y : eye_roi_y + eye_roi_h, lx1 : lx1 + lw]

        right_box = self._detect_eye_in_roi(right_roi_gray, rx1, eye_roi_y)
        if right_box is not None:
            self._last_right_box = right_box
            is_right_open = self._check_eye_open(gray, right_box)
        else:
            right_box = self._last_right_box or (rx1 + int(rw * 0.2), eye_roi_y + int(eye_roi_h * 0.2), int(rw * 0.6), int(eye_roi_h * 0.6))
            is_right_open = False

        left_box = self._detect_eye_in_roi(left_roi_gray, lx1, eye_roi_y)
        if left_box is not None:
            self._last_left_box = left_box
            is_left_open = self._check_eye_open(gray, left_box)
        else:
            left_box = self._last_left_box or (lx1 + int(lw * 0.2), eye_roi_y + int(eye_roi_h * 0.2), int(lw * 0.6), int(eye_roi_h * 0.6))
            is_left_open = False

        right_pts = self._make_eye_points(right_box, is_right_open)
        left_pts = self._make_eye_points(left_box, is_left_open)

        right_ear = self.calculate_ear(right_pts)
        left_ear = self.calculate_ear(left_pts)
        avg_ear = (right_ear + left_ear) / 2.0

        result_data["left_ear"] = left_ear
        result_data["right_ear"] = right_ear
        result_data["avg_ear"] = avg_ear
        result_data["left_eye_points"] = left_pts
        result_data["right_eye_points"] = right_pts

        return result_data

    def _detect_eye_in_roi(self, roi_gray: np.ndarray, offset_x: int, offset_y: int) -> Optional[Tuple[int, int, int, int]]:
        if roi_gray.shape[0] < 20 or roi_gray.shape[1] < 20:
            return None
        eyes = self._eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25)
        )
        if len(eyes) > 0:
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
            return (offset_x + ex, offset_y + ey, ew, eh)
        return None

    def _check_eye_open(self, gray: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        gx, gy, gw, gh = box
        eye_crop = gray[gy : gy + gh, gx : gx + gw]
        if eye_crop.size == 0:
            return False
        std_val = float(np.std(eye_crop))
        return std_val > 18.0

    def _make_eye_points(self, box: Tuple[int, int, int, int], is_open: bool) -> List[Tuple[int, int]]:
        gx, gy, gw, gh = box
        cy = gy + gh // 2
        w_step = max(gw // 3, 1)
        
        if is_open:
            h_open = max(int(gw * 0.30), 6)
            top_y = cy - h_open // 2
            bot_y = cy + h_open // 2
            return [
                (gx, cy),
                (gx + w_step, top_y),
                (gx + 2 * w_step, top_y),
                (gx + gw, cy),
                (gx + 2 * w_step, bot_y),
                (gx + w_step, bot_y),
            ]
        else:
            slit = 2
            return [
                (gx, cy),
                (gx + w_step, cy - slit),
                (gx + 2 * w_step, cy - slit),
                (gx + gw, cy),
                (gx + 2 * w_step, cy + slit),
                (gx + w_step, cy + slit),
            ]

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        if self.use_opencv:
            return self._process_opencv(frame)

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result_data: Dict[str, Any] = {
            "face_detected": False,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "avg_ear": 0.0,
            "left_eye_points": [],
            "right_eye_points": [],
        }

        if self.use_tasks_api:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self._detector.detect(mp_image)
            if res.face_landmarks and len(res.face_landmarks) > 0:
                first_face = res.face_landmarks[0]
                result_data["face_detected"] = True

                left_eye_pts = self._extract_landmark_points(
                    first_face, self.LEFT_EYE_INDICES, width, height
                )
                right_eye_pts = self._extract_landmark_points(
                    first_face, self.RIGHT_EYE_INDICES, width, height
                )

                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                result_data["left_ear"] = left_ear
                result_data["right_ear"] = right_ear
                result_data["avg_ear"] = avg_ear
                result_data["left_eye_points"] = left_eye_pts
                result_data["right_eye_points"] = right_eye_pts
        else:
            frame_rgb.flags.writeable = False
            results = self._face_mesh.process(frame_rgb)
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                first_face = results.multi_face_landmarks[0]
                result_data["face_detected"] = True

                left_eye_pts = self._extract_landmark_points(
                    first_face, self.LEFT_EYE_INDICES, width, height
                )
                right_eye_pts = self._extract_landmark_points(
                    first_face, self.RIGHT_EYE_INDICES, width, height
                )

                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                result_data["left_ear"] = left_ear
                result_data["right_ear"] = right_ear
                result_data["avg_ear"] = avg_ear
                result_data["left_eye_points"] = left_eye_pts
                result_data["right_eye_points"] = right_eye_pts

        return result_data

    @staticmethod
    def draw_eye_contours(
        frame: np.ndarray,
        eye_points: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
    ) -> None:
        if len(eye_points) >= 6:
            pts = np.array(eye_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def close(self) -> None:
        if hasattr(self, "_face_mesh") and self._face_mesh is not None:
            self._face_mesh.close()
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()
