"""
Module: eye_detector.py
Mục đích: Đóng gói toàn bộ logic khởi tạo MediaPipe Face Mesh, trích xuất điểm mốc
           khuôn mặt và tính toán tỷ lệ co mắt EAR (Eye Aspect Ratio) theo chuẩn OOP.
Tối ưu hóa: Phù hợp chạy trên phần cứng nhúng như Raspberry Pi 4.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


class EyeDetector:
    """
    Class xử lý phát hiện mắt và tính toán chỉ số EAR sử dụng MediaPipe Face Mesh.
    """

    # Chỉ số 6 điểm mốc (landmarks) chuẩn cho mắt trái theo MediaPipe Face Mesh
    # P0: Khóe mắt trong, P1, P2: Mí trên, P3: Đuôi mắt, P4, P5: Mí dưới
    LEFT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]

    # Chỉ số 6 điểm mốc chuẩn cho mắt phải theo MediaPipe Face Mesh
    RIGHT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Khởi tạo mô hình MediaPipe Face Mesh.
        """
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Khởi tạo instance Face Mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    @staticmethod
    def calculate_ear(eye_points: List[Tuple[int, int]]) -> float:
        """
        Tính toán tỷ số Eye Aspect Ratio (EAR) theo công thức của Soukupová và Čech (2016):
        EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)

        :param eye_points: Danh sách 6 tọa độ (x, y) của một bên mắt
        :return: Giá trị EAR (float), tỷ lệ mở của mắt
        """
        if len(eye_points) < 6:
            return 0.0

        # Khoảng cách thẳng đứng giữa mí trên và mí dưới
        v1 = math.dist(eye_points[1], eye_points[5])
        v2 = math.dist(eye_points[2], eye_points[4])

        # Khoảng cách nằm ngang giữa hai khóe mắt
        h = math.dist(eye_points[0], eye_points[3])

        # Tránh lỗi chia cho 0 nếu phát hiện lỗi tọa độ
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
        """
        Chuyển đổi tọa độ chuẩn hóa [0, 1] của MediaPipe thành tọa độ pixel thực tế (x, y).
        """
        points: List[Tuple[int, int]] = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            px = int(lm.x * width)
            py = int(lm.y * height)
            points.append((px, py))
        return points

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý 1 khung hình BGR từ camera: nhận diện khuôn mặt, trích xuất điểm mốc mắt và tính EAR.

        :param frame: Khung hình chuẩn BGR (OpenCV format)
        :return: Dictionary chứa kết quả:
                 {
                     "face_detected": bool,
                     "left_ear": float,
                     "right_ear": float,
                     "avg_ear": float,
                     "left_eye_points": List[Tuple[int, int]],
                     "right_eye_points": List[Tuple[int, int]]
                 }
        """
        height, width, _ = frame.shape

        # Chuyển đổi BGR sang RGB cho MediaPipe (tối ưu: đánh dấu writeable=False)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = self._face_mesh.process(frame_rgb)

        result_data: Dict[str, Any] = {
            "face_detected": False,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "avg_ear": 0.0,
            "left_eye_points": [],
            "right_eye_points": [],
        }

        if results.multi_face_landmarks:
            # Lấy khuôn mặt đầu tiên (người lái xe)
            first_face = results.multi_face_landmarks[0]
            result_data["face_detected"] = True

            # Trích xuất tọa độ mắt trái và mắt phải
            left_eye_pts = self._extract_landmark_points(
                first_face, self.LEFT_EYE_INDICES, width, height
            )
            right_eye_pts = self._extract_landmark_points(
                first_face, self.RIGHT_EYE_INDICES, width, height
            )

            # Tính toán EAR cho từng mắt
            left_ear = self.calculate_ear(left_eye_pts)
            right_ear = self.calculate_ear(right_eye_pts)

            # Tỷ lệ EAR trung bình cả hai mắt
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
        """
        Vẽ đường viền đa giác quanh mắt để trực quan hóa trên giao diện camera.
        """
        if len(eye_points) >= 6:
            pts = np.array(eye_points, dtype=np.int32)
            # Khép kín đa giác mắt
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    def close(self) -> None:
        """
        Giải phóng tài nguyên MediaPipe khi đóng ứng dụng.
        """
        if hasattr(self, "_face_mesh") and self._face_mesh is not None:
            self._face_mesh.close()
