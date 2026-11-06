"""
Module: alarm_manager.py
Mục đích: Đóng gói logic phát âm thanh cảnh báo bằng Pygame, quản lý trạng thái phát
           nhằm tránh nghẽn luồng xử lý chính và chống spam âm thanh liên tục.
"""

import os
import time
from typing import Optional
import pygame


class AlarmManager:
    """
    Class quản lý âm thanh cảnh báo sử dụng pygame.mixer.
    """

    def __init__(self, sound_path: str = "assets/alarms/alarm.mp3", volume: float = 1.0) -> None:
        """
        Khởi tạo hệ thống âm thanh.

        :param sound_path: Đường dẫn tới file âm thanh cảnh báo (.mp3 / .wav)
        :param volume: Mức âm lượng từ 0.0 đến 1.0
        """
        self.sound_path = sound_path
        self.volume = max(0.0, min(1.0, volume))
        self._is_initialized = False
        self._last_trigger_time: float = 0.0

        self._init_mixer()

    def _init_mixer(self) -> None:
        """Khởi tạo pygame.mixer an toàn."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self._is_initialized = True
        except pygame.error as err:
            print(f"[AlarmManager - CẢNH BÁO] Không thể khởi tạo pygame mixer: {err}")
            self._is_initialized = False

    def play(self, cooldown: float = 1.0) -> bool:
        """
        Kích hoạt phát âm thanh cảnh báo (non-blocking).

        :param cooldown: Thời gian nghỉ tối thiểu giữa các lần kích hoạt (giây)
        :return: True nếu bắt đầu phát hoặc đang phát, False nếu lỗi
        """
        if not self._is_initialized:
            return False

        current_time = time.time()
        # Tránh spam phát nhạc khi đã có nhạc đang chạy hoặc chưa hết cooldown
        if pygame.mixer.music.get_busy():
            return True

        if current_time - self._last_trigger_time < cooldown:
            return False

        if not os.path.exists(self.sound_path):
            print(f"[AlarmManager - LỖI] Không tìm thấy file âm thanh tại: {self.sound_path}")
            return False

        try:
            pygame.mixer.music.load(self.sound_path)
            pygame.mixer.music.play()
            self._last_trigger_time = current_time
            return True
        except pygame.error as err:
            print(f"[AlarmManager - LỖI] Lỗi phát âm thanh: {err}")
            return False

    def stop(self) -> None:
        """Dừng phát âm thanh cảnh báo."""
        if self._is_initialized and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def is_playing(self) -> bool:
        """Kiểm tra âm thanh cảnh báo có đang phát hay không."""
        return bool(self._is_initialized and pygame.mixer.music.get_busy())

    def cleanup(self) -> None:
        """Giải phóng tài nguyên pygame.mixer khi kết thúc chương trình."""
        if self._is_initialized:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
            self._is_initialized = False
