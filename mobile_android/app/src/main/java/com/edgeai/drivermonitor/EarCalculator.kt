package com.edgeai.drivermonitor

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.hypot

/**
 * Thuật toán tính chỉ số Eye Aspect Ratio (EAR) theo nghiên cứu của Soukupová và Čech (2016).
 * Đề tài: Hệ thống giám sát trạng thái người lái xe (Driver Drowsiness) - Edge AI
 * Sinh viên: Bồ Tấn Phát - MSSV: 123000144 - Đại học Lạc Hồng
 */
object EarCalculator {

    // Chỉ số 6 điểm mốc chuẩn cho mắt trái theo MediaPipe Face Mesh
    val LEFT_EYE_INDICES = listOf(362, 385, 387, 263, 373, 380)

    // Chỉ số 6 điểm mốc chuẩn cho mắt phải theo MediaPipe Face Mesh
    val RIGHT_EYE_INDICES = listOf(33, 160, 158, 133, 153, 144)

    /**
     * Tính toán tỉ số EAR cho 1 bên mắt:
     * EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
     */
    fun calculateEAR(
        landmarks: List<NormalizedLandmark>,
        indices: List<Int>,
        imageWidth: Float,
        imageHeight: Float
    ): Float {
        if (landmarks.size <= indices.maxOrNull()!!) return 0.0f

        val p0 = landmarks[indices[0]] // Góc mắt ngoài
        val p1 = landmarks[indices[1]] // Mí trên góc ngoài
        val p2 = landmarks[indices[2]] // Mí trên góc trong
        val p3 = landmarks[indices[3]] // Góc mắt trong
        val p4 = landmarks[indices[4]] // Mí dưới góc trong
        val p5 = landmarks[indices[5]] // Mí dưới góc ngoài

        val v1 = hypot((p1.x() - p5.x()) * imageWidth, (p1.y() - p5.y()) * imageHeight)
        val v2 = hypot((p2.x() - p4.x()) * imageWidth, (p2.y() - p4.y()) * imageHeight)
        val h = hypot((p0.x() - p3.x()) * imageWidth, (p0.y() - p3.y()) * imageHeight)

        if (h == 0.0f) return 0.0f
        return (v1 + v2) / (2.0f * h)
    }
}
