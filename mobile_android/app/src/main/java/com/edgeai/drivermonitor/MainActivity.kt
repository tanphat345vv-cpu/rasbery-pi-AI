package com.edgeai.drivermonitor

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerOptions
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvFps: TextView
    private lateinit var tvEar: TextView
    private lateinit var tvFrames: TextView
    private lateinit var tvStatus: TextView
    private lateinit var alertBanner: View
    private lateinit var tvAlertTitle: TextView

    private lateinit var cameraExecutor: ExecutorService
    private var faceLandmarker: FaceLandmarker? = null
    private lateinit var alarmPlayer: AlarmPlayer

    private var consecutiveClosedFrames = 0
    private val EAR_CLOSE_THRESHOLD = 0.20f
    private val EAR_OPEN_THRESHOLD = 0.25f
    private val DROWSY_FRAME_THRESHOLD = 20
    private val SLEEP_FRAME_THRESHOLD = 40

    private var frameCount = 0
    private var lastFpsTimestamp = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        tvFps = findViewById(R.id.tvFps)
        tvEar = findViewById(R.id.tvEar)
        tvFrames = findViewById(R.id.tvFrames)
        tvStatus = findViewById(R.id.tvStatus)
        alertBanner = findViewById(R.id.alertBanner)
        tvAlertTitle = findViewById(R.id.tvAlertTitle)

        alarmPlayer = AlarmPlayer(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        setupFaceLandmarker()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun setupFaceLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_landmarker.task")
                .build()

            val options = FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setNumFaces(1)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, _ ->
                    val landmarks = result.faceLandmarks()
                    if (landmarks.isNotEmpty()) {
                        val firstFace = landmarks[0]
                        val width = previewView.width.toFloat()
                        val height = previewView.height.toFloat()

                        val leftEar = EarCalculator.calculateEAR(firstFace, EarCalculator.LEFT_EYE_INDICES, width, height)
                        val rightEar = EarCalculator.calculateEAR(firstFace, EarCalculator.RIGHT_EYE_INDICES, width, height)
                        val avgEar = (leftEar + rightEar) / 2.0f

                        runOnUiThread {
                            updateUI(avgEar, leftEar, rightEar)
                        }
                    } else {
                        runOnUiThread {
                            consecutiveClosedFrames = 0
                            alarmPlayer.stop()
                            alertBanner.visibility = View.GONE
                            tvStatus.text = "KHÔNG PHÁT HIỆN MẶT"
                        }
                    }
                }
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Lỗi tải model AI: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun updateUI(avgEar: Float, leftEar: Float, rightEar: Float) {
        tvEar.text = String.format("%.2f", avgEar)

        // Tính FPS
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsTimestamp >= 1000) {
            tvFps.text = frameCount.toString()
            frameCount = 0
            lastFpsTimestamp = now
        }

        // Logic đánh giá trạng thái
        if (leftEar < EAR_CLOSE_THRESHOLD && rightEar < EAR_CLOSE_THRESHOLD) {
            consecutiveClosedFrames++
            tvFrames.text = consecutiveClosedFrames.toString()

            if (consecutiveClosedFrames >= SLEEP_FRAME_THRESHOLD) {
                // Mức 3: Ngủ gật sâu
                alarmPlayer.play()
                alertBanner.visibility = View.VISIBLE
                alertBanner.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))
                tvAlertTitle.text = "CẢNH BÁO: NGỦ GẬT SÂU!"
                tvStatus.text = "NGỦ GẬT SÂU"
            } else if (consecutiveClosedFrames >= 15) {
                // Mức 2: Giấc ngủ trắng
                alertBanner.visibility = View.VISIBLE
                alertBanner.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark))
                tvAlertTitle.text = "NGUY HIỂM: GIẤC NGỦ TRẮNG!"
                tvStatus.text = "GIẤC NGỦ TRẮNG"
            }
        } else if (avgEar in EAR_CLOSE_THRESHOLD..EAR_OPEN_THRESHOLD) {
            consecutiveClosedFrames++
            tvFrames.text = consecutiveClosedFrames.toString()

            if (consecutiveClosedFrames >= DROWSY_FRAME_THRESHOLD) {
                alertBanner.visibility = View.VISIBLE
                alertBanner.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_orange_light))
                tvAlertTitle.text = "NHẮC NHỞ: MẮT LỜ ĐỜ"
                tvStatus.text = "MẮT LỜ ĐỜ"
            }
        } else {
            consecutiveClosedFrames = 0
            tvFrames.text = "0"
            alarmPlayer.stop()
            alertBanner.visibility = View.GONE
            tvStatus.text = "TỈNH TÁO"
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageProxy(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        val frameTime = System.currentTimeMillis()

        faceLandmarker?.detectAsync(mpImage, frameTime)
        imageProxy.close()
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
        alarmPlayer.release()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
