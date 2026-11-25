package com.edgeai.drivermonitor

import android.content.Context
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/**
 * Quản lý phát âm thanh cảnh báo và rung điện thoại khi tài xế ngủ gật.
 */
class AlarmPlayer(private val context: Context) {

    private var mediaPlayer: MediaPlayer? = null
    private var isPlaying = false

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator?
    }

    init {
        try {
            val alertUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            mediaPlayer = MediaPlayer.create(context, alertUri)?.apply {
                isLooping = true
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun play() {
        if (!isPlaying) {
            mediaPlayer?.start()
            isPlaying = true
        }

        // Rung theo nhịp cảnh báo SOS
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val pattern = longArrayOf(0, 400, 200, 400)
            vibrator?.vibrate(VibrationEffect.createWaveform(pattern, 0))
        } else {
            @Suppress("DEPRECATION")
            vibrator?.vibrate(longArrayOf(0, 400, 200, 400), 0)
        }
    }

    fun stop() {
        if (isPlaying) {
            mediaPlayer?.pause()
            mediaPlayer?.seekTo(0)
            isPlaying = false
        }
        vibrator?.cancel()
    }

    fun release() {
        stop()
        mediaPlayer?.release()
        mediaPlayer = null
    }
}
