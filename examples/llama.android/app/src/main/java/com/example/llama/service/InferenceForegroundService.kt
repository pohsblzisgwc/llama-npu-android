// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.example.llama.MainActivity
import com.example.llama.R

/**
 * Foreground Service for Mobile LLM Inference Engine.
 *
 * Essential for Android 14+ background stability:
 * 1. Elevates process priority (oom_score_adj reduced from ~950 cached to 200).
 * 2. Prevents Android Low Memory Killer (LMK) from prematurely reclaiming the model process.
 * 3. Holds a PowerManager PARTIAL_WAKE_LOCK so CPU doesn't deep-sleep when screen is turned off.
 */
class InferenceForegroundService : Service() {

    companion object {
        private const val TAG = "InferenceFGService"
        const val CHANNEL_ID = "mobile_llm_inference_channel"
        const val NOTIFICATION_ID = 99881

        const val ACTION_START = "com.example.llama.service.START"
        const val ACTION_UPDATE = "com.example.llama.service.UPDATE"
        const val ACTION_STOP = "com.example.llama.service.STOP"

        const val EXTRA_TITLE = "extra_title"
        const val EXTRA_SUBTITLE = "extra_subtitle"

        @Volatile
        var isServiceRunning: Boolean = false
            private set

        fun start(context: Context, title: String? = null, subtitle: String? = null) {
            try {
                val intent = Intent(context, InferenceForegroundService::class.java).apply {
                    action = ACTION_START
                    putExtra(EXTRA_TITLE, title ?: "端侧大模型推理引擎运行中")
                    putExtra(EXTRA_SUBTITLE, subtitle ?: "硬件异构加速运行中 | 维持前台进程常驻")
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start InferenceForegroundService", e)
            }
        }

        fun updateStatus(context: Context, title: String, subtitle: String) {
            if (!isServiceRunning) return
            try {
                val intent = Intent(context, InferenceForegroundService::class.java).apply {
                    action = ACTION_UPDATE
                    putExtra(EXTRA_TITLE, title)
                    putExtra(EXTRA_SUBTITLE, subtitle)
                }
                context.startService(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update InferenceForegroundService", e)
            }
        }

        fun stop(context: Context) {
            try {
                val intent = Intent(context, InferenceForegroundService::class.java).apply {
                    action = ACTION_STOP
                }
                context.stopService(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to stop InferenceForegroundService", e)
            }
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var currentTitle: String = "移动端离线大模型推理"
    private var currentSubtitle: String = "硬件加速活跃 | 前台保活中 (oom_score_adj ~200)"

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "InferenceForegroundService onCreate")
        createNotificationChannel()
        acquireWakeLock()
        isServiceRunning = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: ACTION_START
        if (action == ACTION_STOP) {
            Log.i(TAG, "Received ACTION_STOP, shutting down foreground service...")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val title = intent?.getStringExtra(EXTRA_TITLE) ?: currentTitle
        val subtitle = intent?.getStringExtra(EXTRA_SUBTITLE) ?: currentSubtitle
        currentTitle = title
        currentSubtitle = subtitle

        val notification = buildNotification(title, subtitle)
        startForeground(NOTIFICATION_ID, notification)
        Log.i(TAG, "InferenceForegroundService running in foreground: $title - $subtitle")

        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "InferenceForegroundService onDestroy")
        releaseWakeLock()
        isServiceRunning = false
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun buildNotification(title: String, subtitle: String): Notification {
        val launchIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(subtitle)
            .setSmallIcon(R.mipmap.ic_launcher_round)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "端侧硬件加速大模型推理常驻服务",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "提供高优先级前台进程保活，避免系统因内存回收终止模型推理服务"
                setShowBadge(false)
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
        }
    }

    private fun acquireWakeLock() {
        try {
            if (wakeLock == null) {
                val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
                wakeLock = powerManager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "SnapdragonLLM:InferenceWakeLock"
                ).apply {
                    setReferenceCounted(false)
                    acquire()
                }
                Log.i(TAG, "PowerManager PARTIAL_WAKE_LOCK acquired successfully")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to acquire PARTIAL_WAKE_LOCK", e)
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.let {
                if (it.isHeld) {
                    it.release()
                    Log.i(TAG, "PowerManager PARTIAL_WAKE_LOCK released")
                }
            }
            wakeLock = null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to release PARTIAL_WAKE_LOCK", e)
        }
    }
}
