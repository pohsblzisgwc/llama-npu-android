// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.monitor

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.util.Log
import com.arm.aichat.InferenceStats
import java.io.File
import java.io.RandomAccessFile

data class HardwareSnapshot(
    val cpuUsagePercent: Int = 0,
    val batteryTempC: Float = 0.0f,
    val socTempC: Float = 0.0f,
    val appMemoryMb: Long = 0,
    val totalRamGb: Float = 0.0f,
    val usedRamGb: Float = 0.0f,
    val availRamGb: Float = 0.0f,
    val ramPercent: Int = 0,
    val gpuStatus: String = "未参与 (0% 负载)",
    val activeAccelerator: String = "Qualcomm Hexagon NPU",
    val workloadDescription: String = "纯 NPU 硬件加速",
    val isNpuPrimary: Boolean = true,
    val thermalAlert: String = "正常",
    val thermalSeverity: Int = 0, // 0: Normal, 1: Warm, 2: Hot
    val oomScoreAdj: Int = 0,
    val isIgnoringBattery: Boolean = false,
    val isFgsRunning: Boolean = false
)

class HardwareMonitor(private val context: Context) {

    private var prevTotalCpuTime: Long = 0
    private var prevIdleCpuTime: Long = 0

    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    /**
     * Reads /proc/stat to compute CPU usage percentage
     */
    fun sampleCpuUsage(): Int {
        return try {
            val reader = RandomAccessFile("/proc/stat", "r")
            val load = reader.readLine()
            reader.close()
            val toks = load.split("\\s+".toRegex())
            if (toks.size >= 5) {
                val user = toks[1].toLong()
                val nice = toks[2].toLong()
                val system = toks[3].toLong()
                val idle = toks[4].toLong()
                val iowait = if (toks.size > 5) toks[5].toLong() else 0L
                val irq = if (toks.size > 6) toks[6].toLong() else 0L
                val softirq = if (toks.size > 7) toks[7].toLong() else 0L

                val total = user + nice + system + idle + iowait + irq + softirq
                val totalDelta = total - prevTotalCpuTime
                val idleDelta = idle - prevIdleCpuTime

                prevTotalCpuTime = total
                prevIdleCpuTime = idle

                if (totalDelta > 0) {
                    val usage = ((totalDelta - idleDelta) * 100 / totalDelta).toInt()
                    usage.coerceIn(0, 100)
                } else {
                    0
                }
            } else 0
        } catch (e: Exception) {
            0
        }
    }

    /**
     * Reads battery temperature in Celsius
     */
    fun getBatteryTemperature(): Float {
        return try {
            val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val temp = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
            temp / 10.0f
        } catch (e: Exception) {
            0.0f
        }
    }

    /**
     * Reads SoC / CPU temperature from /sys/class/thermal
     */
    fun getSocTemperature(): Float {
        try {
            val thermalDir = File("/sys/class/thermal")
            if (thermalDir.exists()) {
                val zones = thermalDir.listFiles { _, name -> name.startsWith("thermal_zone") } ?: emptyArray()
                var maxTemp = 0.0f
                for (zone in zones) {
                    val typeFile = File(zone, "type")
                    val tempFile = File(zone, "temp")
                    if (tempFile.exists()) {
                        val tStr = tempFile.readText().trim()
                        var tVal = tStr.toFloatOrNull() ?: continue
                        if (tVal > 1000f) tVal /= 1000f // Millicelsius to Celsius
                        if (tVal in 20.0f..105.0f && tVal > maxTemp) {
                            maxTemp = tVal
                        }
                    }
                }
                if (maxTemp > 0f) return maxTemp
            }
        } catch (e: Exception) {
            // Ignore sysfs permissions
        }
        return getBatteryTemperature()
    }

    /**
     * Reads GPU load / frequency on mobile GPU sysfs
     */
    fun getGpuStatus(): String {
        try {
            val busyFile = File("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")
            if (busyFile.exists()) {
                val busy = busyFile.readText().trim()
                return "GPU: $busy"
            }
            val freqFile = File("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")
            if (freqFile.exists()) {
                val freq = freqFile.readText().trim().toLongOrNull() ?: 0L
                val mhz = freq / 1000000
                return if (mhz > 0) "GPU: ${mhz}MHz" else "GPU: 待机"
            }
        } catch (e: Exception) {
            // Sysfs permissions
        }
        return "GPU: 待机 (未分配计算负载)"
    }

    /**
     * Collects a unified hardware & inference snapshot
     */
    fun takeSnapshot(stats: InferenceStats): HardwareSnapshot {
        val cpuUsage = sampleCpuUsage()
        val batteryTemp = getBatteryTemperature()
        val socTemp = getSocTemperature()

        // RAM stats
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val totalRamGb = memInfo.totalMem.toFloat() / (1024 * 1024 * 1024)
        val availRamGb = memInfo.availMem.toFloat() / (1024 * 1024 * 1024)
        val usedRamGb = totalRamGb - availRamGb
        val ramPercent = if (totalRamGb > 0) ((usedRamGb / totalRamGb) * 100).toInt() else 0

        // App memory (PSS)
        val pssKb = Debug.getPss()
        val appMemMb = pssKb / 1024

        // Workload & thermal attribution
        val maxTemp = maxOf(batteryTemp, socTemp)
        val thermalAlert: String
        val severity: Int
        when {
            maxTemp >= 45.0f -> {
                thermalAlert = "🔴 高温预警 (核心温度较高，建议降低计算负载)"
                severity = 2
            }
            maxTemp >= 40.0f -> {
                thermalAlert = "🟡 运行温热 (处于正常计算负载区间)"
                severity = 1
            }
            else -> {
                thermalAlert = "🟢 运行正常 (处于理想温度区间)"
                severity = 0
            }
        }

        // Hardware offload classification
        val isNpuPrimary = stats.offloadedLayers > 0
        val workloadDesc = when {
            stats.totalLayers == 0 -> "待机 / 未加载模型"
            stats.offloadedLayers == stats.totalLayers ->
                "🟢 全 NPU 硬件加速 (100% 层卸载，能耗效率最优)"
            stats.offloadedLayers > 0 ->
                "🟡 异构协同计算 (${stats.offloadedLayers}/${stats.totalLayers} 层 NPU，其余由 CPU 执行)"
            else ->
                "💻 纯 CPU 推理 (未启用专用硬件加速核心)"
        }

        val oomScoreAdj = ProcessProtectionManager.getOomScoreAdj()
        val isIgnoringBattery = ProcessProtectionManager.isIgnoringBatteryOptimizations(context)
        val isFgsRunning = com.example.llama.service.InferenceForegroundService.isServiceRunning

        return HardwareSnapshot(
            cpuUsagePercent = cpuUsage,
            batteryTempC = batteryTemp,
            socTempC = socTemp,
            appMemoryMb = appMemMb,
            totalRamGb = totalRamGb,
            usedRamGb = usedRamGb,
            availRamGb = availRamGb,
            ramPercent = ramPercent,
            gpuStatus = getGpuStatus(),
            activeAccelerator = stats.activeBackend,
            workloadDescription = workloadDesc,
            isNpuPrimary = isNpuPrimary,
            thermalAlert = thermalAlert,
            thermalSeverity = severity,
            oomScoreAdj = oomScoreAdj,
            isIgnoringBattery = isIgnoringBattery,
            isFgsRunning = isFgsRunning
        )
    }
}
