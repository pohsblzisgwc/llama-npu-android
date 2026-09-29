// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.monitor

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import java.io.File
import java.util.Locale

/**
 * Universal Android Device & Hardware Architecture Profiler.
 *
 * Dynamically detects device SoC, CPU cores, RAM size, and available compute units
 * across Qualcomm Snapdragon, MediaTek Dimensity, Google Tensor, Samsung Exynos, and ARM Mali/Adreno/Immortalis GPUs.
 * Eliminates all hardcoded device model references.
 */
object DeviceHardwareProfiler {

    enum class SocFamily(val displayName: String, val npuName: String, val gpuName: String) {
        SNAPDRAGON("高通骁龙 (Qualcomm Snapdragon)", "Hexagon HTP NPU", "Adreno GPU"),
        DIMENSITY("联发科天玑 (MediaTek Dimensity)", "MediaTek APU", "Mali / Immortalis GPU"),
        GOOGLE_TENSOR("谷歌 Tensor (Google Tensor)", "Google Edge TPU", "Mali GPU"),
        EXYNOS("三星猎户座 (Samsung Exynos)", "Exynos NPU", "Xclipse (AMD RDNA) GPU"),
        GENERIC_ARM("通用 ARMv8/v9 SoC", "端侧 NPU 加速器", "移动 GPU")
    }

    data class DeviceProfile(
        val manufacturer: String,
        val model: String,
        val socFamily: SocFamily,
        val socModelName: String,
        val totalRamGb: Float,
        val cpuCores: Int,
        val recommendedThreads: Int,
        val recommendedContextSize: Int,
        val recommendedHybridLayers: Int,
        val recommendedVmemMb: Int,
        val deviceSummary: String
    )

    fun getProfile(context: Context): DeviceProfile {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val totalRamGb = memInfo.totalMem.toFloat() / (1024 * 1024 * 1024)

        val cpuCores = Runtime.getRuntime().availableProcessors()

        // 1. Detect SoC Model Name & Family dynamically
        val socInfo = detectSocInfo()
        val family = socInfo.first
        val rawSocName = socInfo.second

        // 2. Compute dynamic recommendations based on physical hardware specs
        // Safe context size that will not cause OOM on this device's RAM
        val recCtx = when {
            totalRamGb >= 14.5f -> 4096
            totalRamGb >= 10.5f -> 3072
            totalRamGb >= 6.5f  -> 2048
            else                -> 1024
        }

        // Leave 2-4 cores for Android OS and UI rendering to prevent thermal throttling
        val recThreads = when {
            cpuCores >= 8 -> 4
            cpuCores >= 6 -> 4
            cpuCores >= 4 -> 3
            else          -> maxOf(1, cpuCores - 1)
        }

        // Recommended hybrid offload layers based on RAM headroom
        val recHybridLayers = when {
            totalRamGb >= 14.5f -> 22
            totalRamGb >= 10.5f -> 16
            totalRamGb >= 6.5f  -> 12
            else                -> 8
        }

        // Recommended VMEM for NPU driver (3200 MB for 16GB RAM devices to support Qwen 4B at 4096 context)
        val recVmemMb = when {
            totalRamGb >= 14.5f -> 3200
            totalRamGb >= 10.5f -> 2200
            totalRamGb >= 6.5f  -> 1600
            else                -> 1024
        }

        val manufacturerCapitalized = Build.MANUFACTURER.replaceFirstChar {
            if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString()
        }

        val summary = "$manufacturerCapitalized ${Build.MODEL} • $rawSocName • ${"%.0f".format(totalRamGb)}GB RAM • ${cpuCores}核CPU"

        return DeviceProfile(
            manufacturer = manufacturerCapitalized,
            model = Build.MODEL,
            socFamily = family,
            socModelName = rawSocName,
            totalRamGb = totalRamGb,
            cpuCores = cpuCores,
            recommendedThreads = recThreads,
            recommendedContextSize = recCtx,
            recommendedHybridLayers = recHybridLayers,
            recommendedVmemMb = recVmemMb,
            deviceSummary = summary
        )
    }

    private fun detectSocInfo(): Pair<SocFamily, String> {
        val hardware = Build.HARDWARE.lowercase(Locale.ROOT)
        val board = Build.BOARD.lowercase(Locale.ROOT)
        var socModel = ""

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            socModel = Build.SOC_MODEL.lowercase(Locale.ROOT)
        }

        val cpuInfoHardware = readCpuInfoHardware().lowercase(Locale.ROOT)
        val combined = "$hardware $board $socModel $cpuInfoHardware"

        return when {
            combined.contains("qcom") || combined.contains("qualcomm") ||
            combined.contains("snapdragon") || combined.contains("sm8") ||
            combined.contains("sm7") || combined.contains("taro") ||
            combined.contains("kalama") || combined.contains("pineapple") ||
            combined.contains("sun") -> {
                val cleanName = when {
                    combined.contains("sm8750") || combined.contains("sun") -> "骁龙 8 至尊版 (Snapdragon 8 Elite / HTP v79)"
                    combined.contains("sm8650") || combined.contains("pineapple") -> "骁龙 8 Gen 3 (SM8650 / HTP v75)"
                    combined.contains("sm8550") || combined.contains("kalama") -> "骁龙 8 Gen 2 (SM8550 / HTP v73)"
                    combined.contains("sm8450") || combined.contains("taro") || combined.contains("sm8475") -> "骁龙 8/8+ Gen 1 (SM8450 / HTP v69)"
                    combined.contains("sm7675") || combined.contains("sm7550") -> "骁龙 7+ Gen 3 / 7 Gen 3"
                    else -> "高通骁龙芯片 (Qualcomm Snapdragon)"
                }
                Pair(SocFamily.SNAPDRAGON, cleanName)
            }
            combined.contains("mt") || combined.contains("mediatek") || combined.contains("dimensity") ||
            combined.contains("k69") || combined.contains("k68") -> {
                val cleanName = when {
                    combined.contains("mt6989") || combined.contains("dimensity 9300") -> "联发科天玑 9300 (Dimensity 9300 / APU)"
                    combined.contains("mt6985") || combined.contains("dimensity 9200") -> "联发科天玑 9200 (Dimensity 9200 / APU)"
                    combined.contains("dimensity 8300") || combined.contains("mt6897") -> "联发科天玑 8300 (Dimensity 8300)"
                    else -> "联发科天玑芯片 (MediaTek Dimensity / APU)"
                }
                Pair(SocFamily.DIMENSITY, cleanName)
            }
            combined.contains("tensor") || combined.contains("gs101") ||
            combined.contains("gs201") || combined.contains("zuma") || combined.contains("ripcurrent") -> {
                Pair(SocFamily.GOOGLE_TENSOR, "谷歌 Tensor 芯片 (Google Tensor / TPU)")
            }
            combined.contains("exynos") || combined.contains("s5e9945") ||
            combined.contains("s5e9925") || combined.contains("s5e8845") -> {
                Pair(SocFamily.EXYNOS, "三星猎户座芯片 (Samsung Exynos / Xclipse GPU)")
            }
            else -> {
                val fallbackName = if (socModel.isNotBlank()) socModel.uppercase() else "ARM Cortex 移动平台"
                Pair(SocFamily.GENERIC_ARM, fallbackName)
            }
        }
    }

    private fun readCpuInfoHardware(): String {
        return try {
            val file = File("/proc/cpuinfo")
            if (file.exists() && file.canRead()) {
                file.useLines { lines ->
                    for (line in lines) {
                        if (line.startsWith("Hardware", ignoreCase = true) ||
                            line.startsWith("model name", ignoreCase = true)) {
                            return@useLines line.substringAfter(':').trim()
                        }
                    }
                }
            }
            ""
        } catch (e: Exception) {
            ""
        }
    }
}
