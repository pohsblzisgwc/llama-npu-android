// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.monitor

import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.provider.Settings
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

/**
 * Advanced Process Survival & Priority Manager.
 *
 * Designed for Android 14+ across all brands and SoC architectures (Snapdragon, Dimensity, Tensor, Exynos):
 * 1. Reads real-time Linux kernel oom_score_adj to evaluate kill risk.
 * 2. Manages system battery optimization whitelists (exempt from doze/app standby).
 * 3. Bridges to OEM PowerKeeper / Battery Saver ("无限制") and Autostart settings.
 * 4. Provides Root-level elevation to lock oom_score_adj at -1000 (Immortal / Kernel level).
 * 5. Provides one-click copyable ADB commands to disable Android 12-14 PhantomProcessKiller.
 */
object ProcessProtectionManager {

    private const val TAG = "ProcessProtectionMgr"

    @Volatile
    var isRootImmortalActive: Boolean = false
        private set

    /**
     * Reads current process oom_score_adj from Linux procfs.
     * Lower value = Higher priority (less likely to be killed by Low Memory Killer).
     * -1000 : System-level immortal (Root)
     * 0     : Active Foreground App
     * 200   : Foreground Service (Protected)
     * 700   : Previous App
     * 900+  : Cached Background App (High kill risk)
     */
    fun getOomScoreAdj(): Int {
        return try {
            val file = File("/proc/self/oom_score_adj")
            if (file.exists() && file.canRead()) {
                file.readText().trim().toIntOrNull() ?: 0
            } else {
                0
            }
        } catch (e: Exception) {
            0
        }
    }

    /**
     * Reads current process oom_score from Linux procfs.
     */
    fun getOomScore(): Int {
        return try {
            val file = File("/proc/self/oom_score")
            if (file.exists() && file.canRead()) {
                file.readText().trim().toIntOrNull() ?: 0
            } else {
                0
            }
        } catch (e: Exception) {
            0
        }
    }

    /**
     * Returns true if app is exempt from system battery optimizations (Doze mode).
     */
    fun isIgnoringBatteryOptimizations(context: Context): Boolean {
        return try {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            pm.isIgnoringBatteryOptimizations(context.packageName)
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Directly triggers system dialog to request battery optimization exemption ("无限制").
     */
    fun requestIgnoreBatteryOptimizations(activity: Activity) {
        try {
            val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                data = Uri.parse("package:${activity.packageName}")
            }
            activity.startActivity(intent)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to launch ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS", e)
            try {
                val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                activity.startActivity(intent)
            } catch (e2: Exception) {
                Log.e(TAG, "Failed fallback battery settings", e2)
            }
        }
    }

    /**
     * Attempts to open OEM vendor-specific background power management / autostart settings:
     * Supports Xiaomi/HyperOS, Huawei/Honor, OPPO/Realme/OnePlus, Vivo/iQOO, Samsung, and Android standard.
     */
    fun openVendorBackgroundSettings(activity: Activity): Boolean {
        val intents = listOf(
            // Xiaomi / HyperOS / MIUI PowerKeeper
            Intent().apply {
                component = ComponentName("com.miui.powerkeeper", "com.miui.powerkeeper.ui.HiddenAppsConfigActivity")
                putExtra("package_name", activity.packageName)
                putExtra("package_label", activity.applicationInfo.loadLabel(activity.packageManager))
            },
            // Xiaomi / HyperOS Autostart
            Intent().apply {
                component = ComponentName("com.miui.securitycenter", "com.miui.permcenter.autostart.AutoStartManagementActivity")
            },
            // Huawei / Honor SystemManager
            Intent().apply {
                component = ComponentName("com.huawei.systemmanager", "com.huawei.systemmanager.startupmgr.ui.StartupNormalAppListActivity")
            },
            Intent().apply {
                component = ComponentName("com.huawei.systemmanager", "com.huawei.systemmanager.optimize.process.ProtectActivity")
            },
            // OPPO / Realme / OnePlus
            Intent().apply {
                component = ComponentName("com.coloros.safecenter", "com.coloros.safecenter.permission.startup.StartupAppListActivity")
            },
            // Vivo / iQOO
            Intent().apply {
                component = ComponentName("com.iqoo.secure", "com.iqoo.secure.ui.phoneoptimize.AddWhiteListActivity")
            },
            // Samsung Device Care
            Intent().apply {
                component = ComponentName("com.samsung.android.lool", "com.samsung.android.sm.battery.ui.BatteryActivity")
            },
            // Standard Android Application Details Settings
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.parse("package:${activity.packageName}")
            }
        )

        for (intent in intents) {
            try {
                activity.startActivity(intent)
                return true
            } catch (_: Exception) {
                // Try next OEM candidate
            }
        }
        return false
    }

    /**
     * Checks if Root (su) is available on the device.
     */
    fun checkRootAvailable(): Boolean {
        return try {
            val p = Runtime.getRuntime().exec(arrayOf("su", "-c", "id"))
            val exitCode = p.waitFor()
            exitCode == 0
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Applies Root-level kernel immortal process protection:
     * 1. Locks oom_score_adj to -1000 (OOM killer will NEVER touch this process).
     * 2. Sets Linux process nice priority to -20 (maximum CPU scheduling priority).
     * 3. Disables Android 12-14 PhantomProcessKiller via device_config.
     */
    fun applyRootImmortalProtection(scope: CoroutineScope): Pair<Boolean, String> {
        val pid = Process.myPid()
        return try {
            val commands = arrayOf(
                "echo -1000 > /proc/$pid/oom_score_adj",
                "renice -20 -p $pid",
                "/system/bin/device_config put activity_manager max_phantom_processes 2147483647",
                "setprop persist.sys.fflag.override.settings_enable_monitor_phantom_procs false"
            )

            val p = Runtime.getRuntime().exec("su")
            p.outputStream.bufferedWriter().use { writer ->
                for (cmd in commands) {
                    writer.write(cmd)
                    writer.newLine()
                }
                writer.write("exit\n")
                writer.flush()
            }
            p.waitFor()

            val currentAdj = getOomScoreAdj()
            if (currentAdj == -1000) {
                isRootImmortalActive = true
                startRootKeeperLoop(scope)
                Pair(true, "👑 提权成功！进程 oom_score_adj 已锁定为 -1000 (系统最高保护级别，免受 LMK 终止)")
            } else {
                Pair(false, "Root 命令执行完成，当前 oom_score_adj: $currentAdj")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply root protection", e)
            Pair(false, "Root 提权失败: ${e.message}")
        }
    }

    /**
     * Periodically monitors and re-asserts -1000 so Android ActivityManager cannot override it.
     */
    private fun startRootKeeperLoop(scope: CoroutineScope) {
        scope.launch(Dispatchers.IO) {
            val pid = Process.myPid()
            while (isActive && isRootImmortalActive) {
                try {
                    val adj = getOomScoreAdj()
                    if (adj != -1000) {
                        Log.w(TAG, "Detected oom_score_adj changed to $adj, restoring -1000 via su...")
                        val p = Runtime.getRuntime().exec(arrayOf("su", "-c", "echo -1000 > /proc/$pid/oom_score_adj"))
                        p.waitFor()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Error in Root keeper loop", e)
                }
                delay(12_000) // check every 12 seconds
            }
        }
    }

    /**
     * Generates standard ADB command script for users without Root access.
     */
    fun getAdbCommandGuide(context: Context): String {
        val pkg = context.packageName
        return """
# =======================================================
# Android 移动端大模型进程常驻保活与防终止 ADB 指令 (通用各品牌机型)
# =======================================================

# 1. 禁用 Android 12-14 幽灵进程清理限制 (Phantom Process Killer)
adb shell /system/bin/device_config put activity_manager max_phantom_processes 2147483647
adb shell setprop persist.sys.fflag.override.settings_enable_monitor_phantom_procs false

# 2. 将应用加入系统电池优化白名单 (无限制)
adb shell dumpsys deviceidle whitelist +$pkg

# 3. 授予存储直接映射权限
adb shell appops set $pkg MANAGE_EXTERNAL_STORAGE allow

# 4. (若已获得 Root 权限) 将当前运行中的大模型进程强制锁定为 -1000 最高保护级别:
adb shell "su -c 'echo -1000 > /proc/\$(pidof $pkg)/oom_score_adj'"
        """.trimIndent()
    }
}
