// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

package com.example.llama.settings

import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.util.Log
import com.arm.aichat.HardwareAccelerator
import com.arm.aichat.ModelConfig
import com.arm.aichat.NpuTopology
import com.arm.aichat.KvCacheType
import com.arm.aichat.FlashAttnMode
import org.json.JSONArray
import java.io.File
import java.net.URLDecoder

class ModelSettingsManager(private val context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("snapdragon_llm_settings", Context.MODE_PRIVATE)

    private var activePfd: ParcelFileDescriptor? = null
    private var activeMmprojPfd: ParcelFileDescriptor? = null

    fun getSavedModelConfig(): ModelConfig {
        val profile = com.example.llama.monitor.DeviceHardwareProfiler.getProfile(context)
        val defaultAccelerator = if (profile.socFamily == com.example.llama.monitor.DeviceHardwareProfiler.SocFamily.SNAPDRAGON) {
            HardwareAccelerator.NPU_HEXAGON
        } else {
            HardwareAccelerator.HYBRID_COMBINED
        }
        val defaultLayers = when (defaultAccelerator) {
            HardwareAccelerator.NPU_HEXAGON, HardwareAccelerator.GPU_GENERIC -> 99
            HardwareAccelerator.CPU_ARM -> 0
            HardwareAccelerator.HYBRID_COMBINED -> profile.recommendedHybridLayers
        }

        // Automatic config migration: if upgrading from earlier version where CPU fallback poisoned prefs or 22 layers were hardcoded
        val configVersion = prefs.getInt("config_version", 1)
        if (configVersion < 2) {
            prefs.edit()
                .putInt("config_version", 2)
                .putString("accelerator", defaultAccelerator.name)
                .putInt("n_gpu_layers", defaultLayers)
                .apply()
        }

        val acceleratorStr = prefs.getString("accelerator", defaultAccelerator.name)
        val accelerator = try {
            HardwareAccelerator.valueOf(acceleratorStr ?: defaultAccelerator.name)
        } catch (e: Exception) {
            defaultAccelerator
        }

        val topoStr = prefs.getString("npu_topology", NpuTopology.SINGLE_CORE.name)
        val npuTopology = try {
            val parsed = NpuTopology.valueOf(topoStr ?: NpuTopology.SINGLE_CORE.name)
            // Migrate old DUAL_CORE_GROUPED if necessary
            parsed
        } catch (e: Exception) {
            NpuTopology.SINGLE_CORE
        }

        val mmprojPath = prefs.getString("mmproj_path", null)

        val kvTypeStr = prefs.getString("kv_cache_type", KvCacheType.Q8_0.name)
        val kvCacheType = try {
            KvCacheType.valueOf(kvTypeStr ?: KvCacheType.Q8_0.name)
        } catch (e: Exception) {
            KvCacheType.Q8_0
        }

        val flashAttnStr = prefs.getString("flash_attn_mode", FlashAttnMode.AUTO.name)
        val flashAttnMode = try {
            FlashAttnMode.valueOf(flashAttnStr ?: FlashAttnMode.AUTO.name)
        } catch (e: Exception) {
            FlashAttnMode.AUTO
        }

        val isAdvanced = prefs.getBoolean("is_advanced_mode", false)
        val rawLayers = prefs.getInt("n_gpu_layers", defaultLayers)
        val finalLayers = if (!isAdvanced && (accelerator == HardwareAccelerator.NPU_HEXAGON || accelerator == HardwareAccelerator.GPU_GENERIC)) {
            99
        } else {
            rawLayers
        }

        return ModelConfig(
            nGpuLayers = finalLayers,
            npuTopology = npuTopology,
            nThreads = prefs.getInt("n_threads", profile.recommendedThreads),
            nCtx = prefs.getInt("n_ctx", profile.recommendedContextSize),
            ubatchSize = prefs.getInt("ubatch_size", 256),
            batchSize = prefs.getInt("batch_size", 512),
            temp = prefs.getFloat("temp", 0.7f),
            topP = prefs.getFloat("top_p", 0.9f),
            topK = prefs.getInt("top_k", 40),
            repeatPenalty = prefs.getFloat("repeat_penalty", 1.1f),
            accelerator = accelerator,
            mmprojPath = mmprojPath,
            isAdvancedMode = isAdvanced,
            kvCacheType = kvCacheType,
            flashAttnMode = flashAttnMode,
            coreAffinity = prefs.getBoolean("core_affinity", true),
            useMmap = prefs.getBoolean("use_mmap", true),
            useMlock = prefs.getBoolean("use_mlock", false),
            dma64 = prefs.getBoolean("dma64", true),
            vmemMb = prefs.getInt("vmem_mb", profile.recommendedVmemMb)
        )
    }

    fun saveModelConfig(config: ModelConfig) {
        prefs.edit()
            .putInt("config_version", 2)
            .putString("accelerator", config.accelerator.name)
            .putString("npu_topology", config.npuTopology.name)
            .putInt("n_gpu_layers", config.nGpuLayers)
            .putInt("n_threads", config.nThreads)
            .putInt("n_ctx", config.nCtx)
            .putInt("ubatch_size", config.ubatchSize)
            .putInt("batch_size", config.batchSize)
            .putFloat("temp", config.temp)
            .putFloat("top_p", config.topP)
            .putInt("top_k", config.topK)
            .putFloat("repeat_penalty", config.repeatPenalty)
            .putString("mmproj_path", config.mmprojPath)
            .putBoolean("is_advanced_mode", config.isAdvancedMode)
            .putString("kv_cache_type", config.kvCacheType.name)
            .putString("flash_attn_mode", config.flashAttnMode.name)
            .putBoolean("core_affinity", config.coreAffinity)
            .putBoolean("use_mmap", config.useMmap)
            .putBoolean("use_mlock", config.useMlock)
            .putBoolean("dma64", config.dma64)
            .putInt("vmem_mb", config.vmemMb)
            .apply()
    }

    fun getRecentModelPaths(): List<String> {
        val jsonStr = prefs.getString("recent_models", "[]") ?: "[]"
        val list = mutableListOf<String>()
        try {
            val arr = JSONArray(jsonStr)
            for (i in 0 until arr.length()) {
                val p = arr.getString(i)
                if (p.isNotBlank()) list.add(p)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse recent models", e)
        }
        return list
    }

    fun addRecentModelPath(path: String) {
        val current = getRecentModelPaths().toMutableList()
        current.remove(path)
        current.add(0, path)
        val trimmed = current.take(6)
        val arr = JSONArray(trimmed)
        prefs.edit().putString("recent_models", arr.toString()).apply()
    }

    fun getLastSelectedModelPath(): String? {
        return prefs.getString("last_model_path", null)
    }

    fun setLastSelectedModelPath(path: String) {
        prefs.edit().putString("last_model_path", path).apply()
        addRecentModelPath(path)
    }

    fun getLastSelectedModelUri(): String? {
        return prefs.getString("last_model_uri", null)
    }

    fun setLastSelectedModelUri(uri: String) {
        prefs.edit().putString("last_model_uri", uri).apply()
    }

    fun getLastSelectedDisplayName(): String? {
        return prefs.getString("last_display_name", null)
    }

    fun setLastSelectedDisplayName(name: String) {
        prefs.edit().putString("last_display_name", name).apply()
    }

    fun getLastSelectedMmprojPath(): String? {
        return prefs.getString("mmproj_path", null)
    }

    fun setLastSelectedMmprojPath(path: String?) {
        prefs.edit().putString("mmproj_path", path).apply()
    }

    fun resolveMmprojZeroCopyPath(uri: Uri): Pair<String, String> {
        val displayName = queryDisplayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "mmproj.gguf"
        try {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (e: Exception) {}

        if (uri.scheme == "file") {
            val path = uri.path ?: ""
            return Pair(path, displayName)
        }

        val resolvedPath = resolveRealPath(uri)
        if (resolvedPath != null) {
            val f = File(resolvedPath)
            if (f.exists()) {
                Log.i(TAG, "Zero-copy resolved mmproj primary path: $resolvedPath")
                return Pair(resolvedPath, displayName)
            }
        }

        try {
            activeMmprojPfd?.close()
            val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            if (pfd != null) {
                activeMmprojPfd = pfd
                val fdPath = "/proc/self/fd/${pfd.fd}"
                Log.i(TAG, "Zero-copy opened mmproj via procfs fd: $fdPath (displayName: $displayName)")
                return Pair(fdPath, displayName)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open ParcelFileDescriptor for mmproj zero-copy", e)
        }

        if (resolvedPath != null) {
            return Pair(resolvedPath, displayName)
        }

        throw IllegalArgumentException("无法直接读取该多模态模型文件，请检查存储访问权限")
    }

    /**
     * Resolves a Uri to a zero-copy direct file path without copying to internal storage.
     */
    fun resolveZeroCopyPath(uri: Uri): Pair<String, String> {
        val displayName = queryDisplayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "model.gguf"
        setLastSelectedModelUri(uri.toString())
        setLastSelectedDisplayName(displayName)

        // Try persistable permission grant
        try {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (e: Exception) {}

        // Direct file:// scheme
        if (uri.scheme == "file") {
            val path = uri.path ?: ""
            return Pair(path, displayName)
        }

        // 1. Try DocumentsContract / direct canonical filesystem path resolution
        val resolvedPath = resolveRealPath(uri)
        if (resolvedPath != null) {
            val f = File(resolvedPath)
            if (f.exists()) {
                Log.i(TAG, "Zero-copy resolved primary path: $resolvedPath")
                return Pair(resolvedPath, displayName)
            }
        }

        // 2. Direct fallback: Open ParcelFileDescriptor and use /proc/self/fd/NN
        try {
            activePfd?.close()
            val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            if (pfd != null) {
                activePfd = pfd
                val fdPath = "/proc/self/fd/${pfd.fd}"
                Log.i(TAG, "Zero-copy opened via procfs fd: $fdPath (displayName: $displayName)")
                return Pair(fdPath, displayName)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open ParcelFileDescriptor for zero-copy", e)
        }

        if (resolvedPath != null) {
            return Pair(resolvedPath, displayName)
        }

        throw IllegalArgumentException("无法直接读取该文件，请检查存储访问权限")
    }

    /**
     * Resolves canonical filesystem paths for Android 11+ Scoped Storage Uris
     */
    private fun resolveRealPath(uri: Uri): String? {
        try {
            if (DocumentsContract.isDocumentUri(context, uri)) {
                val docId = DocumentsContract.getDocumentId(uri)
                val authority = uri.authority

                if ("com.android.externalstorage.documents" == authority) {
                    val split = docId.split(":")
                    val type = split.getOrNull(0) ?: ""
                    val relPath = URLDecoder.decode(split.getOrNull(1) ?: "", "UTF-8")
                    return if ("primary".equals(type, ignoreCase = true)) {
                        "/storage/emulated/0/$relPath"
                    } else {
                        "/storage/$type/$relPath"
                    }
                } else if ("com.android.providers.downloads.documents" == authority) {
                    if (docId.startsWith("raw:")) {
                        return docId.removePrefix("raw:")
                    }
                    val id = docId.toLongOrNull()
                    if (id != null) {
                        val contentUri = ContentUris.withAppendedId(
                            Uri.parse("content://downloads/public_downloads"), id
                        )
                        return getDataColumn(contentUri)
                    }
                } else if ("com.android.providers.media.documents" == authority) {
                    val split = docId.split(":")
                    val type = split.getOrNull(0) ?: ""
                    val id = split.getOrNull(1) ?: return null
                    val contentUri = when (type) {
                        "video" -> MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                        "audio" -> MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
                        else -> MediaStore.Files.getContentUri("external")
                    }
                    return getDataColumn(contentUri, "_id=?", arrayOf(id))
                }
            } else if ("content".equals(uri.scheme, ignoreCase = true)) {
                return getDataColumn(uri)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to resolve real path: ${e.message}")
        }
        return null
    }

    private fun getDataColumn(uri: Uri, selection: String? = null, selectionArgs: Array<String>? = null): String? {
        val column = MediaStore.MediaColumns.DATA
        val projection = arrayOf(column)
        try {
            context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(column)
                    if (index != -1) {
                        return cursor.getString(index)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "getDataColumn failed: ${e.message}")
        }
        return null
    }

    /**
     * Prepares and validates a model path before feeding to the native inference engine.
     * Re-opens ParcelFileDescriptor if needed.
     */
    fun preparePathForLoad(inputPath: String): Pair<String, String> {
        if (inputPath.startsWith("content://")) {
            val uri = Uri.parse(inputPath)
            return resolveZeroCopyPath(uri)
        }

        val file = File(inputPath)
        if (file.exists()) {
            return Pair(file.absolutePath, file.name)
        }

        if (inputPath.startsWith("/proc/self/fd/")) {
            if (File(inputPath).exists()) {
                val displayName = getLastSelectedDisplayName() ?: "model.gguf"
                return Pair(inputPath, displayName)
            } else {
                getLastSelectedModelUri()?.let { savedUriStr ->
                    val uri = Uri.parse(savedUriStr)
                    return resolveZeroCopyPath(uri)
                }
            }
        }

        return Pair(inputPath, file.name)
    }

    private fun queryDisplayName(uri: Uri): String? {
        return try {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex != -1 && cursor.moveToFirst()) {
                    cursor.getString(nameIndex)
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    fun closeActivePfd() {
        try {
            activePfd?.close()
            activePfd = null
        } catch (e: Exception) {
            // Ignore
        }
    }

    companion object {
        private const val TAG = "ModelSettingsManager"
    }
}
