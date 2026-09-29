package com.arm.aichat.internal

import android.content.Context
import android.util.Log
import com.arm.aichat.HardwareAccelerator
import com.arm.aichat.InferenceEngine
import com.arm.aichat.InferenceStats
import com.arm.aichat.ModelConfig
import com.arm.aichat.NpuTopology
import com.arm.aichat.UnsupportedArchitectureException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * JNI wrapper for llama.cpp with heterogeneous mobile hardware offloading support (NPU / GPU / CPU).
 */
internal class InferenceEngineImpl private constructor(
    private val nativeLibDir: String
) : InferenceEngine {

    companion object {
        private val TAG = InferenceEngineImpl::class.java.simpleName

        @Volatile
        private var instance: InferenceEngine? = null

        internal fun getInstance(context: Context) =
            instance ?: synchronized(this) {
                val nativeLibDir = context.applicationInfo.nativeLibraryDir
                require(nativeLibDir.isNotBlank()) { "Expected a valid native library path!" }

                try {
                    Log.i(TAG, "Instantiating Snapdragon InferenceEngineImpl...")
                    InferenceEngineImpl(nativeLibDir).also { instance = it }
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(TAG, "Failed to load native library from $nativeLibDir", e)
                    throw e
                }
            }
    }

    /**
     * Native JNI methods
     * @see ai_chat.cpp
     */
    private external fun init(nativeLibDir: String)

    private external fun load(modelPath: String): Int

    private external fun loadExt(modelPath: String, nGpuLayers: Int, npuTopology: String): Int

    private external fun loadAdvanced(
        modelPath: String,
        nGpuLayers: Int,
        npuTopology: String,
        useMmap: Boolean,
        useMlock: Boolean
    ): Int

    private external fun prepare(): Int

    private external fun prepareExt(
        nCtx: Int,
        nThreads: Int,
        ubatchSize: Int,
        temp: Float,
        topP: Float
    ): Int

    private external fun prepareAdvanced(
        nCtx: Int,
        nThreads: Int,
        ubatchSize: Int,
        batchSize: Int,
        temp: Float,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
        kvCacheType: Int,
        flashAttnMode: Int,
        coreAffinity: Boolean,
        dma64: Boolean,
        vmemMb: Int
    ): Int

    private external fun systemInfo(): String

    private external fun nativeGetInferenceStats(): String

    private external fun nativeCancelGeneration(): Unit

    private external fun benchModel(pp: Int, tg: Int, pl: Int, nr: Int): String

    private external fun processSystemPrompt(systemPrompt: String): Int

    private external fun processUserPrompt(userPrompt: String, predictLength: Int): Int

    private external fun loadMmproj(mmprojPath: String): Int

    private external fun processUserPromptWithImage(userPrompt: String, imagePath: String, predictLength: Int): Int

    private external fun generateNextToken(): String?

    private external fun unload()

    private external fun shutdown()

    private external fun nativeGetLastError(): String

    private external fun nativeResetConversation()

    private external fun nativeSetSpeculativePld(enabled: Boolean)

    private val _state =
        MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.Uninitialized)
    override val state: StateFlow<InferenceEngine.State> = _state.asStateFlow()

    override var onHardwareFallback: ((HardwareAccelerator) -> Unit)? = null
    private var lastModelPath: String? = null
    private var lastModelConfig: ModelConfig? = null
    private var lastSystemPrompt: String? = null
    private var currentAccelerator: HardwareAccelerator = HardwareAccelerator.NPU_HEXAGON

    private var _readyForSystemPrompt = false
    @Volatile
    private var _cancelGeneration = false

    @OptIn(ExperimentalCoroutinesApi::class)
    private val llamaDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val llamaScope = CoroutineScope(llamaDispatcher + SupervisorJob())

    init {
        llamaScope.launch {
            try {
                check(_state.value is InferenceEngine.State.Uninitialized) {
                    "Cannot load native library in ${_state.value.javaClass.simpleName}!"
                }
                _state.value = InferenceEngine.State.Initializing
                Log.i(TAG, "Loading native libraries for Snapdragon Hexagon & Multimodal...")
                try { System.loadLibrary("ggml-base") } catch (e: Throwable) {}
                try { System.loadLibrary("ggml") } catch (e: Throwable) {}
                try { System.loadLibrary("ggml-cpu") } catch (e: Throwable) {}
                try { System.loadLibrary("ggml-hexagon") } catch (e: Throwable) {}
                try { System.loadLibrary("llama") } catch (e: Throwable) {}
                try { System.loadLibrary("mtmd") } catch (e: Throwable) {}
                System.loadLibrary("ai-chat")
                init(nativeLibDir)
                _state.value = InferenceEngine.State.Initialized
                Log.i(TAG, "Native library loaded! System info: \n${systemInfo()}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library", e)
                throw e
            }
        }
    }

    override suspend fun loadModel(pathToModel: String) {
        loadModel(pathToModel, ModelConfig())
    }

    override suspend fun loadModel(pathToModel: String, config: ModelConfig) =
        withContext(llamaDispatcher) {
            if (_state.value is InferenceEngine.State.LoadingModel) {
                Log.w(TAG, "Model is already loading, skipping duplicate loadModel call: $pathToModel")
                return@withContext
            }

            if (_state.value is InferenceEngine.State.Uninitialized || _state.value is InferenceEngine.State.Initializing) {
                throw IllegalStateException("Cannot load model before engine is initialized: ${_state.value}")
            }

            if (_state.value !is InferenceEngine.State.Initialized) {
                try {
                    Log.i(TAG, "Resetting previous model/state before loading...")
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.UnloadingModel
                    unload()
                } catch (e: Exception) {
                    Log.w(TAG, "Error during pre-load unload, resetting state", e)
                }
            }

            try {
                Log.i(TAG, "Passing model path to native loader: $pathToModel")
                if (!pathToModel.startsWith("/proc/self/fd/")) {
                    val f = File(pathToModel)
                    if (!f.exists()) {
                        Log.w(TAG, "Notice: Java File.exists() false for $pathToModel, letting native loader proceed.")
                    }
                }

                val offloadLayers = when (config.accelerator) {
                    HardwareAccelerator.CPU_ARM -> 0
                    HardwareAccelerator.HYBRID_COMBINED -> {
                        if (config.nGpuLayers in 1..98) config.nGpuLayers else 22
                    }
                    HardwareAccelerator.NPU_HEXAGON, HardwareAccelerator.GPU_GENERIC -> {
                        if (config.isAdvancedMode && config.nGpuLayers in 1..98) config.nGpuLayers else 99
                    }
                }
                val topology = config.npuTopology.configString

                Log.i(TAG, "Loading GGUF with accelerator=${config.accelerator.name}, offloadLayers=$offloadLayers, topology=$topology, threads=${config.nThreads}...")
                _readyForSystemPrompt = false
                _state.value = InferenceEngine.State.LoadingModel

                loadAdvanced(
                    pathToModel,
                    offloadLayers,
                    topology,
                    config.useMmap,
                    config.useMlock
                ).let {
                    if (it != 0) throw UnsupportedArchitectureException()
                }

                prepareAdvanced(
                    config.nCtx,
                    config.nThreads,
                    config.ubatchSize,
                    config.batchSize,
                    config.temp,
                    config.topP,
                    config.topK,
                    config.repeatPenalty,
                    config.kvCacheType.typeCode,
                    config.flashAttnMode.modeCode,
                    config.coreAffinity,
                    config.dma64,
                    config.vmemMb
                ).let {
                    if (it != 0) throw IOException("Failed to prepare llama context")
                }

                lastModelPath = pathToModel
                lastModelConfig = config
                currentAccelerator = config.accelerator

                if (!config.mmprojPath.isNullOrBlank()) {
                    try {
                        Log.i(TAG, "Loading multimodal projector: ${config.mmprojPath}")
                        val mmCode = loadMmproj(config.mmprojPath)
                        if (mmCode == 0) {
                            Log.i(TAG, "Multimodal projector loaded successfully!")
                        } else {
                            Log.w(TAG, "Failed to load multimodal projector (code $mmCode): ${nativeGetLastError()}")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Exception loading mmproj during loadModel", e)
                    }
                }

                Log.i(TAG, "GGUF Model loaded successfully with Snapdragon NPU offloading!")
                _readyForSystemPrompt = true
                _cancelGeneration = false
                _state.value = InferenceEngine.State.ModelReady
            } catch (e: Exception) {
                Log.e(TAG, (e.message ?: "Error loading model") + "\n" + pathToModel, e)
                _state.value = InferenceEngine.State.Error(e)
                throw e
            }
        }

    override suspend fun loadMmproj(pathToMmproj: String?): Boolean =
        withContext(llamaDispatcher) {
            try {
                if (pathToMmproj.isNullOrBlank()) {
                    loadMmproj("")
                    return@withContext true
                }
                val code = loadMmproj(pathToMmproj)
                if (code == 0) {
                    Log.i(TAG, "loadMmproj succeeded for: $pathToMmproj")
                    true
                } else {
                    Log.e(TAG, "loadMmproj failed with code $code: ${nativeGetLastError()}")
                    false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception loading mmproj", e)
                false
            }
        }

    override suspend fun setSystemPrompt(systemPrompt: String) =
        withContext(llamaDispatcher) {
            require(systemPrompt.isNotBlank()) { "Cannot process empty system prompt!" }
            check(_readyForSystemPrompt) { "System prompt must be set right after model load!" }
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Cannot process system prompt in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Setting system prompt...")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            processSystemPrompt(systemPrompt).let { result ->
                if (result != 0) {
                    RuntimeException("Failed to process system prompt: $result").also {
                        _state.value = InferenceEngine.State.Error(it)
                        throw it
                    }
                }
            }
            lastSystemPrompt = systemPrompt
            Log.i(TAG, "System prompt processed! Ready for conversation.")
            _state.value = InferenceEngine.State.ModelReady
        }

    override fun sendUserPrompt(
        message: String,
        imagePath: String?,
        predictLength: Int,
    ): Flow<String> = flow {
        require(message.isNotEmpty()) { "User prompt cannot be empty!" }
        check(_state.value is InferenceEngine.State.ModelReady) {
            "User prompt discarded due to state: ${_state.value.javaClass.simpleName}"
        }

        try {
            Log.i(TAG, "Processing user prompt (hasImage=${!imagePath.isNullOrBlank()}) on Snapdragon NPU...")
            _readyForSystemPrompt = false
            _cancelGeneration = false
            _state.value = InferenceEngine.State.ProcessingUserPrompt

            var result = if (!imagePath.isNullOrBlank()) {
                processUserPromptWithImage(message, imagePath, predictLength)
            } else {
                processUserPrompt(message, predictLength)
            }
            if (result != 0 && currentAccelerator == HardwareAccelerator.NPU_HEXAGON && lastModelPath != null && imagePath.isNullOrBlank()) {
                val nativeErr = try { nativeGetLastError().trim() } catch (e: Exception) { "" }
                Log.w(TAG, "NPU decode failed (code $result: $nativeErr). Initiating transparent fallback to CPU mode...")
                emit("⚡ [检测到 NPU 共享显存/算子超限 (错误码 $result)，已自动无缝切换至 CPU 模式继续推理...]\n\n")

                val fallbackOk = performCpuFallback()
                if (fallbackOk) {
                    Log.i(TAG, "Fallback to CPU succeeded, re-evaluating prompt...")
                    result = processUserPrompt(message, predictLength)
                }
            }

            if (result != 0) {
                val nativeErr = try { nativeGetLastError().trim() } catch (e: Exception) { "" }
                val detail = if (nativeErr.isNotEmpty()) ": $nativeErr" else ""
                val advice = when {
                    nativeErr.contains("rpcmem", ignoreCase = true) ||
                    nativeErr.contains("fastrpc", ignoreCase = true) ||
                    nativeErr.contains("mmap", ignoreCase = true) ->
                        "\n💡 提示: NPU 共享内存超限 (FastRPC/rpcmem)。请在设置中选择【ARM Cortex CPU】模式或降低 NPU 卸载层数 (如 15-22 层)。"
                    nativeErr.contains("context", ignoreCase = true) || result == 2 ->
                        "\n💡 提示: 模型推理图计算失败或上下文超限，请在设置中适当降低 Context 长度 (如 2048) 或尝试 CPU 模式。"
                    else ->
                        "\n💡 提示: 若 NPU 计算报错，可在侧边栏设置中切换为【CPU 模式】以获得稳定运行体验。"
                }
                val err = "模型推理失败 (错误码 $result$detail)$advice"
                Log.e(TAG, err)
                _state.value = InferenceEngine.State.ModelReady
                throw IOException(err)
            }

            Log.i(TAG, "Prompt evaluated! Streaming tokens...")
            _state.value = InferenceEngine.State.Generating
            while (!_cancelGeneration) {
                val utf8token = generateNextToken()
                if (utf8token != null) {
                    if (utf8token.isNotEmpty()) emit(utf8token)
                } else {
                    val lastErr = try { nativeGetLastError().trim() } catch (e: Exception) { "" }
                    if (lastErr.isNotEmpty() && (lastErr.contains("failed") || lastErr.contains("error") || lastErr.contains("Exception"))) {
                        Log.e(TAG, "Generation interrupted with native error: $lastErr")
                        throw IOException("生成过程异常中断: $lastErr")
                    }
                    break
                }
            }
            if (_cancelGeneration) {
                Log.i(TAG, "Generation stopped by user cancellation.")
            } else {
                Log.i(TAG, "Generation finished.")
            }
            _state.value = InferenceEngine.State.ModelReady
        } catch (e: CancellationException) {
            Log.i(TAG, "Generation flow collection cancelled.")
            _state.value = InferenceEngine.State.ModelReady
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation", e)
            _state.value = InferenceEngine.State.ModelReady
            throw e
        }
    }.flowOn(llamaDispatcher)

    private fun performCpuFallback(): Boolean {
        val path = lastModelPath ?: return false
        val cfg = lastModelConfig ?: ModelConfig()
        return try {
            Log.i(TAG, "performCpuFallback: unloading current NPU context...")
            unload()
            Log.i(TAG, "performCpuFallback: reloading on ARM Cortex CPU (0 layers)...")
            val loadRes = loadExt(path, 0, "CPU")
            if (loadRes != 0) {
                Log.e(TAG, "performCpuFallback: loadExt failed with code $loadRes")
                return false
            }
            val prepRes = prepareAdvanced(
                cfg.nCtx,
                cfg.nThreads,
                256,
                cfg.batchSize,
                cfg.temp,
                cfg.topP,
                cfg.topK,
                cfg.repeatPenalty,
                1, // KV cache type: FP16 for 100% CPU stability
                0, // Flash attention mode: DISABLED for 100% CPU stability
                cfg.coreAffinity,
                false,
                3200
            )
            if (prepRes != 0) {
                Log.e(TAG, "performCpuFallback: prepareAdvanced failed with code $prepRes")
                return false
            }
            lastSystemPrompt?.let { sysPrompt ->
                if (sysPrompt.isNotBlank()) {
                    processSystemPrompt(sysPrompt)
                }
            }
            currentAccelerator = HardwareAccelerator.CPU_ARM
            onHardwareFallback?.invoke(HardwareAccelerator.CPU_ARM)
            Log.i(TAG, "performCpuFallback: CPU fallback successfully configured!")
            true
        } catch (e: Exception) {
            Log.e(TAG, "performCpuFallback failed", e)
            false
        }
    }

    override fun cancelGeneration() {
        _cancelGeneration = true
        try {
            nativeCancelGeneration()
        } catch (e: Exception) {
            Log.w(TAG, "Error triggering native cancelGeneration", e)
        }
    }

    override fun resetConversation() {
        try {
            nativeResetConversation()
        } catch (e: Exception) {
            Log.w(TAG, "Error triggering nativeResetConversation", e)
        }
    }

    override fun getInferenceStats(): InferenceStats {
        return try {
            val jsonStr = nativeGetInferenceStats()
            val obj = JSONObject(jsonStr)
            InferenceStats(
                promptTokens = obj.optInt("prompt_tokens", 0),
                promptTimeMs = obj.optDouble("prompt_time_ms", 0.0),
                promptSpeedTps = obj.optDouble("prompt_speed_tps", 0.0),
                decodeTokens = obj.optInt("decode_tokens", 0),
                decodeTimeMs = obj.optDouble("decode_time_ms", 0.0),
                decodeSpeedTps = obj.optDouble("decode_speed_tps", 0.0),
                offloadedLayers = obj.optInt("offloaded_layers", 0),
                totalLayers = obj.optInt("total_layers", 0),
                activeBackend = obj.optString("active_backend", "Qualcomm Hexagon NPU (HTP v75)"),
                npuTopology = obj.optString("npu_topology", "HTP0"),
                modelLoaded = obj.optBoolean("model_loaded", _state.value is InferenceEngine.State.ModelReady),
                isGenerating = obj.optBoolean("is_generating", !_cancelGeneration)
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse inference stats JSON", e)
            InferenceStats()
        }
    }

    override fun getHardwareReport(): String {
        return try {
            systemInfo()
        } catch (e: Exception) {
            "移动端通用异构硬件加速 (NPU / GPU / CPU)"
        }
    }

    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
        withContext(llamaDispatcher) {
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Benchmark discarded due to state: $state"
            }
            Log.i(TAG, "Running hardware accelerator benchmark (pp: $pp, tg: $tg, pl: $pl, nr: $nr)...")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.Benchmarking
            benchModel(pp, tg, pl, nr).also {
                _state.value = InferenceEngine.State.ModelReady
            }
        }

    override fun cleanUp() {
        _cancelGeneration = true
        runBlocking(llamaDispatcher) {
            when (val state = _state.value) {
                is InferenceEngine.State.ModelReady -> {
                    Log.i(TAG, "Unloading model and releasing NPU resources...")
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.UnloadingModel
                    try { unload() } catch (e: Exception) { Log.w(TAG, "Error in unload()", e) }
                    _state.value = InferenceEngine.State.Initialized
                    Log.i(TAG, "Model unloaded!")
                }
                is InferenceEngine.State.Error -> {
                    Log.i(TAG, "Resetting error state...")
                    try { unload() } catch (e: Exception) {}
                    _state.value = InferenceEngine.State.Initialized
                }
                is InferenceEngine.State.Initialized -> {
                    Log.i(TAG, "Already initialized, nothing to clean up.")
                }
                else -> {
                    try { unload() } catch (e: Exception) {}
                    _state.value = InferenceEngine.State.Initialized
                }
            }
        }
    }

    override fun destroy() {
        _cancelGeneration = true
        runBlocking(llamaDispatcher) {
            _readyForSystemPrompt = false
            when (_state.value) {
                is InferenceEngine.State.Uninitialized -> {}
                is InferenceEngine.State.Initialized -> shutdown()
                else -> { unload(); shutdown() }
            }
        }
        llamaScope.cancel()
    }
}
