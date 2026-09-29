package com.arm.aichat

import com.arm.aichat.InferenceEngine.State
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/**
 * Universal hardware accelerator type for diverse mobile SoC architectures (Snapdragon, Dimensity, Tensor, Exynos, etc.)
 */
enum class HardwareAccelerator(val label: String, val defaultGpuLayers: Int) {
    NPU_HEXAGON("端侧专用 NPU 硬件加速 (高通 Hexagon / 联发科 APU)", 99),
    GPU_GENERIC("GPU 通用图形算力加速 (Adreno / Mali / Immortalis / Xclipse)", 99),
    HYBRID_COMBINED("多硬件异构合并协同调用 (NPU/GPU + CPU 算力并发)", 22),
    CPU_ARM("ARM Cortex CPU (KleidiAI / NEON 全兼容)", 0)
}

/**
 * Mobile NPU Topology & multi-session configuration:
 * - HTP0: Single NPU execution session
 * - HTP0:0,HTP0:1: Dual/multi-session parallel execution for high throughput
 * - 2: Layer-split across virtual sessions
 */
enum class NpuTopology(val configString: String, val displayName: String, val description: String) {
    SINGLE_CORE("HTP0", "标准单会话模式 (HTP0 默认)", "物理 NPU 标准单会话执行，兼容性最佳"),
    DUAL_CORE_GROUPED("HTP0:0,HTP0:1", "多会话并行加速 (HTP0:0,1)", "双虚拟会话并行加速，扩展寻址显存窗口"),
    LAYER_SPLIT("2", "多层拆分模式 (2 Sessions)", "多会话分层卸载，优化大模型内存")
}

enum class KvCacheType(val displayName: String, val typeCode: Int) {
    Q8_0("Q8_0 (推荐: 显存带宽减半，显著加速生成)", 8),
    F16("FP16 (原生精度: 完整 16-bit 浮点缓存)", 1),
    Q4_0("Q4_0 (轻量化: 显存仅占 1/4，优化显存占用)", 2)
}

enum class FlashAttnMode(val displayName: String, val modeCode: Int) {
    AUTO("自动优化 (AUTO - 动态分块注意力机制)", -1),
    ENABLED("强制开启 (ENABLED - 适用于长文本及硬件加速)", 1),
    DISABLED("禁用 (仅特殊 CPU 调试模式使用)", 0)
}

/**
 * Configuration options for GGUF model loading and inference across mobile hardware architectures
 */
data class ModelConfig(
    val nGpuLayers: Int = 99, // Full hardware accelerator offload (all layers)
    val npuTopology: NpuTopology = NpuTopology.SINGLE_CORE,
    val nThreads: Int = 4, // Balanced CPU thread count to prevent thermal throttling
    val nCtx: Int = 4096,
    val ubatchSize: Int = 256, // Safe ubatch size for large context
    val batchSize: Int = 512,
    val temp: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val repeatPenalty: Float = 1.1f,
    val accelerator: HardwareAccelerator = HardwareAccelerator.NPU_HEXAGON,
    val mmprojPath: String? = null,
    // Advanced Mode parameters
    val isAdvancedMode: Boolean = false,
    val kvCacheType: KvCacheType = KvCacheType.Q8_0,
    val flashAttnMode: FlashAttnMode = FlashAttnMode.AUTO,
    val coreAffinity: Boolean = true,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val dma64: Boolean = true,
    val vmemMb: Int = 3200
)

/**
 * Real-time inference performance and hardware metrics
 */
data class InferenceStats(
    val promptTokens: Int = 0,
    val promptTimeMs: Double = 0.0,
    val promptSpeedTps: Double = 0.0,
    val decodeTokens: Int = 0,
    val decodeTimeMs: Double = 0.0,
    val decodeSpeedTps: Double = 0.0,
    val offloadedLayers: Int = 0,
    val totalLayers: Int = 0,
    val activeBackend: String = "端侧硬件加速",
    val npuTopology: String = "HTP0",
    val modelLoaded: Boolean = false,
    val isGenerating: Boolean = false
)

/**
 * Interface defining the core LLM inference operations with universal mobile hardware support.
 */
interface InferenceEngine {
    /**
     * Current state of the inference engine
     */
    val state: StateFlow<State>

    /**
     * Callback invoked when the engine performs automatic hardware fallback (e.g. NPU -> CPU).
     */
    var onHardwareFallback: ((HardwareAccelerator) -> Unit)?

    /**
     * Load a model with default NPU offload configuration.
     */
    suspend fun loadModel(pathToModel: String)

    /**
     * Load a model with customizable Qualcomm Snapdragon NPU configurations.
     */
    suspend fun loadModel(pathToModel: String, config: ModelConfig)

    /**
     * Loads or unloads a multimodal projector model (mmproj) for vision inference.
     */
    suspend fun loadMmproj(pathToMmproj: String?): Boolean

    /**
     * Sends a system prompt to the loaded model.
     */
    suspend fun setSystemPrompt(systemPrompt: String)

    /**
     * Sends a user prompt (optionally with an attached image) to the loaded model and returns a Flow of generated tokens.
     */
    fun sendUserPrompt(
        message: String,
        imagePath: String? = null,
        predictLength: Int = DEFAULT_PREDICT_LENGTH
    ): Flow<String>

    /**
     * Cancels active token generation in progress.
     */
    fun cancelGeneration()

    /**
     * Resets conversation session history and cleans KV cache in native memory.
     */
    fun resetConversation()

    /**
     * Query real-time inference statistics and NPU offload metrics.
     */
    fun getInferenceStats(): InferenceStats

    /**
     * Query hardware and platform diagnosis report.
     */
    fun getHardwareReport(): String

    /**
     * Runs a benchmark with the specified parameters.
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String

    /**
     * Unloads the currently loaded model.
     */
    fun cleanUp()

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    fun destroy()

    /**
     * States of the inference engine
     */
    sealed class State {
        object Uninitialized : State()
        object Initializing : State()
        object Initialized : State()

        object LoadingModel : State()
        object UnloadingModel : State()
        object ModelReady : State()

        object Benchmarking : State()
        object ProcessingSystemPrompt : State()
        object ProcessingUserPrompt : State()

        object Generating : State()

        data class Error(val exception: Exception) : State()
    }

    companion object {
        const val DEFAULT_PREDICT_LENGTH = 1024
    }
}

val State.isUninterruptible
    get() = this is State.Initializing ||
        this is State.LoadingModel ||
        this is State.UnloadingModel ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt

val State.isModelLoaded: Boolean
    get() = this is State.ModelReady ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt ||
        this is State.Generating

class UnsupportedArchitectureException : Exception()
