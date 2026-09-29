// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 The llama.cpp Authors

#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <atomic>
#include <sstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <dlfcn.h>
#include <sched.h>
#include <pthread.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static mtmd_context * g_mtmd_ctx = nullptr;

/**
 * Dynamic Android CPU Core Pinning:
 * Bind inference worker threads strictly to mid & high performance cores to avoid straggler delays.
 */
static bool g_core_affinity_enabled = true;

static void pin_threads_to_performance_cores() {
    if (!g_core_affinity_enabled) {
        return;
    }
    long num_cores = sysconf(_SC_NPROCESSORS_CONF);
    if (num_cores <= 0) num_cores = 8;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (num_cores >= 8) {
        // Pin to big & prime performance cores (e.g. cores 2 to num_cores - 1)
        for (int i = 2; i < num_cores; ++i) {
            CPU_SET(i, &cpuset);
        }
    } else if (num_cores >= 4) {
        for (int i = 1; i < num_cores; ++i) {
            CPU_SET(i, &cpuset);
        }
    } else {
        for (int i = 0; i < num_cores; ++i) {
            CPU_SET(i, &cpuset);
        }
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

/**
 * Universal Mobile AI Inference Defaults
 */
constexpr int   N_THREADS_DEFAULT       = 4;
constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 128;
constexpr int   UBATCH_SIZE_HTP         = 256;
constexpr float DEFAULT_SAMPLER_TEMP    = 0.7f;
constexpr float DEFAULT_SAMPLER_TOP_P   = 0.9f;

static llama_model                      * g_model = nullptr;
static llama_context                    * g_context = nullptr;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler = nullptr;

struct InferenceMetrics {
    int prompt_tokens = 0;
    double prompt_time_ms = 0.0;
    double prompt_speed_tps = 0.0;
    int decode_tokens = 0;
    double decode_time_ms = 0.0;
    double decode_speed_tps = 0.0;
    int offloaded_layers = 0;
    int total_layers = 0;
    int spec_draft_tokens = 0;
    int spec_accepted_tokens = 0;
    std::string active_backend = "Mobile Compute Platform";
    std::string npu_topology = "HTP0";
    std::string adsp_path = "";
};

static InferenceMetrics g_metrics;
static std::atomic<bool> g_cancel_generation{false};
static int64_t g_t_prompt_start = 0;
static int64_t g_t_prompt_end = 0;
static int64_t g_t_decode_start = 0;
static int64_t g_t_decode_end = 0;
static int g_current_n_ctx = DEFAULT_CONTEXT_SIZE;

/**
 * Speculative Prompt Lookup Decoding (PLD) Engine:
 * Employs zero-cost context n-gram matching to draft speculative candidate tokens
 * verified in parallel by the NPU/CPU in a single forward pass without secondary model memory overhead.
 */
static std::vector<llama_token> g_session_tokens;
static std::deque<llama_token>  g_spec_accepted_tokens;
static llama_token              g_spec_last_token = LLAMA_TOKEN_NULL;
static bool                     g_speculative_pld_enabled = false;

static std::vector<llama_token> find_prompt_lookup_draft_tokens(
        const std::vector<llama_token> & session_tokens,
        const llama_vocab * vocab,
        int n_draft = 3,
        int ngram_size = 3) {
    std::vector<llama_token> drafts;
    const int n = (int) session_tokens.size();
    if (n < ngram_size + 1) {
        return drafts;
    }

    // Bound search window to the last 4096 tokens to guarantee <10us search time
    const int max_search_depth = 4096;
    const int min_i = std::max(0, n - max_search_depth);

    // The query ngram is the last ngram_size tokens in session history
    const llama_token * query = &session_tokens[n - ngram_size];

    int match_idx = -1;
    int matched_ngram = ngram_size;

    // 1. Primary search: exact 3-gram match backwards (finds most recent contextual occurrence)
    for (int i = n - ngram_size - 1; i >= min_i; --i) {
        bool match = true;
        for (int k = 0; k < ngram_size; ++k) {
            if (session_tokens[i + k] != query[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            match_idx = i;
            break;
        }
    }

    // 2. Secondary fallback: 2-gram match if 3-gram was not found
    if (match_idx < 0 && ngram_size > 2 && n >= 3) {
        const int small_ngram = 2;
        const llama_token * query2 = &session_tokens[n - small_ngram];
        for (int i = n - small_ngram - 1; i >= min_i; --i) {
            bool match = true;
            for (int k = 0; k < small_ngram; ++k) {
                if (session_tokens[i + k] != query2[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                match_idx = i;
                matched_ngram = small_ngram;
                break;
            }
        }
    }

    if (match_idx >= 0) {
        int start_pos = match_idx + matched_ngram;
        for (int i = 0; i < n_draft && (start_pos + i) < n; ++i) {
            llama_token candidate = session_tokens[start_pos + i];
            if (vocab && llama_vocab_is_eog(vocab, candidate)) {
                break;
            }
            drafts.push_back(candidate);
        }
    }

    return drafts;
}

static std::string get_backend_devices() {
    std::vector<std::string> dev_names;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string reg_name = ggml_backend_reg_name(reg);
        size_t n_devs = ggml_backend_reg_dev_count(reg);
        for (size_t j = 0; j < n_devs; j++) {
            auto *dev = ggml_backend_reg_dev_get(reg, j);
            std::string dname = ggml_backend_dev_name(dev);
            std::string ddesc = ggml_backend_dev_description(dev);
            dev_names.push_back(dname + " (" + ddesc + ")");
        }
    }
    return dev_names.empty() ? "CPU (ARMv8/v9)" : join(dev_names, ", ");
}

static std::string detect_active_hardware() {
    std::string accel_name = "";
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string reg_name = ggml_backend_reg_name(reg);
        if (reg_name.find("HEXAGON") != std::string::npos || reg_name.find("HTP") != std::string::npos) {
            size_t n_devs = ggml_backend_reg_dev_count(reg);
            if (n_devs > 0) {
                auto *dev = ggml_backend_reg_dev_get(reg, 0);
                std::string ddesc = ggml_backend_dev_description(dev);
                accel_name = "Hexagon NPU (" + ddesc + ")";
            } else {
                accel_name = "Hexagon NPU (HTP)";
            }
            break;
        } else if (reg_name.find("VULKAN") != std::string::npos || reg_name.find("OPENCL") != std::string::npos || reg_name.find("GPU") != std::string::npos) {
            accel_name = "Mobile GPU (" + reg_name + ")";
        }
    }

    if (!accel_name.empty() && g_metrics.offloaded_layers > 0) {
        if (g_metrics.total_layers > 0 && g_metrics.offloaded_layers >= g_metrics.total_layers) {
            return accel_name + " [全层加速]";
        }
        return "异构合并调用: " + accel_name + " (" + std::to_string(g_metrics.offloaded_layers) + "层) + CPU (" + 
               std::to_string(std::max(0, g_metrics.total_layers - g_metrics.offloaded_layers)) + "层)";
    }
    return "ARM Cortex CPU (KleidiAI / NEON 全兼容)";
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/);

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    llama_log_set(aichat_android_log_callback, nullptr);

    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Initializing Qualcomm Snapdragon NPU environment: libDir=%s", path_to_backend);
    
    // FastRPC dynamic loader requires ADSP_LIBRARY_PATH to locate libggml-htp-v75.so and DSP modules
    std::string adsp_paths = std::string(path_to_backend) + ";/dsp;/vendor/dsp;/vendor/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp";
    setenv("ADSP_LIBRARY_PATH", adsp_paths.c_str(), 1);
    // Enable 64-bit DMA support for Hexagon HTP v75
    setenv("GGML_HEXAGON_DMA64", "1", 0);
    g_metrics.adsp_path = adsp_paths;

    // Explicitly verify and register libggml-hexagon.so
    std::string hex_lib_path = std::string(path_to_backend) + "/libggml-hexagon.so";
    LOGi("Attempting explicit load of Hexagon NPU backend: %s", hex_lib_path.c_str());
    void * hex_handle = dlopen(hex_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!hex_handle) {
        LOGe("Explicit dlopen(%s) failed: %s", hex_lib_path.c_str(), dlerror());
    } else {
        LOGi("Explicit dlopen(%s) succeeded! Resolving ggml_backend_init...", hex_lib_path.c_str());
        typedef ggml_backend_reg_t (*backend_init_fn_t)(void);
        auto init_fn = (backend_init_fn_t) dlsym(hex_handle, "ggml_backend_init");
        if (init_fn) {
            ggml_backend_reg_t reg = init_fn();
            if (reg) {
                LOGi("Hexagon backend init succeeded, registering backend '%s' with %zu devices",
                     ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
                ggml_backend_register(reg);
            } else {
                LOGe("ggml_backend_init() returned NULL! FastRPC or HTP driver init failed on this device");
            }
        } else {
            LOGe("dlsym(ggml_backend_init) failed: %s", dlerror());
        }
    }

    // Hardware runtime acceleration settings for Qualcomm Snapdragon NPU (HTP/cDSP)
    // 1. FA_SELECT=2: Enable HMX hardware tensor cores for FlashAttention (2: HMX -> HVX -> CPU; 1 disables HMX)
    setenv("GGML_HEXAGON_FA_SELECT", "2", 0);
    // 2. MM_SELECT=2: Enable HMX matrix tensor cores for GEMM
    setenv("GGML_HEXAGON_MM_SELECT", "2", 0);
    // 3. OPBATCH=1280: Maximize batch size to saturate HTP execution pipelines and avoid premature flushes
    setenv("GGML_HEXAGON_OPBATCH", "1280", 0);
    // 4. OPFUSION=1: Enable graph operator fusions (RMS_NORM+MUL, MUL_MAT+ADD, etc.)
    setenv("GGML_HEXAGON_OPFUSION", "1", 0);
    // 5. HOSTBUF=0: Allocate host buffers via standard virtual memory rather than depleting limited physical CMA pool
    setenv("GGML_HEXAGON_HOSTBUF", "0", 0);
    // 6. OPPOLL=0: Disable userspace ring-buffer busy-spin polling to prevent 100% CPU lock and thermal throttling
    setenv("GGML_HEXAGON_OPPOLL", "0", 0);
    // 7. GDN_SELECT=2: HMX tensor units for gated linear activations
    setenv("GGML_HEXAGON_GDN_SELECT", "2", 0);
    // 8. AR_SELECT=2: Fused ALLREDUCE+ADD via DMA
    setenv("GGML_HEXAGON_AR_SELECT", "2", 0);
    // 9. VMEM: Set default virtual memory ceiling (3200MB) for large context models (Qwen 4B / 4096 ctx)
    setenv("GGML_HEXAGON_VMEM", "3200", 0);
    // 10. DMA64: Fast 64-bit DMA descriptors
    setenv("GGML_HEXAGON_DMA64", "1", 0);

    LOGi("Loading dynamic GGML backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    llama_backend_init();
    LOGi("Mobile LLM Backend initialized! Registered devices: %s", get_backend_devices().c_str());
}

static llama_model * load_model_helper(const char * model_path, struct llama_model_params & model_params) {
    llama_model * model = nullptr;
    if (strncmp(model_path, "/proc/self/fd/", 14) == 0) {
        int base_fd = atoi(model_path + 14);
        LOGi("%s: Loading model via procfs file descriptor: %d", __func__, base_fd);
        int fd = dup(base_fd);
        if (fd < 0) {
            LOGe("%s: dup(%d) failed: %s", __func__, base_fd, strerror(errno));
            return nullptr;
        }
        lseek(fd, 0, SEEK_SET);
        FILE * fp = fdopen(fd, "rb");
        if (!fp) {
            LOGe("%s: fdopen(%d) failed: %s", __func__, fd, strerror(errno));
            close(fd);
            return nullptr;
        }
        model = llama_model_load_from_file_ptr(fp, model_params);
        if (!model && model_params.load_mode != LLAMA_LOAD_MODE_NONE) {
            LOGw("%s: Mmap load failed for fd %d, retrying without mmap...", __func__, base_fd);
            int fd2 = dup(base_fd);
            if (fd2 >= 0) {
                lseek(fd2, 0, SEEK_SET);
                FILE * fp2 = fdopen(fd2, "rb");
                if (fp2) {
                    auto params_no_mmap = model_params;
                    params_no_mmap.load_mode = LLAMA_LOAD_MODE_NONE;
                    model = llama_model_load_from_file_ptr(fp2, params_no_mmap);
                }
            }
        }
        return model;
    } else {
        model = llama_model_load_from_file(model_path, model_params);
        if (!model && model_params.load_mode != LLAMA_LOAD_MODE_NONE) {
            LOGw("%s: Mmap load failed for %s, retrying without mmap...", __func__, model_path);
            auto params_no_mmap = model_params;
            params_no_mmap.load_mode = LLAMA_LOAD_MODE_NONE;
            model = llama_model_load_from_file(model_path, params_no_mmap);
        }
        return model;
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadAdvanced(
        JNIEnv *env,
        jobject thiz,
        jstring jmodel_path,
        jint n_gpu_layers,
        jstring jnpu_topology,
        jboolean use_mmap,
        jboolean use_mlock) {
    // Automatically unload previous model if present
    if (g_model != nullptr || g_context != nullptr) {
        LOGi("Unloading previous model before loading new one in native loadAdvanced...");
        Java_com_arm_aichat_internal_InferenceEngineImpl_unload(env, thiz);
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    if (use_mmap && use_mlock) {
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    } else if (use_mmap) {
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    } else if (use_mlock) {
        model_params.load_mode = LLAMA_LOAD_MODE_MLOCK;
    } else {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    std::string topo_str = "HTP0";
    if (jnpu_topology != nullptr) {
        const auto *npu_topology = env->GetStringUTFChars(jnpu_topology, 0);
        if (strlen(npu_topology) > 0) {
            topo_str = npu_topology;
        }
        env->ReleaseStringUTFChars(jnpu_topology, npu_topology);
    }

    // Remap legacy multi-physical core syntax to virtual multi-session
    if (topo_str == "HTP0[0-1]") {
        LOGw("Remapping HTP0[0-1] to HTP0:0,HTP0:1");
        topo_str = "HTP0:0,HTP0:1";
    }

    LOGi("Configuring accelerator topology: GGML_HEXAGON_DEVICES=%s (use_mmap=%d, use_mlock=%d)",
         topo_str.c_str(), (int)use_mmap, (int)use_mlock);
    setenv("GGML_HEXAGON_DEVICES", topo_str.c_str(), 1);
    g_metrics.npu_topology = topo_str;

    const auto *model_path = env->GetStringUTFChars(jmodel_path, 0);
    LOGd("%s: Loading GGUF model: %s (NPU layers=%d, topology=%s)",
         __func__, model_path, n_gpu_layers, topo_str.c_str());

    auto *model = load_model_helper(model_path, model_params);

    // Fallback 1: If multi-core topology (e.g. HTP0:0,HTP0:1) failed on commercial Android cdsp node
    if (!model && topo_str != "HTP0" && n_gpu_layers > 0) {
        LOGw("%s: Topology '%s' failed to load. Retrying with HTP0 single core...", __func__, topo_str.c_str());
        topo_str = "HTP0";
        setenv("GGML_HEXAGON_DEVICES", "HTP0", 1);
        g_metrics.npu_topology = "HTP0 (Auto-Fallback)";
        model = load_model_helper(model_path, model_params);
    }

    // Fallback 2: If Hexagon NPU offload failed completely, try CPU
    if (!model && n_gpu_layers > 0) {
        LOGw("%s: Hexagon NPU offload failed. Retrying with CPU...", __func__);
        model_params.n_gpu_layers = 0;
        model = load_model_helper(model_path, model_params);
        if (model) {
            g_metrics.npu_topology = "CPU";
            n_gpu_layers = 0;
        }
    }

    env->ReleaseStringUTFChars(jmodel_path, model_path);
    if (!model) {
        LOGe("Failed to load GGUF model from %s even after fallbacks", model_path);
        return 1;
    }
    g_model = model;

    // Record model layer metadata
    g_metrics.total_layers = llama_model_n_layer(g_model);
    g_metrics.offloaded_layers = (n_gpu_layers >= g_metrics.total_layers) ? g_metrics.total_layers : std::max(0, (int)n_gpu_layers);
    g_metrics.active_backend = detect_active_hardware();

    LOGi("GGUF Model loaded: Total layers=%d, Offloaded to NPU=%d, Active Backend=%s",
         g_metrics.total_layers, g_metrics.offloaded_layers, g_metrics.active_backend.c_str());
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadExt(
        JNIEnv *env,
        jobject thiz,
        jstring jmodel_path,
        jint n_gpu_layers,
        jstring jnpu_topology) {
    return Java_com_arm_aichat_internal_InferenceEngineImpl_loadAdvanced(
            env, thiz, jmodel_path, n_gpu_layers, jnpu_topology, JNI_TRUE, JNI_FALSE);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_load(JNIEnv *env, jobject thiz, jstring jmodel_path) {
    // Default to hardware offload (all layers, HTP0)
    jstring default_topology = env->NewStringUTF("HTP0");
    jint res = Java_com_arm_aichat_internal_InferenceEngineImpl_loadExt(env, thiz, jmodel_path, 99, default_topology);
    env->DeleteLocalRef(default_topology);
    return res;
}

static llama_context *init_context_ext(
        llama_model *model,
        const int n_ctx = DEFAULT_CONTEXT_SIZE,
        const int n_threads_custom = N_THREADS_DEFAULT,
        const int ubatch_size_custom = 256,
        const int batch_size_custom = 512,
        const int kv_cache_type = 8,
        const int flash_attn_mode = 0) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    const int n_threads = n_threads_custom > 0 ? n_threads_custom : N_THREADS_DEFAULT;
    int eff_ubatch = (ubatch_size_custom > 0) ? ubatch_size_custom : 256;
    if (g_metrics.offloaded_layers > 0 && n_ctx >= 2048 && eff_ubatch > 256) {
        LOGi("%s: Hexagon NPU offload with n_ctx=%d: clamping ubatch from %d to 256 to avoid rpcmem overflow",
             __func__, n_ctx, eff_ubatch);
        eff_ubatch = 256;
    }
    int eff_batch = std::max(batch_size_custom > 0 ? batch_size_custom : 512, eff_ubatch);

    LOGi("%s: Initializing context: threads=%d, n_ctx=%d, n_batch=%d, n_ubatch=%d, kv_type=%d, fa_mode=%d",
         __func__, n_threads, n_ctx, eff_batch, eff_ubatch, kv_cache_type, flash_attn_mode);

    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    int eff_n_ctx = n_ctx;
    if (eff_n_ctx > trained_context_size && trained_context_size > 0) {
        LOGw("%s: Model was trained with %d context size! Enforcing %d context size...",
             __func__, trained_context_size, eff_n_ctx);
        eff_n_ctx = trained_context_size;
    }
    ctx_params.n_ctx = eff_n_ctx;
    ctx_params.n_batch = eff_batch;
    ctx_params.n_ubatch = eff_ubatch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    // Flash Attention: -1 = AUTO, 0 = DISABLED, 1 = ENABLED
    if (flash_attn_mode == 0) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    } else if (flash_attn_mode == 1) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        // Mode is AUTO (-1):
        // On Hexagon NPU offload, FlashAttention is MANDATORY for long contexts (e.g. 4096) to prevent
        // HTP_STATUS_VTCM_TOO_SMALL (5) and non-aligned softmax operator rejection during decode.
        if (g_metrics.offloaded_layers > 0) {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
    }

    // KV Cache Quantization: 8 -> Q8_0, 2 -> Q4_0, 1 -> F16
    ggml_type k_type = (kv_cache_type == 2) ? GGML_TYPE_Q4_0 : ((kv_cache_type == 1) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0);
    // CRITICAL: Hexagon FlashAttention (flash-attn-ops.c) strictly requires F16 or Q8_0 KV cache.
    // If Q4_0 is requested while offloading to Hexagon NPU, promote to Q8_0 so FlashAttention works seamlessly
    // without triggering fallback to non-flash Softmax.
    if (g_metrics.offloaded_layers > 0 && k_type == GGML_TYPE_Q4_0) {
        LOGw("%s: Q4_0 KV cache is unsupported for Hexagon FlashAttention, promoting to Q8_0 for NPU offload", __func__);
        k_type = GGML_TYPE_Q8_0;
    }

    if (g_metrics.offloaded_layers > 0) {
        // Hexagon NPU offload: Hexagon HTP native flash-attn-ops.c supports Q8_0 for BOTH K and V cache.
        // Using Q8_0 for both cuts KV memory footprint and bandwidth in half, boosting decode TPS!
        ctx_params.type_k = k_type;
        ctx_params.type_v = (kv_cache_type == 1) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else {
        // Pure CPU mode: CPU fattn requires F16 V-cache to avoid assert in ggml-cpu/ops.cpp
        ctx_params.type_k = (kv_cache_type == 1) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        ctx_params.type_v = GGML_TYPE_F16;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    auto *context = llama_init_from_model(model, ctx_params);
    if (!context && ctx_params.type_v != GGML_TYPE_F16) {
        LOGw("%s: Selected Q8_0 V-cache init failed, falling back to F16 V-cache...", __func__);
        ctx_params.type_v = GGML_TYPE_F16;
        context = llama_init_from_model(model, ctx_params);
    }
    if (!context && k_type != GGML_TYPE_F16) {
        LOGw("%s: Selected KV cache type %d failed, falling back to default F16 KV cache...", __func__, kv_cache_type);
        ctx_params.type_k = GGML_TYPE_F16;
        ctx_params.type_v = GGML_TYPE_F16;
        context = llama_init_from_model(model, ctx_params);
    }
    // Fallback: If flash attention or initial eff_n_ctx failed
    if (!context && ctx_params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LOGw("%s: Initial llama_init_from_model() failed, retrying with flash_attn=DISABLED...", __func__);
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context = llama_init_from_model(model, ctx_params);
    }
    if (context == nullptr && eff_n_ctx > 2048) {
        LOGw("%s: llama_init_from_model() failed with n_ctx=%d, retrying with n_ctx=2048...", __func__, eff_n_ctx);
        ctx_params.n_ctx = 2048;
        context = llama_init_from_model(model, ctx_params);
    }
    if (context == nullptr && ctx_params.n_ctx > 1024) {
        LOGw("%s: llama_init_from_model() failed with n_ctx=2048, retrying with n_ctx=1024...", __func__);
        ctx_params.n_ctx = 1024;
        context = llama_init_from_model(model, ctx_params);
    }
    if (context == nullptr && ctx_params.n_ctx > 512) {
        LOGw("%s: llama_init_from_model() failed with n_ctx=1024, retrying with n_ctx=512...", __func__);
        ctx_params.n_ctx = 512;
        context = llama_init_from_model(model, ctx_params);
    }
    if (context == nullptr) {
        LOGe("%s: llama_init_from_model() failed all context attempts", __func__);
    }
    g_current_n_ctx = ctx_params.n_ctx;
    return context;
}

static common_sampler *new_sampler_ext(float temp, float top_p, int top_k = 40, float repeat_penalty = 1.1f) {
    common_params_sampling sparams;
    sparams.temp = temp;
    sparams.top_p = top_p;
    sparams.top_k = top_k > 0 ? top_k : 40;
    sparams.penalty_repeat = repeat_penalty;
    return common_sampler_init(g_model, sparams);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepareAdvanced(
        JNIEnv *env,
        jobject /*unused*/,
        jint n_ctx,
        jint n_threads,
        jint ubatch_size,
        jint batch_size,
        jfloat temp,
        jfloat top_p,
        jint top_k,
        jfloat repeat_penalty,
        jint kv_cache_type,
        jint flash_attn_mode,
        jboolean core_affinity,
        jboolean dma64,
        jint vmem_mb) {
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_batch.token) {
        llama_batch_free(g_batch);
        g_batch = {};
    }
    if (g_sampler) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    g_core_affinity_enabled = core_affinity;
    setenv("GGML_HEXAGON_DMA64", dma64 ? "1" : "0", 1);
    setenv("GGML_HEXAGON_VMEM", std::to_string(vmem_mb).c_str(), 1);
    setenv("GGML_HEXAGON_FA_SELECT", "2", 1);
    setenv("GGML_HEXAGON_MM_SELECT", "2", 1);
    setenv("GGML_HEXAGON_OPBATCH", "1280", 1);
    setenv("GGML_HEXAGON_OPFUSION", "1", 1);
    setenv("GGML_HEXAGON_HOSTBUF", "0", 1);
    setenv("GGML_HEXAGON_OPPOLL", "0", 1);
    setenv("GGML_HEXAGON_GDN_SELECT", "2", 1);
    setenv("GGML_HEXAGON_AR_SELECT", "2", 1);

    auto *context = init_context_ext(g_model, n_ctx, n_threads, ubatch_size, batch_size, kv_cache_type, flash_attn_mode);
    if (!context) { return 1; }
    g_context = context;
    int eff_batch = std::max((int)batch_size, (int)ubatch_size);
    g_batch = llama_batch_init(eff_batch, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler_ext(temp, top_p, top_k, repeat_penalty);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepareExt(
        JNIEnv *env,
        jobject thiz,
        jint n_ctx,
        jint n_threads,
        jint ubatch_size,
        jfloat temp,
        jfloat top_p) {
    return Java_com_arm_aichat_internal_InferenceEngineImpl_prepareAdvanced(
            env, thiz, n_ctx, n_threads, ubatch_size, 512, temp, top_p, 40, 1.1f, 8, -1, JNI_TRUE, JNI_TRUE, 3200);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_prepare(JNIEnv *env, jobject thiz) {
    return Java_com_arm_aichat_internal_InferenceEngineImpl_prepareExt(
            env, thiz, DEFAULT_CONTEXT_SIZE, N_THREADS_DEFAULT, UBATCH_SIZE_HTP, DEFAULT_SAMPLER_TEMP, DEFAULT_SAMPLER_TOP_P);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_systemInfo(JNIEnv *env, jobject /*unused*/) {
    std::ostringstream ss;
    ss << "Heterogeneous Mobile SoC Compute Platform (ARM64)\n";
    ss << "Detected GGML Hardware Compute Devices:\n" << get_backend_devices() << "\n\n";
    ss << llama_print_system_info();
    return env->NewStringUTF(ss.str().c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeGetInferenceStats(JNIEnv *env, jobject /*unused*/) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    json << "{"
         << "\"prompt_tokens\":" << g_metrics.prompt_tokens << ","
         << "\"prompt_time_ms\":" << g_metrics.prompt_time_ms << ","
         << "\"prompt_speed_tps\":" << g_metrics.prompt_speed_tps << ","
         << "\"decode_tokens\":" << g_metrics.decode_tokens << ","
         << "\"decode_time_ms\":" << g_metrics.decode_time_ms << ","
         << "\"decode_speed_tps\":" << g_metrics.decode_speed_tps << ","
         << "\"offloaded_layers\":" << g_metrics.offloaded_layers << ","
         << "\"total_layers\":" << g_metrics.total_layers << ","
         << "\"spec_draft_tokens\":" << g_metrics.spec_draft_tokens << ","
         << "\"spec_accepted_tokens\":" << g_metrics.spec_accepted_tokens << ","
         << "\"active_backend\":\"" << g_metrics.active_backend << "\","
         << "\"npu_topology\":\"" << g_metrics.npu_topology << "\""
         << "}";
    return env->NewStringUTF(json.str().c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeCancelGeneration(JNIEnv * /*env*/, jobject /*unused*/) {
    LOGi("Cancelling text generation requested by user");
    g_cancel_generation.store(true);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_benchModel(JNIEnv *env, jobject /*unused*/, jint pp, jint tg,
                                                      jint pl, jint nr) {
    auto *context = init_context_ext(g_model, pp, N_THREADS_DEFAULT, UBATCH_SIZE_HTP);
    if (!context) {
        const auto *const err_msg = "Failed to initialize context for NPU benchmark";
        LOGe("%s", err_msg);
        return env->NewStringUTF(err_msg);
    }

    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Snapdragon NPU Benchmark: prompt processing (pp = %d)", pp);
        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 0, i, {0}, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, g_batch) != 0) {
            LOGe("llama_decode() failed during NPU prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // text generation bench
        LOGi("Snapdragon NPU Benchmark: text generation (tg = %d)", tg);
        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, {j}, true);
            }
            if (llama_decode(context, g_batch) != 0) {
                LOGe("llama_decode() failed during NPU text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;
        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;
        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;
    }

    llama_free(context);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(g_model, model_desc, sizeof(model_desc));

    const auto model_size = double(llama_model_size(g_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(g_model)) / 1e9;
    const auto backend = g_metrics.active_backend;

    std::stringstream result;
    result << std::fixed << std::setprecision(2);
    result << "⚡ Hardware Benchmark Results\n";
    result << "Model: " << model_desc << " (" << model_size << " GiB, " << model_n_params << "B params)\n";
    result << "Hardware Backend: " << backend << " (" << g_metrics.npu_topology << ")\n";
    result << "Prompt Processing (PP " << pp << "): " << pp_avg << " ± " << pp_std << " tokens/s\n";
    result << "Token Generation  (TG " << tg << "): " << tg_avg << " ± " << tg_std << " tokens/s\n";
    return env->NewStringUTF(result.str().c_str());
}

/**
 * Chat and Token Decoding Logic
 */
constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position = 0;
static llama_pos current_position = 0;
static std::string g_last_error_message = "";

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeGetLastError(
        JNIEnv *env,
        jobject /*unused*/
) {
    return env->NewStringUTF(g_last_error_message.c_str());
}

static llama_pos stop_generation_position = 0;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    system_prompt_position = 0;
    current_position = 0;
    g_session_tokens.clear();
    g_spec_accepted_tokens.clear();
    g_spec_last_token = LLAMA_TOKEN_NULL;
    cached_token_chars.clear();
    assistant_ss.str("");
    g_last_error_message.clear();

    if (clear_kv_cache && g_context) {
        llama_memory_clear(llama_get_memory(g_context), false);
    }
}

static void shift_context() {
    if (!g_context) return;
    const int n_discard = (current_position - system_prompt_position) / 2;
    if (n_discard <= 0) {
        LOGw("%s: n_discard is %d <= 0, skipping shift", __func__, n_discard);
        return;
    }
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    std::string formatted;
    bool formatted_ok = false;

    if (g_chat_templates) {
        // 1. Try Jinja template first (supports modern models: Qwen2.5, DeepSeek, Llama-3, etc.)
        try {
            formatted = common_chat_format_single(
                    g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ true);
            formatted_ok = !formatted.empty();
        } catch (const std::exception & e1) {
            LOGw("%s: Jinja format attempt failed: %s", __func__, e1.what());
        } catch (...) {
            LOGw("%s: Jinja format attempt unknown exception", __func__);
        }

        // 2. Try legacy format if Jinja failed
        if (!formatted_ok) {
            try {
                formatted = common_chat_format_single(
                        g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
                formatted_ok = !formatted.empty();
            } catch (const std::exception & e2) {
                LOGw("%s: Legacy format attempt failed: %s", __func__, e2.what());
            } catch (...) {
                LOGw("%s: Legacy format attempt unknown exception", __func__);
            }
        }
    }

    // 3. Robust universal ChatML fallback if templates are unsupported, invalid, or null
    if (!formatted_ok) {
        LOGi("%s: Applying ChatML template fallback for role '%s'", __func__, role.c_str());
        if (role == ROLE_SYSTEM) {
            formatted = "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == ROLE_USER) {
            formatted = "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
        } else {
            formatted = content + "<|im_end|>\n";
        }
    }

    chat_msgs.push_back(new_msg);
    LOGd("%s: Formatted %s message (%zu chars)",
         __func__, role.c_str(), formatted.size());
    return formatted;
}

static void finish_assistant_response() {
    if (!cached_token_chars.empty()) {
        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    }
    std::string response = assistant_ss.str();
    if (!response.empty()) {
        chat_add_and_format(ROLE_ASSISTANT, response);
    }
    g_spec_last_token = LLAMA_TOKEN_NULL;
    g_spec_accepted_tokens.clear();
    cached_token_chars.clear();
    assistant_ss.str("");
}

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
    g_spec_accepted_tokens.clear();
    g_spec_last_token = LLAMA_TOKEN_NULL;
    g_last_error_message.clear();
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);

        if (start_pos + i + cur_batch_size >= g_current_n_ctx - OVERFLOW_HEADROOM) {
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        int decode_result = 0;
        try {
            decode_result = llama_decode(context, batch);
        } catch (const std::exception & de) {
            LOGe("%s: Exception inside llama_decode: %s", __func__, de.what());
            g_last_error_message = de.what();
            decode_result = 1;
        } catch (...) {
            LOGe("%s: Unknown exception inside llama_decode", __func__);
            g_last_error_message = "Unknown native exception during decode";
            decode_result = 1;
        }

        // If batch failed and cur_batch_size > 16, try progressive micro-batch retry
        // This dramatically reduces temporary rpcmem activation tensors on Hexagon NPU
        if (decode_result != 0 && cur_batch_size > 16) {
            LOGw("%s: llama_decode failed on batch of %d tokens (code %d). Retrying chunk in micro-batches of 16...",
                 __func__, cur_batch_size, decode_result);
            common_batch_clear(batch);
            bool micro_success = true;
            for (int mi = 0; mi < cur_batch_size; mi += 16) {
                const int cur_micro_size = std::min(cur_batch_size - mi, 16);
                common_batch_clear(batch);
                for (int mj = 0; mj < cur_micro_size; mj++) {
                    const int token_idx = i + mi + mj;
                    const llama_token token_id = tokens[token_idx];
                    const llama_pos position = start_pos + token_idx;
                    const bool want_logit = compute_last_logit && (token_idx == (int) tokens.size() - 1);
                    common_batch_add(batch, token_id, position, {0}, want_logit);
                }
                int micro_res = 0;
                try {
                    micro_res = llama_decode(context, batch);
                } catch (const std::exception & de) {
                    LOGe("%s: Exception in micro-batch decode: %s", __func__, de.what());
                    micro_res = 1;
                }
                if (micro_res != 0) {
                    LOGe("%s: Micro-batch decode failed at token index %d with code %d",
                         __func__, i + mi, micro_res);
                    micro_success = false;
                    decode_result = micro_res;
                    break;
                }
            }
            if (micro_success) {
                LOGi("%s: Micro-batch retry succeeded! KV cache updated smoothly.", __func__);
                decode_result = 0;
                g_last_error_message.clear();
            }
        }

        if (decode_result) {
            LOGe("%s: llama_decode failed with error code %d", __func__, decode_result);
            if (g_last_error_message.empty()) {
                g_last_error_message = "llama_decode failed with code " + std::to_string(decode_result);
            }
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    try {
        g_last_error_message.clear();
        if (!g_context || !g_model) {
            LOGe("%s: Model or context is null", __func__);
            g_last_error_message = "Model or context is null";
            return 1;
        }

        reset_long_term_states();
        reset_short_term_states();

        const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
        std::string raw_system_prompt(system_prompt ? system_prompt : "");
        env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);
        LOGd("%s: System prompt received (%zu chars)", __func__, raw_system_prompt.size());

        std::string formatted_system_prompt = raw_system_prompt;
        const bool has_chat_template = g_chat_templates && common_chat_templates_was_explicit(g_chat_templates.get());
        if (has_chat_template) {
            formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, raw_system_prompt);
        }

        llama_tokens system_tokens;
        try {
            system_tokens = common_tokenize(g_context, formatted_system_prompt, has_chat_template, has_chat_template);
        } catch (const std::exception & te) {
            LOGw("%s: common_tokenize with template failed: %s, retrying with raw...", __func__, te.what());
            system_tokens = common_tokenize(g_context, raw_system_prompt, false, false);
        }

        const int max_batch_size = g_current_n_ctx - OVERFLOW_HEADROOM;
        if ((int) system_tokens.size() > max_batch_size) {
            LOGe("%s: System prompt too long! %d tokens, max: %d",
                 __func__, (int) system_tokens.size(), max_batch_size);
            g_last_error_message = "System prompt too long (" + std::to_string(system_tokens.size()) + " tokens, max " + std::to_string(max_batch_size) + ")";
            return 1;
        }

        if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position)) {
            LOGe("%s: llama_decode() failed for system prompt", __func__);
            return 2;
        }

        g_session_tokens.insert(g_session_tokens.end(), system_tokens.begin(), system_tokens.end());
        system_prompt_position = current_position = (int) system_tokens.size();
        return 0;
    } catch (const std::exception & e) {
        g_last_error_message = e.what();
        LOGe("%s: Exception caught during system prompt processing: %s", __func__, e.what());
        return 3;
    } catch (...) {
        g_last_error_message = "Unknown exception during system prompt processing";
        LOGe("%s: Unknown exception caught during system prompt processing", __func__);
        return 4;
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    try {
        g_last_error_message.clear();
        pin_threads_to_performance_cores();

        if (!g_context || !g_model) {
            LOGe("%s: Model or context is null", __func__);
            g_last_error_message = "Model or context is null";
            return 1;
        }

        reset_short_term_states();
        g_cancel_generation.store(false);

        const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
        std::string raw_prompt_str(user_prompt ? user_prompt : "");
        env->ReleaseStringUTFChars(juser_prompt, user_prompt);
        LOGd("%s: User prompt received (%zu chars)", __func__, raw_prompt_str.size());

        std::string formatted_user_prompt = raw_prompt_str;
        const bool has_chat_template = g_chat_templates && common_chat_templates_was_explicit(g_chat_templates.get());
        if (has_chat_template) {
            formatted_user_prompt = chat_add_and_format(ROLE_USER, raw_prompt_str);
        }

        // Tokenize with progressive fallback (parse_special=true to parse chat tokens, add_special=false because template manages prefix)
        llama_tokens user_tokens;
        try {
            user_tokens = common_tokenize(g_context, formatted_user_prompt, false, true);
        } catch (const std::exception & te1) {
            LOGw("%s: common_tokenize failed with (false, true): %s. Retrying with (false, false)...",
                 __func__, te1.what());
            try {
                user_tokens = common_tokenize(g_context, formatted_user_prompt, false, false);
            } catch (const std::exception & te2) {
                LOGw("%s: common_tokenize failed on formatted prompt (%s). Tokenizing raw user prompt...",
                     __func__, te2.what());
                user_tokens = common_tokenize(g_context, raw_prompt_str, false, false);
            }
        }

        // Sanitize tokens: ensure all tokens are within vocabulary range [0, n_vocab)
        const llama_vocab * vocab = llama_model_get_vocab(g_model);
        const int n_vocab = llama_vocab_n_tokens(vocab);

        llama_tokens clean_tokens;
        clean_tokens.reserve(user_tokens.size());
        for (llama_token tok : user_tokens) {
            if (tok >= 0 && tok < n_vocab) {
                clean_tokens.push_back(tok);
            } else {
                LOGw("%s: Discarding invalid token ID %d (n_vocab=%d)", __func__, tok, n_vocab);
            }
        }
        user_tokens = std::move(clean_tokens);

        if (user_tokens.empty()) {
            LOGw("%s: user_tokens is empty after tokenization, adding fallback token", __func__);
            llama_token fallback_tok = llama_vocab_bos(vocab);
            if (fallback_tok < 0 || fallback_tok >= n_vocab) {
                fallback_tok = llama_vocab_eos(vocab);
            }
            if (fallback_tok < 0 || fallback_tok >= n_vocab) {
                fallback_tok = 0;
            }
            user_tokens.push_back(fallback_tok);
        }

        const int user_prompt_size = (int) user_tokens.size();
        const int max_batch_size = g_current_n_ctx - OVERFLOW_HEADROOM;
        if (user_prompt_size > max_batch_size) {
            const int skipped_tokens = user_prompt_size - max_batch_size;
            user_tokens.resize(max_batch_size);
            LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
        }

        // Benchmark Prompt Evaluation (Prefill) on Qualcomm NPU / CPU
        g_t_prompt_start = ggml_time_us();
        if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
            LOGe("%s: decode_tokens_in_batches failed for user prompt: %s",
                 __func__, g_last_error_message.c_str());
            if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
                chat_msgs.pop_back();
            }
            return 2;
        }
        g_t_prompt_end = ggml_time_us();

        g_session_tokens.insert(g_session_tokens.end(), user_tokens.begin(), user_tokens.end());
        current_position += user_prompt_size;
        stop_generation_position = current_position + n_predict;

        // Record prefill metrics
        g_metrics.prompt_tokens = user_prompt_size;
        const double pp_sec = double(g_t_prompt_end - g_t_prompt_start) / 1000000.0;
        g_metrics.prompt_time_ms = pp_sec * 1000.0;
        g_metrics.prompt_speed_tps = pp_sec > 0.0 ? (double(user_prompt_size) / pp_sec) : 0.0;

        // Initialize decode generation metrics
        g_t_decode_start = ggml_time_us();
        g_metrics.decode_tokens = 0;
        g_metrics.decode_time_ms = 0.0;
        g_metrics.decode_speed_tps = 0.0;

        LOGi("Snapdragon Prefill: %d tokens in %.2f ms (%.2f tokens/s) on %s",
             g_metrics.prompt_tokens, g_metrics.prompt_time_ms, g_metrics.prompt_speed_tps,
             g_metrics.active_backend.c_str());
        return 0;
    } catch (const std::exception & e) {
        g_last_error_message = e.what();
        LOGe("%s: Exception caught during prompt processing: %s", __func__, e.what());
        if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
            chat_msgs.pop_back();
        }
        return 3;
    } catch (...) {
        g_last_error_message = "Unknown exception caught during prompt processing";
        LOGe("%s: Unknown exception caught during prompt processing", __func__);
        if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
            chat_msgs.pop_back();
        }
        return 4;
    }
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }
    const auto *bytes = (const unsigned char *) string;
    int num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            num = 4;
        } else {
            return false;
        }
        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) { return false; }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeSetSpeculativePld(JNIEnv * /*env*/, jobject /*unused*/, jboolean enabled) {
    g_speculative_pld_enabled = enabled;
    LOGi("Speculative Prompt Lookup Decoding (PLD) acceleration set to: %d", (int)enabled);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    try {
        if (!g_model || !g_context || !g_sampler || !g_batch.token) {
            LOGw("%s: Model, context, sampler or batch is null or uninitialized", __func__);
            return nullptr;
        }

        static thread_local bool s_thread_pinned = false;
        if (!s_thread_pinned) {
            pin_threads_to_performance_cores();
            s_thread_pinned = true;
        }

        if (g_cancel_generation.load()) {
            LOGi("%s: Generation stopped by user cancellation", __func__);
            finish_assistant_response();
            return nullptr;
        }

        // 1. Initial token sampling immediately following prompt prefill
        if (g_spec_last_token == LLAMA_TOKEN_NULL) {
            if (!llama_get_logits_ith(g_context, -1)) {
                LOGe("%s: Logits are not available in llama_context! Cannot sample first token.", __func__);
                g_last_error_message = "llama_get_logits_ith returned null after prefill";
                return nullptr;
            }

            const auto first_token_id = common_sampler_sample(g_sampler, g_context, -1);
            common_sampler_accept(g_sampler, first_token_id, true);

            g_session_tokens.push_back(first_token_id);
            g_spec_last_token = first_token_id;

            g_metrics.decode_tokens = 1;
            g_t_decode_start = ggml_time_us();

            const auto *vocab = llama_model_get_vocab(g_model);
            if (!vocab || llama_vocab_is_eog(vocab, first_token_id)) {
                LOGd("EOG token %d detected on initial token. Generation complete.", first_token_id);
                finish_assistant_response();
                return nullptr;
            }

            auto new_token_chars = common_token_to_piece(g_context, first_token_id);
            cached_token_chars += new_token_chars;

            jstring result = nullptr;
            if (is_valid_utf8(cached_token_chars.c_str())) {
                result = env->NewStringUTF(cached_token_chars.c_str());
                assistant_ss << cached_token_chars;
                cached_token_chars.clear();
            } else {
                result = env->NewStringUTF("");
            }
            return result;
        }

        // 2. Continuation token generation (clean sequential autoregressive forward pass)
        if (current_position >= g_current_n_ctx - OVERFLOW_HEADROOM) {
            LOGw("%s: Context full! Shifting...", __func__);
            shift_context();
        }

        if (current_position >= stop_generation_position) {
            LOGw("%s: Reached stop position: %d", __func__, stop_generation_position);
            finish_assistant_response();
            return nullptr;
        }

        const auto *vocab = llama_model_get_vocab(g_model);
        if (!vocab || llama_vocab_is_eog(vocab, g_spec_last_token)) {
            LOGd("EOG token %d detected. Generation complete.", g_spec_last_token);
            finish_assistant_response();
            return nullptr;
        }

        // Add the previous token to the batch to evaluate its logits at current_position
        common_batch_clear(g_batch);
        common_batch_add(g_batch, g_spec_last_token, current_position, {0}, true);

        int decode_res = 0;
        try {
            decode_res = llama_decode(g_context, g_batch);
        } catch (const std::exception & de) {
            LOGe("%s: Exception in llama_decode during generation: %s", __func__, de.what());
            g_last_error_message = de.what();
            decode_res = 1;
        } catch (...) {
            LOGe("%s: Unknown exception in llama_decode during generation", __func__);
            g_last_error_message = "Unknown native exception in llama_decode";
            decode_res = 1;
        }

        if (decode_res != 0) {
            LOGe("%s: llama_decode() failed with error %d at current_position=%d",
                 __func__, decode_res, current_position);
            g_last_error_message = "llama_decode failed with error " + std::to_string(decode_res) + " at position " + std::to_string(current_position);
            finish_assistant_response();
            return nullptr;
        }

        // Advance sequential position by 1
        current_position++;

        // Sample next candidate token from logits at index -1
        const auto next_token_id = common_sampler_sample(g_sampler, g_context, -1);
        common_sampler_accept(g_sampler, next_token_id, true);

        g_session_tokens.push_back(next_token_id);
        g_spec_last_token = next_token_id;

        g_metrics.decode_tokens++;
        g_t_decode_end = ggml_time_us();
        const double tg_sec = double(g_t_decode_end - g_t_decode_start) / 1000000.0;
        g_metrics.decode_time_ms = tg_sec * 1000.0;
        g_metrics.decode_speed_tps = tg_sec > 0.0 ? (double(g_metrics.decode_tokens) / tg_sec) : 0.0;

        if (llama_vocab_is_eog(vocab, next_token_id)) {
            LOGd("EOG token %d detected. Generation complete.", next_token_id);
            finish_assistant_response();
            return nullptr;
        }

        auto new_token_chars = common_token_to_piece(g_context, next_token_id);
        cached_token_chars += new_token_chars;

        jstring result = nullptr;
        if (is_valid_utf8(cached_token_chars.c_str())) {
            result = env->NewStringUTF(cached_token_chars.c_str());
            assistant_ss << cached_token_chars;
            cached_token_chars.clear();
        } else {
            result = env->NewStringUTF("");
        }
        return result;
    } catch (const std::exception & e) {
        LOGe("%s: Exception caught during token generation: %s", __func__, e.what());
        g_last_error_message = e.what();
        finish_assistant_response();
        return nullptr;
    } catch (...) {
        LOGe("%s: Unknown exception caught during token generation", __func__);
        g_last_error_message = "Unknown exception during token generation";
        finish_assistant_response();
        return nullptr;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_unload(JNIEnv * /*unused*/, jobject /*unused*/) {
    g_cancel_generation.store(true);
    reset_long_term_states();
    reset_short_term_states();

    if (g_mtmd_ctx) {
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    if (g_sampler) {
        common_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    g_chat_templates.reset();
    if (g_batch.token) {
        llama_batch_free(g_batch);
        g_batch = {};
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_metrics.offloaded_layers = 0;
    g_metrics.total_layers = 0;
    g_metrics.prompt_tokens = 0;
    g_metrics.decode_tokens = 0;
    g_metrics.prompt_speed_tps = 0.0;
    g_metrics.decode_speed_tps = 0.0;
    g_metrics.spec_draft_tokens = 0;
    g_metrics.spec_accepted_tokens = 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_loadMmproj(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jmmproj_path
) {
    if (!g_model) {
        LOGe("Cannot load mmproj: base LLM model not loaded yet");
        g_last_error_message = "Base LLM model not loaded yet";
        return 1;
    }
    if (g_mtmd_ctx) {
        LOGi("Freeing existing mtmd context before loading new mmproj");
        mtmd_free(g_mtmd_ctx);
        g_mtmd_ctx = nullptr;
    }
    if (!jmmproj_path) {
        return 0;
    }
    const auto *const mmproj_path = env->GetStringUTFChars(jmmproj_path, nullptr);
    std::string path_str(mmproj_path ? mmproj_path : "");
    env->ReleaseStringUTFChars(jmmproj_path, mmproj_path);

    if (path_str.empty()) {
        return 0;
    }

    LOGi("Initializing multimodal projector from: %s", path_str.c_str());
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false; // CPU/NEON is fastest and safest for CLIP/SigLIP vision encoders on mobile
    mparams.n_threads = N_THREADS_DEFAULT;
    mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    g_mtmd_ctx = mtmd_init_from_file(path_str.c_str(), g_model, mparams);
    if (!g_mtmd_ctx) {
        LOGe("Failed to initialize mtmd context from %s", path_str.c_str());
        g_last_error_message = "Failed to load mmproj projector model from " + path_str;
        return 2;
    }

    LOGi("Successfully loaded mmproj model! Vision support: %d, Audio support: %d",
         mtmd_support_vision(g_mtmd_ctx), mtmd_support_audio(g_mtmd_ctx));
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPromptWithImage(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jstring jimage_path,
        jint n_predict
) {
    try {
        g_last_error_message.clear();
        pin_threads_to_performance_cores();

        if (!g_context || !g_model) {
            LOGe("%s: Model or context is null", __func__);
            g_last_error_message = "Model or context is null";
            return 1;
        }

        if (!g_mtmd_ctx) {
            LOGe("%s: Multimodal projector (mmproj) is not loaded!", __func__);
            g_last_error_message = "未加载多模态投影模型 (mmproj)，请在侧边栏【模型管理与设置】中配置 mmproj GGUF 文件";
            return 5;
        }

        reset_short_term_states();
        g_cancel_generation.store(false);

        const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
        std::string raw_prompt_str(user_prompt ? user_prompt : "");
        env->ReleaseStringUTFChars(juser_prompt, user_prompt);

        const auto *const image_path = env->GetStringUTFChars(jimage_path, nullptr);
        std::string image_path_str(image_path ? image_path : "");
        env->ReleaseStringUTFChars(jimage_path, image_path);

        LOGi("%s: Loading image from: %s", __func__, image_path_str.c_str());
        auto opt = mtmd_helper_init_opt_default();
        auto bmp_wrapper = mtmd_helper_bitmap_init_from_file(g_mtmd_ctx, image_path_str.c_str(), false, opt);
        if (!bmp_wrapper.bitmap) {
            LOGe("%s: Failed to load bitmap from %s", __func__, image_path_str.c_str());
            g_last_error_message = "无法解析或解码图片文件: " + image_path_str;
            return 6;
        }

        // Prepare prompt with media marker
        const std::string marker = mtmd_default_marker();
        std::string prompt_with_marker = raw_prompt_str;
        if (prompt_with_marker.find(marker) == std::string::npos) {
            prompt_with_marker = marker + "\n" + raw_prompt_str;
        }

        std::string formatted_user_prompt = prompt_with_marker;
        const bool has_chat_template = g_chat_templates && common_chat_templates_was_explicit(g_chat_templates.get());
        if (has_chat_template) {
            formatted_user_prompt = chat_add_and_format(ROLE_USER, prompt_with_marker);
        }

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text inp_text = {
            formatted_user_prompt.c_str(),
            formatted_user_prompt.size(),
            /* add_special */ false,
            /* parse_special */ true
        };
        const mtmd_bitmap * bmps[1] = { bmp_wrapper.bitmap };

        int32_t tok_res = mtmd_tokenize(g_mtmd_ctx, chunks, &inp_text, bmps, 1);
        if (tok_res != 0) {
            LOGe("%s: mtmd_tokenize failed: %d", __func__, tok_res);
            g_last_error_message = "多模态分词失败 (mtmd_tokenize error " + std::to_string(tok_res) + ")";
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp_wrapper.bitmap);
            if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
                chat_msgs.pop_back();
            }
            return 2;
        }

        size_t total_tokens = mtmd_helper_get_n_tokens(chunks);
        LOGi("%s: mtmd_tokenize succeeded! Total tokens (including image embeddings): %zu", __func__, total_tokens);

        g_t_prompt_start = ggml_time_us();
        llama_pos new_n_past = current_position;
        int eval_res = mtmd_helper_eval_chunks(
            g_mtmd_ctx,
            g_context,
            chunks,
            current_position,
            0, // seq_id
            BATCH_SIZE,
            true, // logits_last
            &new_n_past
        );
        g_t_prompt_end = ggml_time_us();

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp_wrapper.bitmap);

        if (eval_res != 0) {
            LOGe("%s: mtmd_helper_eval_chunks failed: %d", __func__, eval_res);
            g_last_error_message = "多模态图像与文本编码推理失败 (mtmd_helper_eval_chunks error " + std::to_string(eval_res) + ")";
            if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
                chat_msgs.pop_back();
            }
            return 2;
        }

        current_position = new_n_past;
        stop_generation_position = current_position + n_predict;

        try {
            llama_tokens text_tokens = common_tokenize(g_context, formatted_user_prompt, false, false);
            g_session_tokens.insert(g_session_tokens.end(), text_tokens.begin(), text_tokens.end());
        } catch (...) {
            // Non-critical if tokenization fails
        }

        g_metrics.prompt_tokens = (int)total_tokens;
        const double pp_sec = double(g_t_prompt_end - g_t_prompt_start) / 1000000.0;
        g_metrics.prompt_time_ms = pp_sec * 1000.0;
        g_metrics.prompt_speed_tps = pp_sec > 0.0 ? (double(total_tokens) / pp_sec) : 0.0;

        g_t_decode_start = ggml_time_us();
        g_metrics.decode_tokens = 0;
        g_metrics.decode_time_ms = 0.0;
        g_metrics.decode_speed_tps = 0.0;

        LOGi("Multimodal Prefill: %zu tokens in %.2f ms (%.2f tokens/s)",
             total_tokens, g_metrics.prompt_time_ms, g_metrics.prompt_speed_tps);
        return 0;
    } catch (const std::exception & e) {
        g_last_error_message = e.what();
        LOGe("%s: Exception caught: %s", __func__, e.what());
        if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
            chat_msgs.pop_back();
        }
        return 3;
    } catch (...) {
        g_last_error_message = "Unknown exception in multimodal prompt processing";
        LOGe("%s: Unknown exception caught", __func__);
        if (!chat_msgs.empty() && chat_msgs.back().role == ROLE_USER) {
            chat_msgs.pop_back();
        }
        return 4;
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_nativeResetConversation(JNIEnv * /*env*/, jobject /*unused*/) {
    reset_long_term_states(true);
    reset_short_term_states();
    LOGi("Native conversation and KV cache reset successfully.");
}

extern "C"
JNIEXPORT void JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}
