# llama-npu-android

**llama-npu-android** 是基于 `llama.cpp` 原生 C++ 核心构建的 Android 端侧大语言模型与多模态离线推理应用。深度协同高通骁龙 Hexagon NPU 与多核 ARM Cortex CPU，实现全离线端侧推理加速。

> **AI 开发声明 / AI Development Notice**:
> 本项目由 **AI（Google Antigravity / Gemini）** 辅助生成与协同开发，包含端侧 NPU 硬件加速适配、零拷贝直读架构、多厂商后台保活守护及全离线异构推理优化。
> This project is designed and developed with the assistance of AI (Google Antigravity / Gemini).

---

## 核心特性

- **专用硬件加速**：集成高通骁龙 Hexagon NPU (HTP v73 / v75 / v79) 计算后端，支持配置模型层卸载至 NPU 硬件协同执行；在遇到未适配算子或资源受限时支持平滑降级至 CPU 混合计算。
- **KV 优化与注意力机制**：支持 Q8_0 KV Cache 与 FlashAttention 机制，有效降低端侧自注意力计算与内存带宽开销。
- **通用硬件自适应**：内置通用移动硬件探测器 (`DeviceHardwareProfiler`)，动态识别设备芯片架构并基于物理内存与核心数智能推荐线程数与上下文窗口；在非高通芯片设备上自动采用多线程 ARM NEON CPU 进行推理。
- **零拷贝直读架构**：结合 Android 存储访问框架 (SAF) 与直接文件描述符映射 (`/proc/self/fd/`)，无需复制大型 GGUF 文件即可直接加载，消除磁盘磨损与重复占用。
- **多模态视觉理解**：支持集成载入 `mmproj` 多模态投影模型，实现离线图片多模态问答。
- **多厂商进程保活**：针对 Android 14+ 及各主流品牌深度定制系统（小米、华为、荣耀、OPPO、vivo、三星），提供前台服务守护、电池优化豁免、WakeLock 及可选的系统级保护引导，避免后台推理被终止。
- **100% 物理级纯离线安全**：应用**未申请任何网络访问权限** (`android.permission.INTERNET`)，杜绝任何外部数据泄露风险，保障用户对话隐私。

---

## 开源许可证 (License)

本工程遵循 **MIT 开源许可证** (MIT License)。

```text
MIT License
Copyright (c) 2024-2026 The llama.cpp Authors and Contributors
```

详情请参见同目录下的 [LICENSE](LICENSE) 文件。

### 第三方组件开源声明 (Third-Party Notices)

本项目使用或依赖以下开源库与规范：

1. **Markwon Markdown 渲染引擎** (`io.noties.markwon`):
   - 遵循 **Apache License 2.0** 许可证。
   - Copyright 2017 Dimitry Ivanov.
2. **AndroidX 与 Google Material Design 库**:
   - 遵循 **Apache License 2.0** 许可证。
   - Copyright (C) The Android Open Source Project.
3. **Kotlin 标准库与协程**:
   - 遵循 **Apache License 2.0** 许可证。
   - Copyright 2010-2026 JetBrains s.r.o.

### 硬件驱动说明 (Hardware Runtime Disclaimer)

- 高通 FastRPC 运行时库 (`libcdsprpc.so` / `libadsprpc.so`) 归 Qualcomm Technologies, Inc. 所有。
- **本项目不随附、不分发任何高通专有闭源二进制文件或专有 SDK**。在支持高通芯片的设备上运行时，应用通过 Linux 标准动态链接机制 (`dlopen`) 直接调用设备厂商分区 (`/vendor/lib64/`) 中预装的系统驱动。

---

## 编译与构建 (Building)

### 前置环境
- Android SDK (API Level 36 / Android 16)
- Android NDK 26+ (推荐 `29.0.14206865`)
- CMake 3.22+
- JDK 17+

### 构建指令

```bash
cd examples/llama.android

# 编译 Debug APK
./gradlew assembleDebug --no-daemon

# 输出产物路径
# app/build/outputs/apk/debug/app-debug.apk
```
