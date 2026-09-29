package com.example.llama

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.OpenableColumns
import android.provider.Settings
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import com.example.llama.monitor.ProcessProtectionManager
import com.example.llama.monitor.DeviceHardwareProfiler
import com.example.llama.service.InferenceForegroundService
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.GravityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.HardwareAccelerator
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ModelConfig
import com.arm.aichat.NpuTopology
import com.arm.aichat.FlashAttnMode
import com.arm.aichat.KvCacheType
import com.example.llama.data.ChatMessage
import com.example.llama.data.Conversation
import com.example.llama.data.ConversationRepository
import com.example.llama.monitor.HardwareMonitor
import com.example.llama.settings.ModelSettingsManager
import com.example.llama.ui.ConversationAdapter
import com.google.android.material.card.MaterialCardView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var engine: InferenceEngine
    private lateinit var conversationRepo: ConversationRepository
    private lateinit var settingsManager: ModelSettingsManager
    private lateinit var hardwareMonitor: HardwareMonitor

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var leftDrawer: View
    private lateinit var rightDrawer: View

    // Main Chat Views
    private lateinit var tvMainTitle: TextView
    private lateinit var tvMainSubtitle: TextView
    private lateinit var btnOpenHistory: ImageButton
    private lateinit var btnOpenMonitor: ImageButton
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: TextInputEditText
    private lateinit var userActionFab: FloatingActionButton

    // Left Drawer Views
    private lateinit var btnNewChat: Button
    private lateinit var rvHistoryConversations: RecyclerView
    private lateinit var cardBtnSettings: MaterialCardView
    private lateinit var tvDrawerModelStatus: TextView

    // Right Drawer Views (Hardware Monitor)
    private lateinit var tvWorkloadDesc: TextView
    private lateinit var tvMonLayers: TextView
    private lateinit var tvMonActiveBackend: TextView
    private lateinit var tvMonNpuTopo: TextView
    private lateinit var tvMonGpu: TextView
    private lateinit var tvMonThermalAlert: TextView
    private lateinit var tvMonBatteryTemp: TextView
    private lateinit var tvMonSocTemp: TextView
    private lateinit var tvMonCpuUsage: TextView
    private lateinit var tvMonThreads: TextView
    private lateinit var pbCpuUsage: ProgressBar
    private lateinit var tvMonRamUsage: TextView
    private lateinit var pbRamUsage: ProgressBar
    private lateinit var tvMonAppMem: TextView
    private lateinit var tvMonPrefillSpeed: TextView
    private lateinit var tvMonDecodeSpeed: TextView
    private lateinit var tvMonTtft: TextView
    private lateinit var btnRunBenchmark: Button
    private lateinit var tvBenchmarkResult: TextView

    // Process Protection & Anti-Kill Views
    private lateinit var tvProtOomScore: TextView
    private lateinit var tvProtFgsStatus: TextView
    private lateinit var tvProtBatteryStatus: TextView
    private lateinit var tvProtWakelockStatus: TextView
    private lateinit var switchProtFgs: MaterialSwitch
    private lateinit var btnProtBattery: Button
    private lateinit var btnProtVendorSettings: Button
    private lateinit var btnProtRoot: Button
    private lateinit var btnProtAdb: Button

    // Adapters & State
    private lateinit var messageAdapter: MessageAdapter
    private lateinit var conversationAdapter: ConversationAdapter
    private val currentMessages = mutableListOf<ChatMessage>()
    private var currentConversation: Conversation? = null

    private var activeConfig = ModelConfig()
    private var loadedModelPath: String? = null
    private var loadedModelDisplayName: String = "未加载模型"
    private var isModelReady: Boolean = false
    @Volatile
    private var isModelLoading: Boolean = false
    private var modelLoadingJob: Job? = null
    private var hasAttemptedAutoRestore: Boolean = false
    private var isGenerating: Boolean = false
    private var generationJob: Job? = null
    private var monitorJob: Job? = null

    // Multimodal Views & State
    private lateinit var cardImagePreview: MaterialCardView
    private lateinit var ivImagePreviewThumb: ImageView
    private lateinit var tvImagePreviewName: TextView
    private lateinit var btnRemoveImagePreview: ImageButton
    private lateinit var btnAttachImage: ImageButton

    private var currentAttachedImagePath: String? = null
    private var currentAttachedImageUri: String? = null

    private var settingsDialogView: View? = null

    private val selectModelLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { onModelUriSelected(it) }
    }

    private val selectMmprojLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { onMmprojUriSelected(it) }
    }

    private val selectImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { onImageUriSelected(it) }
    }

    private val requestNotificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.i(TAG, "Notification permission granted for Foreground Service")
        }
    }

    private val requestStoragePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (Environment.isExternalStorageManager()) {
                Toast.makeText(this, "已获得所有文件管理权限，可直接加载模型！", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initRepositoriesAndManagers()
        initViews()
        setupWindowInsets()
        setupDrawerWidths()
        setupConversations()
        setupHardwareMonitor()
        initInferenceEngine()
        checkAndRequestStoragePermission()
        checkAndRequestNotificationPermission()
    }

    private fun checkAndRequestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestNotificationPermissionLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    private fun checkAndRequestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                AlertDialog.Builder(this)
                    .setTitle("需要文件访问权限")
                    .setMessage("为避免反复移动/复制大模型造成存储磨损，并实现骁龙 NPU 零复制直接内存映射，需要授予【所有文件访问权限】。")
                    .setPositiveButton("前往授权") { _, _ ->
                        try {
                            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                                data = Uri.parse("package:$packageName")
                            }
                            requestStoragePermissionLauncher.launch(intent)
                        } catch (e: Exception) {
                            val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                            requestStoragePermissionLauncher.launch(intent)
                        }
                    }
                    .setNegativeButton("稍后", null)
                    .show()
            }
        } else {
            requestPermissions(arrayOf(
                android.Manifest.permission.READ_EXTERNAL_STORAGE,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            ), 1001)
        }
    }

    private fun setupWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(drawerLayout) { _, insets ->
            val statusBars = insets.getInsets(WindowInsetsCompat.Type.statusBars())
            val navBars = insets.getInsets(WindowInsetsCompat.Type.navigationBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())

            // 1. Top bar: apply status bar padding to avoid notification bar collision
            val topBar = findViewById<View>(R.id.main_top_bar)
            topBar?.setPadding(
                topBar.paddingLeft,
                statusBars.top,
                topBar.paddingRight,
                topBar.paddingBottom
            )

            // 2. Bottom input container: accommodate system gesture navigation bar and soft keyboard
            val bottomContainer = findViewById<View>(R.id.bottom_input_container)
            val bottomInset = maxOf(navBars.bottom, ime.bottom)
            bottomContainer?.setPadding(
                bottomContainer.paddingLeft,
                bottomContainer.paddingTop,
                bottomContainer.paddingRight,
                bottomInset + (if (ime.bottom > 0) 4 else 8)
            )

            // 3. Side Drawers: pad with status bar top and navigation bar bottom
            leftDrawer.setPadding(0, statusBars.top, 0, navBars.bottom)
            rightDrawer.setPadding(0, statusBars.top, 0, navBars.bottom)

            insets
        }
    }

    private fun initRepositoriesAndManagers() {
        conversationRepo = ConversationRepository(this)
        settingsManager = ModelSettingsManager(this)
        hardwareMonitor = HardwareMonitor(this)
        activeConfig = settingsManager.getSavedModelConfig()
    }

    private fun initViews() {
        drawerLayout = findViewById(R.id.drawer_layout)
        leftDrawer = findViewById(R.id.nav_left_drawer)
        rightDrawer = findViewById(R.id.nav_right_drawer)

        // Main Chat
        tvMainTitle = findViewById(R.id.tv_main_title)
        tvMainSubtitle = findViewById(R.id.tv_main_subtitle)
        val initialProfile = DeviceHardwareProfiler.getProfile(this)
        tvMainTitle.text = "${initialProfile.manufacturer} ${initialProfile.model}"
        tvMainSubtitle.text = "${initialProfile.socModelName} | 未加载模型"
        btnOpenHistory = findViewById(R.id.btn_open_history)
        btnOpenMonitor = findViewById(R.id.btn_open_monitor)
        messagesRv = findViewById(R.id.messages_rv)
        userInputEt = findViewById(R.id.user_input_et)
        userActionFab = findViewById(R.id.user_action_fab)

        // Multimodal Input Views
        cardImagePreview = findViewById(R.id.card_image_preview)
        ivImagePreviewThumb = findViewById(R.id.iv_image_preview_thumbnail)
        tvImagePreviewName = findViewById(R.id.tv_image_preview_name)
        btnRemoveImagePreview = findViewById(R.id.btn_remove_image_preview)
        btnAttachImage = findViewById(R.id.btn_attach_image)

        btnAttachImage.setOnClickListener {
            selectImageLauncher.launch("image/*")
        }

        btnRemoveImagePreview.setOnClickListener {
            cardImagePreview.visibility = View.GONE
            currentAttachedImagePath = null
            currentAttachedImageUri = null
            if (isGenerating && userInputEt.text.isNullOrBlank()) {
                userActionFab.setImageResource(R.drawable.outline_stop_24)
            }
        }

        // Continuous typing watcher: allows typing while model streams response
        userInputEt.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val hasInput = !s.isNullOrBlank() || currentAttachedImagePath != null
                if (isGenerating) {
                    if (hasInput) {
                        userActionFab.setImageResource(R.drawable.outline_send_24)
                    } else {
                        userActionFab.setImageResource(R.drawable.outline_stop_24)
                    }
                } else {
                    userActionFab.setImageResource(R.drawable.outline_send_24)
                }
            }
        })

        // Left Drawer
        btnNewChat = findViewById(R.id.btn_new_chat)
        rvHistoryConversations = findViewById(R.id.rv_history_conversations)
        cardBtnSettings = findViewById(R.id.card_btn_settings)
        tvDrawerModelStatus = findViewById(R.id.tv_drawer_model_status)

        // Right Drawer
        tvWorkloadDesc = findViewById(R.id.tv_workload_desc)
        tvMonLayers = findViewById(R.id.tv_mon_layers)
        tvMonActiveBackend = findViewById(R.id.tv_mon_active_backend)
        tvMonNpuTopo = findViewById(R.id.tv_mon_npu_topo)
        tvMonGpu = findViewById(R.id.tv_mon_gpu)
        tvMonThermalAlert = findViewById(R.id.tv_mon_thermal_alert)
        tvMonBatteryTemp = findViewById(R.id.tv_mon_battery_temp)
        tvMonSocTemp = findViewById(R.id.tv_mon_soc_temp)
        tvMonCpuUsage = findViewById(R.id.tv_mon_cpu_usage)
        tvMonThreads = findViewById(R.id.tv_mon_threads)
        pbCpuUsage = findViewById(R.id.pb_cpu_usage)
        tvMonRamUsage = findViewById(R.id.tv_mon_ram_usage)
        pbRamUsage = findViewById(R.id.pb_ram_usage)
        tvMonAppMem = findViewById(R.id.tv_mon_app_mem)
        tvMonPrefillSpeed = findViewById(R.id.tv_mon_prefill_speed)
        tvMonDecodeSpeed = findViewById(R.id.tv_mon_decode_speed)
        tvMonTtft = findViewById(R.id.tv_mon_ttft)
        btnRunBenchmark = findViewById(R.id.btn_run_benchmark)
        tvBenchmarkResult = findViewById(R.id.tv_benchmark_result)

        // Process Protection & Anti-Kill Views
        tvProtOomScore = findViewById(R.id.tv_prot_oom_score)
        tvProtFgsStatus = findViewById(R.id.tv_prot_fgs_status)
        tvProtBatteryStatus = findViewById(R.id.tv_prot_battery_status)
        tvProtWakelockStatus = findViewById(R.id.tv_prot_wakelock_status)
        switchProtFgs = findViewById(R.id.switch_prot_fgs)
        btnProtBattery = findViewById(R.id.btn_prot_battery)
        btnProtVendorSettings = findViewById(R.id.btn_prot_vendor_settings)
        btnProtRoot = findViewById(R.id.btn_prot_root)
        btnProtAdb = findViewById(R.id.btn_prot_adb)

        switchProtFgs.isChecked = true
        switchProtFgs.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                val profile = DeviceHardwareProfiler.getProfile(this)
                InferenceForegroundService.start(
                    this,
                    "🟢 硬件就绪 | $loadedModelDisplayName",
                    "${profile.socModelName} | 前台保活中"
                )
            } else {
                InferenceForegroundService.stop(this)
            }
        }

        btnProtBattery.setOnClickListener {
            ProcessProtectionManager.requestIgnoreBatteryOptimizations(this)
        }

        btnProtVendorSettings.setOnClickListener {
            val opened = ProcessProtectionManager.openVendorBackgroundSettings(this)
            if (!opened) {
                Toast.makeText(this, "无法直接打开当前系统后台设置，请前往系统设置手动开启无限制", Toast.LENGTH_SHORT).show()
            }
        }

        btnProtRoot.setOnClickListener {
            val (success, msg) = ProcessProtectionManager.applyRootImmortalProtection(lifecycleScope)
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        }

        btnProtAdb.setOnClickListener {
            val adbText = ProcessProtectionManager.getAdbCommandGuide(this)
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("ADB_AntiKill_Commands", adbText)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(this, "📋 进程常驻保护 ADB 指令已复制到剪贴板！可在电脑终端粘贴执行", Toast.LENGTH_LONG).show()
        }

        // Listeners
        btnOpenHistory.setOnClickListener {
            drawerLayout.openDrawer(GravityCompat.START)
        }
        btnOpenMonitor.setOnClickListener {
            drawerLayout.openDrawer(GravityCompat.END)
        }
        btnNewChat.setOnClickListener {
            createNewChat()
            drawerLayout.closeDrawer(GravityCompat.START)
        }
        cardBtnSettings.setOnClickListener {
            showModelSettingsDialog()
            drawerLayout.closeDrawer(GravityCompat.START)
        }
        btnRunBenchmark.setOnClickListener {
            runNpuBenchmark()
        }

        userActionFab.setOnClickListener {
            if (isGenerating) {
                val pendingText = userInputEt.text?.toString()?.trim() ?: ""
                if (pendingText.isNotEmpty() || currentAttachedImagePath != null) {
                    // User typed a new question while generating: interrupt & send immediately!
                    stopGeneration()
                    sendMessage()
                } else {
                    stopGeneration()
                }
            } else {
                sendMessage()
            }
        }

        messageAdapter = MessageAdapter(currentMessages, this)
        messagesRv.apply {
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true
            }
            adapter = messageAdapter
        }
    }

    /**
     * Scale both left & right drawers to ~82% of screen width (satisfies > 3/4 coverage)
     */
    private fun setupDrawerWidths() {
        val screenWidth = resources.displayMetrics.widthPixels
        val drawerWidth = (screenWidth * 0.82).toInt()
        leftDrawer.layoutParams = leftDrawer.layoutParams.apply { width = drawerWidth }
        rightDrawer.layoutParams = rightDrawer.layoutParams.apply { width = drawerWidth }
    }

    private fun setupConversations() {
        val all = conversationRepo.getAllConversations().toMutableList()
        // Clean up any extraneous empty conversations if more than 1 exist
        val emptyConvs = all.filter { it.messages.isEmpty() }
        if (emptyConvs.size > 1) {
            emptyConvs.drop(1).forEach { conversationRepo.deleteConversation(it.id) }
        }

        val conversations = conversationRepo.getAllConversations().toMutableList()
        val initialConv = if (conversations.isNotEmpty()) {
            conversations.first()
        } else {
            val created = conversationRepo.createNewConversation("新对话")
            conversations.add(created)
            created
        }
        currentConversation = initialConv

        conversationAdapter = ConversationAdapter(
            conversations = conversations,
            activeConversationId = initialConv.id,
            onConversationClick = { conv ->
                switchConversation(conv)
                drawerLayout.closeDrawer(GravityCompat.START)
            },
            onConversationDelete = { conv ->
                deleteConversation(conv)
            }
        )

        rvHistoryConversations.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = conversationAdapter
        }

        loadConversationMessages(initialConv)
    }

    private fun loadConversationMessages(conv: Conversation) {
        currentMessages.clear()
        currentMessages.addAll(conv.messages)
        messageAdapter.notifyDataSetChanged()
        if (currentMessages.isNotEmpty()) {
            messagesRv.scrollToPosition(currentMessages.size - 1)
        }
        tvMainTitle.text = conv.title
    }

    private fun createNewChat() {
        // If current active conversation is already empty (no messages), simply stay in it!
        val current = currentConversation
        if (current != null && current.messages.isEmpty()) {
            loadConversationMessages(current)
            return
        }

        // Check if an empty conversation already exists in storage, reuse it instead of creating duplicate
        val existingEmpty = conversationRepo.getAllConversations().firstOrNull { it.messages.isEmpty() }
        if (existingEmpty != null) {
            switchConversation(existingEmpty)
            return
        }

        val newConv = conversationRepo.createNewConversation("新对话")
        currentConversation = newConv
        conversationAdapter.setActiveConversation(newConv.id)
        conversationAdapter.updateData(conversationRepo.getAllConversations())
        try { engine.resetConversation() } catch (e: Exception) { Log.w(TAG, "resetConversation error", e) }
        loadConversationMessages(newConv)
    }

    private fun switchConversation(conv: Conversation) {
        currentConversation = conv
        conversationAdapter.setActiveConversation(conv.id)
        try { engine.resetConversation() } catch (e: Exception) { Log.w(TAG, "resetConversation error", e) }
        loadConversationMessages(conv)
    }

    private fun deleteConversation(conv: Conversation) {
        conversationRepo.deleteConversation(conv.id)
        val remaining = conversationRepo.getAllConversations()
        if (conv.id == currentConversation?.id) {
            if (remaining.isNotEmpty()) {
                switchConversation(remaining.first())
            } else {
                createNewChat()
            }
        }
        conversationAdapter.updateData(conversationRepo.getAllConversations())
    }

    private fun initInferenceEngine() {
        engine = com.arm.aichat.AiChat.getInferenceEngine(this)
        engine.onHardwareFallback = { newAccel ->
            runOnUiThread {
                activeConfig = activeConfig.copy(
                    accelerator = newAccel,
                    nGpuLayers = if (newAccel == HardwareAccelerator.CPU_ARM) 0 else activeConfig.nGpuLayers
                )
                // Do not permanently save fallback to preferences so next restart still attempts hardware acceleration
                Toast.makeText(this, "加速器显存超限，已临时切换至 CPU 模式稳定运行 (下次启动仍优先尝试硬件加速)", Toast.LENGTH_LONG).show()
            }
        }
        lifecycleScope.launch {
            engine.state.collect { state ->
                when (state) {
                    is InferenceEngine.State.Initializing -> {
                        tvMainSubtitle.text = "硬件推理引擎初始化中..."
                        isModelReady = false
                        userInputEt.isEnabled = false
                        userActionFab.isEnabled = false
                    }
                    is InferenceEngine.State.Initialized -> {
                        tvMainSubtitle.text = "驱动就绪 | 请点击左下角选择模型"
                        isModelReady = false
                        userInputEt.isEnabled = false
                        userActionFab.isEnabled = false
                    }
                    is InferenceEngine.State.UnloadingModel -> {
                        tvMainSubtitle.text = "正在卸载释放旧模型..."
                        isModelReady = false
                        userInputEt.isEnabled = false
                        userActionFab.isEnabled = false
                    }
                    is InferenceEngine.State.LoadingModel -> {
                        tvMainSubtitle.text = "正在加载模型至端侧硬件加速..."
                        isModelReady = false
                        userInputEt.isEnabled = false
                        userActionFab.isEnabled = false
                    }
                    is InferenceEngine.State.ModelReady -> {
                        isModelReady = true
                        userInputEt.isEnabled = true
                        userActionFab.isEnabled = true
                        val accelDesc = when (activeConfig.accelerator) {
                            HardwareAccelerator.NPU_HEXAGON -> "⚡NPU硬件加速 (${activeConfig.nGpuLayers}层)"
                            HardwareAccelerator.GPU_GENERIC -> "⚡GPU加速 (${activeConfig.nGpuLayers}层)"
                            HardwareAccelerator.HYBRID_COMBINED -> "🔀异构混合 (${activeConfig.nGpuLayers}层)"
                            HardwareAccelerator.CPU_ARM -> "💻CPU模式"
                        }
                        tvMainSubtitle.text = "🟢 硬件就绪 [$accelDesc] | $loadedModelDisplayName"
                        tvDrawerModelStatus.text = "模型: $loadedModelDisplayName ($accelDesc)"
                        if (::switchProtFgs.isInitialized && switchProtFgs.isChecked) {
                            val profile = DeviceHardwareProfiler.getProfile(this@MainActivity)
                            InferenceForegroundService.start(
                                this@MainActivity,
                                "🟢 硬件就绪 [$accelDesc] | $loadedModelDisplayName",
                                "${profile.socModelName} | 前台保活中"
                            )
                        }
                    }
                    is InferenceEngine.State.Error -> {
                        isModelReady = false
                        userInputEt.isEnabled = true
                        userActionFab.isEnabled = true
                        tvMainSubtitle.text = "❌ 加载异常: ${state.exception.message}"
                        tvDrawerModelStatus.text = "异常: ${state.exception.message}"
                    }
                    else -> {}
                }
            }
        }

        // Dedicated one-shot auto-restore on initial application startup
        lifecycleScope.launch {
            engine.state.first { it is InferenceEngine.State.Initialized }
            if (!hasAttemptedAutoRestore) {
                hasAttemptedAutoRestore = true
                settingsManager.getLastSelectedModelPath()?.let { lastPath ->
                    if (lastPath.isNotBlank()) {
                        try {
                            val (validPath, name) = settingsManager.preparePathForLoad(lastPath)
                            if (File(validPath).exists() || validPath.startsWith("/proc/self/fd/")) {
                                Log.i(TAG, "Startup auto-restoring remembered model: $name")
                                loadModelPathDirectly(validPath, name)
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not auto restore last model on startup", e)
                        }
                    }
                }
            }
        }
    }

    /**
     * Real-time hardware telemetry loop running in background
     */
    private fun setupHardwareMonitor() {
        monitorJob?.cancel()
        monitorJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                val stats = if (::engine.isInitialized) {
                    engine.getInferenceStats()
                } else {
                    com.arm.aichat.InferenceStats()
                }
                val snap = hardwareMonitor.takeSnapshot(stats)

                withContext(Dispatchers.Main) {
                    tvWorkloadDesc.text = snap.workloadDescription
                    tvMonLayers.text = "• NPU 卸载层数: ${stats.offloadedLayers}/${stats.totalLayers} 层"
                    tvMonActiveBackend.text = "• 活跃后端: ${snap.activeAccelerator}"
                    tvMonNpuTopo.text = "• NPU 拓扑: ${stats.npuTopology}"
                    tvMonGpu.text = "• ${snap.gpuStatus}"

                    tvMonThermalAlert.text = snap.thermalAlert
                    tvMonBatteryTemp.text = String.format("电池: %.1f °C", snap.batteryTempC)
                    tvMonSocTemp.text = String.format("SoC 核心: %.1f °C", snap.socTempC)

                    tvMonCpuUsage.text = "CPU 使用率: ${snap.cpuUsagePercent}%"
                    tvMonThreads.text = "${activeConfig.nThreads} 线程"
                    pbCpuUsage.progress = snap.cpuUsagePercent

                    tvMonRamUsage.text = String.format("系统 RAM: %.1f / %.1f GB (%d%%)", snap.usedRamGb, snap.totalRamGb, snap.ramPercent)
                    pbRamUsage.progress = snap.ramPercent
                    tvMonAppMem.text = "App 进程内存: ${snap.appMemoryMb} MB"

                    if (stats.promptSpeedTps > 0) {
                        tvMonPrefillSpeed.text = String.format("• Prefill (首字吞吐): %.1f tok/s (%.1f ms)", stats.promptSpeedTps, stats.promptTimeMs)
                        tvMonTtft.text = String.format("• 首字时延 (TTFT): %.1f ms", stats.promptTimeMs)
                    }
                    if (stats.decodeSpeedTps > 0) {
                        tvMonDecodeSpeed.text = String.format("• Decode (持续生成): %.1f tok/s", stats.decodeSpeedTps)
                    }

                    // Process Protection Status Telemetry
                    val oomAdj = snap.oomScoreAdj
                    val oomDesc = when {
                        oomAdj <= -1000 -> "$oomAdj (👑 Root 最高保护级别)"
                        oomAdj < 0 -> "$oomAdj (系统最高特权)"
                        oomAdj == 0 -> "0 (前台应用运行中)"
                        oomAdj <= 200 -> "$oomAdj (前台服务受保护，安全)"
                        else -> "$oomAdj (⚠️ 后台优先级较低，易被回收)"
                    }
                    tvProtOomScore.text = "• OOM 评分 (oom_score_adj): $oomDesc"
                    tvProtFgsStatus.text = "• 前台常驻保活服务: ${if (snap.isFgsRunning) "运行中 (常驻通知栏)" else "已停用"}"
                    tvProtBatteryStatus.text = "• 电池无限制白名单: ${if (snap.isIgnoringBattery) "已豁免限制 (无限制)" else "未豁免 (受省电限制)"}"
                    tvProtWakelockStatus.text = "• CPU 唤醒锁 (WakeLock): ${if (snap.isFgsRunning) "已持有 (防熄屏休眠)" else "未持有"}"
                }
                delay(1000)
            }
        }
    }

    private fun safeNotifyItemInserted(position: Int) {
        if (messagesRv.isComputingLayout) {
            messagesRv.post {
                if (position in 0 until currentMessages.size) {
                    messageAdapter.notifyItemInserted(position)
                }
            }
        } else {
            if (position in 0 until currentMessages.size) {
                messageAdapter.notifyItemInserted(position)
            }
        }
    }

    private fun safeNotifyItemChanged(position: Int, payload: Any? = null) {
        if (position < 0 || position >= currentMessages.size) return
        if (messagesRv.isComputingLayout) {
            messagesRv.post {
                if (position in 0 until currentMessages.size) {
                    if (payload != null) {
                        messageAdapter.notifyItemChanged(position, payload)
                    } else {
                        messageAdapter.notifyItemChanged(position)
                    }
                }
            }
        } else {
            if (payload != null) {
                messageAdapter.notifyItemChanged(position, payload)
            } else {
                messageAdapter.notifyItemChanged(position)
            }
        }
    }

    private fun safeScrollToBottom() {
        val target = currentMessages.size - 1
        if (target < 0) return
        if (messagesRv.isComputingLayout) {
            messagesRv.post {
                messagesRv.scrollToPosition(target)
            }
        } else {
            messagesRv.scrollToPosition(target)
        }
    }

    private fun sendMessage() {
        val rawText = userInputEt.text?.toString()?.trim() ?: ""
        val attachedPath = currentAttachedImagePath
        val attachedUri = currentAttachedImageUri

        if (rawText.isEmpty() && attachedPath == null) return

        if (!isModelReady) {
            Toast.makeText(this, "请先在侧边栏点击【模型管理与设置】加载 GGUF 模型！", Toast.LENGTH_LONG).show()
            return
        }

        val text = if (rawText.isEmpty() && attachedPath != null) {
            "请识别分析这张图片的内容。"
        } else {
            rawText
        }

        userInputEt.text?.clear()
        // KEEP userInputEt enabled so user can continuously type while assistant generates!
        userInputEt.isEnabled = true
        userInputEt.hint = "思考生成中，可继续输入下一条..."
        userActionFab.setImageResource(R.drawable.outline_stop_24)
        isGenerating = true

        // Clear attachment preview from input bar
        cardImagePreview.visibility = View.GONE
        currentAttachedImagePath = null
        currentAttachedImageUri = null

        val conv = currentConversation ?: createNewChat().let { currentConversation!! }
        if (conv.messages.isEmpty() && text.length > 2) {
            val titleCandidate = text.take(16).replace("\n", " ")
            conv.title = titleCandidate
            tvMainTitle.text = titleCandidate
        }

        // Add user message with image if attached
        val userMsg = ChatMessage(
            id = UUID.randomUUID().toString(),
            content = text,
            isUser = true,
            imageUri = attachedUri
        )
        currentMessages.add(userMsg)
        conv.messages.add(userMsg)
        conv.updatedAt = System.currentTimeMillis()
        conversationRepo.saveConversation(conv)
        conversationAdapter.updateData(conversationRepo.getAllConversations())
        safeNotifyItemInserted(currentMessages.size - 1)
        safeScrollToBottom()

        // Add assistant placeholder message
        val assistantMsg = ChatMessage(id = UUID.randomUUID().toString(), content = "...", isUser = false)
        currentMessages.add(assistantMsg)
        val assistantIdx = currentMessages.size - 1
        safeNotifyItemInserted(assistantIdx)
        safeScrollToBottom()

        val generationExceptionHandler = CoroutineExceptionHandler { _, throwable ->
            Log.e(TAG, "Unhandled error during generation coroutine", throwable)
            runOnUiThread {
                assistantMsg.content = "⚠️ 推理异常: ${throwable.message}"
                safeNotifyItemChanged(assistantIdx)
                finishGenerationUI(conv, assistantMsg)
            }
        }

        generationJob?.cancel()
        val profile = DeviceHardwareProfiler.getProfile(this@MainActivity)
        InferenceForegroundService.updateStatus(
            this@MainActivity,
            "⚡ 模型推理中...",
            "$loadedModelDisplayName | ${profile.socModelName} 加速中"
        )
        generationJob = lifecycleScope.launch(Dispatchers.Default + generationExceptionHandler) {
            val sb = StringBuilder()
            var lastUpdateTime = 0L
            var hasError = false
            engine.sendUserPrompt(text, attachedPath)
                .catch { e ->
                    Log.e(TAG, "Generation error", e)
                    hasError = true
                    withContext(Dispatchers.Main) {
                        assistantMsg.content = "⚠️ 生成中断: ${e.message}"
                        safeNotifyItemChanged(assistantIdx)
                        finishGenerationUI(conv, assistantMsg)
                        showInferenceErrorRecoveryDialog(e.message ?: "")
                    }
                }
                .onCompletion {
                    if (!hasError) {
                        withContext(Dispatchers.Main) {
                            if (sb.isNotEmpty()) {
                                assistantMsg.content = sb.toString()
                            } else if (assistantMsg.content == "...") {
                                assistantMsg.content = "⚠️ NPU 未返回生成文本 (可尝试调整 Context 长度或重试)"
                            }
                            safeNotifyItemChanged(assistantIdx)
                            safeScrollToBottom()
                            finishGenerationUI(conv, assistantMsg)
                        }
                    }
                }
                .collect { token ->
                    sb.append(token)
                    val now = System.currentTimeMillis()
                    // Throttle UI re-binding to at most once per 65ms using lightweight payload update
                    if (now - lastUpdateTime > 65) {
                        lastUpdateTime = now
                        withContext(Dispatchers.Main) {
                            assistantMsg.content = sb.toString()
                            safeNotifyItemChanged(assistantIdx, MessageAdapter.PAYLOAD_STREAMING)
                            safeScrollToBottom()
                        }
                    }
                }
        }
    }

    private fun finishGenerationUI(conv: Conversation, assistantMsg: ChatMessage) {
        isGenerating = false
        userInputEt.isEnabled = true
        userInputEt.hint = "输入消息与端侧模型对话..."
        val hasInput = !userInputEt.text.isNullOrBlank() || currentAttachedImagePath != null
        userActionFab.setImageResource(if (hasInput) R.drawable.outline_send_24 else R.drawable.outline_send_24)
        if (!conv.messages.any { it.id == assistantMsg.id }) {
            conv.messages.add(assistantMsg)
        }
        conv.updatedAt = System.currentTimeMillis()
        conversationRepo.saveConversation(conv)
        conversationAdapter.updateData(conversationRepo.getAllConversations())
        val profile = DeviceHardwareProfiler.getProfile(this)
        InferenceForegroundService.updateStatus(
            this,
            "🟢 硬件就绪 | $loadedModelDisplayName",
            "${profile.socModelName} | 前台保活中"
        )
    }

    private fun stopGeneration() {
        if (!isGenerating) return
        engine.cancelGeneration()
        generationJob?.cancel()
        isGenerating = false
        userInputEt.isEnabled = true
        userInputEt.hint = "输入消息与骁龙 NPU 对话..."
        userActionFab.setImageResource(R.drawable.outline_send_24)
        currentConversation?.let { conv ->
            if (currentMessages.isNotEmpty()) {
                val lastMsg = currentMessages.last()
                if (!lastMsg.isUser) {
                    safeNotifyItemChanged(currentMessages.size - 1)
                    finishGenerationUI(conv, lastMsg)
                }
            }
        }
        Toast.makeText(this, "已中断模型推理", Toast.LENGTH_SHORT).show()
    }

    private fun showInferenceErrorRecoveryDialog(errMsg: String) {
        if (!isFinishing && !isDestroyed && activeConfig.accelerator != HardwareAccelerator.CPU_ARM) {
            AlertDialog.Builder(this)
                .setTitle("⚡ 硬件加速异常诊断与恢复")
                .setMessage("检测到硬件加速核心执行受限 (通常由于加速器专用显存超限或模型算子未适配)。\n\n是否一键切换为【ARM Cortex CPU (KleidiAI)】模式？\n• CPU 模式支持任意模型与完整上下文\n• 稳定运行、不限驱动内存\n• 自动保留当前模型与对话")
                .setPositiveButton("一键切换至 CPU 模式") { _, _ ->
                    switchToCpuAndReload()
                }
                .setNegativeButton("稍后手动调整", null)
                .show()
        }
    }

    private fun switchToCpuAndReload() {
        val targetPath = loadedModelPath ?: return
        Toast.makeText(this, "正在临时切换至 CPU 模式并重新载入...", Toast.LENGTH_SHORT).show()
        activeConfig = activeConfig.copy(
            accelerator = HardwareAccelerator.CPU_ARM,
            nGpuLayers = 0
        )
        // Do not permanently save fallback to preferences so hardware acceleration remains default on next run
        loadModelPathDirectly(targetPath, loadedModelDisplayName)
    }

    /**
     * Resolves SAF URI to zero-copy direct path without duplicating or moving files!
     */
    private fun onModelUriSelected(uri: Uri) {
        try {
            val (resolvedPath, displayName) = settingsManager.resolveZeroCopyPath(uri)
            settingsDialogView?.findViewById<EditText>(R.id.et_model_path)?.setText(resolvedPath)
            Toast.makeText(this, "已选模型: $displayName (零复制读取)", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to resolve model uri", e)
            Toast.makeText(this, "文件解析失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun onMmprojUriSelected(uri: Uri) {
        try {
            val (resolvedPath, displayName) = settingsManager.resolveMmprojZeroCopyPath(uri)
            settingsDialogView?.findViewById<EditText>(R.id.et_mmproj_path)?.setText(resolvedPath)
            activeConfig = activeConfig.copy(mmprojPath = resolvedPath)
            settingsManager.setLastSelectedMmprojPath(resolvedPath)
            settingsManager.saveModelConfig(activeConfig)
            Toast.makeText(this, "已选视觉模型: $displayName", Toast.LENGTH_SHORT).show()

            if (isModelReady) {
                lifecycleScope.launch(Dispatchers.IO) {
                    val ok = engine.loadMmproj(resolvedPath)
                    withContext(Dispatchers.Main) {
                        if (ok) {
                            Toast.makeText(this@MainActivity, "多模态视觉组件 (mmproj) 已加载就绪！", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(this@MainActivity, "加载 mmproj 失败，请检查是否与主模型匹配", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to resolve mmproj uri", e)
            Toast.makeText(this, "多模态文件解析失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun onImageUriSelected(uri: Uri) {
        try {
            val cacheFile = File(cacheDir, "input_multimodal_${System.currentTimeMillis()}.jpg")
            contentResolver.openInputStream(uri)?.use { input ->
                cacheFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            if (cacheFile.exists() && cacheFile.length() > 0) {
                currentAttachedImagePath = cacheFile.absolutePath
                currentAttachedImageUri = uri.toString()

                val displayName = queryImageDisplayName(uri) ?: cacheFile.name
                tvImagePreviewName.text = displayName
                ivImagePreviewThumb.setImageURI(uri)
                cardImagePreview.visibility = View.VISIBLE

                if (isGenerating) {
                    userActionFab.setImageResource(R.drawable.outline_send_24)
                }
                Toast.makeText(this, "图片已附加: $displayName", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load attached image", e)
            Toast.makeText(this, "无法加载图片: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun queryImageDisplayName(uri: Uri): String? {
        return try {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex != -1 && cursor.moveToFirst()) {
                    cursor.getString(nameIndex)
                } else null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun loadModelPathDirectly(path: String, displayName: String) {
        if (isModelLoading) {
            Log.w(TAG, "Model is already loading in background, ignoring duplicate request for: $path")
            return
        }

        stopGeneration()

        val (validPath, validDisplayName) = try {
            settingsManager.preparePathForLoad(path)
        } catch (e: Exception) {
            Pair(path, displayName)
        }

        loadedModelPath = validPath
        loadedModelDisplayName = validDisplayName
        settingsManager.setLastSelectedModelPath(validPath)

        isModelLoading = true
        modelLoadingJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Loading model with active config: $validPath (Zero-Copy)")
                engine.loadModel(validPath, activeConfig)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "GGUF 模型加载成功！", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "模型加载失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                isModelLoading = false
            }
        }
    }

    /**
     * Dedicated Model Management & Hardware Settings Dialog
     */
    private fun showModelSettingsDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_model_settings, null)
        settingsDialogView = dialogView

        val tvCurrentInfo = dialogView.findViewById<TextView>(R.id.tv_current_model_info)
        val etModelPath = dialogView.findViewById<EditText>(R.id.et_model_path)
        val btnBrowse = dialogView.findViewById<Button>(R.id.btn_browse_model)
        val llRecent = dialogView.findViewById<LinearLayout>(R.id.ll_recent_models)

        val rbHybrid = dialogView.findViewById<RadioButton>(R.id.rb_hybrid)
        val rbNpu = dialogView.findViewById<RadioButton>(R.id.rb_npu)
        val rbGpu = dialogView.findViewById<RadioButton>(R.id.rb_gpu)
        val rbCpu = dialogView.findViewById<RadioButton>(R.id.rb_cpu)
        val rbTopoSingle = dialogView.findViewById<RadioButton>(R.id.rb_topo_single)
        val rbTopoDual = dialogView.findViewById<RadioButton>(R.id.rb_topo_dual)
        val rbTopoLayerSplit = dialogView.findViewById<RadioButton>(R.id.rb_topo_layer_split)

        val tvLabelLayers = dialogView.findViewById<TextView>(R.id.tv_label_layers)
        val sliderLayers = dialogView.findViewById<Slider>(R.id.slider_offload_layers)
        val etThreads = dialogView.findViewById<EditText>(R.id.et_threads)
        val etCtxSize = dialogView.findViewById<EditText>(R.id.et_ctx_size)
        val etMmprojPath = dialogView.findViewById<EditText>(R.id.et_mmproj_path)
        val btnBrowseMmproj = dialogView.findViewById<Button>(R.id.btn_browse_mmproj)

        // Advanced Mode Views
        val switchAdvancedMode = dialogView.findViewById<MaterialSwitch>(R.id.switch_advanced_mode)
        val llAdvancedContainer = dialogView.findViewById<LinearLayout>(R.id.ll_advanced_container)
        val rgKvCache = dialogView.findViewById<RadioGroup>(R.id.rg_kv_cache)
        val rbKvQ80 = dialogView.findViewById<RadioButton>(R.id.rb_kv_q8_0)
        val rbKvF16 = dialogView.findViewById<RadioButton>(R.id.rb_kv_f16)
        val rbKvQ40 = dialogView.findViewById<RadioButton>(R.id.rb_kv_q4_0)
        val rgFlashAttn = dialogView.findViewById<RadioGroup>(R.id.rg_flash_attn)
        val rbFaDisabled = dialogView.findViewById<RadioButton>(R.id.rb_fa_disabled)
        val rbFaAuto = dialogView.findViewById<RadioButton>(R.id.rb_fa_auto)
        val rbFaEnabled = dialogView.findViewById<RadioButton>(R.id.rb_fa_enabled)
        val etUbatchSize = dialogView.findViewById<EditText>(R.id.et_ubatch_size)
        val etBatchSize = dialogView.findViewById<EditText>(R.id.et_batch_size)
        val switchCoreAffinity = dialogView.findViewById<MaterialSwitch>(R.id.switch_core_affinity)
        val etTemp = dialogView.findViewById<EditText>(R.id.et_temp)
        val etTopP = dialogView.findViewById<EditText>(R.id.et_top_p)
        val etTopK = dialogView.findViewById<EditText>(R.id.et_top_k)
        val etRepeatPenalty = dialogView.findViewById<EditText>(R.id.et_repeat_penalty)
        val switchDma64 = dialogView.findViewById<MaterialSwitch>(R.id.switch_dma64)
        val switchMmap = dialogView.findViewById<MaterialSwitch>(R.id.switch_mmap)
        val switchMlock = dialogView.findViewById<MaterialSwitch>(R.id.switch_mlock)
        val etVmemMb = dialogView.findViewById<EditText>(R.id.et_vmem_mb)

        val btnCancel = dialogView.findViewById<Button>(R.id.btn_cancel_settings)
        val btnApply = dialogView.findViewById<Button>(R.id.btn_apply_reload)

        // Populate current values
        val profile = DeviceHardwareProfiler.getProfile(this)
        val tvHardwareInfo = dialogView.findViewById<TextView>(R.id.tv_device_hardware_info)
        tvHardwareInfo?.text = profile.deviceSummary

        tvCurrentInfo.text = if (loadedModelPath != null) {
            "当前加载: $loadedModelDisplayName\n路径: $loadedModelPath"
        } else {
            "当前未加载模型 (请从下方输入路径或点击浏览)"
        }
        etModelPath.setText(loadedModelPath ?: "")
        etMmprojPath?.setText(activeConfig.mmprojPath ?: settingsManager.getLastSelectedMmprojPath() ?: "")

        btnBrowseMmproj?.setOnClickListener {
            checkAndRequestStoragePermission()
            selectMmprojLauncher.launch(arrayOf("*/*"))
        }

        when (activeConfig.accelerator) {
            HardwareAccelerator.HYBRID_COMBINED -> rbHybrid?.isChecked = true
            HardwareAccelerator.NPU_HEXAGON -> rbNpu.isChecked = true
            HardwareAccelerator.GPU_GENERIC -> rbGpu?.isChecked = true
            HardwareAccelerator.CPU_ARM -> rbCpu.isChecked = true
        }

        rbNpu.setOnClickListener {
            if (switchAdvancedMode?.isChecked != true) {
                sliderLayers.value = 99f
                tvLabelLayers.text = "加速器卸载层数: 99 (最大卸载层数)"
            }
        }
        rbGpu?.setOnClickListener {
            if (switchAdvancedMode?.isChecked != true) {
                sliderLayers.value = 99f
                tvLabelLayers.text = "加速器卸载层数: 99 (最大卸载层数)"
            }
        }
        rbHybrid?.setOnClickListener {
            if (switchAdvancedMode?.isChecked != true) {
                sliderLayers.value = profile.recommendedHybridLayers.toFloat()
                tvLabelLayers.text = "加速器卸载层数: ${profile.recommendedHybridLayers} (异构混合推荐)"
            }
        }
        rbCpu.setOnClickListener {
            if (switchAdvancedMode?.isChecked != true) {
                sliderLayers.value = 0f
                tvLabelLayers.text = "加速器卸载层数: 0 (CPU 模式)"
            }
        }

        when (activeConfig.npuTopology) {
            NpuTopology.SINGLE_CORE -> rbTopoSingle.isChecked = true
            NpuTopology.DUAL_CORE_GROUPED -> rbTopoDual.isChecked = true
            NpuTopology.LAYER_SPLIT -> rbTopoLayerSplit.isChecked = true
        }

        sliderLayers.value = activeConfig.nGpuLayers.coerceIn(0, 99).toFloat()
        tvLabelLayers.text = "加速器卸载层数: ${sliderLayers.value.toInt()} (${if (sliderLayers.value.toInt() >= 99) "全部层" else "部分卸载"})"
        sliderLayers.addOnChangeListener { _, value, _ ->
            tvLabelLayers.text = "加速器卸载层数: ${value.toInt()} (${if (value.toInt() >= 99) "全部层" else "部分卸载"})"
        }

        etThreads.setText(activeConfig.nThreads.toString())
        etCtxSize.setText(activeConfig.nCtx.toString())

        // Advanced Mode initialization
        switchAdvancedMode?.isChecked = activeConfig.isAdvancedMode
        llAdvancedContainer?.visibility = if (activeConfig.isAdvancedMode) View.VISIBLE else View.GONE
        switchAdvancedMode?.setOnCheckedChangeListener { _, isChecked ->
            llAdvancedContainer?.visibility = if (isChecked) View.VISIBLE else View.GONE
        }

        when (activeConfig.kvCacheType) {
            KvCacheType.Q8_0 -> rbKvQ80?.isChecked = true
            KvCacheType.F16 -> rbKvF16?.isChecked = true
            KvCacheType.Q4_0 -> rbKvQ40?.isChecked = true
        }

        when (activeConfig.flashAttnMode) {
            FlashAttnMode.DISABLED -> rbFaDisabled?.isChecked = true
            FlashAttnMode.AUTO -> rbFaAuto?.isChecked = true
            FlashAttnMode.ENABLED -> rbFaEnabled?.isChecked = true
        }

        etUbatchSize?.setText(activeConfig.ubatchSize.toString())
        etBatchSize?.setText(activeConfig.batchSize.toString())
        switchCoreAffinity?.isChecked = activeConfig.coreAffinity
        etTemp?.setText(activeConfig.temp.toString())
        etTopP?.setText(activeConfig.topP.toString())
        etTopK?.setText(activeConfig.topK.toString())
        etRepeatPenalty?.setText(activeConfig.repeatPenalty.toString())
        switchDma64?.isChecked = activeConfig.dma64
        switchMmap?.isChecked = activeConfig.useMmap
        switchMlock?.isChecked = activeConfig.useMlock
        etVmemMb?.setText(activeConfig.vmemMb.toString())

        // Presets
        val btnPresetHybrid = dialogView.findViewById<Button>(R.id.btn_preset_hybrid_recommended)
        val btnPresetNpuFull = dialogView.findViewById<Button>(R.id.btn_preset_npu_full)
        val btnPresetCpuFull = dialogView.findViewById<Button>(R.id.btn_preset_cpu_full)

        btnPresetHybrid?.setOnClickListener {
            rbHybrid?.isChecked = true
            rbTopoSingle.isChecked = true
            sliderLayers.value = profile.recommendedHybridLayers.toFloat()
            tvLabelLayers.text = "加速器卸载层数: ${profile.recommendedHybridLayers} (异构合并并发 - 内存安全推荐)"
            etThreads.setText(profile.recommendedThreads.toString())
            etCtxSize.setText(profile.recommendedContextSize.toString())
            rbKvQ80?.isChecked = true
            rbFaDisabled?.isChecked = true
            etUbatchSize?.setText("256")
            etBatchSize?.setText("512")
            switchCoreAffinity?.isChecked = true
            switchDma64?.isChecked = true
            etVmemMb?.setText(profile.recommendedVmemMb.toString())
            Toast.makeText(this, "已匹配当前硬件: ${profile.recommendedHybridLayers}层加速+${profile.recommendedThreads}线程 (${profile.recommendedContextSize}上下文)", Toast.LENGTH_SHORT).show()
        }

        btnPresetNpuFull?.setOnClickListener {
            if (profile.socFamily == DeviceHardwareProfiler.SocFamily.SNAPDRAGON) {
                rbNpu.isChecked = true
            } else {
                rbGpu?.isChecked = true
            }
            rbTopoSingle.isChecked = true
            sliderLayers.value = 99f
            tvLabelLayers.text = "加速器卸载层数: 99 (最大卸载层数)"
            etThreads.setText(profile.recommendedThreads.toString())
            etCtxSize.setText("2048")
            rbKvQ80?.isChecked = true
            rbFaAuto?.isChecked = true
            etUbatchSize?.setText("256")
            etBatchSize?.setText("512")
            Toast.makeText(this, "已应用: NPU 硬件加速预设 (2048 上下文 + 99 层卸载)", Toast.LENGTH_SHORT).show()
        }

        btnPresetCpuFull?.setOnClickListener {
            rbCpu.isChecked = true
            sliderLayers.value = 0f
            tvLabelLayers.text = "加速器卸载层数: 0 (CPU 模式)"
            etThreads.setText(profile.recommendedThreads.toString())
            etCtxSize.setText(profile.recommendedContextSize.toString())
            rbFaAuto?.isChecked = true
            Toast.makeText(this, "已应用: CPU 完整模式 (${"%.0f".format(profile.totalRamGb)}GB RAM 无显存限制)", Toast.LENGTH_SHORT).show()
        }

        // Build recent models list
        val recentModels = settingsManager.getRecentModelPaths()
        llRecent.removeAllViews()
        if (recentModels.isEmpty()) {
            val emptyTv = TextView(this).apply {
                text = "暂无最近模型，建议将 GGUF 放置于 /sdcard/Models/ 目录"
                setTextColor(0xFF64748B.toInt())
                textSize = 11f
            }
            llRecent.addView(emptyTv)
        } else {
            for (path in recentModels) {
                val f = File(path)
                val btn = Button(this, null, android.R.attr.borderlessButtonStyle).apply {
                    val label = if (path.startsWith("/proc/self/fd/")) {
                        settingsManager.getLastSelectedDisplayName() ?: "模型 (Direct FD)"
                    } else {
                        "${f.name} (${if (f.exists()) "%.1f GB".format(f.length() / 1e9) else "Direct File"})"
                    }
                    text = label
                    isAllCaps = false
                    textSize = 11f
                    setTextColor(0xFF38BDF8.toInt())
                    setOnClickListener {
                        etModelPath.setText(path)
                    }
                }
                llRecent.addView(btn)
            }
        }

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        btnBrowse.setOnClickListener {
            checkAndRequestStoragePermission()
            selectModelLauncher.launch(arrayOf("*/*"))
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        btnApply.setOnClickListener {
            val targetPath = etModelPath.text.toString().trim()
            if (targetPath.isEmpty()) {
                Toast.makeText(this, "请输入或选择有效的 GGUF 模型文件！", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val targetMmproj = etMmprojPath?.text?.toString()?.trim()?.ifEmpty { null }

            val newAccel = when {
                rbHybrid?.isChecked == true -> HardwareAccelerator.HYBRID_COMBINED
                rbNpu.isChecked -> HardwareAccelerator.NPU_HEXAGON
                rbGpu?.isChecked == true -> HardwareAccelerator.GPU_GENERIC
                else -> HardwareAccelerator.CPU_ARM
            }
            val newTopo = when {
                rbTopoDual.isChecked -> NpuTopology.DUAL_CORE_GROUPED
                rbTopoLayerSplit.isChecked -> NpuTopology.LAYER_SPLIT
                else -> NpuTopology.SINGLE_CORE
            }
            val isAdvanced = switchAdvancedMode?.isChecked ?: false
            val newLayers = if (!isAdvanced) {
                when (newAccel) {
                    HardwareAccelerator.NPU_HEXAGON, HardwareAccelerator.GPU_GENERIC -> 99
                    HardwareAccelerator.CPU_ARM -> 0
                    HardwareAccelerator.HYBRID_COMBINED -> profile.recommendedHybridLayers
                }
            } else {
                sliderLayers.value.toInt()
            }
            val newThreads = etThreads.text.toString().toIntOrNull() ?: 4
            val newCtx = etCtxSize.text.toString().toIntOrNull() ?: 4096

            val newKvCacheType = when {
                rbKvF16?.isChecked == true -> KvCacheType.F16
                rbKvQ40?.isChecked == true -> KvCacheType.Q4_0
                else -> KvCacheType.Q8_0
            }
            val newFlashAttnMode = when {
                rbFaAuto?.isChecked == true -> FlashAttnMode.AUTO
                rbFaEnabled?.isChecked == true -> FlashAttnMode.ENABLED
                else -> FlashAttnMode.DISABLED
            }
            val newUbatch = etUbatchSize?.text?.toString()?.toIntOrNull() ?: 256
            val newBatch = etBatchSize?.text?.toString()?.toIntOrNull() ?: 512
            val newCoreAffinity = switchCoreAffinity?.isChecked ?: true
            val newTemp = etTemp?.text?.toString()?.toFloatOrNull() ?: 0.7f
            val newTopP = etTopP?.text?.toString()?.toFloatOrNull() ?: 0.9f
            val newTopK = etTopK?.text?.toString()?.toIntOrNull() ?: 40
            val newRepeatPenalty = etRepeatPenalty?.text?.toString()?.toFloatOrNull() ?: 1.1f
            val newDma64 = switchDma64?.isChecked ?: true
            val newMmap = switchMmap?.isChecked ?: true
            val newMlock = switchMlock?.isChecked ?: false
            val newVmemMb = etVmemMb?.text?.toString()?.toIntOrNull() ?: 2800

            activeConfig = activeConfig.copy(
                accelerator = newAccel,
                npuTopology = newTopo,
                nGpuLayers = newLayers,
                nThreads = newThreads,
                nCtx = newCtx,
                mmprojPath = targetMmproj,
                isAdvancedMode = isAdvanced,
                kvCacheType = newKvCacheType,
                flashAttnMode = newFlashAttnMode,
                ubatchSize = newUbatch,
                batchSize = newBatch,
                coreAffinity = newCoreAffinity,
                temp = newTemp,
                topP = newTopP,
                topK = newTopK,
                repeatPenalty = newRepeatPenalty,
                dma64 = newDma64,
                useMmap = newMmap,
                useMlock = newMlock,
                vmemMb = newVmemMb
            )
            settingsManager.setLastSelectedMmprojPath(targetMmproj)
            settingsManager.saveModelConfig(activeConfig)

            dialog.dismiss()
            val displayName = if (targetPath.startsWith("/proc/self/fd/")) {
                settingsManager.getLastSelectedDisplayName() ?: "GGUF 模型"
            } else {
                File(targetPath).name
            }
            loadModelPathDirectly(targetPath, displayName)
        }

        dialog.show()
    }

    private fun runNpuBenchmark() {
        if (!isModelReady) {
            Toast.makeText(this, "请先加载模型后再运行 Benchmark！", Toast.LENGTH_SHORT).show()
            return
        }

        btnRunBenchmark.isEnabled = false
        tvBenchmarkResult.text = "正在运行 Snapdragon NPU 压测 (Prompt 128 / Gen 64)..."

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val report = engine.bench(pp = 128, tg = 64, pl = 1, nr = 2)
                withContext(Dispatchers.Main) {
                    tvBenchmarkResult.text = report
                    btnRunBenchmark.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed", e)
                withContext(Dispatchers.Main) {
                    tvBenchmarkResult.text = "Benchmark 失败: ${e.message}"
                    btnRunBenchmark.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {
        monitorJob?.cancel()
        generationJob?.cancel()
        settingsManager.closeActivePfd()
        if (::engine.isInitialized) {
            engine.destroy()
        }
        super.onDestroy()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
