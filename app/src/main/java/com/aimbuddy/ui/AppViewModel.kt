package com.aimbuddy.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import androidx.annotation.StringRes
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.aimbuddy.BuildConfig
import com.aimbuddy.InstalledModel
import com.aimbuddy.InvalidModelArchiveException
import com.aimbuddy.ModelCatalog
import com.aimbuddy.ModelFormat
import com.aimbuddy.ModelProfileSpec
import com.aimbuddy.ModelSource
import com.aimbuddy.ModelStoreRepository
import com.aimbuddy.ModelZipImporter
import com.aimbuddy.NativeBridge
import com.aimbuddy.Project
import com.aimbuddy.AssistState
import com.aimbuddy.R
import com.aimbuddy.RootShell
import com.aimbuddy.ShizukuStatus
import com.aimbuddy.StoreModelDefinition
import com.aimbuddy.StoreNeedsNewerAppException
import com.aimbuddy.DownloadCancelledException
import com.aimbuddy.TouchHelper
import com.aimbuddy.TouchTest
import com.aimbuddy.TransferFailure
import com.aimbuddy.TransferMeter
import com.aimbuddy.TransferProgress
import com.aimbuddy.runTouchTest
import com.aimbuddy.ui.theme.Accent
import com.aimbuddy.ui.theme.ThemeMode
import com.aimbuddy.update.AppUpdater
import com.aimbuddy.update.DifferentSignerException
import com.aimbuddy.update.UpdateInfo
import com.aimbuddy.update.UpdateProblem
import com.aimbuddy.update.UpdateState
import com.aimbuddy.update.UpdateStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/** An imported pair of files that the detector could not read as a model. */
private class UnreadableModelException(
    message: String = "The detector cannot read that model",
    val tfliteDetail: String? = null,
) : IllegalStateException(message)

/** A short message for the player, shown once. */
data class UiMessage(@StringRes val resId: Int, val args: List<Any> = emptyList())

/** How the app's look is set. Kept in the app's preferences. */
data class Appearance(val mode: ThemeMode = ThemeMode.SYSTEM, val accent: Accent = Accent.BLUE)

/**
 * What the models screens show about the store. `icons` holds the picture file of each model that has one, by id,
 * and `saved` says the list is the one kept by an earlier visit, because the store could not be reached now.
 */
sealed interface StoreState {
    data object Idle : StoreState
    data object Loading : StoreState
    data class Ready(
        val models: List<StoreModelDefinition>,
        val icons: Map<String, String> = emptyMap(),
        val saved: Boolean = false,
    ) : StoreState
    data object Failed : StoreState
    /** The store was read, but its list is written for a newer AimBuddy. */
    data object NeedsNewerApp : StoreState
}

/** The download under way: which model, how far it has got, and whether the player has asked for it to stop. */
data class Transfer(val modelId: String, val progress: TransferProgress = TransferProgress(), val cancelling: Boolean = false)

/**
 * What the models screens show. `transfer` is the model being downloaded, if any. `failedId` is the model whose last
 * download failed and `failure` why, until the player tries again.
 */
data class ModelsState(
    val installed: List<InstalledModel> = emptyList(),
    val activeId: String? = null,
    val transfer: Transfer? = null,
    val failedId: String? = null,
    val failure: TransferFailure? = null,
    val store: StoreState = StoreState.Idle,
) {
    /** The version of the model with this id that is on the phone, or null when it is not there. */
    fun installedVersion(id: String): Int? = installed.firstOrNull { it.id == id }?.version
}

/** The settings as the native side keeps them, read by name. */
class SettingsView(private val values: Map<String, String> = emptyMap()) {
    fun float(name: String, fallback: Float): Float = values[name]?.toFloatOrNull() ?: fallback
    fun int(name: String, fallback: Int): Int = values[name]?.toIntOrNull() ?: fallback
    fun bool(name: String, fallback: Boolean): Boolean = values[name]?.let { it == "1" } ?: fallback

    companion object {
        /** Reads the `name=value` lines the native side returns. */
        fun parse(snapshot: String): SettingsView = SettingsView(
            snapshot.lineSequence()
                .mapNotNull { line -> line.indexOf('=').takeIf { it > 0 }?.let { line.substring(0, it) to line.substring(it + 1).trim() } }
                .toMap()
        )
    }
}

/**
 * Where the two ways of reaching the touchscreen stand. Root can only be known once it has been asked for, by
 * pressing Start or by the touch test, so it is not asked until then.
 */
data class TouchAccess(
    val shizuku: ShizukuStatus = ShizukuStatus.NOT_INSTALLED,
    val root: RootShell.Status = RootShell.Status.NOT_ASKED,
) {
    /** Whether a route is ready to carry the touch helper. */
    val ready: Boolean get() = shizuku == ShizukuStatus.READY || root == RootShell.Status.ALLOWED
}

/**
 * Everything the screens show that is not about the running assist: the models, the settings and the
 * app's look. It outlives the activity, so rotating the phone or changing the language loses nothing.
 */
class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val context: Context get() = getApplication<Application>()
    private val catalog = ModelCatalog(application)
    private val storeRepository = ModelStoreRepository(Project.STORE_URL, File(application.filesDir, "store"))
    private val preferences = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _appearance = MutableStateFlow(readAppearance())
    val appearance: StateFlow<Appearance> = _appearance.asStateFlow()

    private val _settings = MutableStateFlow(SettingsView())
    val settings: StateFlow<SettingsView> = _settings.asStateFlow()

    private val _models = MutableStateFlow(ModelsState())
    val models: StateFlow<ModelsState> = _models.asStateFlow()

    private val _touchAccess = MutableStateFlow(TouchAccess())
    val touchAccess: StateFlow<TouchAccess> = _touchAccess.asStateFlow()

    private val _touchTest = MutableStateFlow<TouchTest>(TouchTest.Idle)
    val touchTest: StateFlow<TouchTest> = _touchTest.asStateFlow()

    private val _messages = MutableSharedFlow<UiMessage>(extraBufferCapacity = 4)
    val messages: SharedFlow<UiMessage> = _messages.asSharedFlow()

    private val updater = AppUpdater(
        context = application,
        // Only a debug build may look for updates anywhere but GitHub (see UPDATE_API_OVERRIDE in app/build.gradle).
        releasesUrl = if (BuildConfig.DEBUG) BuildConfig.UPDATE_API_OVERRIDE.ifEmpty { Project.RELEASES_URL } else Project.RELEASES_URL,
        allowInsecure = BuildConfig.DEBUG,
    )
    val updateStatus: StateFlow<UpdateStatus> = UpdateState.status

    private val _autoUpdateCheck = MutableStateFlow(preferences.getBoolean(KEY_AUTO_UPDATE_CHECK, true))
    val autoUpdateCheck: StateFlow<Boolean> = _autoUpdateCheck.asStateFlow()

    // Said once when a download has been checked, so the screen can open the installer without another tap.
    private val _installRequests = MutableSharedFlow<UpdateStatus.ReadyToInstall>(extraBufferCapacity = 1)
    val installRequests: SharedFlow<UpdateStatus.ReadyToInstall> = _installRequests.asSharedFlow()

    private var saveJob: Job? = null
    private val cancelRequested = AtomicBoolean(false)
    private var pendingParam: Uri? = null
    private var pendingName = "model"

    /** The first read of the settings file. Reading the settings again waits for it, or it could show the defaults. */
    private val settingsLoaded: Job

    init {
        refreshModels()
        // Loading the native library takes a moment, so it is not done while the first screen is drawn.
        settingsLoaded = viewModelScope.launch(Dispatchers.Default) {
            NativeBridge.loadSettings(File(application.filesDir, SETTINGS_FILE).path)
            _settings.value = SettingsView.parse(NativeBridge.settingsSnapshot())
        }
    }

    /** Shows a short message to the player. */
    fun post(message: UiMessage) {
        _messages.tryEmit(message)
    }

    // ---------------------------------------------------------------------------------------
    // Look
    // ---------------------------------------------------------------------------------------

    fun setThemeMode(mode: ThemeMode) = updateAppearance { it.copy(mode = mode) }

    fun setAccent(accent: Accent) = updateAppearance { it.copy(accent = accent) }

    private fun updateAppearance(change: (Appearance) -> Appearance) {
        val updated = change(_appearance.value)
        _appearance.value = updated
        preferences.edit()
            .putString(KEY_THEME_MODE, updated.mode.name)
            .putString(KEY_ACCENT, updated.accent.name)
            .apply()
    }

    private fun readAppearance(): Appearance = Appearance(
        mode = runCatching { ThemeMode.valueOf(preferences.getString(KEY_THEME_MODE, null) ?: "") }.getOrDefault(ThemeMode.SYSTEM),
        accent = runCatching { Accent.valueOf(preferences.getString(KEY_ACCENT, null) ?: "") }.getOrDefault(Accent.BLUE),
    )

    // ---------------------------------------------------------------------------------------
    // Updates
    // ---------------------------------------------------------------------------------------

    /** Whether the app looks for a newer version by itself when it starts. */
    fun setAutoUpdateCheck(enabled: Boolean) {
        _autoUpdateCheck.value = enabled
        preferences.edit().putBoolean(KEY_AUTO_UPDATE_CHECK, enabled).apply()
    }

    /**
     * Looks on GitHub for a newer version. The check that runs by itself happens once each time the
     * app starts, and only when the player allows it; the player can ask for one at any time.
     */
    fun checkForUpdate(manual: Boolean) {
        val current = UpdateState.status.value
        val busy = current is UpdateStatus.Checking || current is UpdateStatus.Downloading ||
            current is UpdateStatus.ReadyToInstall || current is UpdateStatus.Installing
        if (busy || (!manual && (!_autoUpdateCheck.value || current !is UpdateStatus.Unknown))) {
            return
        }
        UpdateState.set(UpdateStatus.Checking)
        work.launch {
            val result = withContext(Dispatchers.IO) { runCatching { updater.findUpdate() } }
            result.exceptionOrNull()?.let { Log.w(TAG, "The update check failed: ${it.message}", it) }
            UpdateState.set(
                result.fold(
                    { info -> if (info != null) UpdateStatus.Available(info) else UpdateStatus.UpToDate },
                    { UpdateStatus.Failed(UpdateProblem.UNREACHABLE, null) },
                )
            )
        }
    }

    /** Downloads the update that is on offer (or that failed) and checks it; the screen then installs it. */
    fun startUpdate() {
        val info: UpdateInfo = when (val status = UpdateState.status.value) {
            is UpdateStatus.Available -> status.info
            is UpdateStatus.Failed -> status.info ?: return
            else -> return
        }
        UpdateState.set(UpdateStatus.Downloading(info, 0f))
        work.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { updater.download(info) { progress -> UpdateState.set(UpdateStatus.Downloading(info, progress)) } }
            }
            result.onSuccess { file ->
                val ready = UpdateStatus.ReadyToInstall(info, file)
                UpdateState.set(ready)
                _installRequests.tryEmit(ready)
            }.onFailure { error ->
                Log.w(TAG, "The update could not be downloaded: ${error.message}", error)
                val problem = when (error) {
                    is DifferentSignerException -> UpdateProblem.DIFFERENT_SIGNER
                    is SecurityException -> UpdateProblem.NOT_VERIFIED
                    else -> UpdateProblem.UNREACHABLE
                }
                UpdateState.set(UpdateStatus.Failed(problem, info))
            }
        }
    }

    /** Hands a downloaded update to the system's installer. */
    fun installUpdate(ready: UpdateStatus.ReadyToInstall) {
        UpdateState.set(UpdateStatus.Installing(ready.info))
        work.launch {
            val result = withContext(Dispatchers.IO) { runCatching { updater.install(ready.file) } }
            result.exceptionOrNull()?.let {
                Log.w(TAG, "The update could not be handed to the installer: ${it.message}", it)
                UpdateState.set(UpdateStatus.Failed(UpdateProblem.NOT_INSTALLED, ready.info))
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Settings
    // ---------------------------------------------------------------------------------------

    /** Re-reads the settings; the assist's menu may have changed them since the screen last looked. */
    fun refreshSettings() {
        viewModelScope.launch(Dispatchers.Default) {
            settingsLoaded.join()
            _settings.value = SettingsView.parse(NativeBridge.settingsSnapshot())
        }
    }

    fun setSetting(name: String, value: String) {
        NativeBridge.setSetting(name, value)
        _settings.value = SettingsView.parse(NativeBridge.settingsSnapshot())
        if (name == "touchBackend") {
            forgetTouchTest()
        }
        saveSoon()
    }

    /** Changes several settings at once, so the screens redraw once. */
    fun setSettings(values: List<Pair<String, String>>) {
        values.forEach { (name, value) -> NativeBridge.setSetting(name, value) }
        _settings.value = SettingsView.parse(NativeBridge.settingsSnapshot())
        saveSoon()
    }

    fun setSetting(name: String, value: Boolean) = setSetting(name, if (value) "1" else "0")

    fun setSetting(name: String, value: Float) = setSetting(name, value.toString())

    fun setSetting(name: String, value: Int) = setSetting(name, value.toString())

    fun resetSettings() {
        NativeBridge.resetSettings()
        _settings.value = SettingsView.parse(NativeBridge.settingsSnapshot())
        forgetTouchTest()
        saveSoon()
    }

    /** Saves the settings once they have stopped changing, so an edit made while the assist is off is not lost if the app is closed. */
    private fun saveSoon() {
        saveJob?.cancel()
        saveJob = viewModelScope.launch(Dispatchers.IO) {
            delay(SAVE_DELAY_MS)
            NativeBridge.saveSettings()
        }
    }

    /** Saves any change that has not been written yet, for when the screen goes out of sight. */
    fun saveSettingsNow() {
        saveJob?.cancel()
        NativeBridge.saveSettings()
    }

    // ---------------------------------------------------------------------------------------
    // Touch access
    // ---------------------------------------------------------------------------------------

    fun updateTouchAccess(shizuku: ShizukuStatus, root: RootShell.Status) {
        _touchAccess.value = TouchAccess(shizuku, root)
    }

    /**
     * Starts the touch helper once and stops it again, to say whether this phone lets AimBuddy move the aim. It is
     * not run while the assist is, which has the touchscreen. The helper is stopped even if the screen is left
     * meanwhile: the work runs to its end, and a press of Start is refused until then (see [touchTest]). The result
     * is dropped when the route was changed while the test ran, because it answers for a choice that is gone.
     */
    fun testTouchAccess() {
        if (_touchTest.value == TouchTest.Running) {
            return
        }
        if (AssistState.state.value.phase != AssistState.Phase.IDLE) {
            post(UiMessage(R.string.touch_test_stop_first))
            return
        }
        _touchTest.value = TouchTest.Running
        viewModelScope.launch(Dispatchers.IO) {
            // Whatever happens below, the state leaves Running: a test that never ended would keep Start refused for good.
            var tested = -1
            var result: TouchTest = TouchTest.Idle
            try {
                settingsLoaded.join()
                tested = NativeBridge.touchBackend()
                val helper = TouchHelper(context)
                result = runTouchTest(tested, helper::start, helper::stop)
            } finally {
                val stillChosen = tested >= 0 && runCatching { NativeBridge.touchBackend() == tested }.getOrDefault(false)
                _touchTest.value = if (stillChosen) result else TouchTest.Idle
                // Root is known to be allowed, or not, only once it has been asked for.
                _touchAccess.update { it.copy(root = RootShell.status()) }
            }
        }
    }

    /** Clears the result of the last test, which is about a choice that has since changed. A test under way is left alone. */
    private fun forgetTouchTest() {
        if (_touchTest.value != TouchTest.Running) {
            _touchTest.value = TouchTest.Idle
        }
    }

    // ---------------------------------------------------------------------------------------
    // Models
    // ---------------------------------------------------------------------------------------

    /** Re-reads which models are installed and which one is in use. */
    fun refreshModels() {
        catalog.tidy()
        migrateLegacyModel()
        _models.update {
            it.copy(installed = catalog.getInstalledModels(), activeId = catalog.getActiveModel()?.id)
        }
    }

    fun useModel(id: String) {
        if (catalog.setActiveModel(id)) {
            refreshModels()
            _messages.tryEmit(UiMessage(R.string.models_next_start))
        }
    }

    fun deleteModel(id: String) {
        catalog.deleteModel(id)
        refreshModels()
    }

    /** Loads the store's list of models in the background, then the pictures of the ones that have one. */
    fun loadStore() {
        if (_models.value.store is StoreState.Loading) {
            return
        }
        _models.update { it.copy(store = StoreState.Loading) }
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { storeRepository.fetchModels() } }
            _models.update {
                it.copy(store = result.fold(
                    { list -> StoreState.Ready(list.models, saved = list.saved) },
                    { error -> if (error is StoreNeedsNewerAppException) StoreState.NeedsNewerApp else StoreState.Failed },
                ))
            }
            result.exceptionOrNull()?.let { Log.w(TAG, "The store could not be read: ${it.message}") }
            result.getOrNull()?.let { loadIcons(it.models) }
        }
    }

    /** Fetches the picture of each model that has one, showing each as it arrives. */
    private suspend fun loadIcons(models: List<StoreModelDefinition>) {
        val icons = mutableMapOf<String, String>()
        for (model in models) {
            val file = withContext(Dispatchers.IO) { storeRepository.iconFor(model) } ?: continue
            icons[model.id] = file.absolutePath
            _models.update { state ->
                val store = state.store
                if (store is StoreState.Ready) state.copy(store = store.copy(icons = icons.toMap())) else state
            }
        }
    }

    /** Asked for by the player: the download under way stops at its next read and leaves nothing behind. */
    fun cancelDownload() {
        if (_models.value.transfer == null) {
            return
        }
        cancelRequested.set(true)
        _models.update { state -> state.copy(transfer = state.transfer?.copy(cancelling = true)) }
        work.launch(Dispatchers.IO) { storeRepository.abort() }   // closing a socket is not done on the main thread
    }

    /** Downloads a model from the store, or the newer version of one that is installed, and makes it the one in use. */
    fun downloadModel(model: StoreModelDefinition) {
        if (_models.value.transfer != null) {
            return
        }
        val wasInUse = _models.value.activeId == model.id
        val isNew = _models.value.installedVersion(model.id) == null
        val isFirstModel = _models.value.installed.isEmpty()
        cancelRequested.set(false)
        _models.update {
            it.copy(
                transfer = Transfer(model.id, TransferProgress(bytesTotal = model.totalSizeBytes)),
                failedId = null,
                failure = null,
            )
        }
        work.launch {
            val meter = TransferMeter()
            var shownAtMs = 0L
            val installed = withContext(Dispatchers.IO) {
                runCatching {
                    val id = catalog.sanitizeModelId(model.id)
                    val files = storeRepository.install(model, File(context.filesDir, "models/$id"), { cancelRequested.get() }) { done, total ->
                        // The screen is told a few times a second, not at every read of the network.
                        val nowMs = SystemClock.elapsedRealtime()
                        if (done == total || nowMs - shownAtMs >= PROGRESS_INTERVAL_MS) {
                            shownAtMs = nowMs
                            val progress = TransferProgress(done, total, meter.add(nowMs, done))
                            _models.update { state -> state.copy(transfer = state.transfer?.copy(progress = progress)) }
                        }
                    }
                    InstalledModel(
                        id = id,
                        title = model.title,
                        description = model.description,
                        source = ModelSource.STORE,
                        paramPath = files.param?.absolutePath.orEmpty(),
                        binPath = files.bin?.absolutePath.orEmpty(),
                        format = model.format,
                        tflitePath = files.tflite?.absolutePath,
                        totalSizeBytes = (files.param?.length() ?: 0L) + (files.bin?.length() ?: 0L) +
                            (files.tflite?.length() ?: 0L),
                        version = model.version,
                    )
                }
            }
            installed.onSuccess {
                // A new model becomes the one in use; an update only replaces the one that was in use.
                catalog.addOrUpdateModel(it, makeActive = isNew || wasInUse)
                // The first model has the screen's own "ready to go" line; a message on top of it would say it twice.
                if (!isFirstModel) {
                    _messages.tryEmit(UiMessage(R.string.models_next_start))
                }
            }.onFailure { error ->
                if (error is DownloadCancelledException) {
                    Log.i(TAG, "Download cancelled")
                } else {
                    Log.w(TAG, "Download failed: ${error.message}")
                    _models.update { it.copy(failedId = model.id, failure = TransferFailure.of(error)) }
                }
            }
            _models.update { it.copy(transfer = null) }
            refreshModels()
        }
    }

    /** Imports a TFLite/ZIP model, or remembers the `.param` file until its matching `.bin` is picked. */
    fun onModelFilePicked(uri: Uri): Boolean {
        val name = displayName(uri)
        val mimeType = context.contentResolver.getType(uri)?.lowercase(Locale.ROOT)
        if (name.endsWith(".zip", ignoreCase = true) ||
            mimeType == "application/zip" ||
            mimeType == "application/x-zip-compressed"
        ) {
            importZip(uri, name)
            return true
        }
        val looksLikeTFLiteByName = name.endsWith(".tflite", ignoreCase = true)
        val looksLikeRenamedTFLite = name.endsWith(".bin", ignoreCase = true) && hasTFLiteHeader(uri)
        if (looksLikeTFLiteByName || looksLikeRenamedTFLite) {
            val importName = if (looksLikeRenamedTFLite) {
                "${name.substringBeforeLast('.', name)}.tflite"
            } else {
                name
            }
            importTFLite(uri, importName)
            return true
        }
        pendingParam = uri
        pendingName = name.substringBeforeLast('.', "model")
        return false
    }

    /** Copies the picked `.param` and `.bin` into the app's storage and makes the pair the model in use. */
    fun onBinPicked(binUri: Uri) {
        val paramUri = pendingParam ?: return
        pendingParam = null
        val name = pendingName
        work.launch {
            val installed = withContext(Dispatchers.IO) {
                val id = catalog.sanitizeModelId("local-$name-${System.currentTimeMillis()}")
                val dir = File(context.filesDir, "models/$id")
                val paramFile = File(dir, "$id.param")
                val binFile = File(dir, "$id.bin")
                runCatching {
                    check(copy(paramUri, paramFile) && copy(binUri, binFile)) { "The files could not be copied" }
                    importedModel(id, name, paramFile, binFile, null)
                }.onFailure { dir.deleteRecursively() }
            }
            finishImport(installed)
        }
    }

    private fun importZip(uri: Uri, fileName: String) {
        val name = fileName.substringBeforeLast('.', fileName).ifBlank { "model" }
        work.launch {
            val installed = withContext(Dispatchers.IO) {
                val idPrefix = catalog.sanitizeModelId("local-$name-${System.currentTimeMillis()}")
                val modelsDir = File(context.filesDir, "models")
                runCatching {
                    val input = context.contentResolver.openInputStream(uri)
                        ?: throw IOException("Could not open the model ZIP")
                    val files = input.use { ModelZipImporter.extractAll(it, modelsDir, idPrefix) }
                    try {
                        files.mapIndexed { index, modelFiles ->
                            val modelName = if (files.size == 1) {
                                name
                            } else {
                                modelFiles.modelName.ifBlank { "$name ${index + 1}" }
                            }
                            if (modelFiles.tflite != null) {
                                importedTFLiteModel(
                                    modelFiles.modelId,
                                    modelName,
                                    modelFiles.tflite,
                                    modelFiles.description,
                                    modelFiles.tfliteFileName ?: modelName,
                                )
                            } else {
                                importedModel(
                                    modelFiles.modelId,
                                    modelName,
                                    checkNotNull(modelFiles.param),
                                    checkNotNull(modelFiles.bin),
                                    modelFiles.description,
                                )
                            }
                        }
                    } catch (error: Throwable) {
                        files.mapNotNull { it.param?.parentFile ?: it.tflite?.parentFile }
                            .distinct()
                            .forEach { it.deleteRecursively() }
                        throw error
                    }
                }
            }
            finishBulkImport(installed)
        }
    }

    private fun importTFLite(uri: Uri, fileName: String) {
        val name = fileName.substringBeforeLast('.', fileName).ifBlank { "model" }
        work.launch {
            val installed = withContext(Dispatchers.IO) {
                val id = catalog.sanitizeModelId("local-$name-${System.currentTimeMillis()}")
                val dir = File(context.filesDir, "models/$id")
                runCatching {
                    val modelFile = File(dir, "$id.tflite")
                    copyTFLite(uri, modelFile)
                    importedTFLiteModel(id, name, modelFile, null, fileName)
                }.onFailure { dir.deleteRecursively() }
            }
            finishImport(installed)
        }
    }

    private fun importedModel(
        id: String,
        name: String,
        paramFile: File,
        binFile: File,
        descriptionFile: File?,
    ): InstalledModel {
        // The metadata must be used for validation too: end-to-end exports and custom input sizes cannot
        // be probed correctly with the default profile.
        val profile = if (descriptionFile == null) {
            ModelProfileSpec.UNKNOWN
        } else {
            try {
                ModelProfileSpec.parse(descriptionFile.readText())
            } catch (e: JSONException) {
                throw InvalidModelArchiveException(
                    "The model JSON is invalid: ${e.message}",
                    InvalidModelArchiveException.Reason.INVALID_METADATA,
                )
            }
        }
        if (!NativeBridge.canReadModel(
                paramFile.absolutePath,
                binFile.absolutePath,
                profile.roles,
                profile.endToEnd,
                profile.inputSize,
                profile.normalizedCoordinates,
            )
        ) {
            throw UnreadableModelException()
        }
        return InstalledModel(
            id = id,
            title = name,
            description = "",
            source = ModelSource.LOCAL,
            paramPath = paramFile.absolutePath,
            binPath = binFile.absolutePath,
            totalSizeBytes = paramFile.length() + binFile.length(),
        )
    }

    private fun importedTFLiteModel(
        id: String,
        name: String,
        modelFile: File,
        descriptionFile: File?,
        profileName: String,
    ): InstalledModel {
        val profile = if (descriptionFile != null) {
            try {
                ModelProfileSpec.parse(descriptionFile.readText())
            } catch (e: JSONException) {
                throw InvalidModelArchiveException(
                    "The model JSON is invalid: ${e.message}",
                    InvalidModelArchiveException.Reason.INVALID_METADATA,
                )
            }
        } else {
            ModelProfileSpec.forTFLiteFileName(profileName) ?: ModelProfileSpec.UNKNOWN
        }
        val rejection = if (!hasTFLiteHeader(modelFile)) {
            "The file has no TFL3 header. Import a complete TensorFlow Lite model, not a PyTorch checkpoint or an NCNN .bin file."
        } else {
            NativeBridge.canReadTFLiteModel(
                modelFile.absolutePath,
                profile.roles,
                profile.endToEnd,
                profile.inputSize,
                profile.normalizedCoordinates,
            )
        }
        if (rejection != null) {
            throw UnreadableModelException(rejection, tfliteDetail = rejection)
        }
        if (descriptionFile == null && profile != ModelProfileSpec.UNKNOWN) {
            val description = JSONObject()
                .put("roles", profile.roles)
                .put("endToEnd", profile.endToEnd)
            if (profile.inputSize > 0) description.put("inputSize", profile.inputSize)
            profile.normalizedCoordinates?.let { description.put("normalizedCoordinates", it) }
            File(modelFile.parentFile, "$id.json").writeText(description.toString())
        }
        return InstalledModel(
            id = id,
            title = name,
            description = "",
            source = ModelSource.LOCAL,
            paramPath = "",
            binPath = "",
            totalSizeBytes = modelFile.length(),
            format = ModelFormat.TFLITE,
            tflitePath = modelFile.absolutePath,
        )
    }

    private fun finishImport(installed: Result<InstalledModel>) {
        val isFirstModel = _models.value.installed.isEmpty()
        installed.onSuccess {
            catalog.addOrUpdateModel(it, makeActive = true)
            if (!isFirstModel) {
                _messages.tryEmit(UiMessage(R.string.models_next_start))
            }
        }.onFailure { error ->
            Log.w(TAG, "Import failed: ${error.message}")
            _messages.tryEmit(UiMessage(importErrorMessage(error)))
        }
        refreshModels()
    }

    private fun importErrorMessage(error: Throwable): Int {
        if (error is UnreadableModelException) {
            return error.tfliteDetail?.let {
                _messages.tryEmit(UiMessage(R.string.models_import_tflite_detail, listOf(it)))
                R.string.models_import_unreadable
            } ?: R.string.models_import_unreadable
            return R.string.models_import_unreadable
        }
        if (error is InvalidModelArchiveException) {
            return when (error.reason) {
                InvalidModelArchiveException.Reason.INVALID_ARCHIVE -> R.string.models_import_zip_invalid
                InvalidModelArchiveException.Reason.AMBIGUOUS_METADATA -> R.string.models_import_zip_metadata_ambiguous
                InvalidModelArchiveException.Reason.INVALID_METADATA -> R.string.models_import_zip_metadata_invalid
                InvalidModelArchiveException.Reason.UNSAFE_PATH -> R.string.models_import_zip_unsafe_path
                InvalidModelArchiveException.Reason.DUPLICATE_PATH -> R.string.models_import_zip_duplicate_path
                InvalidModelArchiveException.Reason.ARCHIVE_LIMIT -> R.string.models_import_zip_limit
            }
        }
        return R.string.models_import_failed
    }

    private fun finishBulkImport(installed: Result<List<InstalledModel>>) {
        val isFirstModel = _models.value.installed.isEmpty()
        installed.onSuccess { models ->
            models.forEachIndexed { index, model ->
                catalog.addOrUpdateModel(model, makeActive = index == models.lastIndex)
            }
            if (!isFirstModel && models.isNotEmpty()) {
                _messages.tryEmit(UiMessage(R.string.models_next_start))
            }
        }.onFailure { error ->
            Log.w(TAG, "Import failed: ${error.message}")
            if (error is UnreadableModelException && error.tfliteDetail != null) {
                _messages.tryEmit(UiMessage(R.string.models_import_tflite_detail, listOf(error.tfliteDetail)))
            } else {
                _messages.tryEmit(UiMessage(importErrorMessage(error)))
            }
        }
        refreshModels()
    }

    private fun copy(from: Uri, to: File): Boolean {
        val directory = to.parentFile ?: return false
        if (!directory.exists() && !directory.mkdirs()) {
            return false
        }
        return context.contentResolver.openInputStream(from)?.use { input ->
            FileOutputStream(to).use { output -> input.copyTo(output) }
            true
        } ?: false
    }

    private fun hasTFLiteHeader(file: File): Boolean {
        val header = ByteArray(TFLITE_HEADER_SIZE)
        return try {
            file.inputStream().use { input ->
                var offset = 0
                while (offset < header.size) {
                    val count = input.read(header, offset, header.size - offset)
                    if (count < 0) break
                    offset += count
                }
                offset == header.size && header.copyOfRange(4, 8).contentEquals(byteArrayOf(
                    TFLITE_MAGIC_0, TFLITE_MAGIC_1, TFLITE_MAGIC_2, TFLITE_MAGIC_3,
                ))
            }
        } catch (error: IOException) {
            false
        }
    }

    private fun copyTFLite(from: Uri, to: File) {
        val directory = to.parentFile ?: throw IOException("Could not create the model folder")
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Could not create the model folder")
        }
        val input = context.contentResolver.openInputStream(from)
            ?: throw IOException("Could not open the TFLite model")
        input.use { source ->
            FileOutputStream(to).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var totalBytes = 0L
                while (true) {
                    val read = source.read(buffer)
                    if (read < 0) break
                    totalBytes += read
                    if (totalBytes > MAX_TFLITE_MODEL_BYTES) {
                        throw IOException("The TFLite model is larger than the 512 MB import limit")
                    }
                    output.write(buffer, 0, read)
                }
            }
        }
    }

    private fun hasTFLiteHeader(uri: Uri): Boolean {
        val header = ByteArray(TFLITE_HEADER_SIZE)
        val read = context.contentResolver.openInputStream(uri)?.use { input ->
            var offset = 0
            while (offset < header.size) {
                val count = input.read(header, offset, header.size - offset)
                if (count < 0) break
                offset += count
            }
            offset
        } ?: return false
        return read == header.size &&
            header[4] == TFLITE_MAGIC_0 &&
            header[5] == TFLITE_MAGIC_1 &&
            header[6] == TFLITE_MAGIC_2 &&
            header[7] == TFLITE_MAGIC_3
    }

    private fun displayName(uri: Uri): String {
        var name = "model"
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) {
                name = cursor.getString(index) ?: name
            }
        }
        return name
    }

    /** A model saved by an older version of the app as a single pair of paths is moved into the catalog. */
    private fun migrateLegacyModel() {
        val param = preferences.getString(KEY_LEGACY_PARAM, null)
        val bin = preferences.getString(KEY_LEGACY_BIN, null)
        if (param.isNullOrBlank() || bin.isNullOrBlank() || !File(param).exists() || !File(bin).exists()) {
            return
        }
        catalog.addOrUpdateModel(
            InstalledModel(
                id = catalog.sanitizeModelId("legacy-local-model"),
                title = "Earlier import",
                description = "",
                source = ModelSource.LOCAL,
                paramPath = param,
                binPath = bin,
                totalSizeBytes = File(param).length() + File(bin).length(),
            ),
            makeActive = false,
        )
        preferences.edit().remove(KEY_LEGACY_PARAM).remove(KEY_LEGACY_BIN).apply()
    }

    private companion object {
        /**
         * Where work runs that has to finish even when the screen is closed halfway: checking for an
         * update, downloading, installing and importing. Their results go into state that outlives
         * the screen (the update state, the catalog), so none of them is left half done.
         */
        val work = CoroutineScope(SupervisorJob() + Dispatchers.Default)

        const val TAG = "AppViewModel"
        const val PREFS_NAME = "aimbuddy_prefs"
        const val KEY_THEME_MODE = "theme_mode"
        const val KEY_ACCENT = "accent"
        const val KEY_AUTO_UPDATE_CHECK = "auto_update_check"
        const val KEY_LEGACY_PARAM = "model_param_path"
        const val KEY_LEGACY_BIN = "model_bin_path"
        const val SETTINGS_FILE = "settings.cfg"
        const val SAVE_DELAY_MS = 400L
        const val PROGRESS_INTERVAL_MS = 250L
        const val MAX_TFLITE_MODEL_BYTES = 512L * 1024 * 1024
        const val TFLITE_HEADER_SIZE = 8
        const val TFLITE_MAGIC_0 = 'T'.code.toByte()
        const val TFLITE_MAGIC_1 = 'F'.code.toByte()
        const val TFLITE_MAGIC_2 = 'L'.code.toByte()
        const val TFLITE_MAGIC_3 = '3'.code.toByte()
    }
}
