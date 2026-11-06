#include "litert_runtime.h"

#include <dlfcn.h>

#include "../settings.h"
#include "../utils/logger.h"

namespace aimbuddy {

namespace {

/// The libraries the Gradle build puts into the APK from the LiteRT packages (see unpackLiteRt).
constexpr const char* kLiteRtLibrary = "libtensorflowlite_jni.so";
constexpr const char* kGpuDelegateLibrary = "libtensorflowlite_gpu_jni.so";

/// Looks one function up. dlsym hands every symbol back untyped; the slot's type is the function's declaration.
template <typename Function>
bool bind(void* library, const char* name, Function* slot) {
    *slot = reinterpret_cast<Function>(dlsym(library, name));
    if (*slot == nullptr) {
        LOGE("LiteRT has no %s", name);
        return false;
    }
    return true;
}

/// Opens a library of the app; null, with the reason logged, when it cannot be opened.
void* openLibrary(const char* name) {
    void* library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char* reason = dlerror();
        LOGE("Could not open %s: %s", name, reason != nullptr ? reason : "no reason given");
    }
    return library;
}

bool bindAll(void* library, LiteRtApi& api) {
    return bind(library, "TfLiteVersion", &api.version) &&
           bind(library, "TfLiteModelCreateFromFile", &api.modelCreateFromFile) &&
           bind(library, "TfLiteModelDelete", &api.modelDelete) &&
           bind(library, "TfLiteInterpreterOptionsCreate", &api.optionsCreate) &&
           bind(library, "TfLiteInterpreterOptionsDelete", &api.optionsDelete) &&
           bind(library, "TfLiteInterpreterOptionsSetNumThreads", &api.optionsSetNumThreads) &&
           bind(library, "TfLiteInterpreterOptionsAddDelegate", &api.optionsAddDelegate) &&
           bind(library, "TfLiteInterpreterOptionsSetErrorReporter", &api.optionsSetErrorReporter) &&
           bind(library, "TfLiteInterpreterCreate", &api.interpreterCreate) &&
           bind(library, "TfLiteInterpreterDelete", &api.interpreterDelete) &&
           bind(library, "TfLiteInterpreterAllocateTensors", &api.allocateTensors) &&
           bind(library, "TfLiteInterpreterInvoke", &api.invoke) &&
           bind(library, "TfLiteInterpreterGetInputTensorCount", &api.inputTensorCount) &&
           bind(library, "TfLiteInterpreterGetInputTensor", &api.inputTensor) &&
           bind(library, "TfLiteInterpreterGetOutputTensorCount", &api.outputTensorCount) &&
           bind(library, "TfLiteInterpreterGetOutputTensor", &api.outputTensor) &&
           bind(library, "TfLiteTensorType", &api.tensorType) &&
           bind(library, "TfLiteTensorNumDims", &api.tensorNumDims) &&
           bind(library, "TfLiteTensorDim", &api.tensorDim) &&
           bind(library, "TfLiteTensorByteSize", &api.tensorByteSize) &&
           bind(library, "TfLiteTensorData", &api.tensorData) &&
           bind(library, "TfLiteTensorQuantizationParams", &api.tensorQuantization) &&
           bind(library, "TfLiteXNNPackDelegateOptionsDefault", &api.xnnpackOptionsDefault) &&
           bind(library, "TfLiteXNNPackDelegateCreate", &api.xnnpackCreate) &&
           bind(library, "TfLiteXNNPackDelegateDelete", &api.xnnpackDelete);
}

bool bindAll(void* library, LiteRtGpuApi& api) {
    return bind(library, "TfLiteGpuDelegateOptionsV2Default", &api.optionsDefault) &&
           bind(library, "TfLiteGpuDelegateV2Create", &api.create) &&
           bind(library, "TfLiteGpuDelegateV2Delete", &api.destroy);
}

/// Opens a library and binds its functions. The library stays open for the life of the process.
template <typename Api>
bool open(const char* name, Api& api) {
    void* library = openLibrary(name);
    if (library == nullptr) {
        return false;
    }
    if (!bindAll(library, api)) {
        dlclose(library);
        return false;
    }
    return true;
}

}  // namespace

const LiteRtApi* liteRt() {
    // Opened once, by whichever thread asks first; C++ makes the other threads wait for it.
    static LiteRtApi api;
    static const bool opened = open(kLiteRtLibrary, api);
    return opened ? &api : nullptr;
}

const LiteRtGpuApi* liteRtGpu() {
    static LiteRtGpuApi api;
    static const bool opened = open(kGpuDelegateLibrary, api);
    return opened ? &api : nullptr;
}

void reportLiteRtError(void* /* userData */, const char* format, va_list args) {
#if ENABLE_WARN_LOGGING
    __android_log_vprint(ANDROID_LOG_WARN, Config::LOG_TAG, format, args);
#else
    (void)format;
    (void)args;
#endif
}

}  // namespace aimbuddy
