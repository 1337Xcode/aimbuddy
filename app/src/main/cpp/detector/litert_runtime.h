#ifndef AIMBUDDY_DETECTOR_LITERT_RUNTIME_H
#define AIMBUDDY_DETECTOR_LITERT_RUNTIME_H

#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>

#include <cstdarg>

namespace aimbuddy {

/**
 * The parts of LiteRT's C API that the TFLite detector calls, looked up in the library when it is
 * opened. The app does not link LiteRT: a player who never loads a .tflite model never loads it,
 * and a phone where it cannot be opened can still run NCNN models.
 */
struct LiteRtApi {
    decltype(&TfLiteVersion) version;
    decltype(&TfLiteModelCreateFromFile) modelCreateFromFile;
    decltype(&TfLiteModelDelete) modelDelete;
    decltype(&TfLiteInterpreterOptionsCreate) optionsCreate;
    decltype(&TfLiteInterpreterOptionsDelete) optionsDelete;
    decltype(&TfLiteInterpreterOptionsSetNumThreads) optionsSetNumThreads;
    decltype(&TfLiteInterpreterOptionsAddDelegate) optionsAddDelegate;
    decltype(&TfLiteInterpreterOptionsSetErrorReporter) optionsSetErrorReporter;
    decltype(&TfLiteInterpreterCreate) interpreterCreate;
    decltype(&TfLiteInterpreterDelete) interpreterDelete;
    decltype(&TfLiteInterpreterAllocateTensors) allocateTensors;
    decltype(&TfLiteInterpreterInvoke) invoke;
    decltype(&TfLiteInterpreterGetInputTensorCount) inputTensorCount;
    decltype(&TfLiteInterpreterGetInputTensor) inputTensor;
    decltype(&TfLiteInterpreterGetOutputTensorCount) outputTensorCount;
    decltype(&TfLiteInterpreterGetOutputTensor) outputTensor;
    decltype(&TfLiteTensorType) tensorType;
    decltype(&TfLiteTensorNumDims) tensorNumDims;
    decltype(&TfLiteTensorDim) tensorDim;
    decltype(&TfLiteTensorByteSize) tensorByteSize;
    decltype(&TfLiteTensorData) tensorData;
    decltype(&TfLiteTensorQuantizationParams) tensorQuantization;
};

/// The GPU delegate's C API, from the delegate's own library.
struct LiteRtGpuApi {
    decltype(&TfLiteGpuDelegateOptionsV2Default) optionsDefault;
    decltype(&TfLiteGpuDelegateV2Create) create;
    decltype(&TfLiteGpuDelegateV2Delete) destroy;
};

/// LiteRT, opened the first time it is asked for. Null, with the reason logged, when it cannot be opened.
const LiteRtApi* liteRt();

/// The GPU delegate, opened the same way. Null when it cannot be opened.
const LiteRtGpuApi* liteRtGpu();

/// Passes LiteRT's own error messages (an unsupported operation, a failed delegate) on to the log.
void reportLiteRtError(void* userData, const char* format, va_list args);

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_LITERT_RUNTIME_H
