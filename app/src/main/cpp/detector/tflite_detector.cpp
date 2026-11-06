#include "tflite_detector.h"

#include <ncnn/mat.h>

#include <chrono>
#include <cmath>

#include "../core/output_agreement.h"
#include "../settings.h"
#include "../utils/logger.h"
#include "backend_timing.h"

namespace aimbuddy {

namespace {

/// Bytes per pixel of the RGBA pictures the capture hands over.
constexpr int kRgbaBytes = 4;

/// Values per pixel of a picture input (red, green, blue).
constexpr int kRgbChannels = 3;

/// The grey of the picture used to probe and time the model: 0.45 of full brightness, as for NCNN models.
constexpr uint8_t kProbeGrey = 115;

/// The most dimensions a picture input or a detection output has.
constexpr int kMaxDims = 4;

/// How LiteRT stores a tensor's values, in the terms the picture writer and the output reader use. False for a type they do not handle.
bool toElement(TfLiteType type, TensorElement* element) {
    switch (type) {
        case kTfLiteFloat32:
            *element = TensorElement::kFloat32;
            return true;
        case kTfLiteInt8:
            *element = TensorElement::kInt8;
            return true;
        case kTfLiteUInt8:
            *element = TensorElement::kUInt8;
            return true;
        default:
            return false;
    }
}

/// Copies a tensor's dimensions out. False when it has more than kMaxDims of them.
bool readDims(const LiteRtApi& api, const TfLiteTensor* tensor, int* dims, int* count) {
    *count = api.tensorNumDims(tensor);
    if (*count < 0 || *count > kMaxDims) {
        return false;
    }
    for (int i = 0; i < *count; ++i) {
        dims[i] = api.tensorDim(tensor, i);
    }
    return true;
}

Quantization toQuantization(const TfLiteQuantizationParams& params) {
    Quantization quantization;
    quantization.scale = params.scale;
    quantization.zeroPoint = params.zero_point;
    return quantization;
}

/// A side by side RGBA picture of one grey.
std::vector<uint8_t> greyPicture(int side) {
    return std::vector<uint8_t>(static_cast<size_t>(side) * side * kRgbaBytes, kProbeGrey);
}

/// One channel of the picture the CPU and the GPU are compared on, as a byte.
uint8_t agreementByte(int x, int y, int channel) {
    return static_cast<uint8_t>(std::lround(agreementPictureValue(x, y, channel) * 255.0f));
}

/// The picture the CPU and the GPU are compared on (see agreementPictureValue), as RGBA bytes.
std::vector<uint8_t> agreementPicture(int side) {
    std::vector<uint8_t> picture(static_cast<size_t>(side) * side * kRgbaBytes, 255);
    uint8_t* pixel = picture.data();
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            pixel[0] = agreementByte(x, y, 0);
            pixel[1] = agreementByte(x, y, 1);
            pixel[2] = agreementByte(x, y, 2);
            pixel += kRgbaBytes;
        }
    }
    return picture;
}

/**
 * A GPU delegate tuned for a model run on every frame: half precision allowed, lowest latency first,
 * and compiled for sustained speed. OpenCL only: the OpenGL backend is slower and its context belongs
 * to the thread that made it. Null when the phone has no usable delegate.
 */
TfLiteDelegate* createGpuDelegate() {
    const LiteRtGpuApi* gpu = liteRtGpu();
    if (gpu == nullptr) {
        return nullptr;
    }
    TfLiteGpuDelegateOptionsV2 options = gpu->optionsDefault();
    options.is_precision_loss_allowed = 1;
    options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
    return gpu->create(&options);
}

}  // namespace

bool TfliteDetector::load(const ModelFiles& files, const ModelProfile& profile) {
    unload();
    api_ = liteRt();
    if (api_ == nullptr) {
        LOGE("LiteRT could not be opened, so a .tflite model cannot be used");
        return false;
    }
    model_ = api_->modelCreateFromFile(files.modelPath.c_str());
    if (model_ == nullptr) {
        LOGE("LiteRT %s cannot read %s as a model", api_->version(), files.modelPath.c_str());
        return false;
    }
    if (!build(cpu_, false) || !readInput(cpu_, profile) || !readOutput(cpu_)) {
        unload();
        return false;
    }
    // A run on a flat grey picture shows how the output is laid out and in what units.
    const int size = inputWriter_.spec().size;
    const std::vector<uint8_t> probe = greyPicture(size);
    if (!run(cpu_, probe.data(), size * kRgbaBytes) ||
        !learnOutput(outputValues(cpu_), outputRows_, outputCols_, size, profile)) {
        LOGE("The model did not produce a readable detection tensor");
        unload();
        return false;
    }
    active_ = &cpu_;
    loaded_ = true;
    LOGI("LiteRT %s loaded the model for the CPU", api_->version());
    return true;
}

void TfliteDetector::unload() {
    if (api_ != nullptr) {
        release(gpu_);
        release(cpu_);
        if (model_ != nullptr) {
            api_->modelDelete(model_);
        }
    }
    model_ = nullptr;
    active_ = nullptr;
    loaded_ = false;
}

bool TfliteDetector::build(Engine& engine, bool onGpu) {
    TfLiteInterpreterOptions* options = api_->optionsCreate();
    if (options == nullptr) {
        return false;
    }
    api_->optionsSetNumThreads(options, fastCoreThreadCount());
    api_->optionsSetErrorReporter(options, reportLiteRtError, nullptr);
    if (onGpu) {
        engine.gpuDelegate = createGpuDelegate();
        if (engine.gpuDelegate == nullptr) {
            api_->optionsDelete(options);
            return false;
        }
        api_->optionsAddDelegate(options, engine.gpuDelegate);
    }
    engine.interpreter = api_->interpreterCreate(model_, options);
    api_->optionsDelete(options);   // the interpreter keeps what it needs of them
    if (engine.interpreter == nullptr || api_->allocateTensors(engine.interpreter) != kTfLiteOk) {
        release(engine);
        return false;
    }
    engine.input = api_->inputTensor(engine.interpreter, 0);
    engine.output = api_->outputTensor(engine.interpreter, 0);
    if (engine.input == nullptr || engine.output == nullptr) {
        release(engine);
        return false;
    }
    return true;
}

void TfliteDetector::release(Engine& engine) {
    if (engine.interpreter != nullptr) {
        api_->interpreterDelete(engine.interpreter);
    }
    // The delegate must outlive the interpreter that runs on it.
    if (engine.gpuDelegate != nullptr) {
        liteRtGpu()->destroy(engine.gpuDelegate);
    }
    engine = Engine();
}

bool TfliteDetector::readInput(const Engine& engine, const ModelProfile& declared) {
    const int inputs = api_->inputTensorCount(engine.interpreter);
    if (inputs != 1) {
        LOGE("The model takes %d inputs; a detector takes one picture", inputs);
        return false;
    }
    InputSpec spec;
    const TfLiteType type = api_->tensorType(engine.input);
    if (!toElement(type, &spec.element)) {
        LOGE("The model's input holds values of LiteRT type %d, which AimBuddy does not write", static_cast<int>(type));
        return false;
    }
    int dims[kMaxDims];
    int count = 0;
    if (!readDims(*api_, engine.input, dims, &count) || !readInputShape(dims, count, &spec.size, &spec.layout)) {
        LOGE("The model's input is not one square RGB picture");
        return false;
    }
    spec.quantization = toQuantization(api_->tensorQuantization(engine.input));
    const size_t bytes = static_cast<size_t>(spec.size) * spec.size * kRgbChannels * elementBytes(spec.element);
    if (api_->tensorByteSize(engine.input) != bytes) {
        LOGE("The model's input has %zu bytes, not %zu", api_->tensorByteSize(engine.input), bytes);
        return false;
    }
    if (!inputWriter_.configure(spec)) {
        LOGE("The model's 8-bit input (scale %g, zero point %d) does not stand for colours from 0 to 1, which is what "
             "AimBuddy gives a model", spec.quantization.scale, spec.quantization.zeroPoint);
        return false;
    }
    if (declared.inputSize > 0 && declared.inputSize != spec.size) {
        LOGW("The model's description says %d px, but the model reads %d px: its own size is used", declared.inputSize,
             spec.size);
    }
    resized_.assign(static_cast<size_t>(spec.size) * spec.size * kRgbaBytes, 0);
    LOGI("Model input: %d px, %s, %s", spec.size, spec.layout == PixelLayout::kPlanar ? "channel planes" : "interleaved",
         spec.element == TensorElement::kFloat32 ? "float" : "8-bit");
    return true;
}

bool TfliteDetector::readOutput(const Engine& engine) {
    const int outputs = api_->outputTensorCount(engine.interpreter);
    if (outputs > 1) {
        LOGI("The model has %d outputs; the first is read as the detections", outputs);
    }
    const TfLiteType type = api_->tensorType(engine.output);
    if (!toElement(type, &outputElement_)) {
        LOGE("The model's output holds values of LiteRT type %d, which AimBuddy does not read", static_cast<int>(type));
        return false;
    }
    int dims[kMaxDims];
    int count = 0;
    if (!readDims(*api_, engine.output, dims, &count) || !readOutputShape(dims, count, &outputRows_, &outputCols_)) {
        LOGE("The model's output is not a table of detections");
        return false;
    }
    outputQuantization_ = toQuantization(api_->tensorQuantization(engine.output));
    const bool integer = outputElement_ != TensorElement::kFloat32;
    if (integer && !(outputQuantization_.scale > 0.0f && std::isfinite(outputQuantization_.scale))) {
        LOGE("The model's integer output has no usable scale (%g)", outputQuantization_.scale);
        return false;
    }
    const size_t values = static_cast<size_t>(outputRows_) * outputCols_;
    if (api_->tensorByteSize(engine.output) != values * elementBytes(outputElement_)) {
        LOGE("The model's output has %zu bytes, not %zu values", api_->tensorByteSize(engine.output), values);
        return false;
    }
    dequantized_.assign(integer ? values : 0, 0.0f);
    return true;
}

bool TfliteDetector::hasTheReadTensors(const Engine& engine) const {
    const InputSpec& input = inputWriter_.spec();
    TensorElement inputElement = TensorElement::kFloat32;
    TensorElement outputElement = TensorElement::kFloat32;
    int dims[kMaxDims];
    int count = 0;
    int size = 0;
    PixelLayout layout = PixelLayout::kInterleaved;
    int rows = 0;
    int cols = 0;
    return toElement(api_->tensorType(engine.input), &inputElement) && inputElement == input.element &&
           readDims(*api_, engine.input, dims, &count) && readInputShape(dims, count, &size, &layout) &&
           size == input.size && layout == input.layout &&
           toElement(api_->tensorType(engine.output), &outputElement) && outputElement == outputElement_ &&
           readDims(*api_, engine.output, dims, &count) && readOutputShape(dims, count, &rows, &cols) &&
           rows == outputRows_ && cols == outputCols_;
}

bool TfliteDetector::run(Engine& engine, const uint8_t* rgba, int stride) {
    void* input = api_->tensorData(engine.input);
    if (input == nullptr) {
        return false;
    }
    inputWriter_.write(rgba, stride, input);
    return api_->invoke(engine.interpreter) == kTfLiteOk;
}

const float* TfliteDetector::outputValues(const Engine& engine) {
    const void* values = api_->tensorData(engine.output);
    if (values == nullptr) {
        return nullptr;
    }
    if (outputElement_ == TensorElement::kFloat32) {
        return static_cast<const float*>(values);
    }
    dequantize(values, outputElement_, outputQuantization_, outputRows_ * outputCols_, dequantized_.data());
    return dequantized_.data();
}

void TfliteDetector::rebuildCpuOnFastCores() {
    // XNNPACK starts its worker threads when the interpreter is made, and a thread keeps the core binding of the
    // thread that started it. The interpreter made at load started them wherever the loading thread could run.
    Engine pinned;
    if (!build(pinned, false) || !hasTheReadTensors(pinned)) {
        release(pinned);
        LOGW("The CPU interpreter could not be made again on the fast cores; the first one is kept");
        return;
    }
    release(cpu_);
    cpu_ = pinned;
    active_ = &cpu_;
}

void TfliteDetector::chooseBackend(const std::function<bool()>& shouldStop) {
    if (!loaded_) {
        return;
    }
    // Bind this thread to the fast cores, then make the CPU interpreter here so its workers are bound too.
    pinToFastCores();
    rebuildCpuOnFastCores();

    const int size = inputWriter_.spec().size;
    const int stride = size * kRgbaBytes;
    const std::vector<uint8_t> probe = greyPicture(size);
    // A stop leaves the CPU interpreter as the backend, and nothing will run on it anyway.
    const float cpuMs = medianRunMs([&] { return run(cpu_, probe.data(), stride); }, shouldStop);
    if (shouldStop()) {
        return;
    }
    // Making the GPU interpreter compiles its programs, which takes seconds on some phones; a stop waits for it.
    if (gpu_.interpreter == nullptr && (!build(gpu_, true) || !hasTheReadTensors(gpu_))) {
        release(gpu_);
    }
    float gpuMs = -1.0f;
    if (gpu_.interpreter != nullptr && !shouldStop()) {
        gpuMs = medianRunMs([&] { return run(gpu_, probe.data(), stride); }, shouldStop);
    }
    if (shouldStop()) {
        return;
    }
    // Fast and wrong is no use: the detector would see nobody, with nothing to say why.
    const bool useGpu = gpu_.interpreter != nullptr && gpuIsFastEnough(cpuMs, gpuMs) && gpuAnswersLikeCpu();
    if (!useGpu) {
        release(gpu_);
    }
    active_ = useGpu ? &gpu_ : &cpu_;
    LOGI("Backend: %s (CPU %.1f ms with %d threads, GPU %.1f ms)", backendName(), cpuMs, fastCoreThreadCount(), gpuMs);
}

bool TfliteDetector::gpuAnswersLikeCpu() {
    const int size = inputWriter_.spec().size;
    const std::vector<uint8_t> picture = agreementPicture(size);
    const int count = outputRows_ * outputCols_;
    const float* cpuValues = run(cpu_, picture.data(), size * kRgbaBytes) ? outputValues(cpu_) : nullptr;
    if (cpuValues == nullptr) {
        LOGW("The picture that checks the GPU against the CPU could not be run on the CPU: the CPU is used");
        return false;
    }
    // A copy: the GPU's answer of an integer model is dequantized into the same buffer.
    const std::vector<float> cpuAnswer(cpuValues, cpuValues + count);
    const float* gpuValues = run(gpu_, picture.data(), size * kRgbaBytes) ? outputValues(gpu_) : nullptr;
    if (gpuValues == nullptr) {
        LOGW("The picture that checks the GPU against the CPU could not be run on the GPU: the CPU is used");
        return false;
    }
    float error = 0.0f;
    const bool agree = outputsAgree(cpuAnswer.data(), gpuValues, count, &error);
    LOGI("The GPU's answer %s the CPU's (largest difference %.1f%% of the range)", agree ? "matches" : "does not match",
         100.0f * error);
    return agree;
}

bool TfliteDetector::detect(const RoiFrame& frame, DetectionBatch& batch) {
    if (!loaded_ || active_ == nullptr) {
        return false;
    }
    const int side = frame.roi.side;
    if (side <= 0 || frame.rgba.size() < static_cast<size_t>(side) * static_cast<size_t>(side) * kRgbaBytes) {
        return false;
    }
    const auto start = std::chrono::steady_clock::now();

    // The region is shrunk to the input size by ncnn's NEON resize, the one the NCNN detector uses, so both
    // formats see the same picture; one table look-up per value then writes it into the model's input.
    const int size = inputWriter_.spec().size;
    const uint8_t* picture = frame.rgba.data();
    if (side != size) {
        ncnn::resize_bilinear_c4(frame.rgba.data(), side, side, side * kRgbaBytes, resized_.data(), size, size,
                                 size * kRgbaBytes);
        picture = resized_.data();
    }
    if (!run(*active_, picture, size * kRgbaBytes)) {
        return false;
    }
    const float* values = outputValues(*active_);
    if (values == nullptr) {
        return false;
    }
    decodeOutput(values, outputRows_, outputCols_, frame, start, batch);
    return true;
}

}  // namespace aimbuddy
