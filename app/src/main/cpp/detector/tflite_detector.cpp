#include "tflite_detector.h"

#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <new>
#include <sstream>

#include "../utils/logger.h"

namespace aimbuddy {

namespace {

constexpr int kMinimumInputSize = 32;
constexpr int kMaximumInputSize = 2048;
constexpr size_t kMaximumOutputElements = 2'000'000;
constexpr int kInferenceThreads = 3;
constexpr int kSmartOutputCount = 6;
constexpr int kSmartDflChannels = 64;
constexpr int kSmartClassCount = 80;
constexpr int kSmartStrides[] = {8, 16, 32};

struct TensorView {
    TensorType type = TensorType::kFloat32;
    size_t elements = 0;
    int rank = 0;
    int dims[4] = {};
    float scale = 0.0f;
    int zeroPoint = 0;
};

std::string shapeString(const TensorView& view) {
    std::ostringstream out;
    out << '[';
    for (int i = 0; i < view.rank; ++i) {
        if (i != 0) out << ',';
        out << view.dims[i];
    }
    out << ']';
    return out.str();
}

const char* typeName(TensorType type) {
    switch (type) {
        case TensorType::kFloat32: return "float32";
        case TensorType::kFloat16: return "float16";
        case TensorType::kInt8: return "int8";
        case TensorType::kUInt8: return "uint8";
    }
    return "unsupported";
}

bool tensorView(const TfLiteTensor* tensor, bool quantized, TensorView* view, std::string* error) {
    if (tensor == nullptr) {
        *error = "The model has a missing input or output tensor";
        return false;
    }
    switch (TfLiteTensorType(tensor)) {
        case kTfLiteFloat32: view->type = TensorType::kFloat32; break;
        case kTfLiteFloat16: view->type = TensorType::kFloat16; break;
        case kTfLiteInt8: view->type = TensorType::kInt8; break;
        case kTfLiteUInt8: view->type = TensorType::kUInt8; break;
        default:
            *error = std::string("TFLite tensor type ") + std::to_string(TfLiteTensorType(tensor)) +
                     " is not supported (use float32, float16, int8 or uint8)";
            return false;
    }
    view->rank = TfLiteTensorNumDims(tensor);
    if (view->rank < 0 || view->rank > 4) {
        *error = "A TFLite input or output has more than four dimensions";
        return false;
    }
    size_t elements = 1;
    for (int i = 0; i < view->rank; ++i) {
        const int dimension = TfLiteTensorDim(tensor, i);
        if (dimension <= 0 || elements > kMaximumOutputElements / static_cast<size_t>(dimension)) {
            *error = "A TFLite tensor has an invalid or excessive size";
            return false;
        }
        view->dims[i] = dimension;
        elements *= static_cast<size_t>(dimension);
    }
    view->elements = elements;
    const TfLiteQuantizationParams quantization = TfLiteTensorQuantizationParams(tensor);
    view->scale = quantization.scale;
    view->zeroPoint = quantization.zero_point;
    if (quantized && (!(view->scale > 0.0f) || !std::isfinite(view->scale))) {
        *error = std::string("A quantized ") + typeName(view->type) +
                 " tensor has a missing or invalid scale. Re-export it with per-tensor quantization.";
        return false;
    }
    const size_t elementSize = view->type == TensorType::kFloat32 ? sizeof(float) :
                               view->type == TensorType::kFloat16 ? sizeof(uint16_t) : sizeof(uint8_t);
    if (view->elements == 0 || TfLiteTensorByteSize(tensor) != view->elements * elementSize) {
        *error = "A TFLite tensor is dynamically sized; export the model with a fixed input and output shape";
        return false;
    }
    return true;
}

bool detectionShape(const TensorView& view, int* rows, int* cols) {
    if (view.rank == 2) {
        *rows = view.dims[0];
        *cols = view.dims[1];
    } else if (view.rank == 3 && view.dims[0] == 1) {
        *rows = view.dims[1];
        *cols = view.dims[2];
    } else {
        return false;
    }
    return *rows > 0 && *cols >= 5 &&
           static_cast<size_t>(*rows) * static_cast<size_t>(*cols) <= kMaximumOutputElements;
}

bool smartShape(const TensorView& view, int* grid, int* channels) {
    if (view.rank != 4 || view.dims[0] != 1 || view.dims[1] != view.dims[2]) return false;
    *grid = view.dims[1];
    *channels = view.dims[3];
    return true;
}

}  // namespace

bool TFLiteDetector::fail(const std::string& message) {
    error_ = message;
    LOGE("TFLite: %s", message.c_str());
    return false;
}

bool TFLiteDetector::load(const char* modelPath, const ModelProfile& profile) {
    unload();
    error_.clear();
    if (modelPath == nullptr || modelPath[0] == '\0') return fail("No TFLite model file was supplied");
    model_ = TfLiteModelCreateFromFile(modelPath);
    if (model_ == nullptr) return fail("TFLite cannot open this file. Check that it is a complete, valid TFLite export");
    options_ = TfLiteInterpreterOptionsCreate();
    if (options_ == nullptr) return fail("Could not prepare the TFLite interpreter");
    TfLiteInterpreterOptionsSetNumThreads(options_, kInferenceThreads);
    TfLiteXNNPackDelegateOptions delegateOptions = TfLiteXNNPackDelegateOptionsDefault();
    delegateOptions.num_threads = kInferenceThreads;
    delegateOptions.flags = TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
    delegate_ = TfLiteXNNPackDelegateCreate(&delegateOptions);
    if (delegate_ == nullptr) return fail("The XNNPACK CPU delegate could not be created");
    TfLiteInterpreterOptionsAddDelegate(options_, delegate_);
    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    if (interpreter_ == nullptr) return fail("TFLite cannot build an interpreter for this graph (unsupported or corrupt model)");
    if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
        return fail("TFLite could not allocate this model's tensors");
    }
    if (!prepareInput() || !prepareOutputs()) return false;
    if (!fillProbe() || !invoke() || !readOutputs() || (smart_ && !combineSmartOutputs())) {
        return fail("TFLite could not run a warm-up inference on this model");
    }
    if (profile.inputSize > 0 && profile.inputSize != inputSize_) {
        return fail("The model description inputSize does not match the TFLite input tensor");
    }
    if (!inferDecodeSpec(output_.data(), outputRows_, outputCols_, profile.endToEndOutput,
                         profile.normalizedCoordinates, &spec_)) {
        return fail("The TFLite output is not a supported YOLO raw, final-detection or pose tensor");
    }
    profile_ = profile;
    if (profile.declaredClassCount == 0) {
        profile_ = profileWithDefaultRoles(profile_, spec_);
    } else if (spec_.layout != OutputLayout::kEndToEnd &&
               profile_.declaredClassCount != spec_.classCount) {
        LOGW("TFLite model has %d classes but its description lists %d", spec_.classCount, profile_.declaredClassCount);
    }
    profile_.endToEndOutput = spec_.layout == OutputLayout::kEndToEnd ||
                              spec_.layout == OutputLayout::kPoseEndToEnd;
    profile_.inputSize = inputSize_;
    loaded_ = true;
    LOGI("Loaded TFLite %dx%d input, output [%d,%d], %s", inputSize_, inputSize_, outputRows_, outputCols_, backendName());
    return true;
}

bool TFLiteDetector::prepareInput() {
    if (TfLiteInterpreterGetInputTensorCount(interpreter_) != 1) return fail("The TFLite model must have exactly one input tensor");
    input_ = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    TensorView view;
    if (!tensorView(input_, TfLiteTensorType(input_) != kTfLiteFloat32 && TfLiteTensorType(input_) != kTfLiteFloat16,
                    &view, &error_)) {
        return fail("Invalid TFLite input: " + error_);
    }
    if (view.rank != 4 || view.dims[0] != 1) {
        return fail("The TFLite input must be [1,height,height,3] (NHWC) or [1,3,height,height] (NCHW), but is " +
                    shapeString(view));
    }
    inputChannelsFirst_ = view.dims[1] == 3 && view.dims[2] == view.dims[3];
    const bool channelsLast = view.dims[3] == 3 && view.dims[1] == view.dims[2];
    if (!inputChannelsFirst_ && !channelsLast) {
        return fail("The TFLite input must be square RGB in NHWC or NCHW layout, but is " + shapeString(view));
    }
    inputSize_ = inputChannelsFirst_ ? view.dims[2] : view.dims[1];
    if (inputSize_ < kMinimumInputSize || inputSize_ > kMaximumInputSize) {
        return fail("The TFLite input side must be between 32 and 2048 pixels");
    }
    inputType_ = view.type;
    inputScale_ = view.scale;
    inputZeroPoint_ = view.zeroPoint;
    const size_t count = static_cast<size_t>(inputSize_) * inputSize_ * 3;
    if (count > SIZE_MAX / sizeof(float)) return fail("The TFLite input is too large");
    try {
        rgb_.resize(count);
    } catch (const std::bad_alloc&) {
        return fail("Not enough memory to prepare the TFLite input");
    }
    return true;
}

bool TFLiteDetector::prepareOutputs() {
    const int count = TfLiteInterpreterGetOutputTensorCount(interpreter_);
    if (count != 1 && count != kSmartOutputCount) {
        return fail("The model must have one detection output or the six supported EdgeFirst YOLO outputs");
    }
    try {
        outputs_.resize(count);
        for (int i = 0; i < count; ++i) {
            const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter_, i);
            TensorView view;
            if (!tensorView(tensor, TfLiteTensorType(tensor) != kTfLiteFloat32 && TfLiteTensorType(tensor) != kTfLiteFloat16,
                            &view, &error_)) {
                return fail("Invalid TFLite output " + std::to_string(i) + ": " + error_);
            }
            outputs_[i].tensor = tensor;
            outputs_[i].type = view.type;
            outputs_[i].scale = view.scale;
            outputs_[i].zeroPoint = view.zeroPoint;
            outputs_[i].values.resize(view.elements);
        }
    } catch (const std::bad_alloc&) {
        return fail("Not enough memory for TFLite output buffers");
    }
    if (count == 1) {
        TensorView view;
        std::string error;
        if (!tensorView(outputs_[0].tensor, false, &view, &error) ||
            !detectionShape(view, &outputRows_, &outputCols_)) {
            return fail("The detection output must have shape [rows,columns] or [1,rows,columns], but is " + shapeString(view));
        }
    } else {
        smart_ = true;
        if (!prepareSmartOutputs()) return false;
        int anchors = 0;
        for (const SmartScale& scale : smartScales_) anchors += scale.grid * scale.grid;
        outputRows_ = 5;
        outputCols_ = anchors;
        if (static_cast<size_t>(anchors) > kMaximumOutputElements / 5) return fail("The EdgeFirst output is too large");
    }
    try {
        output_.resize(static_cast<size_t>(outputRows_) * outputCols_);
    } catch (const std::bad_alloc&) {
        return fail("Not enough memory to combine TFLite outputs");
    }
    return true;
}

bool TFLiteDetector::prepareSmartOutputs() {
    try {
        smartScales_.clear();
        for (int wanted = 0; wanted < 3; ++wanted) {
            const int stride = kSmartStrides[wanted];
            if (inputSize_ % stride != 0) return fail("The EdgeFirst strides do not match the input size");
            const int grid = inputSize_ / stride;
            SmartScale scale{stride, grid, -1, -1};
            for (int i = 0; i < static_cast<int>(outputs_.size()); ++i) {
                TensorView view;
                std::string error;
                if (!tensorView(outputs_[i].tensor, false, &view, &error)) return fail(error);
                int candidateGrid = 0;
                int channels = 0;
                if (!smartShape(view, &candidateGrid, &channels) || candidateGrid != grid) continue;
                if (channels == kSmartDflChannels && scale.boxes < 0) scale.boxes = i;
                if (channels == kSmartClassCount && scale.scores < 0) scale.scores = i;
            }
            if (scale.boxes < 0 || scale.scores < 0) {
                return fail("The six-output EdgeFirst YOLO must provide DFL64 and COCO80 tensors at strides 8, 16 and 32");
            }
            smartScales_.push_back(scale);
        }
    } catch (const std::bad_alloc&) {
        return fail("Not enough memory to prepare the EdgeFirst output map");
    }
    return true;
}

bool TFLiteDetector::fillProbe() {
    // A neutral square exercises the same input conversion and output path as a real frame.
    return resize(nullptr, 0, 0);
}

bool TFLiteDetector::resize(const uint8_t* rgba, int side, int stride) {
    if (rgba == nullptr) {
        std::fill(rgb_.begin(), rgb_.end(), 114);
    } else {
        if (!resizer_.configure(side, inputSize_)) return fail("Not enough memory to prepare the image resize map");
        resizer_.resize(rgba, stride, rgb_.data());
    }
    if (!rgbToTensor(rgb_.data(), static_cast<size_t>(inputSize_) * inputSize_, inputType_, inputScale_, inputZeroPoint_,
                     inputChannelsFirst_, TfLiteTensorData(input_))) {
        return fail("Could not prepare the TFLite input tensor");
    }
    return true;
}

bool TFLiteDetector::invoke() {
    return interpreter_ != nullptr && TfLiteInterpreterInvoke(interpreter_) == kTfLiteOk;
}

bool TFLiteDetector::readOutputs() {
    for (Output& output : outputs_) {
        TensorView view;
        std::string error;
        if (!tensorView(output.tensor, output.type == TensorType::kInt8 || output.type == TensorType::kUInt8,
                        &view, &error) || view.elements != output.values.size()) {
            return fail("The TFLite output changed shape after warm-up; fixed-shape exports are required");
        }
        float* destination = smart_ ? output.values.data() : output_.data();
        if (!tensorToFloat(TfLiteTensorData(output.tensor), output.type, output.values.size(), output.scale,
                           output.zeroPoint, destination)) {
            return fail("Could not read a TFLite output tensor");
        }
    }
    return true;
}

bool TFLiteDetector::combineSmartOutputs() {
    SmartYoloScale scales[3];
    for (int i = 0; i < 3; ++i) {
        const SmartScale& scale = smartScales_[i];
        scales[i] = {scale.stride, scale.grid, outputs_[scale.boxes].values.data(), outputs_[scale.scores].values.data()};
    }
    return decodeSmartYolo(inputSize_, scales, 3, output_.data(), output_.size());
}

void TFLiteDetector::unload() {
    loaded_ = false;
    if (interpreter_ != nullptr) TfLiteInterpreterDelete(interpreter_);
    if (delegate_ != nullptr) TfLiteXNNPackDelegateDelete(delegate_);
    if (options_ != nullptr) TfLiteInterpreterOptionsDelete(options_);
    if (model_ != nullptr) TfLiteModelDelete(model_);
    interpreter_ = nullptr;
    delegate_ = nullptr;
    options_ = nullptr;
    model_ = nullptr;
    input_ = nullptr;
    rgb_.clear();
    resizer_.clear();
    outputs_.clear();
    smartScales_.clear();
    output_.clear();
    inputSize_ = 0;
    outputRows_ = 0;
    outputCols_ = 0;
    smart_ = false;
    profile_ = ModelProfile();
    spec_ = DecodeSpec();
}

bool TFLiteDetector::detect(const RoiFrame& frame, DetectionBatch& batch) {
    if (!loaded_) return false;
    const int side = frame.roi.side;
    if (side <= 0 || frame.rgba.size() < static_cast<size_t>(side) * side * 4) return false;
    const auto start = std::chrono::steady_clock::now();
    if (!resize(frame.rgba.data(), side, side * 4)) return false;
    if (!invoke() || !readOutputs() || (smart_ && !combineSmartOutputs())) return false;
    DecodeParams params;
    params.minScore = minScore_.load(std::memory_order_relaxed);
    params.modelSize = inputSize_;
    ModelBox boxes[kMaxDetections];
    const int count = decodeDetections(output_.data(), outputRows_, outputCols_, spec_, params, profile_, boxes, kMaxDetections);
    const float screenPerCapture = frame.geometry.screenPerCapture();
    batch.count = count;
    for (int i = 0; i < count; ++i) batch.detections[i] = mapToScreen(boxes[i], frame.roi, frame.geometry, inputSize_);
    batch.scanned.left = static_cast<float>(frame.roi.x) * screenPerCapture;
    batch.scanned.top = static_cast<float>(frame.roi.y) * screenPerCapture;
    batch.scanned.right = static_cast<float>(frame.roi.x + frame.roi.side) * screenPerCapture;
    batch.scanned.bottom = static_cast<float>(frame.roi.y + frame.roi.side) * screenPerCapture;
    batch.captureNs = frame.captureTimeNs;
    batch.inferenceMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    return true;
}

}  // namespace aimbuddy
