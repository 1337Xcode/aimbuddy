#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include "detector/tflite_detector.h"
#include "test_harness.h"

using namespace aimbuddy;

namespace {

constexpr int kSide = 32;
constexpr int kInputElements = kSide * kSide * 3;
constexpr int kOutputElements = 300 * 6;
constexpr float kFloatBox[] = {4.0f, 6.0f, 28.0f, 30.0f, 0.9f, 0.0f};
constexpr float kQuantizedBox[] = {2.0f, 3.0f, 14.0f, 15.0f, 0.75f, 0.0f};

TfLiteType inputType = kTfLiteFloat32;
TfLiteType outputType = kTfLiteFloat32;
bool channelsFirst = false;
bool inputValid = true;
int invocations = 0;
std::vector<uint8_t> frame;
std::vector<uint8_t> inputData;
std::vector<uint8_t> outputData;
TfLiteTensor inputTensor{};
TfLiteTensor outputTensor{};
struct {
    int32_t size;
    int32_t data[4];
} inputDimensions{}, outputDimensions{};

void resetFake(TfLiteType type, bool planar) {
    inputType = type;
    outputType = type == kTfLiteInt8 ? kTfLiteInt8 : kTfLiteFloat32;
    channelsFirst = planar;
    inputValid = true;
    invocations = 0;
    const size_t inputBytes = kInputElements * (inputType == kTfLiteFloat32 ? sizeof(float) : sizeof(int8_t));
    const size_t outputBytes = kOutputElements * (outputType == kTfLiteFloat32 ? sizeof(float) : sizeof(int8_t));
    inputData.assign(inputBytes, 0);
    outputData.assign(outputBytes, 0);
    frame.resize(kSide * kSide * 4);
    for (size_t pixel = 0; pixel < kSide * kSide; ++pixel) {
        frame[pixel * 4] = static_cast<uint8_t>(pixel % 256);
        frame[pixel * 4 + 1] = static_cast<uint8_t>((pixel * 3) % 256);
        frame[pixel * 4 + 2] = static_cast<uint8_t>((255 - pixel) % 256);
        frame[pixel * 4 + 3] = 255;
    }
    inputDimensions = channelsFirst ? decltype(inputDimensions){4, {1, 3, kSide, kSide}}
                                    : decltype(inputDimensions){4, {1, kSide, kSide, 3}};
    outputDimensions = {3, {1, 300, 6}};
    inputTensor = {};
    inputTensor.type = inputType;
    inputTensor.dims = reinterpret_cast<TfLiteIntArray*>(&inputDimensions);
    inputTensor.data.data = inputData.data();
    inputTensor.bytes = inputData.size();
    inputTensor.params = inputType == kTfLiteInt8 ? TfLiteQuantizationParams{1.0f / 255.0f, -128}
                                                  : TfLiteQuantizationParams{};
    outputTensor = {};
    outputTensor.type = outputType;
    outputTensor.dims = reinterpret_cast<TfLiteIntArray*>(&outputDimensions);
    outputTensor.data.data = outputData.data();
    outputTensor.bytes = outputData.size();
    outputTensor.params = outputType == kTfLiteInt8 ? TfLiteQuantizationParams{0.25f, 0}
                                                    : TfLiteQuantizationParams{};
    const float* expected = outputType == kTfLiteInt8 ? kQuantizedBox : kFloatBox;
    for (int i = 0; i < 6; ++i) {
        if (outputType == kTfLiteFloat32) {
            reinterpret_cast<float*>(outputData.data())[i] = expected[i];
        } else {
            outputData[i] = static_cast<uint8_t>(static_cast<int8_t>(std::lround(expected[i] / 0.25f)));
        }
    }
}

bool inputMatchesFrame() {
    for (size_t pixel = 0; pixel < kSide * kSide; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const uint8_t value = frame[pixel * 4 + channel];
            const size_t index = channelsFirst ? channel * kSide * kSide + pixel : pixel * 3 + channel;
            if (inputType == kTfLiteFloat32) {
                if (reinterpret_cast<const float*>(inputData.data())[index] != value / 255.0f) return false;
            } else {
                const int expected = std::clamp(static_cast<int>(std::lround(value / 255.0f / (1.0f / 255.0f) - 128)), -128, 127);
                if (static_cast<int8_t>(inputData[index]) != expected) return false;
            }
        }
    }
    return true;
}

void runDetectorCase(TfLiteType type, bool planar) {
    resetFake(type, planar);
    TFLiteDetector detector;
    ModelProfile profile = ModelProfile::defaultsFor(1);
    profile.endToEndOutput = true;
    profile.normalizedCoordinates = 0;
    profile.inputSize = kSide;
    if (!detector.load("fake", profile)) {
        CHECK(false);
        return;
    }
    detector.setMinScore(0.25f);
    RoiFrame roi;
    roi.rgba = frame;
    roi.roi = {0, 0, kSide};
    roi.geometry = {kSide, kSide, kSide, kSide};
    DetectionBatch batch;
    CHECK(detector.detect(roi, batch));
    CHECK(invocations == 2);
    CHECK(inputValid);
    CHECK(batch.count == 1);
    if (batch.count == 1) {
        const float* expected = type == kTfLiteInt8 ? kQuantizedBox : kFloatBox;
        CHECK_NEAR(batch.detections[0].x, expected[0], 1e-4);
        CHECK_NEAR(batch.detections[0].y, expected[1], 1e-4);
        CHECK_NEAR(batch.detections[0].width, expected[2] - expected[0], 1e-4);
        CHECK_NEAR(batch.detections[0].height, expected[3] - expected[1], 1e-4);
        CHECK_NEAR(batch.detections[0].score, expected[4], 1e-4);
    }
}

}  // namespace

TEST(tflite_detector_reads_the_single_float_output_into_detection) {
    runDetectorCase(kTfLiteFloat32, false);
}

TEST(tflite_detector_reads_the_single_quantized_output_and_planar_input) {
    runDetectorCase(kTfLiteInt8, true);
}

struct TfLiteModel {};
struct TfLiteInterpreterOptions {};
struct TfLiteInterpreter {};

extern "C" {

TfLiteModel* TfLiteModelCreateFromFile(const char*) { return new TfLiteModel{}; }
void TfLiteModelDelete(TfLiteModel* model) { delete model; }
TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() { return new TfLiteInterpreterOptions{}; }
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options) { delete options; }
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions*, int32_t) {}
void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions*, TfLiteOpaqueDelegate*) {}
TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteModel*, const TfLiteInterpreterOptions*) { return new TfLiteInterpreter{}; }
void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter) { delete interpreter; }
TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter*) { return kTfLiteOk; }
TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter*) {
    ++invocations;
    if (invocations > 1) inputValid = inputMatchesFrame();
    return inputValid ? kTfLiteOk : kTfLiteError;
}
int32_t TfLiteInterpreterGetInputTensorCount(const TfLiteInterpreter*) { return 1; }
int32_t TfLiteInterpreterGetOutputTensorCount(const TfLiteInterpreter*) { return 1; }
TfLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter*, int32_t) { return &inputTensor; }
const TfLiteTensor* TfLiteInterpreterGetOutputTensor(const TfLiteInterpreter*, int32_t) { return &outputTensor; }
TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) { return tensor->type; }
int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) { return tensor->dims->size; }
int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t index) { return tensor->dims->data[index]; }
size_t TfLiteTensorByteSize(const TfLiteTensor* tensor) { return tensor->bytes; }
void* TfLiteTensorData(const TfLiteTensor* tensor) { return tensor->data.data; }
TfLiteQuantizationParams TfLiteTensorQuantizationParams(const TfLiteTensor* tensor) { return tensor->params; }
TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault() { return {}; }
TfLiteDelegate* TfLiteXNNPackDelegateCreate(const TfLiteXNNPackDelegateOptions*) { return new TfLiteDelegate{}; }
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) { delete delegate; }

}  // extern "C"
