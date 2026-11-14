#ifndef AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
#define AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H

#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/core/c/common.h>

#include <atomic>
#include <string>
#include <vector>

#include "../core/detection_decoder.h"
#include "../core/tflite_tensor.h"
#include "detector.h"

namespace aimbuddy {

/// Runs a fixed-shape detection graph directly through the TFLite C API. All buffers are prepared at load time.
class TFLiteDetector final : public Detector {
public:
    TFLiteDetector() = default;
    ~TFLiteDetector() override { unload(); }
    TFLiteDetector(const TFLiteDetector&) = delete;
    TFLiteDetector& operator=(const TFLiteDetector&) = delete;

    bool load(const char* modelPath, const ModelProfile& profile);
    void unload() override;
    bool isLoaded() const override { return loaded_; }
    bool detect(const RoiFrame& frame, DetectionBatch& batch) override;
    void setMinScore(float score) override { minScore_.store(score, std::memory_order_relaxed); }
    int inputSize() const override { return inputSize_; }
    const ModelProfile& profile() const override { return profile_; }
    const char* backendName() const override { return "TFLite CPU (XNNPACK)"; }
    const std::string& error() const { return error_; }

private:
    struct Output {
        const TfLiteTensor* tensor = nullptr;
        TensorType type = TensorType::kFloat32;
        float scale = 0.0f;
        int zeroPoint = 0;
        std::vector<float> values;
    };
    struct SmartScale {
        int stride = 0;
        int grid = 0;
        int boxes = -1;
        int scores = -1;
    };

    bool fail(const std::string& message);
    bool prepareInput();
    bool prepareOutputs();
    bool prepareSmartOutputs();
    bool invoke();
    bool readOutputs();
    bool combineSmartOutputs();
    void resize(const uint8_t* rgba, int side, int stride);
    void fillProbe();
    std::vector<uint8_t> rgb_;

    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* options_ = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteDelegate* delegate_ = nullptr;
    TfLiteTensor* input_ = nullptr;
    TensorType inputType_ = TensorType::kFloat32;
    bool inputChannelsFirst_ = false;
    float inputScale_ = 0.0f;
    int inputZeroPoint_ = 0;
    std::vector<Output> outputs_;
    std::vector<SmartScale> smartScales_;
    std::vector<float> output_;
    int inputSize_ = 0;
    int outputRows_ = 0;
    int outputCols_ = 0;
    ModelProfile profile_;
    DecodeSpec spec_;
    bool loaded_ = false;
    bool smart_ = false;
    std::string error_;
    std::atomic<float> minScore_{0.12f};
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
