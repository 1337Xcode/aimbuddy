#ifndef AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
#define AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H

#include <jni.h>
#include <ncnn/allocator.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include "../capture/roi_capture.h"
#include "../core/detection_batch.h"
#include "../core/detection_decoder.h"
#include "../core/model_profile.h"

namespace aimbuddy {

class TFLiteDetector {
public:
    TFLiteDetector() = default;
    ~TFLiteDetector() { unload(); }
    TFLiteDetector(const TFLiteDetector&) = delete;
    TFLiteDetector& operator=(const TFLiteDetector&) = delete;

    bool load(JNIEnv* env, jobject runner, const ModelProfile& profile);
    void unload();
    bool isLoaded() const { return loaded_; }
    bool detect(const RoiFrame& frame, DetectionBatch& batch);
    void setMinScore(float score) { minScore_.store(score, std::memory_order_relaxed); }
    int inputSize() const { return inputSize_; }
    const ModelProfile& profile() const { return profile_; }
    const char* backendName() const { return backend_; }

private:
    bool runInference(JNIEnv* env, std::vector<float>& output);

    JavaVM* jvm_ = nullptr;
    jobject runner_ = nullptr;
    jobject inputBuffer_ = nullptr;
    jmethodID inputSizeMethod_ = nullptr;
    jmethodID outputShapeMethod_ = nullptr;
    jmethodID runInferenceMethod_ = nullptr;
    jmethodID closeMethod_ = nullptr;
    jmethodID backendMethod_ = nullptr;
    char backend_[64] = "TFLite CPU (XNNPACK)";
    std::vector<uint8_t> rgb_;
    std::vector<float> output_;
    ncnn::UnlockedPoolAllocator inputPool_;
    int inputSize_ = 0;
    int outputRows_ = 0;
    int outputCols_ = 0;
    ModelProfile profile_;
    DecodeSpec spec_;
    bool loaded_ = false;
    bool loggedJavaFailure_ = false;
    std::atomic<float> minScore_{0.12f};
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
