#include "tflite_detector.h"

#include <ncnn/mat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <new>
#include <cstdio>
#include <cstring>

#include "../utils/logger.h"

namespace aimbuddy {

namespace {

constexpr int kMinimumInputSize = 32;
constexpr int kMaximumInputSize = 2048;
constexpr size_t kMaximumOutputElements = 2'000'000;

class ScopedJNIEnv {
public:
    explicit ScopedJNIEnv(JavaVM* vm) : vm_(vm) {
        if (vm_ == nullptr) return;
        const jint status = vm_->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (status == JNI_EDETACHED && vm_->AttachCurrentThread(&env_, nullptr) == JNI_OK) {
            attached_ = true;
        } else if (status != JNI_OK) {
            env_ = nullptr;
        }
    }

    ~ScopedJNIEnv() {
        if (attached_) vm_->DetachCurrentThread();
    }

    JNIEnv* get() const { return env_; }

private:
    JavaVM* vm_ = nullptr;
    JNIEnv* env_ = nullptr;
    bool attached_ = false;
};

bool clearJavaException(JNIEnv* env, const char* operation) {
    if (!env->ExceptionCheck()) return false;
    env->ExceptionDescribe();
    env->ExceptionClear();
    LOGE("TFLite Java call failed: %s", operation);
    return true;
}

}  // namespace

bool TFLiteDetector::load(JNIEnv* env, jobject runner, const ModelProfile& profile) {
    unload();
    if (env == nullptr || runner == nullptr || env->GetJavaVM(&jvm_) != JNI_OK) {
        LOGE("TFLite model runner is unavailable");
        return false;
    }

    runner_ = env->NewGlobalRef(runner);
    if (runner_ == nullptr || clearJavaException(env, "retaining the model runner")) {
        unload();
        return false;
    }

    jclass runnerClass = env->GetObjectClass(runner_);
    if (runnerClass == nullptr || clearJavaException(env, "reading the model runner class")) {
        unload();
        return false;
    }
    inputSizeMethod_ = env->GetMethodID(runnerClass, "nativeInputSize", "()I");
    outputShapeMethod_ = env->GetMethodID(runnerClass, "nativeOutputShape", "()[I");
    runInferenceMethod_ = env->GetMethodID(runnerClass, "nativeRunInference", "(Ljava/nio/ByteBuffer;)[F");
    closeMethod_ = env->GetMethodID(runnerClass, "close", "()V");
    backendMethod_ = env->GetMethodID(runnerClass, "nativeBackend", "()Ljava/lang/String;");
    env->DeleteLocalRef(runnerClass);
    if (inputSizeMethod_ == nullptr || outputShapeMethod_ == nullptr || runInferenceMethod_ == nullptr ||
        closeMethod_ == nullptr || backendMethod_ == nullptr || clearJavaException(env, "finding model runner methods")) {
        unload();
        return false;
    }

    auto backend = static_cast<jstring>(env->CallObjectMethod(runner_, backendMethod_));
    if (clearJavaException(env, "reading the inference backend") || backend == nullptr) {
        unload();
        return false;
    }
    const char* name = env->GetStringUTFChars(backend, nullptr);
    if (name == nullptr) {
        env->DeleteLocalRef(backend);
        unload();
        return false;
    }
    std::snprintf(backend_, sizeof(backend_), "%s", name);
    env->ReleaseStringUTFChars(backend, name);
    env->DeleteLocalRef(backend);

    inputSize_ = env->CallIntMethod(runner_, inputSizeMethod_);
    if (clearJavaException(env, "reading the model input size") ||
        inputSize_ < kMinimumInputSize || inputSize_ > kMaximumInputSize ||
        (profile.inputSize > 0 && profile.inputSize != inputSize_)) {
        LOGE("TFLite model input size is invalid or does not match its description");
        unload();
        return false;
    }

    auto shape = static_cast<jintArray>(env->CallObjectMethod(runner_, outputShapeMethod_));
    if (clearJavaException(env, "reading the model output shape") || shape == nullptr) {
        unload();
        return false;
    }
    const jsize rank = env->GetArrayLength(shape);
    jint dimensions[3] = {};
    if (rank == 2 || rank == 3) env->GetIntArrayRegion(shape, 0, rank, dimensions);
    env->DeleteLocalRef(shape);
    if (clearJavaException(env, "reading the model output dimensions")) {
        unload();
        return false;
    }
    if (rank == 2) {
        outputRows_ = dimensions[0];
        outputCols_ = dimensions[1];
    } else if (rank == 3 && dimensions[0] == 1) {
        outputRows_ = dimensions[1];
        outputCols_ = dimensions[2];
    }
    if (outputRows_ <= 0 || outputCols_ <= 0 ||
        static_cast<size_t>(outputRows_) * static_cast<size_t>(outputCols_) > kMaximumOutputElements) {
        LOGE("TFLite model output shape is not supported");
        unload();
        return false;
    }

    try {
        rgb_.assign(static_cast<size_t>(inputSize_) * inputSize_ * 3, 115);
        output_.resize(static_cast<size_t>(outputRows_) * outputCols_);
    } catch (const std::bad_alloc&) {
        LOGE("Not enough memory to prepare the TFLite model");
        unload();
        return false;
    }
    jobject localBuffer = env->NewDirectByteBuffer(rgb_.data(), static_cast<jlong>(rgb_.size()));
    if (localBuffer == nullptr || clearJavaException(env, "creating the model input buffer")) {
        unload();
        return false;
    }
    inputBuffer_ = env->NewGlobalRef(localBuffer);
    env->DeleteLocalRef(localBuffer);
    if (inputBuffer_ == nullptr || clearJavaException(env, "retaining the model input buffer")) {
        unload();
        return false;
    }

    if (!runInference(env, output_) ||
        !inferDecodeSpec(output_.data(), outputRows_, outputCols_, profile.endToEndOutput,
                         profile.normalizedCoordinates, &spec_)) {
        LOGE("TFLite model output is not a detection tensor AimBuddy can read");
        unload();
        return false;
    }

    profile_ = profile;
    if (profile_.declaredClassCount == 0) {
        const int normalized = profile_.normalizedCoordinates;
        const bool endToEnd = profile_.endToEndOutput;
        profile_ = ModelProfile::defaultsFor(spec_.layout == OutputLayout::kPose ||
                                                     spec_.layout == OutputLayout::kPoseEndToEnd
                                                 ? 1
                                                 : spec_.classCount);
        profile_.endToEndOutput = endToEnd;
        profile_.normalizedCoordinates = normalized;
    } else if (spec_.layout != OutputLayout::kEndToEnd && profile_.declaredClassCount != spec_.classCount) {
        LOGW("TFLite model has %d classes but its description lists %d", spec_.classCount,
             profile_.declaredClassCount);
    }
    profile_.inputSize = inputSize_;
    loaded_ = true;
    return true;
}

void TFLiteDetector::unload() {
    if (runner_ != nullptr) {
        ScopedJNIEnv access(jvm_);
        JNIEnv* env = access.get();
        if (env == nullptr) {
            LOGE("Could not attach to Java to release the TFLite model");
        } else {
            if (closeMethod_ != nullptr) {
                env->CallVoidMethod(runner_, closeMethod_);
                clearJavaException(env, "closing the model runner");
            }
            if (inputBuffer_ != nullptr) env->DeleteGlobalRef(inputBuffer_);
            env->DeleteGlobalRef(runner_);
        }
    } else if (inputBuffer_ != nullptr) {
        ScopedJNIEnv access(jvm_);
        if (access.get() != nullptr) access.get()->DeleteGlobalRef(inputBuffer_);
    }
    jvm_ = nullptr;
    runner_ = nullptr;
    inputBuffer_ = nullptr;
    inputSizeMethod_ = nullptr;
    outputShapeMethod_ = nullptr;
    runInferenceMethod_ = nullptr;
    closeMethod_ = nullptr;
    backendMethod_ = nullptr;
    std::snprintf(backend_, sizeof(backend_), "%s", "TFLite CPU (XNNPACK)");
    rgb_.clear();
    output_.clear();
    inputPool_.clear();
    inputSize_ = 0;
    outputRows_ = 0;
    outputCols_ = 0;
    profile_ = ModelProfile();
    spec_ = DecodeSpec();
    loaded_ = false;
    loggedJavaFailure_ = false;
}

bool TFLiteDetector::runInference(JNIEnv* env, std::vector<float>& output) {
    if (env == nullptr || runner_ == nullptr || inputBuffer_ == nullptr) return false;
    auto result = static_cast<jfloatArray>(env->CallObjectMethod(runner_, runInferenceMethod_, inputBuffer_));
    if (clearJavaException(env, "running inference") || result == nullptr) {
        if (!loggedJavaFailure_) {
            LOGE("TFLite inference failed; check the model input and output tensors");
            loggedJavaFailure_ = true;
        }
        return false;
    }
    const jsize count = env->GetArrayLength(result);
    const size_t expected = static_cast<size_t>(outputRows_) * outputCols_;
    if (static_cast<size_t>(count) != expected) {
        env->DeleteLocalRef(result);
        LOGE("TFLite returned %d values, expected %zu", count, expected);
        return false;
    }
    env->GetFloatArrayRegion(result, 0, count, output.data());
    env->DeleteLocalRef(result);
    return !clearJavaException(env, "copying model output");
}

bool TFLiteDetector::detect(const RoiFrame& frame, DetectionBatch& batch) {
    if (!loaded_) return false;
    const int side = frame.roi.side;
    if (side <= 0 || frame.rgba.size() < static_cast<size_t>(side) * side * 4) return false;

    const auto start = std::chrono::steady_clock::now();
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(frame.rgba.data(), ncnn::Mat::PIXEL_RGBA2RGB, side, side,
                                                    side * 4, inputSize_, inputSize_, &inputPool_);
    if (input.empty()) return false;
    for (int y = 0; y < inputSize_; ++y) {
        for (int x = 0; x < inputSize_; ++x) {
            const size_t pixel = (static_cast<size_t>(y) * inputSize_ + x) * 3;
            const size_t plane = static_cast<size_t>(y) * inputSize_ + x;
            for (int channel = 0; channel < 3; ++channel) {
                const int value = static_cast<int>(std::lround(input.channel(channel)[plane]));
                rgb_[pixel + channel] = static_cast<uint8_t>(std::clamp(value, 0, 255));
            }
        }
    }

    ScopedJNIEnv access(jvm_);
    if (!runInference(access.get(), output_)) return false;

    DecodeParams params;
    params.minScore = minScore_.load(std::memory_order_relaxed);
    params.modelSize = inputSize_;
    ModelBox boxes[kMaxDetections];
    const int count =
        decodeDetections(output_.data(), outputRows_, outputCols_, spec_, params, profile_, boxes, kMaxDetections);

    const float screenPerCapture = frame.geometry.screenPerCapture();
    batch.count = count;
    for (int i = 0; i < count; ++i) {
        batch.detections[i] = mapToScreen(boxes[i], frame.roi, frame.geometry, inputSize_);
    }
    batch.scanned.left = static_cast<float>(frame.roi.x) * screenPerCapture;
    batch.scanned.top = static_cast<float>(frame.roi.y) * screenPerCapture;
    batch.scanned.right = static_cast<float>(frame.roi.x + frame.roi.side) * screenPerCapture;
    batch.scanned.bottom = static_cast<float>(frame.roi.y + frame.roi.side) * screenPerCapture;
    batch.captureNs = frame.captureTimeNs;
    batch.inferenceMs =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    return true;
}

}  // namespace aimbuddy
