#include "detector.h"

#include "../settings.h"
#include "../utils/logger.h"
#include "ncnn_detector.h"

namespace aimbuddy {

bool modelFormatFromCode(int code, ModelFormat* format) {
    switch (code) {
        case static_cast<int>(ModelFormat::kNcnn):
            *format = ModelFormat::kNcnn;
            return true;
        default:
            return false;
    }
}

bool Detector::learnOutput(const float* data, int rows, int cols, int inputSize, const ModelProfile& declared) {
    DecodeSpec spec;
    if (!inferDecodeSpec(data, rows, cols, declared.endToEndOutput, &spec)) {
        return false;
    }
    const int anchors = spec.layout == OutputLayout::kChannelsFirst ? cols : rows;
    if (spec.layout != OutputLayout::kEndToEnd && inferInputSize(anchors) != inputSize) {
        LOGW("Model output has %d anchors, unusual for a %d px input", anchors, inputSize);
    }
    if (declared.declaredClassCount != 0 && spec.layout != OutputLayout::kEndToEnd &&
        declared.declaredClassCount != spec.classCount) {
        LOGW("Model has %d classes but its description lists %d", spec.classCount, declared.declaredClassCount);
    }
    spec_ = spec;
    inputSize_ = inputSize;
    // No description saved with the model: the roles are guessed from how many classes it has.
    profile_ = settleProfile(declared, spec);
    LOGI("Model reads %dx%d input, output %dx%d, %d classes, %s coordinates", inputSize, inputSize, rows, cols,
         spec.classCount, spec.normalized ? "normalised" : "pixel");
    return true;
}

void Detector::decodeOutput(const float* data, int rows, int cols, const RoiFrame& frame,
                            std::chrono::steady_clock::time_point start, DetectionBatch& batch) const {
    DecodeParams params;
    params.minScore = minScore_.load(std::memory_order_relaxed);
    params.modelSize = inputSize_;
    decodeIntoBatch(data, rows, cols, spec_, params, profile_, frame.roi, frame.geometry, &batch);
    batch.captureNs = frame.captureTimeNs;
    batch.inferenceMs =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::unique_ptr<Detector> makeDetector(ModelFormat format) {
    switch (format) {
        case ModelFormat::kNcnn:
            return std::make_unique<NcnnDetector>();
    }
    return nullptr;
}

}  // namespace aimbuddy
