#ifndef AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
#define AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H

#include <functional>
#include <vector>

#include "../core/model_tensors.h"
#include "detector.h"
#include "litert_runtime.h"

namespace aimbuddy {

/**
 * Runs a .tflite export of a YOLO-style detector with LiteRT.
 *
 * The model says how it wants its picture (channels per pixel or in planes, floats or 8-bit
 * integers) and how its answer is stored, and the detector writes and reads exactly that. On the CPU
 * LiteRT runs the model through XNNPACK; the GPU delegate (OpenCL, half precision) is timed against
 * it on the inference thread and kept only when it is clearly faster and gives the CPU's answer, as
 * for NCNN models. The picture and the answer pass through buffers made at load; the one call that
 * may allocate per frame is ncnn's resize, which the NCNN path makes too.
 */
class TfliteDetector final : public Detector {
public:
    TfliteDetector() = default;
    ~TfliteDetector() override { unload(); }

    bool load(const ModelFiles& files, const ModelProfile& profile) override;
    void chooseBackend(const std::function<bool()>& shouldStop) override;
    bool detect(const RoiFrame& frame, DetectionBatch& batch) override;
    const char* backendName() const override { return active_ == &gpu_ ? "GPU (LiteRT)" : "CPU (LiteRT)"; }

private:
    /// One interpreter of the model: on the CPU, or with the GPU delegate, which it owns.
    struct Engine {
        TfLiteInterpreter* interpreter = nullptr;
        TfLiteDelegate* gpuDelegate = nullptr;
        TfLiteTensor* input = nullptr;
        const TfLiteTensor* output = nullptr;
    };

    void unload();
    bool build(Engine& engine, bool onGpu);
    void release(Engine& engine);
    /// Makes the CPU interpreter again on the calling thread, which must be bound to the fast cores already.
    void rebuildCpuOnFastCores();
    /// Reads the model's picture input from an engine: its size, layout and how it stores values.
    bool readInput(const Engine& engine, const ModelProfile& declared);
    /// Reads the model's detection output from an engine: its shape and how it stores values.
    bool readOutput(const Engine& engine);
    /// Whether an engine's tensors are the ones read at load, so one picture writer and one reader serve both engines.
    bool hasTheReadTensors(const Engine& engine) const;
    /// Writes an input-sized RGBA picture into the engine's input and runs the model.
    bool run(Engine& engine, const uint8_t* rgba, int stride);
    /// The engine's last answer as floats: the tensor itself for a float model, a dequantized copy otherwise.
    const float* outputValues(const Engine& engine);
    /// Runs one picture through both engines and says whether the GPU gave the CPU's answer (see outputsAgree).
    bool gpuAnswersLikeCpu();

    const LiteRtApi* api_ = nullptr;
    TfLiteModel* model_ = nullptr;
    Engine cpu_;
    Engine gpu_;
    Engine* active_ = nullptr;
    InputWriter inputWriter_;
    TensorElement outputElement_ = TensorElement::kFloat32;
    Quantization outputQuantization_;
    int outputRows_ = 0;
    int outputCols_ = 0;
    std::vector<uint8_t> resized_;     ///< The region shrunk to the model's input size, when it is larger.
    std::vector<float> dequantized_;   ///< The answer of an integer model, as floats.
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_TFLITE_DETECTOR_H
