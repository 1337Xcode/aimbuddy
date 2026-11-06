#ifndef AIMBUDDY_DETECTOR_DETECTOR_H
#define AIMBUDDY_DETECTOR_DETECTOR_H

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "../capture/roi_capture.h"
#include "../core/detection_batch.h"
#include "../core/detection_decoder.h"
#include "../core/model_profile.h"

namespace aimbuddy {

/// The formats a detection model comes in. The numbers are the ones Kotlin passes (ModelFiles.formatCode).
enum class ModelFormat : int {
    kNcnn = 0,
};

/// The format Kotlin names by `code`; false for a number this build does not know.
bool modelFormatFromCode(int code, ModelFormat* format);

/// Where a model's files are.
struct ModelFiles {
    ModelFormat format = ModelFormat::kNcnn;
    std::string modelPath;     ///< The NCNN .param file.
    std::string weightsPath;   ///< The NCNN .bin file.
};

/**
 * Runs the user's detection model on the centre region of each captured frame.
 *
 * Users train their own models, so nothing about a model is assumed beyond what it tells us: the
 * input size, the output layout and the coordinate units are worked out when it loads, and what each
 * class means comes from the ModelProfile. A subclass per format runs the network; what its answer
 * means, and turning it into targets on the screen, is the same for every format and lives here.
 *
 * All methods except setMinScore() must be called from a single thread at a time.
 */
class Detector {
public:
    virtual ~Detector() = default;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /**
     * Loads the model and works out its input size and output layout with a trial run. Returns
     * false, leaving the detector unloaded, if the files are not there or the model cannot be used.
     */
    virtual bool load(const ModelFiles& files, const ModelProfile& profile) = 0;

    /**
     * Times the model on each processor it can run on and keeps the fastest. Call it on the thread
     * that will run detect(): it also pins that thread to the fast cores.
     *
     * The timing takes seconds on a slow phone, and whoever stops the assist waits for it, so
     * `shouldStop` is asked between its steps: once it says true the timing ends and the CPU stays
     * the backend. A step already under way (loading the network onto the GPU) is not cut short.
     */
    virtual void chooseBackend(const std::function<bool()>& shouldStop) = 0;

    /**
     * Detects targets in one captured region. Fills `batch` with boxes in screen pixels, strongest
     * first. Returns false if the model is not loaded or the network failed.
     */
    virtual bool detect(const RoiFrame& frame, DetectionBatch& batch) = 0;

    /// What runs the network, for the readout. It starts with "GPU" when a graphics processor does.
    virtual const char* backendName() const = 0;

    bool isLoaded() const { return loaded_; }

    /// Width and height of the network input, in pixels.
    int inputSize() const { return inputSize_; }

    /// What each class means, with defaults filled in for a model that came without a description.
    const ModelProfile& profile() const { return profile_; }

    /// Lowest score worth reporting; weaker candidates are dropped while decoding. Thread safe.
    void setMinScore(float score) { minScore_.store(score, std::memory_order_relaxed); }

protected:
    Detector() = default;

    /**
     * Learns from the output of a run on a probe picture how the output is laid out and what each
     * class means, and keeps that with the input size. Returns false when the tensor cannot be a
     * detector's output.
     */
    bool learnOutput(const float* data, int rows, int cols, int inputSize, const ModelProfile& declared);

    /// Decodes one output tensor into `batch`, stamped with the frame's time and the time since `start`.
    void decodeOutput(const float* data, int rows, int cols, const RoiFrame& frame,
                      std::chrono::steady_clock::time_point start, DetectionBatch& batch) const;

    bool loaded_ = false;
    int inputSize_ = 0;
    ModelProfile profile_;
    DecodeSpec spec_;

private:
    std::atomic<float> minScore_{0.12f};
};

/// A detector for models of the given format, not loaded yet.
std::unique_ptr<Detector> makeDetector(ModelFormat format);

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_DETECTOR_H
