#ifndef AIMBUDDY_DETECTOR_NCNN_DETECTOR_H
#define AIMBUDDY_DETECTOR_NCNN_DETECTOR_H

#include <ncnn/allocator.h>
#include <ncnn/net.h>

#include <functional>
#include <string>

#include "detector.h"

namespace aimbuddy {

/**
 * Runs an NCNN export of a YOLO-style detector (a .param and a .bin file).
 *
 * Phones differ a lot, so the detector loads the model for both the CPU and (if present) the GPU and,
 * on the inference thread, times both and keeps the faster. The GPU is shared with the game, so it
 * must be clearly faster to be chosen, and it must give the CPU's answer to a test picture: a driver
 * that is fast and wrong is not used.
 */
class NcnnDetector final : public Detector {
public:
    NcnnDetector() = default;
    ~NcnnDetector() override { unload(); }

    bool load(const ModelFiles& files, const ModelProfile& profile) override;
    void chooseBackend(const std::function<bool()>& shouldStop) override;
    bool detect(const RoiFrame& frame, DetectionBatch& batch) override;
    const char* backendName() const override { return backend_ == Backend::kVulkan ? "GPU (Vulkan)" : "CPU"; }

private:
    enum class Backend { kCpu, kVulkan };

    /// Releases the model and every buffer it holds.
    void unload();
    /// Switches to an already loaded backend. Returns false if that backend is not available.
    bool useBackend(Backend backend);
    bool loadNet(ncnn::Net& net, const char* paramPath, const char* binPath) const;
    bool runOnce(ncnn::Net& net, const ncnn::Mat& input, ncnn::Mat& output) const;
    bool probeOutput(ncnn::Net& net, const ModelProfile& declared);
    float medianLatencyMs(ncnn::Net& net, const std::function<bool()>& shouldStop);
    /// Runs one picture through both networks and says whether the GPU gave the CPU's answer (see outputsAgree).
    bool gpuAnswersLikeCpu();

    ncnn::Net cpuNet_;
    ncnn::Net gpuNet_;
    ncnn::Net* active_ = nullptr;
    ncnn::UnlockedPoolAllocator inputPool_;  ///< Recycles the per-frame input image instead of reallocating it.

    bool gpuLoaded_ = false;
    Backend backend_ = Backend::kCpu;
    int inputIndex_ = 0;
    int outputIndex_ = 0;
    std::string paramPath_;
    std::string binPath_;
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_NCNN_DETECTOR_H
