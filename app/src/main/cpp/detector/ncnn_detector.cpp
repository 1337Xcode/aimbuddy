#include "ncnn_detector.h"

#include <ncnn/gpu.h>

#include <chrono>
#include <iterator>
#include <vector>

#include "../core/output_agreement.h"
#include "../settings.h"
#include "../utils/logger.h"
#include "backend_timing.h"

namespace aimbuddy {

namespace {

/// Network input sizes tried, in order, when the model description does not state one.
constexpr int kProbeInputSizes[] = {256, 320, 384, 416, 512, 640};

/// What ncnn's GpuInfo::type() says of a Vulkan device that is the CPU in disguise (SwiftShader on an
/// emulator). It cannot beat the CPU path, and loading the network onto it takes about a minute.
constexpr int kGpuTypeCpu = 3;

/// Grey level of the synthetic image used to probe and time the network.
constexpr float kProbeGrey = 0.45f;

/// Options that suit both backends: low memory, half precision where the hardware allows it.
void applyCommonOptions(ncnn::Option& opt) {
    opt.lightmode = true;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_sgemm_convolution = true;
    opt.use_winograd_convolution = true;
    opt.openmp_blocktime = kSpinWaitMs;
    opt.flush_denormals = 3;  // tiny float values are flushed to zero, which is faster on ARM
}

/// A flat grey image of the given size, used to probe and time the network.
ncnn::Mat makeProbeImage(int size) {
    ncnn::Mat image(size, size, 3);
    image.fill(kProbeGrey);
    return image;
}

/// Pointer to the first value of a detection tensor, whether ncnn stored it as 2D or as one channel.
const float* tensorData(const ncnn::Mat& tensor) {
    return tensor.dims == 3 ? static_cast<const float*>(tensor.channel(0).data)
                            : static_cast<const float*>(tensor.data);
}

}  // namespace

bool NcnnDetector::loadNet(ncnn::Net& net, const char* paramPath, const char* binPath) const {
    if (paramPath == nullptr || binPath == nullptr || paramPath[0] == '\0' || binPath[0] == '\0') {
        LOGE("There are no model files to read");
        return false;
    }
    if (net.load_param(paramPath) != 0 || net.load_model(binPath) != 0) {
        LOGE("Could not read the model files %s and %s", paramPath, binPath);
        net.clear();
        return false;
    }
    return true;
}

bool NcnnDetector::runOnce(ncnn::Net& net, const ncnn::Mat& input, ncnn::Mat& output) const {
    ncnn::Extractor extractor = net.create_extractor();
    return extractor.input(inputIndex_, input) == 0 && extractor.extract(outputIndex_, output) == 0;
}

bool NcnnDetector::probeOutput(ncnn::Net& net, const ModelProfile& declared) {
    std::vector<int> sizes;
    if (declared.inputSize > 0) {
        sizes.push_back(declared.inputSize);
    } else {
        sizes.assign(std::begin(kProbeInputSizes), std::end(kProbeInputSizes));
    }

    // A model exported with a fixed shape only accepts its own input size, so the first size that
    // runs is the right one. A fully convolutional model accepts any size and gets the first.
    for (const int size : sizes) {
        ncnn::Mat output;
        if (!runOnce(net, makeProbeImage(size), output) || output.empty() || output.c != 1 ||
            output.elempack != 1 || output.elemsize != sizeof(float)) {
            continue;
        }
        if (learnOutput(tensorData(output), output.h, output.w, size, declared)) {
            return true;
        }
    }
    LOGE("The model did not produce a readable detection tensor");
    return false;
}

bool NcnnDetector::load(const ModelFiles& files, const ModelProfile& profile) {
    unload();
    paramPath_ = files.modelPath;
    binPath_ = files.weightsPath;

    applyCommonOptions(cpuNet_.opt);
    cpuNet_.opt.use_vulkan_compute = false;
    cpuNet_.opt.num_threads = fastCoreThreadCount();
    if (!loadNet(cpuNet_, paramPath_.c_str(), binPath_.c_str()) || cpuNet_.input_indexes().empty() ||
        cpuNet_.output_indexes().empty()) {
        unload();
        return false;
    }
    inputIndex_ = cpuNet_.input_indexes()[0];
    outputIndex_ = cpuNet_.output_indexes()[0];

    if (!probeOutput(cpuNet_, profile)) {
        unload();
        return false;
    }

    active_ = &cpuNet_;
    backend_ = Backend::kCpu;
    loaded_ = true;
    return true;
}

void NcnnDetector::unload() {
    cpuNet_.clear();
    gpuNet_.clear();
    inputPool_.clear();
    active_ = nullptr;
    loaded_ = false;
    gpuLoaded_ = false;
    backend_ = Backend::kCpu;
}

float NcnnDetector::medianLatencyMs(ncnn::Net& net, const std::function<bool()>& shouldStop) {
    const ncnn::Mat probe = makeProbeImage(inputSize_);
    ncnn::Mat output;
    return medianRunMs([&] { return runOnce(net, probe, output); }, shouldStop);
}

bool NcnnDetector::useBackend(Backend backend) {
    if (!loaded_) {
        return false;
    }
    if (backend == Backend::kVulkan) {
        if (!gpuLoaded_) {
            return false;
        }
        active_ = &gpuNet_;
    } else {
        active_ = &cpuNet_;
    }
    backend_ = backend;
    return true;
}

void NcnnDetector::chooseBackend(const std::function<bool()>& shouldStop) {
    if (!loaded_) {
        return;
    }
    // Bind this thread and ncnn's CPU workers to the fast cores; the little cores stay free for the game and system.
    pinToFastCores();

    // A stop leaves the CPU net that load() set up as the backend, and nothing will run on it anyway.
    const float cpuMs = medianLatencyMs(cpuNet_, shouldStop);
    if (shouldStop()) {
        return;
    }
    float gpuMs = -1.0f;
    if (ncnn::get_gpu_count() > 0 && ncnn::get_gpu_info().type() != kGpuTypeCpu && !gpuLoaded_) {
        applyCommonOptions(gpuNet_.opt);
        gpuNet_.opt.use_vulkan_compute = true;
        gpuNet_.set_vulkan_device(ncnn::get_default_gpu_index());
        gpuLoaded_ = loadNet(gpuNet_, paramPath_.c_str(), binPath_.c_str());
        if (!gpuLoaded_) {
            gpuNet_.clear();
        }
    }
    if (shouldStop()) {
        return;
    }
    if (gpuLoaded_) {
        gpuMs = medianLatencyMs(gpuNet_, shouldStop);
        // A stop cuts the timing short, which is no failure: the GPU net, slow to load on some phones, stays loaded.
        if (gpuMs < 0.0f && !shouldStop()) {
            gpuNet_.clear();
            gpuLoaded_ = false;
        }
    }
    if (shouldStop()) {
        return;
    }

    bool useGpu = gpuLoaded_ && gpuIsFastEnough(cpuMs, gpuMs);
    if (useGpu && !gpuAnswersLikeCpu()) {
        // Fast and wrong is no use: the detector would see nobody, with nothing to say why.
        useGpu = false;
        gpuNet_.clear();
        gpuLoaded_ = false;
    }
    useBackend(useGpu ? Backend::kVulkan : Backend::kCpu);
    LOGI("Backend: %s (CPU %.1f ms with %d threads, GPU %.1f ms)", backendName(), cpuMs,
         cpuNet_.opt.num_threads, gpuMs);
}

bool NcnnDetector::gpuAnswersLikeCpu() {
    ncnn::Mat input(inputSize_, inputSize_, 3);
    for (int channel = 0; channel < 3; ++channel) {
        float* plane = input.channel(channel);
        for (int y = 0; y < inputSize_; ++y) {
            for (int x = 0; x < inputSize_; ++x) {
                plane[y * inputSize_ + x] = agreementPictureValue(x, y, channel);
            }
        }
    }
    ncnn::Mat cpuOutput;
    ncnn::Mat gpuOutput;
    if (!runOnce(cpuNet_, input, cpuOutput) || !runOnce(gpuNet_, input, gpuOutput)) {
        LOGW("The picture that checks the GPU against the CPU could not be run on both: the CPU is used");
        return false;
    }
    // The CPU's answer was held to this shape when the model was loaded (probeOutput); the GPU's has to match it.
    if (cpuOutput.empty() || gpuOutput.empty() || gpuOutput.c != 1 || gpuOutput.elempack != 1 ||
        gpuOutput.elemsize != sizeof(float) || gpuOutput.w != cpuOutput.w || gpuOutput.h != cpuOutput.h) {
        LOGW("The GPU's answer has another shape than the CPU's: the CPU is used");
        return false;
    }
    float error = 0.0f;
    const bool agree = outputsAgree(tensorData(cpuOutput), tensorData(gpuOutput), cpuOutput.w * cpuOutput.h, &error);
    LOGI("The GPU's answer %s the CPU's (largest difference %.1f%% of the range)", agree ? "matches" : "does not match",
         100.0f * error);
    return agree;
}

bool NcnnDetector::detect(const RoiFrame& frame, DetectionBatch& batch) {
    if (!loaded_ || active_ == nullptr) {
        return false;
    }
    const int side = frame.roi.side;
    if (side <= 0 || frame.rgba.size() < static_cast<size_t>(side) * static_cast<size_t>(side) * 4) {
        return false;
    }
    const auto start = std::chrono::steady_clock::now();

    // One pass converts RGBA to RGB and resizes the region to the network input size.
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(frame.rgba.data(), ncnn::Mat::PIXEL_RGBA2RGB, side, side,
                                                    side * 4, inputSize_, inputSize_, &inputPool_);
    static const float kMean[3] = {0.0f, 0.0f, 0.0f};
    static const float kScale[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(kMean, kScale);

    ncnn::Mat output;
    if (!runOnce(*active_, input, output) || output.empty() || output.elempack != 1 ||
        output.elemsize != sizeof(float)) {
        return false;
    }

    decodeOutput(tensorData(output), output.h, output.w, frame, start, batch);
    return true;
}

}  // namespace aimbuddy
