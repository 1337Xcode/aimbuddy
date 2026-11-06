#include "backend_timing.h"

#include <ncnn/cpu.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace aimbuddy {

namespace {

/// Runs thrown away before timing, so caches and GPU pipelines are warm, then the runs that are timed.
constexpr int kBenchmarkWarmupRuns = 3;
constexpr int kBenchmarkTimedRuns = 12;

/// CPU worker threads are capped so the game keeps most of the fast cores.
constexpr int kMaxCpuThreads = 3;

}  // namespace

int fastCoreThreadCount() {
    const int big = ncnn::get_big_cpu_count();
    if (big > 0) {
        return std::min(big, kMaxCpuThreads);
    }
    return std::max(1, std::min(ncnn::get_cpu_count() / 2, kMaxCpuThreads));
}

void pinToFastCores() {
    ncnn::set_cpu_powersave(2);
    ncnn::set_kmp_blocktime(kSpinWaitMs);
}

float medianRunMs(const std::function<bool()>& runOnce, const std::function<bool()>& shouldStop) {
    for (int i = 0; i < kBenchmarkWarmupRuns; ++i) {
        if (shouldStop() || !runOnce()) {
            return -1.0f;
        }
    }
    std::vector<float> times;
    times.reserve(kBenchmarkTimedRuns);
    for (int i = 0; i < kBenchmarkTimedRuns; ++i) {
        if (shouldStop()) {
            return -1.0f;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!runOnce()) {
            return -1.0f;
        }
        times.push_back(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

}  // namespace aimbuddy
