#ifndef AIMBUDDY_DETECTOR_BACKEND_TIMING_H
#define AIMBUDDY_DETECTOR_BACKEND_TIMING_H

#include <functional>

namespace aimbuddy {

/// The GPU is shared with the game, so it must beat the CPU by a clear margin to be used: its time
/// must be under this share of the CPU's.
constexpr float kGpuMustBeFasterBy = 0.80f;

/// How long idle CPU workers spin before sleeping. A library default of a few hundred milliseconds
/// would keep every worker busy between 30 to 60 Hz frames, wasting power and heat for no latency gain.
constexpr int kSpinWaitMs = 1;

/// Worker threads for CPU inference: the fast cores, but at most three, so the game keeps the rest.
int fastCoreThreadCount();

/**
 * Binds the calling thread to the fast cores, leaving the little cores to the game and the system.
 * A thread the caller starts afterwards inherits the binding, which is how an inference library's
 * own worker threads end up on the fast cores too.
 */
void pinToFastCores();

/**
 * Median time of one run in milliseconds, after a few runs that warm caches and GPU pipelines up.
 * Returns a negative number when a run fails or `shouldStop` says true between two runs.
 */
float medianRunMs(const std::function<bool()>& runOnce, const std::function<bool()>& shouldStop);

/// Whether a GPU that took `gpuMs` a run is worth taking over a CPU that took `cpuMs`. A negative time is a failed timing.
inline bool gpuIsFastEnough(float cpuMs, float gpuMs) {
    return cpuMs > 0.0f && gpuMs >= 0.0f && gpuMs < cpuMs * kGpuMustBeFasterBy;
}

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_BACKEND_TIMING_H
