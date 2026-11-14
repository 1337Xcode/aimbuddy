#ifndef AIMBUDDY_CORE_DISPLAY_BOUNDS_H
#define AIMBUDDY_CORE_DISPLAY_BOUNDS_H

#include <algorithm>
#include <cmath>

namespace aimbuddy {

/**
 * Validates and clamps one point in physical display pixels. Invalid geometry or non-finite
 * coordinates are rejected; silently replacing either with zero would send a finger to an edge.
 */
inline bool clampDisplayPoint(float* x, float* y, int displayWidth, int displayHeight) {
    if (x == nullptr || y == nullptr || displayWidth <= 0 || displayHeight <= 0 ||
        !std::isfinite(*x) || !std::isfinite(*y)) {
        return false;
    }
    // The valid sample range is inclusive: a 2400x1080 display accepts x in [0, 2399].
    *x = std::clamp(*x, 0.0f, static_cast<float>(displayWidth - 1));
    *y = std::clamp(*y, 0.0f, static_cast<float>(displayHeight - 1));
    return true;
}

}  // namespace aimbuddy

#endif  // AIMBUDDY_CORE_DISPLAY_BOUNDS_H
