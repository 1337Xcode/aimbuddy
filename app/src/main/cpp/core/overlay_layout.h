#ifndef AIMBUDDY_CORE_OVERLAY_LAYOUT_H
#define AIMBUDDY_CORE_OVERLAY_LAYOUT_H

#include <algorithm>
#include <cmath>

#include "tracker.h"

namespace aimbuddy {

struct OverlayLabelRect {
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

struct OverlayLabelPosition {
    float x = 0.0f;
    float y = 0.0f;
    bool visible = false;
};

/// Places short labels without covering target outlines, other labels, or the screen edge.
class OverlayLabelLayout {
public:
    OverlayLabelLayout(float viewportWidth, float viewportHeight)
        : viewportWidth_(viewportWidth), viewportHeight_(viewportHeight) {}

    void reserve(const OverlayLabelRect& obstacle) {
        if (count_ < static_cast<int>(sizeof(occupied_) / sizeof(occupied_[0])) && valid(obstacle)) {
            occupied_[count_++] = obstacle;
        }
    }

    OverlayLabelPosition place(const OverlayLabelRect& target, float width, float height) {
        constexpr float margin = 2.0f;
        constexpr float gap = 3.0f;
        if (!valid(target) || !(width > 0.0f) || !(height > 0.0f) || !std::isfinite(width) ||
            !std::isfinite(height) || !(viewportWidth_ >= width + 2.0f * margin) ||
            !(viewportHeight_ >= height + 2.0f * margin)) {
            return {};
        }

        const float maxX = viewportWidth_ - width - margin;
        const float maxY = viewportHeight_ - height - margin;
        const float centerX = (target.left + target.right) * 0.5f - width * 0.5f;
        const float centerY = (target.top + target.bottom) * 0.5f - height * 0.5f;
        auto fitted = [=](float x, float y) {
            const float left = std::clamp(x, margin, maxX);
            const float top = std::clamp(y, margin, maxY);
            return OverlayLabelRect{left, top, left + width, top + height};
        };
        const OverlayLabelRect choices[] = {
            fitted(centerX, target.top - gap - height),
            fitted(centerX, target.bottom + gap),
            fitted(target.right + gap, centerY),
            fitted(target.left - gap - width, centerY),
        };

        for (const OverlayLabelRect& choice : choices) {
            const OverlayLabelRect halo = expanded(choice, 1.0f);
            bool blocked = false;
            for (int i = 0; i < count_ && !blocked; ++i) {
                blocked = intersects(halo, occupied_[i]);
            }
            if (!blocked) {
                reserve(halo);
                return {choice.left, choice.top, true};
            }
        }
        return {};
    }

private:
    static bool valid(const OverlayLabelRect& rect) {
        return std::isfinite(rect.left) && std::isfinite(rect.top) && std::isfinite(rect.right) &&
               std::isfinite(rect.bottom) && rect.right >= rect.left && rect.bottom >= rect.top;
    }

    static OverlayLabelRect expanded(const OverlayLabelRect& rect, float amount) {
        return {rect.left - amount, rect.top - amount, rect.right + amount, rect.bottom + amount};
    }

    static bool intersects(const OverlayLabelRect& a, const OverlayLabelRect& b) {
        return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    }

    float viewportWidth_;
    float viewportHeight_;
    OverlayLabelRect occupied_[kMaxTracks * 2]{};
    int count_ = 0;
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_CORE_OVERLAY_LAYOUT_H
