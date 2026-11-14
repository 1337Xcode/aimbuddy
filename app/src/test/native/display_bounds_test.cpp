#include <cmath>
#include <limits>

#include "core/display_bounds.h"
#include "test_harness.h"

using aimbuddy::clampDisplayPoint;

TEST(display_points_stay_inside_the_inclusive_sample_bounds) {
    float x = -20.0f;
    float y = 4000.0f;
    CHECK(clampDisplayPoint(&x, &y, 2400, 1080));
    CHECK_NEAR(x, 0.0f, 0.0f);
    CHECK_NEAR(y, 1079.0f, 0.0f);

    x = 1200.25f;
    y = 540.5f;
    CHECK(clampDisplayPoint(&x, &y, 2400, 1080));
    CHECK_NEAR(x, 1200.25f, 0.0f);
    CHECK_NEAR(y, 540.5f, 0.0f);

    x = 2400.0f;
    y = -0.01f;
    CHECK(clampDisplayPoint(&x, &y, 2400, 1080));
    CHECK_NEAR(x, 2399.0f, 0.0f);
    CHECK_NEAR(y, 0.0f, 0.0f);
}

TEST(non_finite_display_points_are_rejected_not_relocated) {
    constexpr float infinity = std::numeric_limits<float>::infinity();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    const float invalid[] = {infinity, -infinity, nan};
    for (float value : invalid) {
        float x = value;
        float y = 500.0f;
        CHECK(!clampDisplayPoint(&x, &y, 2400, 1080));
        x = 500.0f;
        y = value;
        CHECK(!clampDisplayPoint(&x, &y, 2400, 1080));
    }
}

TEST(display_points_require_known_positive_display_geometry) {
    float x = 10.0f;
    float y = 20.0f;
    CHECK(!clampDisplayPoint(&x, &y, 0, 1080));
    CHECK(!clampDisplayPoint(&x, &y, 2400, -1));
    CHECK(!clampDisplayPoint(nullptr, &y, 2400, 1080));
    CHECK(!clampDisplayPoint(&x, nullptr, 2400, 1080));
}
