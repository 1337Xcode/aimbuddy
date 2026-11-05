#include "core/screen_mapping.h"
#include "core/touch_protocol.h"
#include "test_harness.h"

using namespace aimbuddy::touch;

namespace {

// The touchscreen of the phone this was developed on: 8640 x 19200 raw units, built in portrait.
const RawRange kPanel{0, 8639, 0, 19199};

// The same phone's display: 1080 x 2400 pixels upright, so 2400 x 1080 when turned on its side.
constexpr int kLongEdge = 2400;
constexpr int kShortEdge = 1080;

}  // namespace

TEST(mapping_follows_the_panel_when_the_display_is_upright) {
    RawPoint topLeft = displayToRaw(0, 0, kShortEdge, kLongEdge, 0, kPanel);
    CHECK(topLeft.x == 0 && topLeft.y == 0);
    RawPoint bottomRight = displayToRaw(kShortEdge, kLongEdge, kShortEdge, kLongEdge, 0, kPanel);
    CHECK(bottomRight.x == kPanel.maxX && bottomRight.y == kPanel.maxY);
    RawPoint middle = displayToRaw(540, 1200, kShortEdge, kLongEdge, 0, kPanel);
    CHECK_NEAR(middle.x, 4319.5, 1.0);
    CHECK_NEAR(middle.y, 9599.5, 1.0);
}

TEST(mapping_turns_the_corners_for_a_quarter_turn_counter_clockwise) {
    // Rotation 1: the panel's top edge is on the player's left, so the display's top left corner
    // is the panel's top right.
    RawPoint topLeft = displayToRaw(0, 0, kLongEdge, kShortEdge, 1, kPanel);
    CHECK(topLeft.x == kPanel.maxX && topLeft.y == 0);
    RawPoint topRight = displayToRaw(kLongEdge, 0, kLongEdge, kShortEdge, 1, kPanel);
    CHECK(topRight.x == kPanel.maxX && topRight.y == kPanel.maxY);
    RawPoint bottomLeft = displayToRaw(0, kShortEdge, kLongEdge, kShortEdge, 1, kPanel);
    CHECK(bottomLeft.x == 0 && bottomLeft.y == 0);
}

TEST(mapping_turns_the_corners_for_a_quarter_turn_clockwise) {
    // Rotation 3: the display's top left corner is the panel's bottom left.
    RawPoint topLeft = displayToRaw(0, 0, kLongEdge, kShortEdge, 3, kPanel);
    CHECK(topLeft.x == 0 && topLeft.y == kPanel.maxY);
    RawPoint topRight = displayToRaw(kLongEdge, 0, kLongEdge, kShortEdge, 3, kPanel);
    CHECK(topRight.x == 0 && topRight.y == 0);
    RawPoint bottomLeft = displayToRaw(0, kShortEdge, kLongEdge, kShortEdge, 3, kPanel);
    CHECK(bottomLeft.x == kPanel.maxX && bottomLeft.y == kPanel.maxY);
}

TEST(mapping_flips_both_axes_when_upside_down) {
    RawPoint topLeft = displayToRaw(0, 0, kShortEdge, kLongEdge, 2, kPanel);
    CHECK(topLeft.x == kPanel.maxX && topLeft.y == kPanel.maxY);
    RawPoint bottomRight = displayToRaw(kShortEdge, kLongEdge, kShortEdge, kLongEdge, 2, kPanel);
    CHECK(bottomRight.x == 0 && bottomRight.y == 0);
}

TEST(mapping_keeps_positions_in_proportion_after_a_turn) {
    // A point a quarter of the way along the display's width stays a quarter of the way along the
    // panel's height after a quarter turn.
    RawPoint point = displayToRaw(kLongEdge * 0.25f, kShortEdge * 0.5f, kLongEdge, kShortEdge, 1, kPanel);
    CHECK_NEAR(point.y, 0.25 * kPanel.maxY, 1.0);
    CHECK_NEAR(point.x, 0.5 * kPanel.maxX, 1.0);
}

TEST(mapping_clamps_positions_outside_the_display) {
    RawPoint far = displayToRaw(-500, 99999, kShortEdge, kLongEdge, 0, kPanel);
    CHECK(far.x == 0 && far.y == kPanel.maxY);
}

TEST(mapping_respects_a_panel_that_does_not_start_at_zero) {
    const RawRange offset{100, 1099, 200, 2199};
    RawPoint topLeft = displayToRaw(0, 0, 1000, 2000, 0, offset);
    CHECK(topLeft.x == 100 && topLeft.y == 200);
    RawPoint bottomRight = displayToRaw(1000, 2000, 1000, 2000, 0, offset);
    CHECK(bottomRight.x == 1099 && bottomRight.y == 2199);
}

TEST(mapping_treats_odd_rotation_numbers_as_turns) {
    RawPoint a = displayToRaw(10, 20, kShortEdge, kLongEdge, 4, kPanel);
    RawPoint b = displayToRaw(10, 20, kShortEdge, kLongEdge, 0, kPanel);
    CHECK(a.x == b.x && a.y == b.y);
    RawPoint c = displayToRaw(10, 20, kLongEdge, kShortEdge, -1, kPanel);
    RawPoint d = displayToRaw(10, 20, kLongEdge, kShortEdge, 3, kPanel);
    CHECK(c.x == d.x && c.y == d.y);
}

TEST(mapping_runs_backwards_to_the_place_a_touch_started_from) {
    // A position sent to the panel and read back is where it began, to within a raw unit, for every turn.
    for (int rotation = 0; rotation < 4; ++rotation) {
        const bool sideways = rotation % 2 == 1;
        const int width = sideways ? kLongEdge : kShortEdge;
        const int height = sideways ? kShortEdge : kLongEdge;
        for (const float shareX : {0.0f, 0.2f, 0.5f, 0.93f, 1.0f}) {
            for (const float shareY : {0.0f, 0.31f, 0.5f, 0.77f, 1.0f}) {
                const float x = shareX * static_cast<float>(width);
                const float y = shareY * static_cast<float>(height);
                const RawPoint raw = displayToRaw(x, y, width, height, rotation, kPanel);
                const DisplayPoint back = rawToDisplay(raw.x, raw.y, width, height, rotation, kPanel);
                CHECK_NEAR(back.x, x, 0.3);
                CHECK_NEAR(back.y, y, 0.3);
            }
        }
    }
}

TEST(mapping_reads_a_raw_corner_as_the_display_corner_it_is) {
    // Rotation 1: the panel's top edge is on the player's left, so its top right corner is the display's top left.
    const DisplayPoint topLeft = rawToDisplay(kPanel.maxX, 0, kLongEdge, kShortEdge, 1, kPanel);
    CHECK_NEAR(topLeft.x, 0.0, 1e-3);
    CHECK_NEAR(topLeft.y, 0.0, 1e-3);
    // Rotation 3: the panel's top edge is on the player's right, so its top left corner is the display's top right.
    const DisplayPoint topRight = rawToDisplay(0, 0, kLongEdge, kShortEdge, 3, kPanel);
    CHECK_NEAR(topRight.x, kLongEdge, 1e-3);
    CHECK_NEAR(topRight.y, 0.0, 1e-3);
    // Raw units outside the panel's range are brought onto it.
    const DisplayPoint far = rawToDisplay(-500, 99999, kShortEdge, kLongEdge, 0, kPanel);
    CHECK_NEAR(far.x, 0.0, 1e-3);
    CHECK_NEAR(far.y, kLongEdge, 1e-3);
}

TEST(protocol_packets_keep_their_size_and_floats) {
    using namespace aimbuddy::touch::protocol;
    CHECK(sizeof(RequestPacket) == 16);
    CHECK(sizeof(ReportPacket) == 16);
    const RequestPacket move = makeAimRequest(Request::kAimMove, 1234.5f, -0.25f);
    CHECK(move.type == static_cast<uint8_t>(Request::kAimMove));
    CHECK_NEAR(bitsToFloat(move.a), 1234.5, 0.0);
    CHECK_NEAR(bitsToFloat(move.b), -0.25, 0.0);
    const ReportPacket status = makeStatus(State::kMerging, Reason::kNone);
    CHECK(status.type == static_cast<uint8_t>(Report::kStatus));
    CHECK(status.state == static_cast<uint8_t>(State::kMerging));
    const ReportPacket ready = makeReady(10, 8639, 19199);
    CHECK(ready.a == 10 && ready.b == 8639 && ready.c == 19199);
    const ReportPacket player = makePlayer(2, 120, 640);
    CHECK(player.type == static_cast<uint8_t>(Report::kPlayer));
    CHECK(player.a == 2 && player.b == 120 && player.c == 640);
}

// A display made smaller than its panel (`wm size`, as screen stretching apps do) is shown in a frame in the
// middle of the panel, and Android maps touches into that frame only. The expected values below are what an
// emulator with a 1080 x 2400 panel reported for `wm size 1080x1206`: the input viewport's physical frame was
// rows 597 to 1803, and a touch at panel row 1497 arrived at display row 900.
namespace {

const RawRange kEmulatorPanel{0, 32767, 0, 32767};
constexpr int kPanelShort = 1080;
constexpr int kPanelLong = 2400;
constexpr int kForcedLong = 1206;

}  // namespace

TEST(frame_of_a_shrunk_display_is_the_one_android_letterboxes_it_into) {
    const PanelFrame upright = panelFrameFor(kPanelShort, kForcedLong, 0, kPanelShort, kPanelLong);
    CHECK_NEAR(upright.left, 0.0, 1e-6);
    CHECK_NEAR(upright.width, 1.0, 1e-6);
    CHECK_NEAR(upright.top, 597.0 / 2400.0, 1e-6);
    CHECK_NEAR(upright.height, 1206.0 / 2400.0, 1e-6);
    // Turned on its side, the frame runs across the panel's long side, which is now the width.
    const PanelFrame sideways = panelFrameFor(kForcedLong, kPanelShort, 1, kPanelShort, kPanelLong);
    CHECK_NEAR(sideways.left, 597.0 / 2400.0, 1e-6);
    CHECK_NEAR(sideways.width, 1206.0 / 2400.0, 1e-6);
    CHECK_NEAR(sideways.top, 0.0, 1e-6);
    CHECK_NEAR(sideways.height, 1.0, 1e-6);
}

TEST(frame_of_a_display_narrower_than_its_panel_has_bars_at_the_sides) {
    // 540 x 2400 on a 1080 x 2400 panel: the height fills it, and the frame is 540 wide from column 270.
    const PanelFrame frame = panelFrameFor(540, kPanelLong, 0, kPanelShort, kPanelLong);
    CHECK_NEAR(frame.left, 270.0 / 1080.0, 1e-6);
    CHECK_NEAR(frame.width, 540.0 / 1080.0, 1e-6);
    CHECK_NEAR(frame.top, 0.0, 1e-6);
    CHECK_NEAR(frame.height, 1.0, 1e-6);
}

TEST(frame_is_the_whole_panel_when_the_display_fills_it_or_the_panel_is_unknown) {
    const PanelFrame full = panelFrameFor(kPanelShort, kPanelLong, 0, kPanelShort, kPanelLong);
    CHECK(full.left == 0.0f && full.top == 0.0f && full.width == 1.0f && full.height == 1.0f);
    const PanelFrame unknown = panelFrameFor(kPanelShort, kForcedLong, 0, 0, 0);
    CHECK(unknown.left == 0.0f && unknown.top == 0.0f && unknown.width == 1.0f && unknown.height == 1.0f);
}

TEST(mapping_puts_a_touch_inside_the_frame_of_a_shrunk_display) {
    // Upright: display row 900 is panel row 597 + 900 = 1497, which is 1497 / 2400 of the raw range.
    const PanelFrame upright = panelFrameFor(kPanelShort, kForcedLong, 0, kPanelShort, kPanelLong);
    const RawPoint middle = displayToRaw(540, 900, kPanelShort, kForcedLong, 0, kEmulatorPanel, upright);
    CHECK(middle.x == 16384);   // 0.5 * 32767, rounded
    CHECK(middle.y == 20438);   // 1497 / 2400 * 32767 = 20438.4
    // The top of the display is the top of the frame, not the top of the panel.
    const RawPoint top = displayToRaw(540, 0, kPanelShort, kForcedLong, 0, kEmulatorPanel, upright);
    CHECK(top.y == 8151);       // 597 / 2400 * 32767 = 8150.8
    // Sideways (rotation 1), display column 900 is the same place on the panel's long side.
    const PanelFrame sideways = panelFrameFor(kForcedLong, kPanelShort, 1, kPanelShort, kPanelLong);
    const RawPoint turned = displayToRaw(900, 540, kForcedLong, kPanelShort, 1, kEmulatorPanel, sideways);
    CHECK(turned.x == 16384);
    CHECK(turned.y == 20438);
}

TEST(mapping_reads_a_touch_in_the_frame_back_as_the_display_place_it_is) {
    const PanelFrame sideways = panelFrameFor(kForcedLong, kPanelShort, 1, kPanelShort, kPanelLong);
    for (const float x : {0.0f, 301.5f, 900.0f, 1206.0f}) {
        for (const float y : {0.0f, 540.0f, 1080.0f}) {
            const RawPoint raw = displayToRaw(x, y, kForcedLong, kPanelShort, 1, kEmulatorPanel, sideways);
            const DisplayPoint back = rawToDisplay(raw.x, raw.y, kForcedLong, kPanelShort, 1, kEmulatorPanel, sideways);
            CHECK_NEAR(back.x, x, 0.1);
            CHECK_NEAR(back.y, y, 0.1);
        }
    }
    // A finger in the bar beside the frame is read as being at the frame's edge.
    const PanelFrame upright = panelFrameFor(kPanelShort, kForcedLong, 0, kPanelShort, kPanelLong);
    const DisplayPoint inBar = rawToDisplay(16384, 100, kPanelShort, kForcedLong, 0, kEmulatorPanel, upright);
    CHECK_NEAR(inBar.y, 0.0, 1e-3);
}

TEST(mapping_is_unchanged_for_a_display_of_the_panels_own_shape) {
    // A phone set to a lower resolution of the same shape (1080 x 2400 shown on a 1440 x 3200 panel) has no bars, so a
    // touch goes where it went before the frame existed: the frame is the whole panel and every raw value is the same.
    for (int rotation = 0; rotation < 4; ++rotation) {
        const bool sideways = rotation % 2 == 1;
        const int width = sideways ? kLongEdge : kShortEdge;
        const int height = sideways ? kShortEdge : kLongEdge;
        const PanelFrame frame = panelFrameFor(width, height, rotation, 1440, 3200);
        CHECK(frame.left == 0.0f && frame.top == 0.0f && frame.width == 1.0f && frame.height == 1.0f);
        for (const float shareX : {0.0f, 0.37f, 1.0f}) {
            for (const float shareY : {0.0f, 0.61f, 1.0f}) {
                const float x = shareX * static_cast<float>(width);
                const float y = shareY * static_cast<float>(height);
                const RawPoint withFrame = displayToRaw(x, y, width, height, rotation, kPanel, frame);
                const RawPoint withoutFrame = displayToRaw(x, y, width, height, rotation, kPanel);
                CHECK(withFrame.x == withoutFrame.x && withFrame.y == withoutFrame.y);
            }
        }
    }
}
