#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>

#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

#include "core/detection_decoder.h"
#include "core/tflite_tensor.h"
#include "test_harness.h"

using namespace aimbuddy;

// LiteRT 2.3.2 returns this struct by value; its old 32-byte declaration overran the caller's stack.
static_assert(sizeof(TfLiteXNNPackDelegateOptions) == 64, "XNNPack options must match the LiteRT 2.3.2 ABI");
static_assert(offsetof(TfLiteXNNPackDelegateOptions, runtime_flags) == 4, "Unexpected runtime_flags offset");
static_assert(offsetof(TfLiteXNNPackDelegateOptions, flags) == 8, "Unexpected flags offset");
static_assert(offsetof(TfLiteXNNPackDelegateOptions, weights_cache) == 16, "Unexpected weights_cache offset");
static_assert(offsetof(TfLiteXNNPackDelegateOptions, weight_cache_file_path) == 32, "Unexpected cache path offset");
static_assert(offsetof(TfLiteXNNPackDelegateOptions, weight_cache_provider) == 48, "Unexpected cache provider offset");

namespace {

// Both layouts represent one 40 x 80 body centred at (100, 120), and one duplicate.
constexpr int kAnchors = 50;
const std::vector<float> kFloat = [] {
    std::vector<float> values(5 * kAnchors, 0.0f);
    values[1] = 100.0f;
    values[kAnchors + 1] = 120.0f;
    values[2 * kAnchors + 1] = 40.0f;
    values[3 * kAnchors + 1] = 80.0f;
    values[4 * kAnchors + 1] = 0.9f;
    values[2] = 101.0f;
    values[kAnchors + 2] = 120.0f;
    values[2 * kAnchors + 2] = 41.0f;
    values[3 * kAnchors + 2] = 81.0f;
    values[4 * kAnchors + 2] = 0.7f;
    return values;
}();

std::vector<float> transpose(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (int row = 0; row < 5; ++row) {
        for (int anchor = 0; anchor < kAnchors; ++anchor) result[anchor * 5 + row] = input[row * kAnchors + anchor];
    }
    return result;
}

void checkDecoded(const float* data, int rows, int cols) {
    DecodeSpec spec;
    CHECK(inferDecodeSpec(data, rows, cols, false, 0, &spec));
    ModelBox boxes[4];
    DecodeParams params;
    params.modelSize = 640;
    const int count = decodeDetections(data, rows, cols, spec, params,
                                       ModelProfile::defaultsFor(1), boxes, 4);
    CHECK(count == 1);
    CHECK_NEAR(boxes[0].left, 80.0, 1e-3);
    CHECK_NEAR(boxes[0].top, 80.0, 1e-3);
    CHECK_NEAR(boxes[0].right, 120.0, 1e-3);
    CHECK_NEAR(boxes[0].bottom, 160.0, 1e-3);
    CHECK_NEAR(boxes[0].score, 1.0, 1e-3);
}

}  // namespace

TEST(tflite_tensor_converts_float16_special_values_and_subnormals) {
    CHECK(float16ToFloat(0x3c00) == 1.0f);
    CHECK(float16ToFloat(0xc000) == -2.0f);
    CHECK_NEAR(float16ToFloat(0x3555), 0.33325195, 1e-7);
    CHECK_NEAR(float16ToFloat(0x0001), 5.9604645e-8, 1e-12);
    CHECK(std::isinf(float16ToFloat(0x7c00)));
    CHECK(std::isnan(float16ToFloat(0x7e00)));
    CHECK(float16ToFloat(0x8000) == 0.0f && std::signbit(float16ToFloat(0x8000)));
    for (float value : {0.0f, 1.0f, -2.0f, 0.33325195f, 5.9604645e-8f, 65504.0f}) {
        CHECK_NEAR(float16ToFloat(floatToFloat16(value)), value, std::max(1e-7, std::fabs(value) * 1e-3));
    }
    CHECK(floatToFloat16(1.0f) == 0x3c00);
    CHECK(floatToFloat16(-2.0f) == 0xc000);
    CHECK(floatToFloat16(5.9604645e-8f) == 0x0001);
    CHECK(std::isinf(float16ToFloat(floatToFloat16(1000000.0f))));
    CHECK(std::isnan(float16ToFloat(floatToFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

TEST(tflite_tensor_converts_each_supported_type_without_an_intermediate_vector) {
    const uint16_t half[] = {0x3c00, 0x4000, 0xc000};
    const int8_t signedValues[] = {-128, 0, 127};
    const uint8_t unsignedValues[] = {0, 128, 255};
    const float floats[] = {0.5f, -1.0f, 4.0f};
    float output[3];
    CHECK(tensorToFloat(half, TensorType::kFloat16, 3, 0.0f, 0, output));
    CHECK_NEAR(output[1], 2.0, 1e-6);
    CHECK(tensorToFloat(signedValues, TensorType::kInt8, 3, 0.02f, -3, output));
    CHECK_NEAR(output[0], -2.5, 1e-5);
    CHECK_NEAR(output[2], 2.6, 1e-5);
    CHECK(tensorToFloat(unsignedValues, TensorType::kUInt8, 3, 0.01f, 128, output));
    CHECK_NEAR(output[0], -1.28, 1e-5);
    CHECK_NEAR(output[2], 1.27, 1e-5);
    CHECK(tensorToFloat(floats, TensorType::kFloat32, 3, 0.0f, 0, output));
    CHECK(output[2] == 4.0f);
    CHECK(!tensorToFloat(signedValues, TensorType::kInt8, 3, 0.0f, 0, output));
    CHECK(!tensorToFloat(nullptr, TensorType::kFloat32, 3, 0.0f, 0, output));
}

TEST(tflite_input_preserves_rgb_order_in_nchw_and_nhwc) {
    const uint8_t rgb[] = {255, 0, 0, 0, 128, 0, 0, 0, 64, 255, 128, 64};
    float output[12] = {};
    CHECK(rgbToTensor(rgb, 4, TensorType::kFloat32, 0.0f, 0, true, output));
    const float expected[] = {1, 0, 0, 1, 0, 128.0f / 255, 0, 128.0f / 255,
                              0, 0, 64.0f / 255, 64.0f / 255};
    for (int i = 0; i < 12; ++i) CHECK_NEAR(output[i], expected[i], 1e-6);

    CHECK(rgbToTensor(rgb, 4, TensorType::kFloat32, 0.0f, 0, false, output));
    for (int i = 0; i < 12; ++i) CHECK_NEAR(output[i], rgb[i] / 255.0f, 1e-6);
}

TEST(tflite_input_quantizes_planar_rgb_with_the_tensor_parameters) {
    const uint8_t rgb[] = {255, 0, 0, 0, 128, 0, 0, 0, 64, 255, 128, 64};
    int8_t output[12] = {};
    CHECK(rgbToTensor(rgb, 4, TensorType::kInt8, 1.0f / 255.0f, -128, true, output));
    CHECK(output[0] == 127);
    CHECK(output[4] == -128);
    CHECK(output[5] == 0);
    CHECK(output[8] == -128);
    CHECK(output[11] == -64);
}

TEST(tflite_resize_reuses_the_same_ncnn_sampling_with_padded_rows) {
    const uint8_t pixels[2][2][3] = {{{0, 0, 0}, {40, 80, 120}}, {{80, 120, 160}, {160, 200, 240}}};
    uint8_t rgba[24];
    std::memset(rgba, 255, sizeof(rgba));
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            for (int c = 0; c < 3; ++c) rgba[y * 12 + x * 4 + c] = pixels[y][x][c];
        }
    }
    const int expected[9][3] = {{0, 0, 0}, {20, 40, 60}, {40, 80, 120},
                                {40, 60, 80}, {70, 100, 130}, {100, 140, 180},
                                {80, 120, 160}, {120, 160, 200}, {160, 200, 240}};
    RgbResizer resizer;
    CHECK(resizer.configure(2, 3));
    CHECK(resizer.configure(2, 3));
    uint8_t output[27] = {};
    resizer.resize(rgba, 12, output);
    for (int i = 0; i < 9; ++i) {
        for (int c = 0; c < 3; ++c) CHECK(output[i * 3 + c] == expected[i][c]);
    }
}

TEST(tflite_quantized_tensors_decode_in_both_yolo_layouts) {
    const float scale = 1.0f;
    auto quantize = [scale](const std::vector<float>& values, std::vector<uint8_t>* output) {
        output->resize(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            (*output)[i] = static_cast<uint8_t>(std::clamp(std::lround(values[i] / scale), 0L, 255L));
        }
    };
    std::vector<uint8_t> quantized;
    quantize(kFloat, &quantized);
    std::vector<float> decoded(kFloat.size());
    CHECK(tensorToFloat(quantized.data(), TensorType::kUInt8, quantized.size(), scale, 0, decoded.data()));
    checkDecoded(decoded.data(), 5, kAnchors);

    const std::vector<float> transposed = transpose(kFloat);
    quantize(transposed, &quantized);
    CHECK(tensorToFloat(quantized.data(), TensorType::kUInt8, quantized.size(), scale, 0, decoded.data()));
    checkDecoded(decoded.data(), kAnchors, 5);
}
