#include <cstdint>
#include <vector>

#include "core/model_tensors.h"
#include "test_harness.h"

using namespace aimbuddy;

namespace {

/// A 2 by 2 RGBA picture, row by row, with a padding byte after each row (stride 9) and alpha set
/// to values that must never reach the tensor.
const uint8_t kPicture[] = {
    0, 255, 51, 7, 102, 153, 204, 7, 0,   // row 0: (0, 255, 51), (102, 153, 204), padding
    255, 0, 0, 7, 1, 2, 3, 7, 0,          // row 1: (255, 0, 0), (1, 2, 3), padding
};
constexpr int kStride = 9;

InputSpec specOf(PixelLayout layout, TensorElement element, float scale = 0.0f, int zeroPoint = 0) {
    InputSpec spec;
    spec.size = 2;
    spec.layout = layout;
    spec.element = element;
    spec.quantization.scale = scale;
    spec.quantization.zeroPoint = zeroPoint;
    return spec;
}

}  // namespace

TEST(input_shape_tells_interleaved_from_planar) {
    int size = 0;
    PixelLayout layout = PixelLayout::kPlanar;
    const int nhwc[] = {1, 256, 256, 3};
    CHECK(readInputShape(nhwc, 4, &size, &layout));
    CHECK(size == 256);
    CHECK(layout == PixelLayout::kInterleaved);

    const int nchw[] = {1, 3, 320, 320};
    CHECK(readInputShape(nchw, 4, &size, &layout));
    CHECK(size == 320);
    CHECK(layout == PixelLayout::kPlanar);
}

TEST(input_shape_refuses_what_the_region_cannot_fill) {
    int size = 0;
    PixelLayout layout = PixelLayout::kPlanar;
    const int notSquare[] = {1, 256, 320, 3};
    const int batchOfTwo[] = {2, 256, 256, 3};
    const int fourChannels[] = {1, 256, 256, 4};
    const int noBatch[] = {256, 256, 3};
    CHECK(!readInputShape(notSquare, 4, &size, &layout));
    CHECK(!readInputShape(batchOfTwo, 4, &size, &layout));
    CHECK(!readInputShape(fourChannels, 4, &size, &layout));
    CHECK(!readInputShape(noBatch, 3, &size, &layout));
}

TEST(output_shape_is_read_as_rows_and_columns) {
    int rows = 0;
    int cols = 0;
    const int raw[] = {1, 7, 1344};
    CHECK(readOutputShape(raw, 3, &rows, &cols));
    CHECK(rows == 7);
    CHECK(cols == 1344);
    const int flat[] = {300, 6};
    CHECK(readOutputShape(flat, 2, &rows, &cols));
    CHECK(rows == 300);
    CHECK(cols == 6);

    const int batchOfTwo[] = {2, 7, 1344};
    const int fourDims[] = {1, 1, 7, 1344};
    const int empty[] = {1, 0, 1344};
    CHECK(!readOutputShape(batchOfTwo, 3, &rows, &cols));
    CHECK(!readOutputShape(fourDims, 4, &rows, &cols));
    CHECK(!readOutputShape(empty, 3, &rows, &cols));
}

TEST(float_input_is_written_pixel_by_pixel) {
    InputWriter writer;
    CHECK(writer.configure(specOf(PixelLayout::kInterleaved, TensorElement::kFloat32)));
    std::vector<float> tensor(12, -1.0f);
    writer.write(kPicture, kStride, tensor.data());
    const float expected[] = {0.0f, 1.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 0.0f, 0.0f, 1.0f / 255, 2.0f / 255, 3.0f / 255};
    for (int i = 0; i < 12; ++i) {
        CHECK_NEAR(tensor[i], expected[i], 1e-6);
    }
}

TEST(float_input_is_written_plane_by_plane) {
    InputWriter writer;
    CHECK(writer.configure(specOf(PixelLayout::kPlanar, TensorElement::kFloat32)));
    std::vector<float> tensor(12, -1.0f);
    writer.write(kPicture, kStride, tensor.data());
    // All four reds, then the greens, then the blues.
    const float expected[] = {0.0f, 0.4f, 1.0f, 1.0f / 255, 1.0f, 0.6f, 0.0f, 2.0f / 255, 0.2f, 0.8f, 0.0f, 3.0f / 255};
    for (int i = 0; i < 12; ++i) {
        CHECK_NEAR(tensor[i], expected[i], 1e-6);
    }
}

TEST(signed_input_is_quantized_with_the_model_scale) {
    // The usual full-integer export: one step per byte value, zero point -128.
    InputWriter writer;
    CHECK(writer.configure(specOf(PixelLayout::kInterleaved, TensorElement::kInt8, 1.0f / 255, -128)));
    std::vector<int8_t> tensor(12, 0);
    writer.write(kPicture, kStride, tensor.data());
    const int expected[] = {-128, 127, -77, -26, 25, 76, 127, -128, -128, -127, -126, -125};
    for (int i = 0; i < 12; ++i) {
        CHECK(tensor[i] == expected[i]);
    }
}

TEST(unsigned_input_is_quantized_and_kept_in_range) {
    // Half a step per byte value: the brighter half of the bytes would overflow and is held at 255.
    InputWriter writer;
    CHECK(writer.configure(specOf(PixelLayout::kPlanar, TensorElement::kUInt8, 1.0f / 510, 0)));
    std::vector<uint8_t> tensor(12, 0);
    writer.write(kPicture, kStride, tensor.data());
    const int expected[] = {0, 204, 255, 2, 255, 255, 0, 4, 102, 255, 0, 6};
    for (int i = 0; i < 12; ++i) {
        CHECK(tensor[i] == expected[i]);
    }
}

TEST(integer_input_without_a_scale_is_refused) {
    InputWriter writer;
    CHECK(!writer.configure(specOf(PixelLayout::kInterleaved, TensorElement::kInt8, 0.0f, -128)));
    CHECK(!writer.configure(specOf(PixelLayout::kInterleaved, TensorElement::kUInt8, -0.5f, 0)));
}

TEST(quantized_output_is_turned_back_into_real_values) {
    Quantization quantization;
    quantization.scale = 0.5f;
    quantization.zeroPoint = -128;
    const int8_t stored[] = {-128, 0, 127};
    float real[3] = {};
    dequantize(stored, TensorElement::kInt8, quantization, 3, real);
    CHECK_NEAR(real[0], 0.0, 1e-6);
    CHECK_NEAR(real[1], 64.0, 1e-6);
    CHECK_NEAR(real[2], 127.5, 1e-6);

    quantization.zeroPoint = 10;
    const uint8_t storedUnsigned[] = {10, 0, 255};
    dequantize(storedUnsigned, TensorElement::kUInt8, quantization, 3, real);
    CHECK_NEAR(real[0], 0.0, 1e-6);
    CHECK_NEAR(real[1], -5.0, 1e-6);
    CHECK_NEAR(real[2], 122.5, 1e-6);
}
