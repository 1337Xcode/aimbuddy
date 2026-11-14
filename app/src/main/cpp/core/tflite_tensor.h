#ifndef AIMBUDDY_CORE_TFLITE_TENSOR_H
#define AIMBUDDY_CORE_TFLITE_TENSOR_H

#include <cstddef>
#include <cstdint>

namespace aimbuddy {

/// The numeric formats accepted at a TFLite model's input and output tensors.
enum class TensorType : unsigned char { kFloat32, kFloat16, kInt8, kUInt8 };

/// Converts packed RGB bytes to float or quantized model input. A planar tensor stores
/// all red pixels, then green, then blue; an interleaved tensor keeps each pixel's RGB together.
bool rgbToTensor(const uint8_t* rgb, size_t pixelCount, TensorType type, float scale, int zeroPoint,
                 bool channelsFirst, void* out);

/// Converts one tensor to float without allocating. Returns false for an invalid count or missing data.
bool tensorToFloat(const void* data, TensorType type, size_t count, float scale, int zeroPoint, float* out);

/// Converts a finite single-precision value to the nearest IEEE half, saturating overflow to infinity.
/// Converts an IEEE 754 half-precision value to single precision. No floating-point extensions are required.
float float16ToFloat(uint16_t value);

/// Converts a finite single-precision value to the nearest IEEE half, saturating overflow to infinity.
uint16_t floatToFloat16(float value);

/// Resizes an RGBA square to packed NHWC RGB, with NCNN-compatible half-pixel bilinear sampling.
/// `out` must hold inputSize * inputSize * 3 bytes. No allocation is performed.
void resizeRgbRgba(const uint8_t* rgba, int side, int stride, int inputSize, uint8_t* out);

/// One pair of EdgeFirst YOLO11 feature maps: 4 DFL distributions and COCO class logits per cell.
struct SmartYoloScale {
    int stride = 0;
    int grid = 0;
    const float* boxes = nullptr;
    const float* scores = nullptr;
};

/// Combines three fixed feature-map pairs into a channels-first [5, anchors] person tensor.
bool decodeSmartYolo(int inputSize, const SmartYoloScale* scales, size_t count,
                     float* output, size_t outputElements);

}  // namespace aimbuddy

#endif  // AIMBUDDY_CORE_TFLITE_TENSOR_H
