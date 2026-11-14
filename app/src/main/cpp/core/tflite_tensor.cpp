#include "tflite_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace aimbuddy {

float float16ToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
    int exponent = (value >> 10) & 0x1f;
    uint32_t fraction = value & 0x03ff;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            // Renormalize a half subnormal into the float exponent range.
            exponent = 1;
            while ((fraction & 0x0400) == 0) {
                fraction <<= 1;
                --exponent;
            }
            fraction &= 0x03ff;
            const int floatExponent = exponent + 127 - 15;
            bits = sign | (static_cast<uint32_t>(floatExponent) << 23) | (fraction << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (fraction << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (fraction << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t floatToFloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int exponent = static_cast<int>((bits >> 23) & 0xff);
    const uint32_t fraction = bits & 0x7fffff;
    if (exponent == 255) {
        return static_cast<uint16_t>(sign | 0x7c00 | (fraction == 0 ? 0 : 0x200));
    }
    const int halfExponent = exponent - 127 + 15;
    if (halfExponent >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    if (halfExponent <= 0) {
        if (halfExponent < -10) return static_cast<uint16_t>(sign);
        const uint32_t mantissa = fraction | 0x800000;
        const int shift = 14 - halfExponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1);
        if (remainder > (1u << (shift - 1)) ||
            (remainder == (1u << (shift - 1)) && (half & 1) != 0)) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(halfExponent) << 10) | (fraction >> 13);
    const uint32_t remainder = fraction & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0)) ++half;
    return static_cast<uint16_t>(sign | half);
}

bool rgbToTensor(const uint8_t* rgb, size_t pixelCount, TensorType type, float scale, int zeroPoint,
                 bool channelsFirst, void* out) {
    if (rgb == nullptr || out == nullptr || pixelCount > SIZE_MAX / 3) return false;
    const bool quantized = type == TensorType::kInt8 || type == TensorType::kUInt8;
    if (quantized && (!(scale > 0.0f) || !std::isfinite(scale))) return false;
    const int minimum = type == TensorType::kInt8 ? -128 : 0;
    const int maximum = type == TensorType::kInt8 ? 127 : 255;
    for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float normalized = rgb[pixel * 3 + channel] / 255.0f;
            const size_t index = channelsFirst ? channel * pixelCount + pixel : pixel * 3 + channel;
            if (type == TensorType::kFloat32) {
                static_cast<float*>(out)[index] = normalized;
            } else if (type == TensorType::kFloat16) {
                static_cast<uint16_t*>(out)[index] = floatToFloat16(normalized);
            } else {
                const int value = std::clamp(static_cast<int>(std::lround(normalized / scale + zeroPoint)), minimum, maximum);
                if (type == TensorType::kInt8) {
                    static_cast<int8_t*>(out)[index] = static_cast<int8_t>(value);
                } else {
                    static_cast<uint8_t*>(out)[index] = static_cast<uint8_t>(value);
                }
            }
        }
    }
    return true;
}

bool tensorToFloat(const void* data, TensorType type, size_t count, float scale, int zeroPoint, float* out) {
    if (data == nullptr || out == nullptr || count > SIZE_MAX / sizeof(float)) return false;
    switch (type) {
        case TensorType::kFloat32: {
            const float* input = static_cast<const float*>(data);
            for (size_t i = 0; i < count; ++i) out[i] = input[i];
            return true;
        }
        case TensorType::kFloat16: {
            const uint16_t* input = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < count; ++i) out[i] = float16ToFloat(input[i]);
            return true;
        }
        case TensorType::kInt8: {
            if (!(scale > 0.0f) || !std::isfinite(scale)) return false;
            const int8_t* input = static_cast<const int8_t*>(data);
            for (size_t i = 0; i < count; ++i) out[i] = (static_cast<int>(input[i]) - zeroPoint) * scale;
            return true;
        }
        case TensorType::kUInt8: {
            if (!(scale > 0.0f) || !std::isfinite(scale)) return false;
            const uint8_t* input = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < count; ++i) out[i] = (static_cast<int>(input[i]) - zeroPoint) * scale;
            return true;
        }
    }
    return false;
}

namespace {

// Identical coefficients and intermediate rounding to ncnn::resize_bilinear_c3.
void sourceCoordinate(int destination, int source, int target, int* coordinate, int* weight0, int* weight1) {
    float fraction = static_cast<float>((destination + 0.5) * static_cast<double>(source) / target - 0.5);
    int pixel = static_cast<int>(std::floor(fraction));
    fraction -= pixel;
    if (pixel < 0) {
        pixel = 0;
        fraction = 0.0f;
    }
    if (pixel >= source - 1) {
        pixel = source - 2;
        fraction = 1.0f;
    }
    *coordinate = pixel;
    *weight0 = static_cast<int>((1.0f - fraction) * 2048.0f + 0.5f);
    *weight1 = static_cast<int>(fraction * 2048.0f + 0.5f);
}

float distributionExpectation(const float* values) {
    constexpr int kBins = 16;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < kBins; ++i) maximum = std::max(maximum, values[i]);
    float weighted = 0.0f;
    float total = 0.0f;
    for (int i = 0; i < kBins; ++i) {
        const float probability = std::exp(values[i] - maximum);
        weighted += i * probability;
        total += probability;
    }
    return total > 0.0f && std::isfinite(total) ? weighted / total : 0.0f;
}

float sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

}  // namespace

void resizeRgbRgba(const uint8_t* rgba, int side, int stride, int inputSize, uint8_t* out) {
    if (rgba == nullptr || out == nullptr || side < 2 || inputSize <= 0 || stride < side * 4) return;
    for (int y = 0; y < inputSize; ++y) {
        int sy, b0, b1;
        sourceCoordinate(y, side, inputSize, &sy, &b0, &b1);
        const uint8_t* row0 = rgba + static_cast<size_t>(sy) * stride;
        const uint8_t* row1 = row0 + stride;
        for (int x = 0; x < inputSize; ++x) {
            int sx, a0, a1;
            sourceCoordinate(x, side, inputSize, &sx, &a0, &a1);
            const uint8_t* source0 = row0 + sx * 4;
            const uint8_t* source1 = row1 + sx * 4;
            for (int channel = 0; channel < 3; ++channel) {
                const int top = (source0[channel] * a0 + source0[4 + channel] * a1) >> 4;
                const int bottom = (source1[channel] * a0 + source1[4 + channel] * a1) >> 4;
                *out++ = static_cast<uint8_t>((top * b0 + bottom * b1 + (1 << 21)) >> 22);
            }
        }
    }
}

bool decodeSmartYolo(int inputSize, const SmartYoloScale* scales, size_t count,
                     float* output, size_t outputElements) {
    constexpr int kBins = 16;
    constexpr int kClasses = 80;
    if (inputSize <= 0 || scales == nullptr || count != 3 || output == nullptr) return false;
    int anchors = 0;
    for (size_t i = 0; i < count; ++i) {
        if (scales[i].stride <= 0 || scales[i].grid <= 0 || scales[i].grid * scales[i].stride != inputSize ||
            scales[i].boxes == nullptr || scales[i].scores == nullptr || anchors > 1000000) return false;
        anchors += scales[i].grid * scales[i].grid;
    }
    if (outputElements != static_cast<size_t>(anchors) * 5) return false;
    int offset = 0;
    for (size_t i = 0; i < count; ++i) {
        const SmartYoloScale& scale = scales[i];
        const int cells = scale.grid * scale.grid;
        for (int y = 0; y < scale.grid; ++y) {
            for (int x = 0; x < scale.grid; ++x) {
                const int cell = y * scale.grid + x;
                const int anchor = offset + cell;
                const float* distribution = scale.boxes + static_cast<size_t>(cell) * 4 * kBins;
                const float left = distributionExpectation(distribution);
                const float top = distributionExpectation(distribution + kBins);
                const float right = distributionExpectation(distribution + 2 * kBins);
                const float bottom = distributionExpectation(distribution + 3 * kBins);
                output[anchor] = (x + 0.5f + (right - left) * 0.5f) * scale.stride / inputSize;
                output[anchors + anchor] = (y + 0.5f + (bottom - top) * 0.5f) * scale.stride / inputSize;
                output[2 * anchors + anchor] = (left + right) * scale.stride / inputSize;
                output[3 * anchors + anchor] = (top + bottom) * scale.stride / inputSize;
                output[4 * anchors + anchor] = sigmoid(scale.scores[static_cast<size_t>(cell) * kClasses]);
            }
        }
        offset += cells;
    }
    return true;
}

}  // namespace aimbuddy
