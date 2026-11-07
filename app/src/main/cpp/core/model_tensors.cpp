#include "model_tensors.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aimbuddy {

namespace {

/// Values per pixel of a picture input (red, green, blue), and bytes per pixel of the RGBA frames.
constexpr int kChannels = 3;
constexpr int kRgbaBytes = 4;

/// The channel value a byte stands for: 0 to 1, as the models were trained on.
float channelValue(int byte) { return static_cast<float>(byte) * (1.0f / 255.0f); }

/// Stores a real value in an integer type T: divided by the scale, rounded, shifted by the zero point, kept in range.
template <typename T>
T quantize(float value, const Quantization& quantization) {
    const long stored = std::lround(value / quantization.scale) + quantization.zeroPoint;
    const long lowest = std::numeric_limits<T>::min();
    const long highest = std::numeric_limits<T>::max();
    return static_cast<T>(std::clamp(stored, lowest, highest));
}

/// Writes the pixels as red, green, blue one after the other, row by row.
template <typename T>
void writeInterleaved(const uint8_t* rgba, int size, int stride, const T* table, T* out) {
    for (int y = 0; y < size; ++y) {
        const uint8_t* pixel = rgba + static_cast<long>(y) * stride;
        for (int x = 0; x < size; ++x) {
            out[0] = table[pixel[0]];
            out[1] = table[pixel[1]];
            out[2] = table[pixel[2]];
            pixel += kRgbaBytes;
            out += kChannels;
        }
    }
}

/// Writes all the red values, then all the green, then all the blue.
template <typename T>
void writePlanar(const uint8_t* rgba, int size, int stride, const T* table, T* out) {
    const long plane = static_cast<long>(size) * size;
    T* red = out;
    T* green = out + plane;
    T* blue = out + 2 * plane;
    for (int y = 0; y < size; ++y) {
        const uint8_t* pixel = rgba + static_cast<long>(y) * stride;
        for (int x = 0; x < size; ++x) {
            *red++ = table[pixel[0]];
            *green++ = table[pixel[1]];
            *blue++ = table[pixel[2]];
            pixel += kRgbaBytes;
        }
    }
}

template <typename T>
void writePicture(const uint8_t* rgba, int stride, const InputSpec& spec, const T* table, void* tensor) {
    T* out = static_cast<T*>(tensor);
    if (spec.layout == PixelLayout::kPlanar) {
        writePlanar(rgba, spec.size, stride, table, out);
    } else {
        writeInterleaved(rgba, spec.size, stride, table, out);
    }
}

/// Steps of the quantization that the ends of its range may be away from the colour range they stand for.
constexpr float kRangeToleranceSteps = 2.0f;

/// The colour ranges an 8-bit input is quantized for: 0 to 1 (Ultralytics' exports), raw pixels from 0 to 255 (a
/// model that scales them itself) and -1 to 1 (older quantized detectors). Lowest and highest value of each.
constexpr float kColourRanges[][2] = {{0.0f, 1.0f}, {0.0f, 255.0f}, {-1.0f, 1.0f}};

/// The colour range that the lowest and highest value T can store stand for, if it is one of kColourRanges.
template <typename T>
bool colourRangeOf(const Quantization& quantization, ColourRange* range) {
    const float lowest = quantization.scale * static_cast<float>(static_cast<int>(std::numeric_limits<T>::min()) - quantization.zeroPoint);
    const float highest = quantization.scale * static_cast<float>(static_cast<int>(std::numeric_limits<T>::max()) - quantization.zeroPoint);
    const float tolerance = kRangeToleranceSteps * quantization.scale;
    for (const auto& candidate : kColourRanges) {
        if (std::fabs(lowest - candidate[0]) <= tolerance && std::fabs(highest - candidate[1]) <= tolerance) {
            range->lowest = candidate[0];
            range->highest = candidate[1];
            return true;
        }
    }
    return false;
}

/// The colour range an input is filled in. A float input gets 0 to 1, as the models were trained; an integer one
/// gets the range its quantization stands for. False for an integer input without a usable scale, or whose range is
/// none of those (only up to 0.5, say), which this writer would fill wrongly.
bool colourRangeFor(TensorElement element, const Quantization& quantization, ColourRange* range) {
    if (element == TensorElement::kFloat32) {
        *range = ColourRange();
        return true;
    }
    if (!(quantization.scale > 0.0f && std::isfinite(quantization.scale))) {
        return false;
    }
    return element == TensorElement::kInt8 ? colourRangeOf<int8_t>(quantization, range)
                                           : colourRangeOf<uint8_t>(quantization, range);
}

template <typename T>
void dequantizeAll(const T* values, const Quantization& quantization, int count, float* out) {
    for (int i = 0; i < count; ++i) {
        out[i] = static_cast<float>(static_cast<int>(values[i]) - quantization.zeroPoint) * quantization.scale;
    }
}

}  // namespace

int elementBytes(TensorElement element) {
    return element == TensorElement::kFloat32 ? static_cast<int>(sizeof(float)) : 1;
}

bool readInputShape(const int* dims, int dimCount, int* size, PixelLayout* layout) {
    if (dims == nullptr || dimCount != 4 || dims[0] != 1) {
        return false;
    }
    // The channel count is checked last, so a picture of 3 x 3 pixels counts as interleaved; no network reads one.
    if (dims[3] == kChannels && dims[1] == dims[2] && dims[1] > 0) {
        *size = dims[1];
        *layout = PixelLayout::kInterleaved;
        return true;
    }
    if (dims[1] == kChannels && dims[2] == dims[3] && dims[2] > 0) {
        *size = dims[2];
        *layout = PixelLayout::kPlanar;
        return true;
    }
    return false;
}

bool flexibleInputShape(const int* dims, int dimCount, int side, int resized[4]) {
    if (dims == nullptr || dimCount != 4 || dims[0] != 1 || side <= 0) {
        return false;
    }
    if (dims[3] == kChannels && dims[1] == 1 && dims[2] == 1) {
        resized[0] = 1;
        resized[1] = side;
        resized[2] = side;
        resized[3] = kChannels;
        return true;
    }
    if (dims[1] == kChannels && dims[2] == 1 && dims[3] == 1) {
        resized[0] = 1;
        resized[1] = kChannels;
        resized[2] = side;
        resized[3] = side;
        return true;
    }
    return false;
}

bool readOutputShape(const int* dims, int dimCount, int* rows, int* cols) {
    if (dims == nullptr || (dimCount != 2 && dimCount != 3) || (dimCount == 3 && dims[0] != 1)) {
        return false;
    }
    *rows = dims[dimCount - 2];
    *cols = dims[dimCount - 1];
    return *rows > 0 && *cols > 0;
}

bool InputWriter::configure(const InputSpec& spec) {
    const bool integer = spec.element != TensorElement::kFloat32;
    ColourRange range;
    if (spec.size <= 0 || !colourRangeFor(spec.element, spec.quantization, &range)) {
        return false;
    }
    spec_ = spec;
    range_ = range;
    for (int byte = 0; byte < 256; ++byte) {
        const float value = range.lowest + channelValue(byte) * (range.highest - range.lowest);
        floats_[byte] = value;
        signedBytes_[byte] = integer ? quantize<int8_t>(value, spec.quantization) : 0;
        unsignedBytes_[byte] = integer ? quantize<uint8_t>(value, spec.quantization) : 0;
    }
    return true;
}

void InputWriter::write(const uint8_t* rgba, int stride, void* tensor) const {
    switch (spec_.element) {
        case TensorElement::kFloat32:
            writePicture(rgba, stride, spec_, floats_, tensor);
            break;
        case TensorElement::kInt8:
            writePicture(rgba, stride, spec_, signedBytes_, tensor);
            break;
        case TensorElement::kUInt8:
            writePicture(rgba, stride, spec_, unsignedBytes_, tensor);
            break;
    }
}

void dequantize(const void* values, TensorElement element, const Quantization& quantization, int count, float* out) {
    switch (element) {
        case TensorElement::kFloat32:
            std::copy_n(static_cast<const float*>(values), count, out);
            break;
        case TensorElement::kInt8:
            dequantizeAll(static_cast<const int8_t*>(values), quantization, count, out);
            break;
        case TensorElement::kUInt8:
            dequantizeAll(static_cast<const uint8_t*>(values), quantization, count, out);
            break;
    }
}

}  // namespace aimbuddy
