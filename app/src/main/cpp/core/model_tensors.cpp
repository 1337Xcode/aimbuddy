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
    if (spec.size <= 0 || (integer && !(spec.quantization.scale > 0.0f && std::isfinite(spec.quantization.scale)))) {
        return false;
    }
    spec_ = spec;
    for (int byte = 0; byte < 256; ++byte) {
        const float value = channelValue(byte);
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
