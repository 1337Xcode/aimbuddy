#ifndef AIMBUDDY_CORE_MODEL_TENSORS_H
#define AIMBUDDY_CORE_MODEL_TENSORS_H

#include <cstdint>

namespace aimbuddy {

/**
 * Moves a picture into a network's input tensor and numbers out of its output tensor, for a runtime
 * that takes the tensors exactly as the model declares them (LiteRT). Models differ in both: the
 * channels interleaved per pixel or in planes, and values as floats or as 8-bit integers with a scale
 * and a zero point. Nothing here allocates, so it can run on every frame.
 */

/// How the values of a tensor are stored.
enum class TensorElement : unsigned char { kFloat32, kInt8, kUInt8 };

/// How a picture input orders its values: per pixel (height, width, channel) or per channel plane (channel, height, width).
enum class PixelLayout : unsigned char { kInterleaved, kPlanar };

/// How an integer tensor stands for real values: real = (stored - zeroPoint) * scale.
struct Quantization {
    float scale = 0.0f;
    int zeroPoint = 0;
};

/// Everything needed to fill a picture input: its side in pixels, its layout and how its values are stored.
struct InputSpec {
    int size = 0;
    PixelLayout layout = PixelLayout::kInterleaved;
    TensorElement element = TensorElement::kFloat32;
    Quantization quantization;   ///< Only read for an integer tensor.
};

/// The real values the darkest and the brightest colour are given as.
struct ColourRange {
    float lowest = 0.0f;
    float highest = 1.0f;
};

/// Bytes of one value of a tensor.
int elementBytes(TensorElement element);

/**
 * Reads the shape of a picture input: [1, S, S, 3] interleaved or [1, 3, S, S] planar. False for any
 * other shape: a batch of more than one, a picture that is not square, or one that is not RGB.
 */
bool readInputShape(const int* dims, int dimCount, int* size, PixelLayout* layout);

/**
 * The shape to give a picture input whose height and width are left open, as models exported with a flexible size
 * are (LiteRT reports those sides as 1 until the input is resized): [1, side, side, 3] or [1, 3, side, side], written
 * to `resized`. False for an input whose size is fixed, or that is not one RGB picture.
 */
bool flexibleInputShape(const int* dims, int dimCount, int side, int resized[4]);

/// Reads the shape of a detection output as rows by columns: [1, R, C] or [R, C]. False for any other shape.
bool readOutputShape(const int* dims, int dimCount, int* rows, int* cols);

/**
 * Writes RGBA pictures into an input tensor, each channel value scaled to the colour range the input stands for (0 to
 * 1 as the models were trained, or the range an 8-bit input is quantized for), then stored as the tensor stores it. The 256 values a channel can have are converted once,
 * so a frame costs one table look-up per value.
 */
class InputWriter {
public:
    /**
     * Prepares the tables for one input. False when an integer input has no usable scale, or when its range stands
     * for none of 0 to 1, 0 to 255 and -1 to 1, which this writer would fill wrongly.
     */
    bool configure(const InputSpec& spec);

    /// Writes a spec.size by spec.size RGBA picture, `stride` bytes a row, into `tensor` (alpha is dropped).
    void write(const uint8_t* rgba, int stride, void* tensor) const;

    const InputSpec& spec() const { return spec_; }
    const ColourRange& range() const { return range_; }

private:
    InputSpec spec_;
    ColourRange range_;
    float floats_[256] = {};
    int8_t signedBytes_[256] = {};
    uint8_t unsignedBytes_[256] = {};
};

/// Turns `count` stored values of an integer tensor back into the real values they stand for.
void dequantize(const void* values, TensorElement element, const Quantization& quantization, int count, float* out);

}  // namespace aimbuddy

#endif  // AIMBUDDY_CORE_MODEL_TENSORS_H
