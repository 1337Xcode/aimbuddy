package com.aimbuddy

import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

internal class AndroidTFLiteDelegate {
    private var gpu: GpuDelegate? = null
    private var nnapi: NnApiDelegate? = null
    private var compatibility: CompatibilityList? = null

    init {
        try {
            val support = CompatibilityList()
            compatibility = support
            if (support.isDelegateSupportedOnThisDevice) {
                // GPU 2.14 exposes CompatibilityList but not the GpuDelegateFactory.Options superclass
                // needed to customize its options on the compile classpath. Its defaults are device-specific.
                gpu = GpuDelegate()
            }
        } catch (error: Throwable) {
            Log.i(TAG, "GPU delegate unavailable: ${error.message}")
            gpu?.close()
            gpu = null
        }
        if (gpu == null) {
            try {
                nnapi = NnApiDelegate(
                    NnApiDelegate.Options()
                        .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                        .setAllowFp16(true)
                        .setUseNnapiCpu(false),
                )
            } catch (error: Throwable) {
                Log.i(TAG, "NNAPI delegate unavailable: ${error.message}")
                nnapi = null
            }
        }
    }

    val hasGpu: Boolean
        get() = gpu != null
    val hasNnapi: Boolean
        get() = nnapi != null

    fun addTo(options: Interpreter.Options) {
        gpu?.let(options::addDelegate)
        nnapi?.let(options::addDelegate)
    }

    fun close() {
        gpu?.close()
        nnapi?.close()
        compatibility?.close()
        gpu = null
        nnapi = null
        compatibility = null
    }

    private companion object {
        const val TAG = "TFLiteModelRunner"
    }
}

class TFLiteModelRunner(modelPath: String) : AutoCloseable {
    private data class OutputTensor(
        val shape: IntArray,
        val type: DataType,
        val scale: Float,
        val zeroPoint: Int,
        val buffer: ByteBuffer,
        val values: FloatArray,
    )

    private var backend = "TFLite CPU (XNNPACK)"
    private var delegate: AndroidTFLiteDelegate? = null
    private val interpreter: Interpreter

    init {
        var driver: AndroidTFLiteDelegate? = null
        var prepared: Interpreter? = null
        try {
            driver = AndroidTFLiteDelegate()
            prepared = Interpreter(
                File(modelPath),
                Interpreter.Options()
                    .setNumThreads(INFERENCE_THREADS)
                    .setUseXNNPACK(true)
                    .also(driver::addTo),
            )
            delegate = driver
            backend = when {
                driver.hasGpu -> "TFLite GPU / CPU fallback"
                driver.hasNnapi -> "TFLite NNAPI / CPU fallback"
                else -> "TFLite CPU (XNNPACK)"
            }
        } catch (error: RuntimeException) {
            prepared?.close()
            driver?.close()
            Log.i("TFLiteModelRunner", "Accelerated model initialization failed; using XNNPACK CPU: ${error.message}")
            prepared = Interpreter(
                File(modelPath),
                Interpreter.Options().setNumThreads(INFERENCE_THREADS).setUseXNNPACK(true),
            )
            backend = "TFLite CPU (XNNPACK)"
        }
        interpreter = checkNotNull(prepared)
    }

    fun nativeBackend(): String = backend

    private val inputSize: Int
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputShape: IntArray
    private val inputBuffer: ByteBuffer
    private val outputTensors: List<OutputTensor>
    private val outputMap: MutableMap<Int, Any>
    private val smartScales: List<TFLiteSmartYoloScale>?
    private val smartOutput: FloatArray?
    private var closed = false

    init {
        try {
            require(interpreter.inputTensorCount == 1) { "The model must have one input tensor" }
            require(interpreter.outputTensorCount == 1 || interpreter.outputTensorCount == SMART_OUTPUT_COUNT) {
                "The model must have one output tensor or the supported EdgeFirst YOLO11 smart outputs"
            }

            val input = interpreter.getInputTensor(0)
            val inputDimensions = input.shape()
            require(inputDimensions.size == 4 && inputDimensions[0] == 1 &&
                inputDimensions[1] == inputDimensions[2] && inputDimensions[3] == 3
            ) { "The model input must be a square NHWC RGB image" }
            inputSize = inputDimensions[1]
            require(inputSize in MIN_INPUT_SIZE..MAX_INPUT_SIZE) { "The model input size is unsupported" }
            inputType = input.dataType()
            require(inputType in SUPPORTED_TYPES) { "The model input tensor type is unsupported" }
            val inputQuantization = input.quantizationParams()
            inputScale = inputQuantization.scale
            inputZeroPoint = inputQuantization.zeroPoint
            if (inputType != DataType.FLOAT32) {
                require(inputScale > 0f) { "The model input has invalid quantization parameters" }
            }

            outputTensors = (0 until interpreter.outputTensorCount).map { index ->
                val tensor = interpreter.getOutputTensor(index)
                val type = tensor.dataType()
                require(type in SUPPORTED_TYPES) { "The model output tensor type is unsupported" }
                val quantization = tensor.quantizationParams()
                if (type != DataType.FLOAT32) {
                    require(quantization.scale > 0f) { "The model output has invalid quantization parameters" }
                }
                val elementCount = tensor.shape().fold(1L) { count, dimension -> count * dimension }
                require(elementCount in 1..MAX_OUTPUT_ELEMENTS) { "The model output tensor is too large" }
                OutputTensor(
                    shape = tensor.shape(),
                    type = type,
                    scale = quantization.scale,
                    zeroPoint = quantization.zeroPoint,
                    buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder()),
                    values = FloatArray(elementCount.toInt()),
                )
            }

            inputBuffer = ByteBuffer.allocateDirect(input.numBytes()).order(ByteOrder.nativeOrder())
            outputMap = mutableMapOf<Int, Any>().apply {
                outputTensors.forEachIndexed { index, tensor -> put(index, tensor.buffer) }
            }
            if (outputTensors.size == 1) {
                outputShape = TFLiteModelContract.detectionOutputShape(outputTensors.single().shape)
                val outputElementCount = outputShape.fold(1L) { count, dimension -> count * dimension }
                require(outputElementCount in 6..MAX_OUTPUT_ELEMENTS) { "The model output tensor is too large" }
                smartScales = null
                smartOutput = null
            } else {
                val scales = findSmartYoloScales(outputTensors, inputSize)
                    ?: throw IllegalArgumentException("The EdgeFirst YOLO11 smart output tensors are not supported")
                smartScales = scales
                val anchors = scales.sumOf { it.gridSize * it.gridSize }
                outputShape = intArrayOf(5, anchors)
                smartOutput = FloatArray(5 * anchors)
            }
        } catch (e: RuntimeException) {
            interpreter.close()
            delegate?.close()
            delegate = null
            throw e
        }
    }

    fun nativeInputSize(): Int = inputSize

    fun nativeOutputShape(): IntArray = outputShape.copyOf()

    @Synchronized
    fun nativeRunInference(rgbPixels: ByteBuffer): FloatArray {
        check(!closed) { "The model is closed" }
        require(rgbPixels.isDirect && rgbPixels.capacity() == inputSize * inputSize * CHANNELS) {
            "The detector supplied an invalid image buffer"
        }

        val pixels = rgbPixels.duplicate()
        pixels.clear()
        inputBuffer.clear()
        when (inputType) {
            DataType.FLOAT32 -> while (pixels.hasRemaining()) {
                inputBuffer.putFloat((pixels.get().toInt() and 0xff) / 255f)
            }
            DataType.INT8 -> while (pixels.hasRemaining()) {
                inputBuffer.put(quantize(pixels.get().toInt() and 0xff, Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()))
            }
            DataType.UINT8 -> while (pixels.hasRemaining()) {
                inputBuffer.put(quantize(pixels.get().toInt() and 0xff, 0, 0xff))
            }
            else -> error("The model input tensor type is unsupported")
        }
        inputBuffer.rewind()

        if (outputTensors.size == 1) {
            val output = outputTensors.single()
            output.buffer.clear()
            interpreter.run(inputBuffer, output.buffer)
            readOutput(output)
            return output.values
        }

        outputTensors.forEach { it.buffer.clear() }
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputMap)
        outputTensors.forEach(::readOutput)
        val merged = checkNotNull(smartOutput)
        check(TFLiteSmartYoloOutput.decode(inputSize, checkNotNull(smartScales), merged)) {
            "The EdgeFirst YOLO11 smart output tensors could not be decoded"
        }
        return merged
    }

    private fun readOutput(output: OutputTensor) {
        output.buffer.rewind()
        when (output.type) {
            DataType.FLOAT32 -> output.values.indices.forEach { output.values[it] = output.buffer.float }
            DataType.INT8 -> output.values.indices.forEach {
                output.values[it] = (output.buffer.get().toInt() - output.zeroPoint) * output.scale
            }
            DataType.UINT8 -> output.values.indices.forEach {
                output.values[it] = ((output.buffer.get().toInt() and 0xff) - output.zeroPoint) * output.scale
            }
            else -> error("The model output tensor type is unsupported")
        }
    }

    private fun findSmartYoloScales(
        tensors: List<OutputTensor>,
        inputSize: Int,
    ): List<TFLiteSmartYoloScale>? {
        val byGrid = mutableMapOf<Int, MutableList<OutputTensor>>()
        for (tensor in tensors) {
            val shape = tensor.shape
            if (shape.size != 4 || shape[0] != 1 || shape[1] != shape[2]) return null
            byGrid.getOrPut(shape[1]) { mutableListOf() }.add(tensor)
        }
        if (byGrid.size != 3) return null

        val scales = mutableListOf<TFLiteSmartYoloScale>()
        for ((gridSize, pair) in byGrid) {
            if (pair.size != 2 || gridSize <= 0 || inputSize % gridSize != 0) return null
            val stride = inputSize / gridSize
            if (stride !in SUPPORTED_SMART_STRIDES) return null
            val boxes = pair.singleOrNull { it.shape[3] == TFLiteSmartYoloOutput.DFL_BINS * 4 } ?: return null
            val scores = pair.singleOrNull { it.shape[3] == TFLiteSmartYoloOutput.COCO_CLASS_COUNT } ?: return null
            scales += TFLiteSmartYoloScale(stride, gridSize, boxes.values, scores.values)
        }
        return scales.sortedBy { it.stride }.takeIf { scales ->
            scales.map { it.stride } == SUPPORTED_SMART_STRIDES
        }
    }

    private fun quantize(pixel: Int, minimum: Int, maximum: Int): Byte {
        val value = ((pixel / 255f) / inputScale + inputZeroPoint).roundToInt().coerceIn(minimum, maximum)
        return value.toByte()
    }

    @Synchronized
    override fun close() {
        if (!closed) {
            interpreter.close()
            delegate?.close()
            delegate = null
            closed = true
        }
    }

    private companion object {
        const val CHANNELS = 3
        const val INFERENCE_THREADS = 3
        const val MIN_INPUT_SIZE = 32
        const val MAX_INPUT_SIZE = 2048
        const val MAX_OUTPUT_ELEMENTS = 2_000_000L
        const val SMART_OUTPUT_COUNT = 6
        val SUPPORTED_SMART_STRIDES = listOf(8, 16, 32)
        val SUPPORTED_TYPES = setOf(DataType.FLOAT32, DataType.INT8, DataType.UINT8)
    }
}
