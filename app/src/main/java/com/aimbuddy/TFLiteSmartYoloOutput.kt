package com.aimbuddy

import kotlin.math.exp

internal data class TFLiteSmartYoloScale(
    val stride: Int,
    val gridSize: Int,
    val boxes: FloatArray,
    val scores: FloatArray,
)

internal object TFLiteSmartYoloOutput {
    const val DFL_BINS = 16
    const val COCO_CLASS_COUNT = 80
    private const val BOX_COORDINATES = 4
    private const val PERSON_CLASS = 0

    fun decode(inputSize: Int, scales: List<TFLiteSmartYoloScale>, output: FloatArray): Boolean {
        if (inputSize <= 0 || scales.size != 3) return false
        val anchorCount = scales.sumOf { it.gridSize * it.gridSize }
        if (output.size != (BOX_COORDINATES + 1) * anchorCount) return false

        var anchorOffset = 0
        for (scale in scales) {
            val cellCount = scale.gridSize * scale.gridSize
            if (inputSize != scale.gridSize * scale.stride ||
                scale.boxes.size != cellCount * BOX_COORDINATES * DFL_BINS ||
                scale.scores.size != cellCount * COCO_CLASS_COUNT
            ) {
                return false
            }

            for (y in 0 until scale.gridSize) {
                for (x in 0 until scale.gridSize) {
                    val cell = y * scale.gridSize + x
                    val anchor = anchorOffset + cell
                    val boxBase = cell * BOX_COORDINATES * DFL_BINS
                    val left = distributionExpectation(scale.boxes, boxBase)
                    val top = distributionExpectation(scale.boxes, boxBase + DFL_BINS)
                    val right = distributionExpectation(scale.boxes, boxBase + 2 * DFL_BINS)
                    val bottom = distributionExpectation(scale.boxes, boxBase + 3 * DFL_BINS)
                    val normalizer = inputSize.toFloat()

                    output[anchor] = (x + 0.5f + (right - left) * 0.5f) * scale.stride / normalizer
                    output[anchorCount + anchor] =
                        (y + 0.5f + (bottom - top) * 0.5f) * scale.stride / normalizer
                    output[2 * anchorCount + anchor] = (left + right) * scale.stride / normalizer
                    output[3 * anchorCount + anchor] = (top + bottom) * scale.stride / normalizer
                    output[4 * anchorCount + anchor] =
                        sigmoid(scale.scores[cell * COCO_CLASS_COUNT + PERSON_CLASS])
                }
            }
            anchorOffset += cellCount
        }
        return anchorOffset == anchorCount
    }

    private fun distributionExpectation(values: FloatArray, offset: Int): Float {
        var maximum = Float.NEGATIVE_INFINITY
        for (bin in 0 until DFL_BINS) {
            maximum = maxOf(maximum, values[offset + bin])
        }

        var weighted = 0f
        var total = 0f
        for (bin in 0 until DFL_BINS) {
            val probability = exp(values[offset + bin] - maximum)
            weighted += bin * probability
            total += probability
        }
        return weighted / total
    }

    private fun sigmoid(value: Float): Float =
        if (value >= 0f) {
            1f / (1f + exp(-value))
        } else {
            val exponential = exp(value)
            exponential / (1f + exponential)
        }
}
