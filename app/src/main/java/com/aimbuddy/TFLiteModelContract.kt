package com.aimbuddy

/** Validates the small, flat detection tensor consumed by the native decoder. */
internal object TFLiteModelContract {
    private const val YOLO_POSE_CHANNELS = 4 + 1 + 17 * 3

    fun detectionOutputShape(dimensions: IntArray): IntArray {
        val shape = when {
            dimensions.size == 3 && dimensions[0] == 1 -> dimensions.copyOfRange(1, 3)
            dimensions.size == 2 -> dimensions.copyOf()
            else -> throw IllegalArgumentException("The model output must be a two-dimensional detection tensor")
        }
        require(shape.all { it > 0 }) { "The model output dimensions must be positive" }
        return shape
    }
}
