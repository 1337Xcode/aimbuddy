package com.aimbuddy

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class TFLiteModelContractTest {
    @Test
    fun acceptsFlatAndBatchedDetectionOutputs() {
        assertArrayEquals(
            intArrayOf(5, 8400),
            TFLiteModelContract.detectionOutputShape(intArrayOf(1, 5, 8400)),
        )
        assertArrayEquals(
            intArrayOf(300, 6),
            TFLiteModelContract.detectionOutputShape(intArrayOf(300, 6)),
        )
    }

    @Test
    fun acceptsStandardYoloPoseInEitherLayout() {
        assertArrayEquals(
            intArrayOf(56, 8400),
            TFLiteModelContract.detectionOutputShape(intArrayOf(1, 56, 8400)),
        )
        assertArrayEquals(
            intArrayOf(8400, 56),
            TFLiteModelContract.detectionOutputShape(intArrayOf(1, 8400, 56)),
        )
        assertArrayEquals(
            intArrayOf(300, 57),
            TFLiteModelContract.detectionOutputShape(intArrayOf(1, 300, 57)),
        )
    }

    @Test
    fun rejectsOutputsWithoutATwoDimensionalDetectionLayout() {
        assertThrows(IllegalArgumentException::class.java) {
            TFLiteModelContract.detectionOutputShape(intArrayOf(5, 8400, 1))
        }
    }
}
