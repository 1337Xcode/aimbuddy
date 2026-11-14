package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TFLiteSmartYoloOutputTest {
    @Test
    fun combinesEdgeFirstSmartOutputsIntoPersonDetectionTensor() {
        val scales = listOf(8, 4, 2).mapIndexed { index, gridSize ->
            val stride = listOf(8, 16, 32)[index]
            val cellCount = gridSize * gridSize
            val boxes = FloatArray(cellCount * 4 * TFLiteSmartYoloOutput.DFL_BINS) { -20f }
            val scores = FloatArray(cellCount * TFLiteSmartYoloOutput.COCO_CLASS_COUNT) { -20f }
            for (cell in 0 until cellCount) {
                for (side in 0 until 4) {
                    boxes[cell * 4 * TFLiteSmartYoloOutput.DFL_BINS + side * TFLiteSmartYoloOutput.DFL_BINS + 1] = 20f
                }
                scores[cell * TFLiteSmartYoloOutput.COCO_CLASS_COUNT] = 2f
                scores[cell * TFLiteSmartYoloOutput.COCO_CLASS_COUNT + 1] = 10f
            }
            TFLiteSmartYoloScale(stride, gridSize, boxes, scores)
        }
        val output = FloatArray(5 * (64 + 16 + 4))

        assertTrue(TFLiteSmartYoloOutput.decode(64, scales, output))
        assertEquals(0.0625f, output[0], 0.001f)
        assertEquals(0.25f, output[84 + 64 + 16], 0.001f)
        assertEquals(0.25f, output[2 * (64 + 16 + 4)], 0.001f)
        assertEquals(0.25f, output[3 * (64 + 16 + 4)], 0.001f)
        assertEquals(0.8808f, output[4 * 84], 0.001f)
    }

    @Test
    fun rejectsInvalidScaleShapes() {
        val scale = TFLiteSmartYoloScale(
            stride = 8,
            gridSize = 8,
            boxes = FloatArray(8 * 8 * 4 * TFLiteSmartYoloOutput.DFL_BINS),
            scores = FloatArray(8 * 8 * TFLiteSmartYoloOutput.COCO_CLASS_COUNT),
        )

        assertFalse(TFLiteSmartYoloOutput.decode(64, listOf(scale), FloatArray(5 * 84)))
        assertFalse(TFLiteSmartYoloOutput.decode(64, listOf(scale, scale, scale), FloatArray(5 * 84)))
    }
}
