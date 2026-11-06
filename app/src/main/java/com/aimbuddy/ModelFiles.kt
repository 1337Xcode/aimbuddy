package com.aimbuddy

/**
 * Where a model's files are, which also says what runs it. [formatCode] is the number the native
 * side knows the format by (`ModelFormat` in `detector/detector.h`).
 */
sealed class ModelFiles {
    abstract val formatCode: Int

    /** The file the format is read from first. */
    abstract val modelPath: String

    /** The weights, for a format that keeps them apart from [modelPath]; null otherwise. */
    abstract val weightsPath: String?

    /** Every file of the model. */
    val paths: List<String>
        get() = listOfNotNull(modelPath, weightsPath)

    /** An NCNN export: the network in a `.param` file and its weights in a `.bin` file. */
    data class Ncnn(val paramPath: String, val binPath: String) : ModelFiles() {
        override val formatCode: Int get() = 0
        override val modelPath: String get() = paramPath
        override val weightsPath: String get() = binPath
    }

    /** A TFLite (LiteRT) export: one `.tflite` file holding the network and its weights. */
    data class Tflite(val path: String) : ModelFiles() {
        override val formatCode: Int get() = 1
        override val modelPath: String get() = path
        override val weightsPath: String? get() = null
    }
}
