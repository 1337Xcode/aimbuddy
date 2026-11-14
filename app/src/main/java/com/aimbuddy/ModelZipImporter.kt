package com.aimbuddy

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.Locale
import java.util.zip.ZipInputStream
import org.json.JSONException
import org.json.JSONObject

internal class InvalidModelArchiveException(
    message: String,
    val reason: Reason = Reason.INVALID_ARCHIVE,
) : IOException(message) {
    enum class Reason {
        INVALID_ARCHIVE, AMBIGUOUS_METADATA, INVALID_METADATA, UNSAFE_PATH, DUPLICATE_PATH, ARCHIVE_LIMIT
    }
}

internal data class ModelZipFiles(
    val param: File?,
    val bin: File?,
    val tflite: File?,
    val tfliteFileName: String?,
    val description: File?,
    val modelId: String,
    val modelName: String,
)

internal object ModelZipImporter {
    private const val MAX_ENTRIES = 512
    private const val MAX_DESCRIPTION_FILES = 64
    private const val MAX_MODEL_FILE_BYTES = 512L * 1024 * 1024
    private const val MAX_DESCRIPTION_BYTES = 1024L * 1024
    private const val MAX_TOTAL_BYTES = 1024L * 1024 * 1024

    private data class ArchiveFile(val path: String, val file: File)

    private data class PairCandidate(
        val directory: String,
        val stem: String,
        var param: ArchiveFile? = null,
        var bin: ArchiveFile? = null,
    )

    private data class TFLiteCandidate(
        val directory: String,
        val stem: String,
        val model: ArchiveFile,
    )

    private data class SupportedModel(
        val directory: String,
        val stem: String,
        val param: ArchiveFile?,
        val bin: ArchiveFile?,
        val tflite: ArchiveFile?,
    )

    fun extract(input: InputStream, destination: File, modelId: String): ModelZipFiles {
        require(modelId.matches(Regex("[a-z0-9._-]+"))) { "Invalid model id" }
        if (!destination.exists() && !destination.mkdirs()) {
            throw IOException("Could not create the model folder")
        }

        val staging = File(destination, ".zip-extract-${System.nanoTime()}")
        if (!staging.mkdir()) {
            throw IOException("Could not create the ZIP staging folder")
        }

        try {
            val files = readRelevantFiles(input, staging)
            val models = findSupportedModels(files)
            if (models.size != 1) {
                throw InvalidModelArchiveException(
                    if (models.isEmpty()) {
                        "The ZIP has no matching .param and .bin files or a .tflite model"
                    } else {
                        "The ZIP has more than one supported model"
                    }
                )
            }

            val model = models.single()
            val parent = model.directory
            val normalizedStem = model.stem.lowercase(Locale.ROOT)
            val description = findDescription(files, parent, normalizedStem, allowSharedMetadata = true)
            val paramFile = model.param?.let { File(destination, "$modelId.param") }
            val binFile = model.bin?.let { File(destination, "$modelId.bin") }
            val tfliteFile = model.tflite?.let { File(destination, "$modelId.tflite") }
            val descriptionFile = description?.let { File(destination, "$modelId.json") }

            if (model.param != null && model.bin != null) {
                model.param.file.copyTo(checkNotNull(paramFile))
                model.bin.file.copyTo(checkNotNull(binFile))
            } else {
                checkNotNull(model.tflite).file.copyTo(checkNotNull(tfliteFile))
            }
            if (description != null && descriptionFile != null) {
                description.file.copyTo(descriptionFile)
            }
            return ModelZipFiles(
                paramFile,
                binFile,
                tfliteFile,
                model.tflite?.path?.substringAfterLast('/'),
                descriptionFile,
                modelId,
                model.stem,
            )
        } finally {
            staging.deleteRecursively()
        }
    }

    fun extractAll(input: InputStream, destinationRoot: File, modelIdPrefix: String): List<ModelZipFiles> {
        require(modelIdPrefix.matches(Regex("[a-z0-9._-]+"))) { "Invalid model id prefix" }
        if (!destinationRoot.exists() && !destinationRoot.mkdirs()) {
            throw IOException("Could not create the model folder")
        }

        val staging = File(destinationRoot, ".zip-extract-${System.nanoTime()}")
        if (!staging.mkdir()) {
            throw IOException("Could not create the ZIP staging folder")
        }

        val createdModelDirs = mutableListOf<File>()
        try {
            val files = readRelevantFiles(input, staging)
            val models = findSupportedModels(files)
            if (models.isEmpty()) {
                throw InvalidModelArchiveException("The ZIP has no matching .param and .bin files or a .tflite model")
            }
            val modelCountByDirectory = models.groupingBy { it.directory }.eachCount()
            return models.mapIndexed { index, model ->
                val modelId = if (models.size == 1) modelIdPrefix else "$modelIdPrefix-${index + 1}"
                val modelDir = File(destinationRoot, modelId)
                if (!modelDir.mkdir()) {
                    throw IOException("Could not create the model folder")
                }
                createdModelDirs += modelDir
                val description = findDescription(
                    files = files,
                    directory = model.directory,
                    stem = model.stem.lowercase(Locale.ROOT),
                    allowSharedMetadata = modelCountByDirectory[model.directory] == 1,
                )
                val paramFile = model.param?.let { File(modelDir, "$modelId.param") }
                val binFile = model.bin?.let { File(modelDir, "$modelId.bin") }
                val tfliteFile = model.tflite?.let { File(modelDir, "$modelId.tflite") }
                val descriptionFile = description?.let { File(modelDir, "$modelId.json") }
                if (model.param != null && model.bin != null) {
                    model.param.file.copyTo(checkNotNull(paramFile))
                    model.bin.file.copyTo(checkNotNull(binFile))
                } else {
                    checkNotNull(model.tflite).file.copyTo(checkNotNull(tfliteFile))
                }
                if (description != null && descriptionFile != null) {
                    description.file.copyTo(descriptionFile)
                }
                ModelZipFiles(
                    param = paramFile,
                    bin = binFile,
                    tflite = tfliteFile,
                    tfliteFileName = model.tflite?.path?.substringAfterLast('/'),
                    description = descriptionFile,
                    modelId = modelId,
                    modelName = model.stem,
                )
            }
        } catch (error: Throwable) {
            createdModelDirs.forEach { it.deleteRecursively() }
            throw error
        } finally {
            staging.deleteRecursively()
        }
    }

    private fun findDescription(
        files: Map<String, ArchiveFile>,
        directory: String,
        stem: String,
        allowSharedMetadata: Boolean,
    ): ArchiveFile? {
        files[siblingPath(directory, "$stem.json")]?.let { return it }
        if (allowSharedMetadata) {
            files[siblingPath(directory, "model.json")]?.let { return it }
        }

        val candidates = files.values.filter { file ->
            file.path.substringBeforeLast('/', "") == directory &&
                file.path.endsWith(".json") && hasProfileFields(file.file)
        }
        if (!allowSharedMetadata && candidates.isNotEmpty()) {
            throw InvalidModelArchiveException(
                "The ZIP has model settings that cannot be matched to a specific model",
                InvalidModelArchiveException.Reason.AMBIGUOUS_METADATA,
            )
        }
        if (allowSharedMetadata && candidates.size > 1) {
            throw InvalidModelArchiveException(
                "The ZIP has multiple possible model descriptions",
                InvalidModelArchiveException.Reason.AMBIGUOUS_METADATA,
            )
        }
        return candidates.singleOrNull()
    }

    private fun hasProfileFields(file: File): Boolean = try {
        val json = JSONObject(file.readText())
        listOf("roles", "classes", "inputSize", "endToEnd", "normalizedCoordinates").any { json.has(it) }
    } catch (_: JSONException) {
        false
    }

    private fun readRelevantFiles(input: InputStream, staging: File): Map<String, ArchiveFile> {
        val files = linkedMapOf<String, ArchiveFile>()
        var totalBytes = 0L
        var entries = 0
        var descriptions = 0

        ZipInputStream(input).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                entries++
                if (entries > MAX_ENTRIES) {
                    throw InvalidModelArchiveException("The ZIP has too many entries", InvalidModelArchiveException.Reason.ARCHIVE_LIMIT)
                }

                val path = safeEntryPath(entry.name)
                val name = path.substringAfterLast('/')
                val extension = name.substringAfterLast('.', "").lowercase(Locale.ROOT)
                val isRelevant = !entry.isDirectory && extension in setOf("param", "bin", "tflite", "json")
                if (extension == "json" && isRelevant) {
                    descriptions++
                    if (descriptions > MAX_DESCRIPTION_FILES) {
                        throw InvalidModelArchiveException("The ZIP has too many JSON files", InvalidModelArchiveException.Reason.ARCHIVE_LIMIT)
                    }
                }
                val target = if (isRelevant) File(staging, "entry-$entries") else null
                val fileLimit = if (extension == "json") MAX_DESCRIPTION_BYTES else MAX_MODEL_FILE_BYTES
                var fileBytes = 0L
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                val output = target?.outputStream()?.buffered()
                try {
                    while (true) {
                        val read = zip.read(buffer)
                        if (read < 0) break
                        fileBytes += read
                        totalBytes += read
                        if (totalBytes > MAX_TOTAL_BYTES) {
                            throw InvalidModelArchiveException("The ZIP expands beyond the allowed size", InvalidModelArchiveException.Reason.ARCHIVE_LIMIT)
                        }
                        if (isRelevant && fileBytes > fileLimit) {
                            throw InvalidModelArchiveException("A model file in the ZIP is too large", InvalidModelArchiveException.Reason.ARCHIVE_LIMIT)
                        }
                        output?.write(buffer, 0, read)
                    }
                } finally {
                    output?.close()
                }
                zip.closeEntry()

                if (isRelevant && extension in setOf("param", "bin", "tflite")) {
                    val key = path.lowercase(Locale.ROOT)
                    if (files.containsKey(key)) {
                        throw InvalidModelArchiveException("The ZIP contains duplicate model file paths", InvalidModelArchiveException.Reason.DUPLICATE_PATH)
                    }
                    files[key] = ArchiveFile(key, checkNotNull(target))
                } else if (isRelevant && extension == "json") {
                    val key = path.lowercase(Locale.ROOT)
                    if (files.containsKey(key)) {
                        throw InvalidModelArchiveException("The ZIP contains duplicate model file paths", InvalidModelArchiveException.Reason.DUPLICATE_PATH)
                    }
                    files[key] = ArchiveFile(key, checkNotNull(target))
                }
            }
        }
        return files
    }

    private fun findPairs(files: Map<String, ArchiveFile>): List<PairCandidate> {
        val candidates = linkedMapOf<String, PairCandidate>()
        for (archiveFile in files.values) {
            val name = archiveFile.path.substringAfterLast('/')
            val extension = name.substringAfterLast('.', "")
            if (extension != "param" && extension != "bin") continue

            val stem = name.substringBeforeLast('.')
            if (stem.isBlank()) continue
            val directory = archiveFile.path.substringBeforeLast('/', "")
            val key = "$directory/$stem"
            val candidate = candidates.getOrPut(key) { PairCandidate(directory, stem) }
            if (extension == "param") candidate.param = archiveFile else candidate.bin = archiveFile
        }

        return candidates.values.filter { it.param != null && it.bin != null }
    }

    private fun findTFLiteModels(files: Map<String, ArchiveFile>): List<TFLiteCandidate> =
        files.values.mapNotNull { archiveFile ->
            val name = archiveFile.path.substringAfterLast('/')
            if (!name.endsWith(".tflite", ignoreCase = true)) return@mapNotNull null
            TFLiteCandidate(
                directory = archiveFile.path.substringBeforeLast('/', ""),
                stem = name.substringBeforeLast('.'),
                model = archiveFile,
            )
        }

    private fun findSupportedModels(files: Map<String, ArchiveFile>): List<SupportedModel> {
        val pairs = findPairs(files).map {
            SupportedModel(
                directory = it.directory,
                stem = it.stem,
                param = it.param,
                bin = it.bin,
                tflite = null,
            )
        }
        val tfliteModels = findTFLiteModels(files).map {
            SupportedModel(
                directory = it.directory,
                stem = it.stem,
                param = null,
                bin = null,
                tflite = it.model,
            )
        }
        return (pairs + tfliteModels).sortedWith(compareBy({ it.directory }, { it.stem.lowercase(Locale.ROOT) }))
    }

    private fun safeEntryPath(rawPath: String): String {
        val path = rawPath.replace('\\', '/')
        if (path.startsWith('/') || path.matches(Regex("^[a-zA-Z]:.*"))) {
            throw InvalidModelArchiveException("The ZIP contains an absolute path", InvalidModelArchiveException.Reason.UNSAFE_PATH)
        }
        val parts = path.split('/')
        if (parts.any { it == ".." }) {
            throw InvalidModelArchiveException("The ZIP contains a parent path", InvalidModelArchiveException.Reason.UNSAFE_PATH)
        }
        return parts.filter { it.isNotEmpty() && it != "." }.joinToString("/")
    }

    private fun siblingPath(directory: String, name: String): String =
        if (directory.isEmpty()) name else "$directory/$name"
}
