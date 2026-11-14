package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class ModelZipImporterTest {

    @get:Rule
    val folder = TemporaryFolder()

    @Test
    fun importsNestedMatchingFilesAndKeepsTheirDescription() {
        val archive = zip(
            "my-model/model.param" to "network".toByteArray(),
            "my-model/model.bin" to "weights".toByteArray(),
            "my-model/model.json" to """{"roles":["body"],"inputSize":320,"endToEnd":true}""".toByteArray(),
            "my-model/readme.txt" to "ignored".toByteArray(),
        )
        val output = folder.newFolder("installed")

        val files = ModelZipImporter.extract(ByteArrayInputStream(archive), output, "local-model-1")

        assertEquals("network", checkNotNull(files.param).readText())
        assertEquals("weights", checkNotNull(files.bin).readText())
        assertEquals(null, files.tflite)
        assertNotNull(files.description)
        val profile = ModelProfileSpec.parse(files.description!!.readText())
        assertEquals(320, profile.inputSize)
        assertEquals("body", profile.roles)
        assertEquals(true, profile.endToEnd)
    }

    @Test
    fun importsRootLevelFilesAndPairNamedDescription() {
        val archive = zip(
            "detector.param" to "network".toByteArray(),
            "detector.bin" to "weights".toByteArray(),
            "detector.json" to """{"inputSize":640}""".toByteArray(),
        )

        val files = ModelZipImporter.extract(
            ByteArrayInputStream(archive),
            folder.newFolder("root-model"),
            "local-model-root",
        )

        assertEquals("network", checkNotNull(files.param).readText())
        assertEquals(640, ModelProfileSpec.parse(files.description!!.readText()).inputSize)
    }

    @Test
    fun importsTFLiteModelAndItsDescription() {
        val archive = zip(
            "models/valorant_int8_256.tflite" to byteArrayOf(1, 2, 3),
            "models/model.json" to """{"roles":["head"],"inputSize":256}""".toByteArray(),
        )

        val files = ModelZipImporter.extract(
            ByteArrayInputStream(archive),
            folder.newFolder("tflite-model"),
            "local-tflite-model",
        )

        assertEquals(null, files.param)
        assertEquals(null, files.bin)
        assertEquals("local-tflite-model.tflite", checkNotNull(files.tflite).name)
        assertEquals("valorant_int8_256.tflite", files.tfliteFileName)
        assertEquals("head", ModelProfileSpec.parse(checkNotNull(files.description).readText()).roles)
    }

    @Test
    fun importsUniquelyIdentifiableSettingsWithOtherNamesAndIgnoresReadme() {
        val archive = zip(
            "export/detector.param" to "network".toByteArray(),
            "export/detector.bin" to "weights".toByteArray(),
            "export/settings.json" to """{"roles":["head"],"inputSize":256}""".toByteArray(),
            "export/other.json" to """{"license":"example"}""".toByteArray(),
            "export/README.md" to "Model information".toByteArray(),
            "other/settings.json" to """{"inputSize":640}""".toByteArray(),
        )
        val output = folder.newFolder("named-settings")

        val files = ModelZipImporter.extract(ByteArrayInputStream(archive), output, "local-model")

        assertEquals(256, ModelProfileSpec.parse(checkNotNull(files.description).readText()).inputSize)
        assertEquals(setOf("local-model.param", "local-model.bin", "local-model.json"), output.list()!!.toSet())
    }

    @Test
    fun importsUniquelyIdentifiableTfliteSettings() {
        val archive = zip(
            "model.tflite" to byteArrayOf(1),
            "settings.json" to """{"classes":{"0":"head"}}""".toByteArray(),
            "README.txt" to "Information".toByteArray(),
        )

        val files = ModelZipImporter.extract(
            ByteArrayInputStream(archive), folder.newFolder("tflite-settings"), "local-model",
        )

        assertEquals("head", ModelProfileSpec.parse(checkNotNull(files.description).readText()).roles)
    }

    @Test
    fun rejectsAmbiguousUnnamedSettingsWithoutInstallingFiles() {
        val archive = zip(
            "detector.param" to byteArrayOf(1),
            "detector.bin" to byteArrayOf(2),
            "first.json" to """{"inputSize":320}""".toByteArray(),
            "second.json" to """{"roles":["head"]}""".toByteArray(),
        )
        val output = folder.newFolder("ambiguous-settings")

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), output, "local-model")
            throw AssertionError("Expected ambiguous settings to be rejected")
        } catch (e: InvalidModelArchiveException) {
            assertEquals(InvalidModelArchiveException.Reason.AMBIGUOUS_METADATA, e.reason)
        }
        assertEquals(emptyList<String>(), output.list()!!.toList())
    }

    @Test
    fun namedSettingsTakePrecedenceOverOtherJson() {
        val archive = zip(
            "detector.tflite" to byteArrayOf(1),
            "detector.json" to """{"inputSize":256}""".toByteArray(),
            "model.json" to """{"inputSize":320}""".toByteArray(),
            "other.json" to """{"inputSize":640}""".toByteArray(),
        )

        val files = ModelZipImporter.extract(
            ByteArrayInputStream(archive), folder.newFolder("preferred-settings"), "local-model",
        )

        assertEquals(256, ModelProfileSpec.parse(checkNotNull(files.description).readText()).inputSize)
    }

    @Test
    fun rejectsUnsafeReadmePaths() {
        val archive = zip(
            "model.tflite" to byteArrayOf(1),
            "../README.md" to "Information".toByteArray(),
        )
        val output = folder.newFolder("unsafe-readme")

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), output, "local-model")
            throw AssertionError("Expected an unsafe README path to be rejected")
        } catch (e: InvalidModelArchiveException) {
            assertEquals(InvalidModelArchiveException.Reason.UNSAFE_PATH, e.reason)
        }
        assertFalse(folder.root.resolve("README.md").exists())
        assertEquals(emptyList<String>(), output.list()!!.toList())
    }

    @Test
    fun rejectsDuplicatePathsIgnoringCase() {
        val archive = zip(
            "model.tflite" to byteArrayOf(1),
            "MODEL.TFLITE" to byteArrayOf(2),
        )
        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), folder.newFolder("duplicates"), "local-model")
            throw AssertionError("Expected duplicate paths to be rejected")
        } catch (e: InvalidModelArchiveException) {
            assertEquals(InvalidModelArchiveException.Reason.DUPLICATE_PATH, e.reason)
        }
    }

    @Test
    fun rejectsTooManyEntriesIncludingIgnoredFiles() {
        val entries = (1..513).map { "readme-$it.txt" to byteArrayOf(1) }.toTypedArray()
        try {
            ModelZipImporter.extract(ByteArrayInputStream(zip(*entries)), folder.newFolder("too-many"), "local-model")
            throw AssertionError("Expected the entry limit to be enforced")
        } catch (e: InvalidModelArchiveException) {
            assertEquals(InvalidModelArchiveException.Reason.ARCHIVE_LIMIT, e.reason)
        }
    }

    @Test
    fun rejectsArchivesContainingMoreThanOneSupportedModel() {
        val archive = zip(
            "model.param" to byteArrayOf(1),
            "model.bin" to byteArrayOf(2),
            "extra.tflite" to byteArrayOf(3),
        )

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), folder.newFolder("mixed"), "local-mixed-model")
            throw AssertionError("Expected multiple supported models to be rejected")
        } catch (_: InvalidModelArchiveException) {
        }
    }

    @Test
    fun rejectsAnArchiveWithoutAMatchingPair() {
        val archive = zip(
            "model.param" to "network".toByteArray(),
            "other.bin" to "weights".toByteArray(),
        )

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), folder.newFolder("empty"), "local-model-2")
            throw AssertionError("Expected an invalid archive")
        } catch (_: InvalidModelArchiveException) {
        }
    }

    @Test
    fun rejectsAmbiguousPairs() {
        val archive = zip(
            "first/model.param" to byteArrayOf(1),
            "first/model.bin" to byteArrayOf(2),
            "second/model.param" to byteArrayOf(3),
            "second/model.bin" to byteArrayOf(4),
        )

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), folder.newFolder("ambiguous"), "local-model-3")
            throw AssertionError("Expected an ambiguous archive to be rejected")
        } catch (_: InvalidModelArchiveException) {
        }
    }

    @Test
    fun extractsAllModelsWithTheirOwnSettings() {
        val archive = zip(
            "pack/alpha.param" to "alpha-net".toByteArray(),
            "pack/alpha.bin" to "alpha-weights".toByteArray(),
            "pack/alpha.json" to """{"roles":["body"],"inputSize":320}""".toByteArray(),
            "pack/beta.param" to "beta-net".toByteArray(),
            "pack/beta.bin" to "beta-weights".toByteArray(),
            "pack/beta.json" to """{"roles":["head"],"inputSize":640}""".toByteArray(),
        )

        val models = ModelZipImporter.extractAll(
            ByteArrayInputStream(archive),
            folder.newFolder("multi"),
            "local-batch",
        )

        assertEquals(2, models.size)
        assertEquals(setOf("local-batch-1", "local-batch-2"), models.map { it.modelId }.toSet())
        assertEquals(setOf("alpha", "beta"), models.map { it.modelName }.toSet())
        assertEquals(setOf(320, 640), models.map { ModelProfileSpec.parse(checkNotNull(it.description).readText()).inputSize }.toSet())
    }

    @Test
    fun rejectsSharedModelJsonWhenMoreThanOneModelIsInTheSameDirectory() {
        val archive = zip(
            "pack/alpha.param" to byteArrayOf(1),
            "pack/alpha.bin" to byteArrayOf(2),
            "pack/beta.param" to byteArrayOf(3),
            "pack/beta.bin" to byteArrayOf(4),
            "pack/model.json" to """{"inputSize":320}""".toByteArray(),
        )

        try {
            ModelZipImporter.extractAll(ByteArrayInputStream(archive), folder.newFolder("multi-shared"), "local-batch")
            throw AssertionError("Expected ambiguous settings in multi-model ZIP to be rejected")
        } catch (e: InvalidModelArchiveException) {
            assertEquals(InvalidModelArchiveException.Reason.AMBIGUOUS_METADATA, e.reason)
        }
    }

    @Test
    fun rejectsParentPathsWithoutWritingOutsideTheDestination() {
        val archive = zip(
            "../escape/model.param" to byteArrayOf(1),
            "../escape/model.bin" to byteArrayOf(2),
        )
        val output = folder.newFolder("safe")

        try {
            ModelZipImporter.extract(ByteArrayInputStream(archive), output, "local-model-4")
            throw AssertionError("Expected an unsafe archive path to be rejected")
        } catch (_: InvalidModelArchiveException) {
        }
        assertFalse(folder.root.resolve("escape").exists())
    }

    private fun zip(vararg entries: Pair<String, ByteArray>): ByteArray {
        val bytes = ByteArrayOutputStream()
        ZipOutputStream(bytes).use { output ->
            entries.forEach { (name, content) ->
                output.putNextEntry(ZipEntry(name))
                output.write(content)
                output.closeEntry()
            }
        }
        return bytes.toByteArray()
    }
}
