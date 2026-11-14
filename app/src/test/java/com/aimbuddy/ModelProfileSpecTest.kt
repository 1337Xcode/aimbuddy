package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * A model's description comes from a file next to it, which the player may have written or imported.
 * Whatever it says must never reach the detector as a size or a list it cannot work with.
 */
class ModelProfileSpecTest {

    @get:Rule
    val folder = TemporaryFolder()

    @Test
    fun anInputSizeNoNetworkHasIsLeftForTheDetectorToWorkOut() {
        for (size in listOf(100000, 31, 0, -256, Int.MAX_VALUE)) {
            assertEquals("size $size", 0, ModelProfileSpec.parse("""{"inputSize": $size}""").inputSize)
        }
    }

    @Test
    fun aRealInputSizeIsKept() {
        for (size in listOf(32, 160, 256, 640, 1280, 2048)) {
            assertEquals("size $size", size, ModelProfileSpec.parse("""{"inputSize": $size}""").inputSize)
        }
    }

    @Test
    fun rolesCanBeAListOrOneText() {
        assertEquals("body,head", ModelProfileSpec.parse("""{"roles": ["body", "head"]}""").roles)
        assertEquals("body,head", ModelProfileSpec.parse("""{"roles": ["enemy", "head"]}""").roles)
        assertEquals("body,ignore", ModelProfileSpec.parse("""{"roles": "body,ignore"}""").roles)
        assertEquals("body,ignore", ModelProfileSpec.parse("""{"roles": "person,teammate"}""").roles)
        assertEquals(null, ModelProfileSpec.parse("""{"roles": ["teammate","ally"]}""").roles)
        assertEquals(null, ModelProfileSpec.parse("""{"roles": ""}""").roles)
        assertEquals(null, ModelProfileSpec.parse("{}").roles)
    }

    @Test
    fun classMapFallsBackToUnknownWhenNoEnemyClassIsRecognized() {
        assertEquals(
            null,
            ModelProfileSpec.parse("""{"classes":{"0":"teammate","1":"ally"}}""").roles,
        )
    }

    @Test
    fun aModelWithNoDescriptionIsUnknown() {
        val param = folder.newFile("model.param")

        assertEquals(ModelProfileSpec.UNKNOWN, ModelProfileSpec.readFor(param.path))
        assertEquals(ModelProfileSpec.UNKNOWN, ModelProfileSpec.readFor(null))
    }

    @Test
    fun aDescriptionThatIsNotJsonIsIgnored() {
        val param = folder.newFile("model.param")
        File(folder.root, "model.json").writeText("this is not json")

        assertEquals(ModelProfileSpec.UNKNOWN, ModelProfileSpec.readFor(param.path))
    }

    @Test
    fun theDescriptionNamedForTheParamFileWinsOverModelJson() {
        val param = folder.newFile("tiny.param")
        File(folder.root, "model.json").writeText("""{"inputSize": 320}""")
        File(folder.root, "tiny.json").writeText("""{"inputSize": 160, "endToEnd": true}""")

        val spec = ModelProfileSpec.readFor(param.path)

        assertEquals(160, spec.inputSize)
        assertTrue(spec.endToEnd)
        assertFalse(ModelProfileSpec.readFor(folder.newFile("other.param").path).endToEnd)
    }

    @Test
    fun coordinateUnitsAreExplicitOrLeftForTheDetectorToWorkOut() {
        assertEquals(null, ModelProfileSpec.parse("{}").normalizedCoordinates)
        assertEquals(true, ModelProfileSpec.parse("""{"normalizedCoordinates":true}""").normalizedCoordinates)
        assertEquals(false, ModelProfileSpec.parse("""{"normalizedCoordinates":false}""").normalizedCoordinates)
    }

    @Test
    fun knownTFLiteFilesGetTheirProjectClassRoles() {
        assertEquals(
            "head",
            ModelProfileSpec.forTFLiteFileName("valorant_int8_256.tflite")?.roles,
        )
        assertEquals(
            "body,head",
            ModelProfileSpec.forTFLiteFileName("valorant_256_v26n.tflite")?.roles,
        )
        assertEquals(null, ModelProfileSpec.forTFLiteFileName("custom.tflite"))
    }

    @Test
    fun knownTFLiteFilesKeepTheirNetworkInputAndPoseUnits() {
        val valorant = checkNotNull(ModelProfileSpec.forTFLiteFileName("valorant_256_v26n.tflite"))
        assertEquals(256, valorant.inputSize)
        assertEquals(null, valorant.normalizedCoordinates)

        val pose = checkNotNull(ModelProfileSpec.forTFLiteFileName("yolo26n-pose.tflite"))
        assertEquals("body", pose.roles)
        assertEquals(640, pose.inputSize)
        assertEquals(false, pose.normalizedCoordinates)
    }

    @Test
    fun aTFLiteDescriptionOverridesFilenameDefaults() {
        val model = folder.newFile("valorant_int8_256.tflite")
        File(folder.root, "valorant_int8_256.json")
            .writeText("""{"roles":["body"],"inputSize":256}""")

        val spec = ModelProfileSpec.readForTFLite(model.path)

        assertEquals("body", spec.roles)
        assertEquals(256, spec.inputSize)
    }
}
