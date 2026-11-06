package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

/** The list of models is read, changed and written back; two changes made at the same time must both survive. */
class ModelCatalogTest {

    @get:Rule
    val folder = TemporaryFolder()

    private fun modelWithFiles(id: String): InstalledModel {
        val param = File(folder.root, "$id.param").also { it.writeText("param") }
        val bin = File(folder.root, "$id.bin").also { it.writeText("bin") }
        return InstalledModel(
            id = id,
            title = "Model $id",
            description = "",
            source = ModelSource.STORE,
            files = ModelFiles.Ncnn(param.path, bin.path),
            totalSizeBytes = 8,
            version = 1,
        )
    }

    @Test
    fun modelsAddedAtTheSameTimeAllStayOnTheList() {
        val catalog = ModelCatalog(File(folder.root, "models"), InMemoryPreferences())
        val perThread = 150
        val models = (0 until 2 * perThread).map { modelWithFiles("model-$it") }
        val together = CountDownLatch(1)

        val adders = listOf(models.take(perThread), models.drop(perThread)).map { share ->
            thread {
                together.await()
                share.forEach { catalog.addOrUpdateModel(it, makeActive = false) }
            }
        }
        together.countDown()
        adders.forEach { it.join() }

        assertEquals(models.map { it.id }.sorted(), catalog.getInstalledModels().map { it.id }.sorted())
    }

    @Test
    fun aModelAddedAgainReplacesItself() {
        val catalog = ModelCatalog(File(folder.root, "models"), InMemoryPreferences())
        val first = modelWithFiles("general")

        catalog.addOrUpdateModel(first)
        catalog.addOrUpdateModel(first.copy(title = "Newer", version = 2))

        val listed = catalog.getInstalledModels().single()
        assertEquals("Newer", listed.title)
        assertEquals(2, listed.version)
        assertTrue(catalog.getActiveModel()?.id == "general")
    }

    @Test
    fun aListSavedByAnEarlierVersionLoadsTheSameModels() {
        val param = File(folder.root, "general.param").also { it.writeText("param") }
        val bin = File(folder.root, "general.bin").also { it.writeText("bin") }
        val preferences = InMemoryPreferences()
        val savedBefore = """[{"id":"general","title":"AimBuddy General","description":"","source":"STORE",""" +
            """"paramPath":"${param.path.replace("\\", "\\\\")}","binPath":"${bin.path.replace("\\", "\\\\")}",""" +
            """"totalSizeBytes":8,"version":2}]"""
        preferences.edit().putString("installed_models_json", savedBefore).apply()

        val listed = ModelCatalog(File(folder.root, "models"), preferences).getInstalledModels().single()

        assertEquals(ModelFiles.Ncnn(param.path, bin.path), listed.files)
        assertEquals(2, listed.version)
    }

    @Test
    fun aTfliteModelIsSavedSoThatEarlierVersionsSkipIt() {
        val model = File(folder.root, "mine.tflite").also { it.writeText("TFL3") }
        val preferences = InMemoryPreferences()
        val catalog = ModelCatalog(File(folder.root, "models"), preferences)
        catalog.addOrUpdateModel(
            InstalledModel(
                id = "mine",
                title = "Mine",
                description = "",
                source = ModelSource.LOCAL,
                files = ModelFiles.Tflite(model.path),
                totalSizeBytes = 4,
            )
        )

        val listed = ModelCatalog(File(folder.root, "models"), preferences).getInstalledModels().single()
        assertEquals(ModelFiles.Tflite(model.path), listed.files)
        assertEquals(listOf(model.path), listed.files.paths)
        // An earlier version reads only paramPath and binPath, and leaves an entry without them off its list.
        val saved = org.json.JSONArray(preferences.getString("installed_models_json", "[]")).getJSONObject(0)
        assertFalse(saved.has("paramPath"))
        assertFalse(saved.has("binPath"))
    }
}
