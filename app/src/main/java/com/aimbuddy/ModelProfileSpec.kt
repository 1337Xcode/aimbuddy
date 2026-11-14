package com.aimbuddy

import android.util.Log
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * What a model says about itself, read from a small JSON file saved next to it.
 *
 * Players train their own models, so the app cannot guess what the classes mean. The file
 * has four optional fields:
 *
 * ```
 * { "roles": ["body", "head"], "endToEnd": false, "inputSize": 256, "normalizedCoordinates": true }
 * ```
 *
 * - `roles` lists what each class is, in class order: "body" (an enemy), "head" (an enemy's head)
 *   or "ignore" (anything that must not be highlighted, such as teammates).
 * - `endToEnd` is true for exports that already remove duplicate boxes (shape N x 6).
 * - `inputSize` is the network input size in pixels, 32 to 2048; leave it out to have it detected.
 * - `normalizedCoordinates` is true for fractional output, false for pixels, or omitted to detect it.
 *
 * Raw `[1, 56, 8400]` / `[1, 8400, 56]` and final `[1, 300, 57]` pose outputs are
 * recognized from their shape: one person class followed by 17 COCO keypoints. They
 * always use `body`, never a head-only sidecar. Official YOLO pose uses model pixels.
 *
 * Without a file, one class means "body" and two classes mean "body" then "head".
 */
data class ModelProfileSpec(
    val roles: String?,
    val endToEnd: Boolean,
    val inputSize: Int,
    val normalizedCoordinates: Boolean? = null,
) {
    companion object {
        private const val TAG = "ModelProfileSpec"

        /** The smallest and the largest network input, in pixels, that a description may state. */
        private const val MIN_INPUT_SIZE = 32
        private const val MAX_INPUT_SIZE = 2048

        /** Used when a model comes without a description. */
        val UNKNOWN = ModelProfileSpec(roles = null, endToEnd = false, inputSize = 0)

        /**
         * Reads the description for the model whose `.param` file is [paramPath].
         * `<name>.json` next to the `.param` file wins over `model.json` in the same folder.
         */
        fun readFor(paramPath: String?): ModelProfileSpec {
            if (paramPath.isNullOrBlank()) return UNKNOWN
            val param = File(paramPath)
            return readDescription(param)
        }

        fun readForTFLite(modelPath: String?, profileName: String? = null): ModelProfileSpec {
            if (modelPath.isNullOrBlank()) return UNKNOWN
            val model = File(modelPath)
            val described = readDescription(model)
            return if (described != UNKNOWN) described else
                forTFLiteFileName(profileName ?: model.name) ?: UNKNOWN
        }

        private fun readDescription(model: File): ModelProfileSpec {
            val candidates = listOf(
                File(model.parentFile, model.nameWithoutExtension + ".json"),
                File(model.parentFile, "model.json"),
            )
            val file = candidates.firstOrNull { it.isFile } ?: return UNKNOWN
            return try {
                parse(file.readText())
            } catch (e: JSONException) {
                Log.w(TAG, "Ignoring unreadable model description ${file.name}: ${e.message}")
                UNKNOWN
            } catch (e: IOException) {
                Log.w(TAG, "Could not read the model description ${file.name}: ${e.message}")
                UNKNOWN
            }
        }

        /** Parses the JSON text of a description. Unknown fields are ignored. */
        fun parse(json: String): ModelProfileSpec {
            val root = JSONObject(json)
            val roles = when (val raw = root.opt("roles")) {
                is JSONArray -> canonicalizeRoles((0 until raw.length()).map { raw.optString(it) })
                is String -> canonicalizeRoles(raw.split(','))
                else -> rolesFromClasses(root.optJSONObject("classes"))
            }
            return ModelProfileSpec(
                roles = roles?.takeIf { it.isNotBlank() },
                endToEnd = root.optBoolean("endToEnd", false),
                // A number no network input has is not a size: the detector works it out instead of allocating it.
                inputSize = root.optInt("inputSize", 0).takeIf { it in MIN_INPUT_SIZE..MAX_INPUT_SIZE } ?: 0,
                normalizedCoordinates = if (root.has("normalizedCoordinates")) {
                    root.getBoolean("normalizedCoordinates")
                } else {
                    null
                },
            )
        }

        fun forTFLiteFileName(name: String): ModelProfileSpec? {
            val isOfficialPose = name.equals("yolo26n-pose.tflite", ignoreCase = true)
            val size = if (isOfficialPose) {
                640
            } else {
                Regex("_(192|256|320|640)(?:_|\\.)").find(name)?.groupValues?.get(1)?.toIntOrNull() ?: 0
            }
            val roles = when {
                isOfficialPose -> "body"
                name.startsWith("yolov8n_", ignoreCase = true) ||
                name.startsWith("yolov8n_", ignoreCase = true) ||
                    name.startsWith("valorant_int8_", ignoreCase = true) -> "head"
                name.equals("valorant_256_v26n.tflite", ignoreCase = true) ||
                    name.equals("valorant_256_v26s.tflite", ignoreCase = true) -> "body,head"
                name.startsWith("dawan_0121_v11s_", ignoreCase = true) ->
                    "body,head,ignore,ignore,ignore,ignore,ignore"
                name.startsWith("codm_256_", ignoreCase = true) -> "body,ignore"
                else -> return null
            }
            return ModelProfileSpec(
                roles = roles,
                endToEnd = false,
                inputSize = size,
                // The known raw YOLO exports and the official final COCO pose both use pixel coordinates.
                normalizedCoordinates = if (isOfficialPose) false else null,
            )
        }

        private fun rolesFromClasses(classes: JSONObject?): String? {
            if (classes == null) return null
            val entries = mutableMapOf<Int, String>()
            val keys = classes.keys()
            while (keys.hasNext()) {
                val index = keys.next().toIntOrNull() ?: continue
                if (index in 0 until MAX_CLASSES) {
                    entries[index] = classes.optString(index.toString())
                }
            }
            val lastIndex = entries.keys.maxOrNull() ?: return null
            return canonicalizeRoles((0..lastIndex).map { index -> entries[index].orEmpty() })
        }

        private fun canonicalizeRoles(tokens: List<String>): String? {
            if (tokens.isEmpty()) return null
            var recognized = false
            val canonical = tokens.map { token ->
                val role = roleFromToken(token)
                if (role != "ignore") {
                    recognized = true
                }
                role
            }
            return if (recognized) canonical.joinToString(",") else null
        }

        private fun roleFromToken(token: String): String {
            val word = token.trim().lowercase()
            if (word.isBlank()) return "ignore"
            return when {
                word == "head" || word.contains("head") -> "head"
                word == "body" ||
                    word == "person" ||
                    word == "player" ||
                    word == "enemy" ||
                    word == "target" ||
                    word == "character" ||
                    word == "torso" ||
                    word.contains("enemy") ||
                    word.contains("person") ||
                    word.contains("player") -> "body"
                else -> "ignore"
            }
        }

        private const val MAX_CLASSES = 16
    }
}
