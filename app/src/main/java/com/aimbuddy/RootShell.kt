package com.aimbuddy

import android.util.Log
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Talking to the superuser binary, for phones that have root.
 *
 * Root is only used to start the touch helper program. Nothing here changes system settings or
 * file permissions: the helper opens the devices it needs itself.
 */
object RootShell {
    private const val TAG = "RootShell"

    /** What is known of root for this app. It is only learned by asking, and asking may show the player a prompt. */
    enum class Status { NOT_ASKED, ALLOWED, UNAVAILABLE }

    /** The two ways a superuser binary takes a command: `su -c "cmd"` (Magisk, SuperSU) and `su 0 sh -c "cmd"` (the Android source tree). */
    private enum class Style { COMMAND_FLAG, USER_FIRST }

    @Volatile
    private var status = Status.NOT_ASKED

    @Volatile
    private var style = Style.COMMAND_FLAG

    /** What the last request for root found, in this run. */
    fun status(): Status = status

    /** The words that run `script` as root in the given style. */
    private fun suCommand(style: Style, script: String): List<String> = when (style) {
        Style.COMMAND_FLAG -> listOf("su", "-c", script)
        Style.USER_FIRST -> listOf("su", "0", "sh", "-c", script)
    }

    /**
     * Asks the superuser binary to run `id` and checks that it answers as root. The first call may
     * show the player a permission prompt, so it waits up to `timeoutSeconds` for an answer.
     */
    fun requestRoot(timeoutSeconds: Long): Boolean {
        if (status == Status.ALLOWED) {
            return true
        }
        for (candidate in Style.entries) {
            val answer = tryRoot(candidate, timeoutSeconds)
            if (answer == Answer.GRANTED) {
                style = candidate
                status = Status.ALLOWED
                Log.i(TAG, "Root granted ($candidate)")
                return true
            }
            if (answer == Answer.TIMED_OUT) {
                status = Status.UNAVAILABLE
                return false   // waiting for the player's answer in a prompt: do not ask again
            }
            if (answer == Answer.CANCELLED) {
                return false   // the assist was stopped while asking: nothing was learned, and no second prompt is shown
            }
        }
        Log.i(TAG, "Root refused or not available")
        status = Status.UNAVAILABLE
        return false
    }

    /** What the superuser binary said to one attempt. */
    private enum class Answer { GRANTED, REFUSED, TIMED_OUT, CANCELLED }

    /**
     * Asks for root in one style and reports how it went. A thread interrupted while it waits is being told to stop
     * (the service does that when the assist ends): the request is withdrawn and the interrupt stays set.
     */
    private fun tryRoot(candidate: Style, timeoutSeconds: Long): Answer {
        val process = try {
            ProcessBuilder(suCommand(candidate, "id")).redirectErrorStream(true).start()
        } catch (e: IOException) {
            Log.i(TAG, "No superuser binary: ${e.message}")
            return Answer.REFUSED
        }
        val answered = try {
            process.waitFor(timeoutSeconds, TimeUnit.SECONDS)
        } catch (_: InterruptedException) {
            process.destroyForcibly()
            // Throwing the exception cleared the interrupt; set it again so the caller can see it.
            Thread.currentThread().interrupt()
            return Answer.CANCELLED
        }
        if (!answered) {
            process.destroyForcibly()
            Log.w(TAG, "Root request timed out after ${timeoutSeconds}s")
            return Answer.TIMED_OUT
        }
        val output = try {
            process.inputStream.bufferedReader().use { it.readText() }
        } catch (e: IOException) {
            Log.w(TAG, "Could not read the answer to the root request: ${e.message}")
            return Answer.REFUSED
        }
        return if (process.exitValue() == 0 && output.contains("uid=0")) Answer.GRANTED else Answer.REFUSED
    }

    /** Starts `command` as root, with its output merged into one stream. */
    fun start(command: List<String>): Process {
        val line = command.joinToString(" ") { quote(it) }
        return ProcessBuilder(suCommand(style, "exec $line")).redirectErrorStream(true).start()
    }

    private fun quote(word: String): String = "'" + word.replace("'", "'\\''") + "'"
}
