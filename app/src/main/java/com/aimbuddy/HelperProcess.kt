package com.aimbuddy

import android.util.Log
import java.io.IOException
import java.io.InputStream
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.concurrent.thread

/**
 * A running touch helper program, as seen by whoever started it.
 *
 * The helper prints one line, "listening 41234", on its standard output once it is ready for the app
 * to connect, and the port in it was picked by the system when the helper bound it, so no other program
 * can hold it. Everything else it has to say, why it stopped included, goes to its error output. This
 * class reads both, finds that line, and keeps the last lines so the player can be told why it stopped.
 * [TouchHelper] uses it for both routes, which differ only in how the process is started ([ShizukuProcess],
 * whose error output is a pipe of its own, or [RootShell], which joins the two).
 */
class HelperProcess(private val process: Process) {

    private val lock = Any()
    private val lines = ArrayDeque<String>()
    private val port = CompletableFuture<Int>()

    init {
        // Read both outputs as they come, so neither pipe fills up and stalls the helper.
        thread(name = "aimtouch-output", isDaemon = true) {
            try {
                readLines(process.inputStream) { line ->
                    LISTENING.matchEntire(line.trim())?.let { port.complete(it.groupValues[1].toInt()) }
                }
            } finally {
                port.complete(NOT_LISTENING)   // the output ended: the helper is gone (no effect if it already spoke)
            }
        }
        thread(name = "aimtouch-errors", isDaemon = true) {
            readLines(process.errorStream) {}
        }
    }

    /** Keeps every line of [stream] in the log and hands it to [onLine], until the stream ends. */
    private fun readLines(stream: InputStream, onLine: (String) -> Unit) {
        try {
            stream.bufferedReader().forEachLine { line ->
                if (BuildConfig.DEBUG) {
                    Log.d(TAG, line)   // the helper runs with --verbose in debug builds
                }
                synchronized(lock) {
                    lines.addLast(line)
                    while (lines.size > MAX_LINES) {
                        lines.removeFirst()
                    }
                }
                onLine(line)
            }
        } catch (_: IOException) {
            // The helper was stopped while this thread waited for its output: there is nothing left to read.
        }
    }

    /**
     * Waits for the helper to say where it listens. Returns the port, or [NOT_LISTENING] if it ended or took too long.
     * A thread interrupted while it waits is being told to stop (the service does that when the assist ends): the
     * answer is then [NOT_LISTENING] and the interrupt stays set, so the caller tries no other route. An exception
     * thrown here would end a background thread, and with it the whole app.
     */
    fun awaitPort(timeoutMs: Long): Int = try {
        port.get(timeoutMs, TimeUnit.MILLISECONDS)
    } catch (_: TimeoutException) {
        NOT_LISTENING
    } catch (_: InterruptedException) {
        // Throwing the exception cleared the interrupt; set it again so the caller can see it.
        Thread.currentThread().interrupt()
        NOT_LISTENING
    }

    /** The last lines the helper printed. */
    fun log(): String = synchronized(lock) { lines.joinToString("\n") }

    /** Ends the helper. It also ends by itself when the app disconnects. */
    fun stop() {
        process.destroy()
    }

    companion object {
        private const val TAG = "aimtouch"

        /** The answer when the helper never said where it listens. */
        const val NOT_LISTENING = -1

        private const val MAX_LINES = 40
        private val LISTENING = Regex("listening (\\d{1,5})")
    }
}
