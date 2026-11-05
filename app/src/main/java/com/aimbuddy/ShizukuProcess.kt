package com.aimbuddy

import android.os.ParcelFileDescriptor
import android.os.RemoteException
import android.util.Log
import moe.shizuku.server.IRemoteProcess
import moe.shizuku.server.IShizukuService
import rikka.shizuku.Shizuku
import java.io.InputStream
import java.io.OutputStream

/**
 * A program that Shizuku's server started as the shell user, seen as a [Process].
 *
 * The server runs it with a plain `Runtime.exec` and ends it when this app's process dies. A Shizuku
 * user service would start a Java process first, and on Transsion phones (Tecno, Infinix, itel) with
 * Shizuku 13.6.0 that process dies while it starts (RikkaApps/Shizuku#2048), so the helper never ran.
 * The client library keeps its own wrapper for this call private, which is why this class exists.
 */
internal class ShizukuProcess private constructor(private val remote: IRemoteProcess) : Process() {

    // The pipes are fetched here, while start() runs inside its caller's error handling: fetched later, on a thread
    // that reads the output, a server that died in between would throw where nothing catches it and end the app.
    private val output: OutputStream = ParcelFileDescriptor.AutoCloseOutputStream(remote.outputStream)
    private val input: InputStream = ParcelFileDescriptor.AutoCloseInputStream(remote.inputStream)
    private val errors: InputStream = ParcelFileDescriptor.AutoCloseInputStream(remote.errorStream)

    override fun getOutputStream(): OutputStream = output

    override fun getInputStream(): InputStream = input

    override fun getErrorStream(): InputStream = errors

    override fun waitFor(): Int = remote.waitFor()

    override fun exitValue(): Int = remote.exitValue()

    override fun isAlive(): Boolean = try {
        remote.alive()
    } catch (e: RemoteException) {
        Log.w(TAG, "Shizuku did not answer whether the helper runs: ${e.message}")
        false   // the server is gone, and the program went with it
    }

    override fun destroy() = endQuietly(remote)

    companion object {
        private const val TAG = "TouchHelper"

        /** Starts [command] through Shizuku. Throws when Shizuku is not running, refuses, or cannot run the program. */
        fun start(command: List<String>): Process {
            val service = IShizukuService.Stub.asInterface(Shizuku.getBinder())
                ?: throw IllegalStateException("Shizuku is not running.")
            val remote = service.newProcess(command.toTypedArray(), null, null)
            try {
                return ShizukuProcess(remote)
            } catch (e: RemoteException) {
                endQuietly(remote)   // started, but its output cannot be read: it must not be left running
                throw e
            }
        }

        /** Ends [remote] when the server can still be reached; if it cannot, the program already went with it. */
        private fun endQuietly(remote: IRemoteProcess) {
            try {
                remote.destroy()
            } catch (e: RemoteException) {
                Log.w(TAG, "Could not end the helper through Shizuku: ${e.message}")
            }
        }
    }
}
