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

    override fun getOutputStream(): OutputStream = ParcelFileDescriptor.AutoCloseOutputStream(remote.outputStream)

    override fun getInputStream(): InputStream = ParcelFileDescriptor.AutoCloseInputStream(remote.inputStream)

    override fun getErrorStream(): InputStream = ParcelFileDescriptor.AutoCloseInputStream(remote.errorStream)

    override fun waitFor(): Int = remote.waitFor()

    override fun exitValue(): Int = remote.exitValue()

    override fun isAlive(): Boolean = try {
        remote.alive()
    } catch (e: RemoteException) {
        Log.w(TAG, "Shizuku did not answer whether the helper runs: ${e.message}")
        false   // the server is gone, and the program went with it
    }

    override fun destroy() {
        try {
            remote.destroy()
        } catch (e: RemoteException) {
            // The server is gone: it ended its programs, this one included, when it went.
            Log.w(TAG, "Could not end the helper through Shizuku: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "TouchHelper"

        /** Starts [command] through Shizuku. Throws when Shizuku is not running, refuses, or cannot run the program. */
        fun start(command: List<String>): Process {
            val service = IShizukuService.Stub.asInterface(Shizuku.getBinder())
                ?: throw IllegalStateException("Shizuku is not running.")
            return ShizukuProcess(service.newProcess(command.toTypedArray(), null, null))
        }
    }
}
