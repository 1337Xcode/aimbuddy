package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import java.io.InputStream
import java.io.OutputStream
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit

/**
 * The app learns where the touch helper listens from the line it prints. Stopping the assist while the
 * helper is still starting interrupts the thread that waits for that line, and the wait has to give up
 * quietly: an exception there ended the whole app in 0.3.5-beta.1.
 */
class TouchHelperTest {

    private data class WaitResult(val port: Int, val interruptKept: Boolean)

    /**
     * A program whose standard output is what the test writes to [output] and whose error output is [errors], kept
     * apart as Shizuku keeps them. It never ends unless the test closes them.
     */
    private class FakeProgram : Process() {
        val output = PipedOutputStream()
        val errors = PipedOutputStream()
        private val outputReadEnd = PipedInputStream(output)
        private val errorsReadEnd = PipedInputStream(errors)

        override fun getInputStream(): InputStream = outputReadEnd
        override fun getOutputStream(): OutputStream = OutputStream.nullOutputStream()
        override fun getErrorStream(): InputStream = errorsReadEnd
        override fun waitFor(): Int = 0
        override fun exitValue(): Int = 0
        override fun destroy() {
            output.close()
            errors.close()
        }
    }

    @Test
    fun theWaitEndsWithThePortTheHelperPrints() {
        val program = FakeProgram()
        val helper = HelperProcess(program)

        program.output.write("aimtouch 1.0\nlistening 41234\n".toByteArray())
        program.output.flush()

        assertEquals(41234, helper.awaitPort(10_000))
        program.destroy()
    }

    @Test
    fun aHelperThatEndsWithoutSayingWhereItListensGivesNotListeningAndKeepsWhyFromItsErrorOutput() {
        val program = FakeProgram()
        val helper = HelperProcess(program)

        // The helper says why it stops on its error output, as say() in touch_server.cpp does.
        program.errors.write("aimtouch: no touchscreen found\n".toByteArray())
        program.destroy()

        assertEquals(HelperProcess.NOT_LISTENING, helper.awaitPort(10_000))
        assertEquals("aimtouch: no touchscreen found", awaitLog(helper))
    }

    /** The log once the reader threads have caught up: they run apart from the wait for the port. */
    private fun awaitLog(helper: HelperProcess): String {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (helper.log().isEmpty() && System.nanoTime() < deadline) {
            Thread.sleep(1)
        }
        return helper.log()
    }

    @Test
    fun aHelperThatSaysNothingInTimeGivesNotListeningAndLeavesNoInterrupt() {
        val program = FakeProgram()
        val helper = HelperProcess(program)

        assertEquals(HelperProcess.NOT_LISTENING, helper.awaitPort(0))
        assertFalse(Thread.currentThread().isInterrupted)
        program.destroy()
    }

    @Test
    fun aThreadInterruptedWhileWaitingGivesUpAndKeepsTheInterrupt() {
        val program = FakeProgram()
        val helper = HelperProcess(program)
        val wait = FutureTask {
            val port = helper.awaitPort(60_000)
            WaitResult(port, interruptKept = Thread.currentThread().isInterrupted)
        }
        val waiter = Thread(wait)
        waiter.start()
        waitUntilParked(waiter)

        waiter.interrupt()

        // An exception thrown by the wait comes out of get() as an ExecutionException and fails the test.
        assertEquals(WaitResult(HelperProcess.NOT_LISTENING, interruptKept = true), wait.get(10, TimeUnit.SECONDS))
        program.destroy()
    }

    /** Returns once [thread] sits inside its wait, so the interrupt lands during the wait and not before it. */
    private fun waitUntilParked(thread: Thread) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (thread.state != Thread.State.TIMED_WAITING) {
            check(System.nanoTime() < deadline) { "the thread never started waiting (it is ${thread.state})" }
            Thread.sleep(1)
        }
    }
}
