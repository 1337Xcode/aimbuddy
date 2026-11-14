package com.aimbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OverlayWindowPolicyTest {

    @Test
    fun theWindowIsAlwaysThereWhenCompatibilityModeIsOff() {
        assertTrue(fullScreenWindowWanted(touchCompatMode = false, menuOpen = false, menuPending = false))
        assertTrue(fullScreenWindowWanted(touchCompatMode = false, menuOpen = true, menuPending = false))
    }

    @Test
    fun compatibilityModeKeepsNoWindowWhileTheMenuIsShut() {
        assertFalse(fullScreenWindowWanted(touchCompatMode = true, menuOpen = false, menuPending = false))
    }

    @Test
    fun compatibilityModeHasTheWindowWhileTheMenuIsOpenOrOpening() {
        assertTrue(fullScreenWindowWanted(touchCompatMode = true, menuOpen = true, menuPending = false))
        assertTrue(fullScreenWindowWanted(touchCompatMode = true, menuOpen = false, menuPending = true))
    }

    @Test
    fun compatibilityMenuLifecycleAttachesThenDetachesTheWindow() {
        // (menuOpen, menuPending) -> window wanted
        val steps = listOf(
            (false to false) to false,   // closed: no window
            (false to true) to true,     // asked for: the window attaches and the menu waits for its renderer
            (true to false) to true,     // open
            (false to false) to false,   // closed again: the window detaches
        )
        steps.forEach { (state, expected) ->
            assertEquals(expected, fullScreenWindowWanted(true, menuOpen = state.first, menuPending = state.second))
        }
    }
}
