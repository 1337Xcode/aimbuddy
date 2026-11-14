package com.aimbuddy

/**
 * Whether the full screen overlay window should be on the screen.
 *
 * Normally it always is: it draws the targets. In touch compatibility mode nothing is drawn over the game, so the window
 * exists only for the menu: from the moment the menu is asked for (the window needs a moment to get its surface) until
 * the menu is closed.
 */
internal fun fullScreenWindowWanted(touchCompatMode: Boolean, menuOpen: Boolean, menuPending: Boolean): Boolean =
    !touchCompatMode || menuOpen || menuPending
