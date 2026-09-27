"""Point warnings at the user's line, however many exptrail frames are in between."""

from __future__ import annotations

import os
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _in_package(filename: str) -> bool:
    return os.path.abspath(filename).startswith(_PACKAGE_DIR)


def user_stacklevel() -> int:
    """``stacklevel`` for a ``warnings.warn`` made by the caller of this function.

    Walks up to the first frame outside the exptrail package, so the warning
    names the user's ``with Run(...)``, ``run.start()`` or ``@track``-decorated
    call, whichever path led here.
    """
    frame = sys._getframe(1)  # the function that is about to call warnings.warn
    level = 1
    while frame is not None and _in_package(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    return level
