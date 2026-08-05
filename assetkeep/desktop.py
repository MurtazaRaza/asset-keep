"""The two things this tool has to ask the desktop itself to do.

Everything else here is portable by construction - SQLite, Pillow, numpy and a
browser behave the same everywhere - but "show me where this file is" and "tell
me how to install the missing system dependency" are not, and both are visible
enough that getting them wrong on one platform is the difference between a tool
that works and one that looks broken.

They live in one module rather than behind ``sys.platform`` branches at the call
sites, because the call sites are a FastAPI endpoint and a CLI command and
neither is somewhere a platform branch can be exercised without standing up the
thing around it. Here the command is built by a pure function and run by a thin
one, so the branch a Mac can never take is still a branch a Mac can test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: How long to wait for the file manager before giving up. It is being asked to
#: raise a window, not to do work, so a hang means something is wrong rather
#: than something is slow.
REVEAL_TIMEOUT = 10

#: What to run to install each optional system dependency, per platform. The
#: value is a command a person can paste, which is the only form of this advice
#: worth printing: "install ffmpeg" is not help.
#:
#: Windows has no assimp package worth naming - there is no winget or choco
#: entry, and the three real routes (vcpkg, conda-forge, the official installer)
#: all end with a DLL somewhere this tool then has to be told about - so that
#: one hint points at the README instead of pretending otherwise.
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "darwin": {
        "assimp": "brew install assimp",
        "ffmpeg": "brew install ffmpeg",
        "ffprobe": "brew install ffmpeg",
    },
    "win32": {
        "assimp": "vcpkg install assimp   (see README: Windows)",
        "ffmpeg": "winget install Gyan.FFmpeg",
        "ffprobe": "winget install Gyan.FFmpeg",
    },
    "linux": {
        "assimp": "sudo apt install libassimp-dev",
        "ffmpeg": "sudo apt install ffmpeg",
        "ffprobe": "sudo apt install ffmpeg",
    },
}


def platform_key(platform: str | None = None) -> str:
    """The key into :data:`INSTALL_HINTS` for a ``sys.platform`` string.

    Every Linux and BSD collapses to ``linux``, because the advice is the same
    shape on all of them and getting the package manager exactly right is not
    something this module can do from a platform string anyway.

    >>> platform_key("darwin"), platform_key("win32")
    ('darwin', 'win32')
    >>> platform_key("linux"), platform_key("freebsd14")
    ('linux', 'linux')
    """
    platform = platform or sys.platform
    if platform.startswith("win"):
        return "win32"
    if platform == "darwin":
        return "darwin"
    return "linux"


def install_hint(tool: str, platform: str | None = None) -> str:
    """A pasteable command that installs ``tool``, or ``''`` if none is known.

    >>> install_hint("ffmpeg", "darwin")
    'brew install ffmpeg'
    >>> install_hint("ffmpeg", "win32")
    'winget install Gyan.FFmpeg'
    >>> install_hint("nothing-like-this", "darwin")
    ''
    """
    return INSTALL_HINTS[platform_key(platform)].get(tool, "")


def install_hints(platform: str | None = None) -> dict[str, str]:
    """Every hint for one platform, for the UI to show beside a missing tier."""
    return dict(INSTALL_HINTS[platform_key(platform)])


def reveal_command(path: Path, platform: str | None = None) -> str | list[str]:
    """What to run to show ``path`` to a person, selected in their file manager.

    A string on Windows and a list everywhere else, and that asymmetry is the
    point rather than an oversight. ``explorer`` takes the path as the tail of
    ``/select,`` in a single argument, and :func:`subprocess.list2cmdline` quotes
    any argument containing a space - which for a path with a space in it
    produces ``"/select,C:\\Program Files\\x.png"``, a quoted blob explorer
    parses as a filename and fails to find. Handing Windows a command line
    already in the form it documents avoids the round trip. It is safe to build
    by concatenation because ``"`` is not a legal character in a Windows path,
    so there is nothing in ``path`` that can end the quoted section early.

    Linux gets the containing folder rather than the file: selecting one item is
    a per-file-manager flag, and ``xdg-open`` on the parent is the thing that
    works on all of them.

    The examples use a bare filename so that they assert the same thing on
    every platform - a path with a separator in it renders differently here
    depending on which machine is reading the docstring, which is the trap this
    whole module exists to stay out of. ``tests/test_desktop.py`` covers the
    quoting of a path that does have separators and spaces in it.

    >>> reveal_command(Path("hero.png"), "darwin")
    ['open', '-R', 'hero.png']
    >>> reveal_command(Path("art/hero.png"), "linux")
    ['xdg-open', 'art']
    >>> reveal_command(Path("hero.png"), "win32")
    'explorer /select,"hero.png"'
    """
    key = platform_key(platform)
    if key == "darwin":
        return ["open", "-R", str(path)]
    if key == "win32":
        return f'explorer /select,"{path}"'
    return ["xdg-open", str(path.parent)]


def reveal(path: Path) -> None:
    """Show ``path`` in the file manager. Never raises, never blocks for long.

    Failure is silent here on purpose: the caller already knows the file exists,
    every supported platform has a file manager, and the only remaining reasons
    to fail are a headless session or a desktop without ``xdg-open`` - neither
    of which the person clicking the button can do anything about, and neither
    of which is worth turning a browse into a traceback.
    """
    command = reveal_command(path)
    try:
        subprocess.run(
            command,
            check=False,
            timeout=REVEAL_TIMEOUT,
            # A string command on Windows goes straight to CreateProcess; shell
            # is off so nothing in the path is ever interpreted.
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def extend_library_search_path(directories) -> None:
    """Make ``directories`` findable by a native-library loader, per platform.

    The variable to write differs, and so does the separator, and so does the
    consequence of getting it wrong: on Windows a native binding reads ``PATH``
    and a ``LD_LIBRARY_PATH`` entry is simply ignored, so the library sits on
    disk and the feature reports itself unavailable with no error anywhere.

    Directories are prepended in the order given and never duplicated, so
    calling this twice is a no-op rather than a slowly growing environment.
    """
    variable = "PATH" if platform_key() == "win32" else "LD_LIBRARY_PATH"
    wanted = [str(d) for d in directories if str(d)]

    existing = [p for p in os.environ.get(variable, "").split(os.pathsep) if p]
    os.environ[variable] = os.pathsep.join(dict.fromkeys(wanted + existing))

    # Windows only, and belt-and-braces on top of PATH: it is what lets a DLL
    # opened from one of these folders resolve its own dependencies - an assimp
    # build next to the zlib it was linked against - which PATH alone no longer
    # guarantees under the default secure search mode.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    for directory in wanted:
        try:
            add_dll_directory(directory)
        except OSError:
            continue
