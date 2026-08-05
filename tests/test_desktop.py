"""The platform branches, tested from whichever platform is running.

This file exists because the project is now developed on two machines and the
interesting code here is exactly the code one of them can never execute. Every
test passes a platform in rather than reading ``sys.platform``, so the Windows
behaviour is asserted on macOS and the macOS behaviour on Windows, and a change
that breaks the other machine fails here rather than after a push and a clone.

Nothing in this file runs a command. ``reveal_command`` is pure and its output
is the whole contract; actually spawning Explorer during a test run would open
a window per test on the one platform where it would work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assetkeep import desktop

PLATFORMS = ("darwin", "win32", "linux")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_every_platform_can_reveal(platform):
    """No platform falls through to "unsupported". That was the old behaviour."""
    command = desktop.reveal_command(Path("/art/hero.png"), platform)
    assert command


def test_windows_reveal_is_one_command_line_not_an_argument_list():
    """A list would be quoted by list2cmdline into a form explorer cannot parse.

    ``subprocess`` quotes any argument containing a space, which turns
    ``/select,C:\\Program Files\\x.png`` into one quoted blob that explorer
    reads as a filename and fails to find. The path has to arrive already in
    the form explorer documents, which means building the command line here.
    """
    source = Path("C:/Program Files/Art/hero.png")
    command = desktop.reveal_command(source, "win32")

    assert isinstance(command, str)
    # Compared against str(source) rather than a literal, because the literal
    # would be right on exactly one of the two machines this is run from.
    assert command == f'explorer /select,"{source}"'
    assert "Program Files" in command


def test_macos_reveal_selects_the_file():
    source = Path("/art/hero.png")
    assert desktop.reveal_command(source, "darwin") == ["open", "-R", str(source)]


def test_linux_reveal_opens_the_containing_folder():
    """Selecting one item is a per-file-manager flag; the parent works on all."""
    source = Path("/art/hero.png")
    assert desktop.reveal_command(source, "linux") == ["xdg-open", str(source.parent)]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_install_hints_cover_both_optional_system_dependencies(platform):
    hints = desktop.install_hints(platform)
    assert hints["ffmpeg"] and hints["assimp"]
    # The hint is a command to paste, not a description of one. A hint that
    # says "install ffmpeg" is the same as no hint.
    assert "brew" in hints["ffmpeg"] or "winget" in hints["ffmpeg"] or "apt" in hints["ffmpeg"]


def test_no_hint_mentions_a_package_manager_from_another_platform():
    assert "brew" not in desktop.install_hints("win32")["ffmpeg"]
    assert "winget" not in desktop.install_hints("darwin")["ffmpeg"]


def test_unknown_tool_has_no_hint_rather_than_a_wrong_one():
    assert desktop.install_hint("libfoo", "darwin") == ""


def test_search_path_uses_the_variable_the_running_platform_reads(monkeypatch):
    """POSIX reads LD_LIBRARY_PATH, Windows reads PATH, and the separator differs.

    Asserted against the platform actually running rather than a parameter,
    because this one writes to the real environment and the variable it writes
    is what a native binding will go and read moments later.
    """
    import os

    variable = "PATH" if desktop.platform_key() == "win32" else "LD_LIBRARY_PATH"
    monkeypatch.setenv(variable, f"/already/here{os.pathsep}/and/here")

    desktop.extend_library_search_path(["/new/one", "/already/here"])

    entries = os.environ[variable].split(os.pathsep)
    assert entries[0] == "/new/one"
    # Prepended, and never duplicated: calling this twice must not grow the
    # environment without bound.
    assert entries.count("/already/here") == 1
    assert "/and/here" in entries


def test_search_path_is_idempotent(monkeypatch):
    import os

    variable = "PATH" if desktop.platform_key() == "win32" else "LD_LIBRARY_PATH"
    monkeypatch.setenv(variable, "")

    desktop.extend_library_search_path(["/a", "/b"])
    first = os.environ[variable]
    desktop.extend_library_search_path(["/a", "/b"])

    assert os.environ[variable] == first
