"""Test-wide guards.

Neither of these is defensive tidiness. Every path in :class:`Config` has a
working default under ``~/AssetKeep``, which is the right behaviour for the tool
and a trap for a test.

**Writes.** A fixture that overrides ``db_path`` and forgets ``vault_path``
writes into the real home directory and passes, because it then asserts against
the same default it just polluted. That happened, and nothing in the suite
noticed - the test was green while files accumulated in ``~``. So the suite
refuses to touch the defaults at all, and says which test did it.

**Reads.** ``models_path`` has the same default and the same problem in reverse.
Nothing writes there during a test, so the write guard never fires; but the
optional tier asks whether the weights are present, and a developer who has run
``assetkeep model download`` would then get a scan that queues embedding jobs
while everybody else gets one that does not. A suite whose behaviour depends on
what is installed in somebody's home directory is not a suite. So a config left
at the default gets redirected into ``tmp_path``, where the answer is reliably
"nothing is installed".

**The machine's ollama.** The same trap once more, and this one is not even in
the home directory: the VLM tier asks a server on localhost what it is holding.
On a machine where ollama happens to be running with moondream pulled, every
capability check answers differently and a test asserting that captions are
refused passes for everybody except the person who implemented them. So the
tier is reported absent by default, and a test that wants it present says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assetkeep.config import Config
from assetkeep.tagging import clip, vlm

#: Directories no test may create or write into. These are the defaults a
#: Config gets when a fixture does not override them.
PROTECTED = (
    Config().db_path.parent,
    Config().source_path.parent,
)


def _snapshot() -> dict:
    return {
        path: sorted(p.name for p in path.rglob("*")) if path.exists() else None
        for path in PROTECTED
    }


@pytest.fixture(autouse=True)
def models_are_never_the_real_ones(tmp_path, monkeypatch):
    """Point a defaulted ``models_path`` at an empty directory in tmp_path."""
    default = Config().models_path
    real = clip.directory

    def directory(config):
        if Path(config.models_path) == default:
            return tmp_path / "models" / clip.FOLDER
        return real(config)

    monkeypatch.setattr(clip, "directory", directory)


@pytest.fixture(autouse=True)
def ollama_is_never_the_real_one(monkeypatch):
    """Report the captioning tier absent, whatever this machine is running."""
    monkeypatch.setattr(
        vlm, "installed_models", lambda config, timeout=vlm.PROBE_TIMEOUT: None
    )


@pytest.fixture(autouse=True)
def home_is_off_limits():
    """Fail any test that creates or changes anything under the real defaults."""
    before = _snapshot()
    yield
    after = _snapshot()

    for path, contents in after.items():
        if contents != before[path]:
            pytest.fail(
                f"this test wrote to {path}, which is the real user directory. "
                f"A Config in a fixture needs db_path, vault_path, thumbs_path "
                f"and source_path all pointed at tmp_path."
            )
