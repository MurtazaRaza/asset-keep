"""Content hashing: the thing an asset's identity is actually made of.

Paths move constantly in game development - reorganising ``Assets/``,
re-exporting, copying a pack into a second project - so a library keyed on paths
rots within a month. Keyed on content, a moved file is silently re-linked on the
next scan and duplicates collapse on their own.

blake2b rather than SHA-256 because it is faster in pure CPython on this
hardware and nothing here is a security boundary; this hash answers "are these
the same bytes", not "did someone tamper with them".
"""

from __future__ import annotations

from hashlib import blake2b
from pathlib import Path

#: 128 bits, as 32 hex characters. Full-width blake2b would be 128 characters in
#: every path, every URL and every log line, to guard against a collision that
#: at library scale is not going to happen: a birthday collision at 128 bits
#: needs on the order of 10^19 assets.
DIGEST_SIZE = 16

#: Large enough that syscall overhead disappears, small enough to stay off the
#: heap's radar when several scans run at once.
CHUNK_SIZE = 64 * 1024


def hash_file(path: Path) -> str:
    """Hex digest of a file's contents.

    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as d:
    ...     f = pathlib.Path(d, "a.txt"); _ = f.write_text("hello")
    ...     g = pathlib.Path(d, "b.txt"); _ = g.write_text("hello")
    ...     hash_file(f) == hash_file(g), len(hash_file(f))
    (True, 32)
    """
    digest = blake2b(digest_size=DIGEST_SIZE)
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Same digest for content already in memory, as vault imports produce."""
    return blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def shard(content_hash: str) -> tuple[str, str]:
    """Two-level directory shard for the thumbnail store.

    256 x 256 buckets keeps any one directory to a few dozen entries at the
    scale this tool is built for, which matters on APFS where directory
    listings degrade long before the filesystem complains.

    >>> shard("abcdef0123456789abcdef0123456789")
    ('ab', 'cd')
    """
    if len(content_hash) < 4:
        raise ValueError(f"hash too short to shard: {content_hash!r}")
    return content_hash[:2], content_hash[2:4]
