"""What a URL says about itself.

The one asset kind with no bytes of its own. An asset store page, a tutorial, a
palette someone posted: these are half of what "where did that come from" means
in practice, and until now the library could hold the sprite but not the page it
came off.

This module only fetches and parses. Nothing here touches the database, for the
same reason the other probes do not: what a page claims about itself and what
the library decides to record are two separate decisions, and only the first one
involves the network. :mod:`assetkeep.reference` makes the second.

**A fetch is best effort and never raises.** A dead link, a page behind a login,
a server that hangs: each of those produces a reference with a title taken from
the URL rather than an error somebody has to clear. The URL is the asset; the
metadata is a nicety that either arrived or did not.

Three limits, all of them because the response comes from somewhere untrusted:
only ``http`` and ``https`` are followed, so a pasted ``file:///etc/passwd``
cannot be turned into a preview; the HTML is read to a cap rather than to the
end; and the preview image is capped again and rejected unless the server calls
it an image. Nothing here is a security boundary - this is a local tool fetching
a URL its user typed - but a hostile page should cost a timeout, not a disk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse, urlunparse

from .. import __version__

log = logging.getLogger(__name__)

#: Sent so a site that blocks unidentified clients gets something to allow, and
#: so anybody reading their logs can tell what this is.
USER_AGENT = f"AssetKeep/{__version__} (+local asset index)"

#: Long enough for a slow asset store, short enough that pasting a dead link
#: does not feel like a hang.
TIMEOUT = 12.0

#: How much HTML is read before the parse gives up. Everything wanted here lives
#: in ``<head>``; a megabyte is several times the largest real one, and the cap
#: is what stops a server that streams forever from being able to.
MAX_HTML_BYTES = 1024 * 1024

#: How large a preview image may be. Above this it is not a thumbnail source,
#: it is a download.
MAX_IMAGE_BYTES = 12 * 1024 * 1024

SAFE_SCHEMES = ("http", "https")


@dataclass
class Page:
    """What one URL yielded, ready for the library to record."""

    #: The URL after redirects, which is the one worth storing: shortened and
    #: tracking-laden links resolve to something a person can recognise a year
    #: later.
    url: str
    title: str = ""
    description: str = ""
    #: ``og:site_name``, falling back to the host. This is what ends up in
    #: ``source_name``, and the host alone is a perfectly good answer.
    site_name: str = ""
    #: Absolute URL of the preview image, already resolved against the page.
    image_url: str | None = None
    #: Set when the page could not be read. The reference is still made.
    error: str | None = None


def fetch(url: str, timeout: float = TIMEOUT) -> Page:
    """Read a URL and extract what a library entry needs from it.

    A URL that is itself an image is a case worth handling rather than an edge:
    pasting a link straight to a reference JPEG is how most of these arrive, and
    there is no HTML to parse. The image becomes its own preview and the
    filename its title.
    """
    import httpx

    normalised = normalise(url)
    if normalised is None:
        return Page(url=url, title=url, error=f"not a fetchable URL: {url}")

    page = Page(url=normalised, title=fallback_title(normalised))
    page.site_name = host_of(normalised)

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
        ) as client:
            with client.stream("GET", normalised) as response:
                response.raise_for_status()
                page.url = str(response.url)
                page.site_name = host_of(page.url)
                content_type = response.headers.get("content-type", "").lower()

                if content_type.startswith("image/"):
                    page.title = fallback_title(page.url)
                    page.image_url = page.url
                    return page

                body = _read_capped(response, MAX_HTML_BYTES)
    except Exception as exc:  # noqa: BLE001 - a dead link is not a failure
        log.info("could not fetch %s: %s", normalised, exc)
        page.error = f"{type(exc).__name__}: {exc}"
        return page

    _apply(page, parse_html(body))
    return page


def _read_capped(response, limit: int) -> str:
    """Decode at most ``limit`` bytes of a streaming response."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= limit:
            break
    encoding = response.charset_encoding or "utf-8"
    return b"".join(chunks)[:limit].decode(encoding, errors="replace")


def _apply(page: Page, found: dict[str, str]) -> None:
    """Fold parsed metadata onto a page, keeping the fallbacks it arrived with.

    Order matters and is not arbitrary. ``og:title`` is written for sharing and
    is nearly always cleaner than ``<title>``, which carries the site name and a
    tagline; the ``<title>`` is the fallback, and the URL's own last segment is
    the fallback for that.
    """
    title = found.get("og:title") or found.get("title") or ""
    if title.strip():
        page.title = _collapse(title)

    description = found.get("og:description") or found.get("description") or ""
    page.description = _collapse(description)

    site = found.get("og:site_name")
    if site and site.strip():
        page.site_name = _collapse(site)

    image = (
        found.get("og:image")
        or found.get("twitter:image")
        or found.get("icon")
    )
    if image:
        candidate = urljoin(page.url, image.strip())
        if urlparse(candidate).scheme in SAFE_SCHEMES:
            page.image_url = candidate


def fetch_image(url: str, timeout: float = TIMEOUT) -> bytes | None:
    """Download a preview image, or ``None`` if it is not one or is too large.

    The content type is trusted to say no and not to say yes: a server calling
    a page ``image/png`` still has to survive Pillow opening it, which is what
    :func:`assetkeep.thumbs.store` does with the bytes this returns.
    """
    import httpx

    if urlparse(url).scheme not in SAFE_SCHEMES:
        return None

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not content_type.startswith("image/"):
                    log.info("preview at %s is %s, not an image", url, content_type)
                    return None

                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_IMAGE_BYTES:
                        log.info("preview at %s exceeds the size cap", url)
                        return None
                return bytes(data)
    except Exception as exc:  # noqa: BLE001 - no preview is not a failed add
        log.info("could not fetch the preview at %s: %s", url, exc)
        return None


# --- URLs -------------------------------------------------------------------


#: A host, and nothing that merely parses as one. Unicode letters are in, so an
#: IDN survives; braces, quotes and spaces are out.
_HOST = re.compile(r"^[^\W_]([\w\-.]*[^\W_])?$", re.UNICODE)


def normalise(url: str) -> str | None:
    """A fetchable, comparable form of a URL, or ``None`` if it is not one.

    A bare host gets ``https://``, because that is what pasting from a browser's
    address bar produces once the scheme has been helpfully hidden. The fragment
    goes, since ``#section`` addresses a place on a page rather than a different
    page. Nothing else is touched: query strings distinguish real pages, and a
    trailing slash is the server's business, not this function's.

    The host is checked rather than merely parsed, and that check is here
    because of a bug it would have caught: a UI handler passed a click event
    where a URL was expected, ``urlparse`` was perfectly happy to read
    ``https://{'isTrusted': true}`` as a host, and a reference to it was
    created. Anything without a scheme must also look like a domain, since a
    bare word is far more likely to be a mistake than a LAN hostname - and
    ``http://nas:8080/x``, typed with its scheme, still works.

    >>> normalise("https://kenney.nl/assets/platformer-pack#tiles")
    'https://kenney.nl/assets/platformer-pack'
    >>> normalise("kenney.nl/assets")
    'https://kenney.nl/assets'
    >>> normalise("HTTPS://Kenney.NL/Assets")
    'https://kenney.nl/Assets'
    >>> normalise("http://localhost:8765/thing")
    'http://localhost:8765/thing'
    >>> normalise("file:///etc/passwd") is None
    True
    >>> normalise("{'isTrusted': true}") is None
    True
    >>> normalise("not a url") is None
    True
    >>> normalise("   ") is None
    True
    """
    text = (url or "").strip()
    if not text:
        return None

    parsed = urlparse(text)
    explicit = bool(parsed.scheme)
    if not explicit:
        parsed = urlparse(f"https://{text}")
    if parsed.scheme not in SAFE_SCHEMES or not parsed.netloc:
        return None

    try:
        host = parsed.hostname
    except ValueError:  # a malformed port, or a broken IPv6 literal
        return None
    if not host or not _HOST.match(host):
        return None
    if not explicit and "." not in host:
        return None

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def host_of(url: str) -> str:
    """The host, without ``www.`` or a port.

    >>> host_of("https://www.artstation.com/artwork/xyz")
    'artstation.com'
    >>> host_of("http://localhost:8080/notes")
    'localhost'
    """
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def fallback_title(url: str) -> str:
    """A title for a page that has not been read, or has nothing to say.

    The last meaningful path segment, which is usually the slug somebody would
    recognise, and the host when there is no path at all.

    >>> fallback_title("https://kenney.nl/assets/platformer-pack-redux")
    'platformer pack redux'
    >>> fallback_title("https://example.com/refs/castle_wall.jpg")
    'castle wall'
    >>> fallback_title("https://kenney.nl/")
    'kenney.nl'
    """
    parsed = urlparse(url)
    segments = [part for part in PurePosixPath(parsed.path).parts if part != "/"]
    if not segments:
        return host_of(url)

    stem = PurePosixPath(segments[-1]).stem or segments[-1]
    return _collapse(stem.replace("-", " ").replace("_", " ")) or host_of(url)


def _collapse(text: str) -> str:
    """One line, no runs of whitespace, and not unboundedly long.

    >>> _collapse("  Kenney  \\n  Platformer   Pack ")
    'Kenney Platformer Pack'
    """
    return " ".join(text.split())[:500]


# --- parsing ----------------------------------------------------------------


class _HeadParser(HTMLParser):
    """Collects the handful of tags that carry a page's own description.

    Stops at ``</head>`` rather than reading the document, which is not an
    optimisation: a body containing ``<title>`` inside an inline SVG is common,
    and taking the last one seen would replace a good title with a bad one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: dict[str, str] = {}
        self._in_title = False
        self.done = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if self.done:
            return
        attributes = {name.lower(): (value or "") for name, value in attrs}

        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or ""
            ).lower()
            content = attributes.get("content", "")
            # First wins: a page repeating og:image lists its best one first,
            # and the later ones are usually per-section decoration.
            if key and content and key not in self.found:
                self.found[key] = content
        elif tag == "link":
            relations = attributes.get("rel", "").lower().split()
            if "icon" in relations and "icon" not in self.found:
                self.found["icon"] = attributes.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self.done = True

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.done and "title" not in self.found:
            self.found["title"] = data


def parse_html(body: str) -> dict[str, str]:
    """Metadata keys a page declares about itself, first occurrence winning.

    >>> found = parse_html('''
    ...   <html><head><title>Kenney - Platformer Pack</title>
    ...   <meta property="og:title" content="Platformer Pack Redux">
    ...   <meta property="og:image" content="/img/pack.png">
    ...   <meta name="description" content="A pack of tiles.">
    ...   </head><body><title>ignored</title></body></html>''')
    >>> found["title"], found["og:title"]
    ('Kenney - Platformer Pack', 'Platformer Pack Redux')
    >>> found["og:image"], found["description"]
    ('/img/pack.png', 'A pack of tiles.')
    """
    parser = _HeadParser()
    try:
        parser.feed(body)
    except Exception as exc:  # noqa: BLE001 - malformed HTML is normal HTML
        log.debug("html parse stopped early: %s", exc)
    return parser.found
