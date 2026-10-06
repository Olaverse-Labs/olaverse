import http.client
import os
import re
import sys
import time
import urllib.request
import urllib.error

# Default Hugging Face repository hosting the serialized model JSON files
HF_REPO_ID = "olaverse/otk-bpe-50k"

# Downloads stream into "<cache file>.part" in chunks of this size and are moved
# into place only once complete, so the cache never holds a half-written file.
_DOWNLOAD_CHUNK = 1 << 20          # 1 MB
# A dropped connection is retried this many times (after the first attempt),
# each time resuming from the bytes already on disk with a Range request.
_DOWNLOAD_RETRIES = 5
# Seconds without a byte before a read is abandoned and retried.
_DOWNLOAD_TIMEOUT = 60


class ModelDownloadError(RuntimeError):
    """A model file could not be downloaded.

    ``status`` is the HTTP status code when the server answered with an error,
    and ``None`` when the failure was in the connection itself.
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class ModelNotFoundError(ModelDownloadError):
    """The server answered 404: the file does not exist in that repository.

    Kept apart from other download failures so a caller that treats a file as
    optional can tell "not there" from "could not reach it".
    """

def _get_hf_token():
    """
    Resolve a Hugging Face access token for private or gated repositories.

    HF_TOKEN wins, then the token store written by `huggingface-cli login`
    (HF_HOME/token, ~/.cache/huggingface/token). Reading the store matters
    because most people authenticate with the CLI and never export HF_TOKEN,
    and without it a private repo fails with a bare 401.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token.strip()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates = [os.path.join(hf_home, "token")]
    else:
        cache_base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        candidates = [os.path.join(cache_base, "huggingface", "token")]

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if token:
                return token
        except OSError:
            continue
    return None


def get_cache_dir():
    """
    Get the default cross-platform local cache directory for olaverse models.
    Respects XDG_CACHE_HOME, falls back to ~/.cache/olaverse/models.
    """
    cache_base = os.environ.get("XDG_CACHE_HOME")
    if not cache_base:
        cache_base = os.path.expanduser("~/.cache")
    return os.path.join(cache_base, "olaverse", "models")

def _usable(path):
    """True for a non-empty regular file.

    An empty file is what an interrupted download used to leave behind. It exists,
    so it would be returned and then fail to load, so it counts as missing.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _resolve_url(repo_id, filename):
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _ssl_context():
    import ssl
    try:
        # macOS Python builds often ship without root certificates; certifi
        # (already pulled in via the requests dependency) provides a CA
        # bundle so verification can stay on.
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    # Escape hatch for environments where verification cannot succeed
    # (e.g. corporate TLS interception with no CA bundle available).
    if os.environ.get("OLAVERSE_INSECURE_SSL") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_content_range(value):
    """``(start, total)`` from ``bytes START-END/TOTAL``; ``total`` is None when unknown.

    Also reads the ``bytes */TOTAL`` form of a 416 answer, where ``start`` is None.
    """
    match = re.match(r"\s*bytes\s+(?:(\d+)-\d+|\*)/(\d+|\*)", value or "")
    if not match:
        return None, None
    start = int(match.group(1)) if match.group(1) is not None else None
    total = int(match.group(2)) if match.group(2) != "*" else None
    return start, total


class _IncompleteDownload(Exception):
    """The body ended before Content-Length bytes arrived."""


def _stream_to_part(url, part_path, token, ctx):
    """Download ``url`` into ``part_path``, resuming and retrying; return the byte count.

    Each attempt asks only for what is missing (``Range: bytes=<size>-``) when a
    partial file exists. A server that ignores Range and sends the whole file
    again simply restarts it. The result is checked against the size the server
    announced. Raises ``ModelDownloadError`` (``ModelNotFoundError`` for a 404)
    once the retries are used up or for an error that retrying cannot fix.
    """
    attempt = 0
    while True:
        offset = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if offset:
            req.add_header("Range", f"bytes={offset}-")

        try:
            expected = None
            with urllib.request.urlopen(req, context=ctx, timeout=_DOWNLOAD_TIMEOUT) as response:
                headers = getattr(response, "headers", None)
                length = _int_or_none(headers.get("Content-Length")) if headers is not None else None
                resumed = offset and getattr(response, "status", None) == 206
                if resumed:
                    start, total = _parse_content_range(headers.get("Content-Range"))
                    if start is not None and start != offset:
                        raise _IncompleteDownload(
                            f"server resumed at byte {start}, not {offset}")
                    expected = total if total is not None else (
                        offset + length if length is not None else None)
                else:
                    # A 200: the whole body again, so whatever was on disk is discarded.
                    expected = length
                with open(part_path, "ab" if resumed else "wb") as out:
                    while True:
                        chunk = response.read(_DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)

            size = os.path.getsize(part_path)
            if expected is not None and size != expected:
                raise _IncompleteDownload(f"received {size} of {expected} bytes")
            if size == 0:
                raise _IncompleteDownload("the server sent an empty file")
            return size

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                # Range not satisfiable: the partial file is either already the
                # whole file, or does not belong to this one.
                _, total = _parse_content_range(exc.headers.get("Content-Range") if exc.headers else None)
                if total is not None and total == offset and offset > 0:
                    return offset
                if os.path.exists(part_path):
                    os.remove(part_path)
                error = exc
            elif exc.code == 404:
                raise ModelNotFoundError(f"HTTP 404 for {url}", status=404) from exc
            elif 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise ModelDownloadError(f"HTTP {exc.code} for {url}", status=exc.code) from exc
            else:
                error = exc
        except (urllib.error.URLError, OSError, http.client.HTTPException,
                _IncompleteDownload) as exc:
            error = exc

        attempt += 1
        if attempt > _DOWNLOAD_RETRIES:
            status = error.code if isinstance(error, urllib.error.HTTPError) else None
            raise ModelDownloadError(f"{error}", status=status) from error
        print(f"Download interrupted ({error}); retry {attempt}/{_DOWNLOAD_RETRIES}, "
              f"resuming from byte {os.path.getsize(part_path) if os.path.exists(part_path) else 0}...",
              file=sys.stderr)
        time.sleep(min(2 ** (attempt - 1), 30))


def get_model_path(filename, repo_id=None):
    """
    Get the path to a model file.
    Resolves locations in the following order:
    1. Custom directory specified via OLAVERSE_MODELS_DIR environment variable.
    2. Local cache directory (~/.cache/olaverse/models/).
    3. Bundled local package directory (olaverse/models/) for development/fallback.
    4. Downloads the model on-demand from Hugging Face if not found locally.

    An empty file in any of those places counts as missing.

    Downloads go to ``<cache path>.part`` in 1 MB chunks, are retried up to 5
    times after a dropped connection (each retry resumes where the last stopped),
    are checked against the size the server announced, and are moved into place
    with ``os.replace`` only when complete. The cache path therefore never holds a
    partial file. An unfinished ``.part`` file is resumed by the next call.

    Args:
        filename (str): Name of the model file (with or without .json extension).
        repo_id (str, optional): Hugging Face repo ID to download from.

    Returns:
        str: Absolute path to the resolved model file.

    Raises:
        ModelNotFoundError: the repository has no such file (HTTP 404).
        ModelDownloadError: any other download failure. Both are ``RuntimeError``
            subclasses, so existing ``except RuntimeError`` handlers still work.
    """
    # 1. Check custom models directory specified via environment variable
    custom_dir = os.environ.get("OLAVERSE_MODELS_DIR")
    if custom_dir:
        custom_path = os.path.join(custom_dir, filename)
        if _usable(custom_path):
            return custom_path

    # 2. Check the local cache directory
    # Cache is namespaced by repo_id (when given) so repos that reuse generic
    # filenames (e.g. every olaverse/prism-* repo ships "model.py") don't
    # collide with each other in a shared flat cache directory.
    cache_dir = get_cache_dir()
    cache_path = os.path.join(cache_dir, repo_id, filename) if repo_id else os.path.join(cache_dir, filename)
    if _usable(cache_path):
        return cache_path

    # 3. Check the local package directory (development fallback)
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    pkg_path = os.path.join(pkg_dir, filename)
    if _usable(pkg_path):
        return pkg_path

    # 4. Download from Hugging Face
    if repo_id is None:
        repo_id = os.environ.get("OLAVERSE_HF_REPO", HF_REPO_ID)

    os.makedirs(os.path.dirname(cache_path) or cache_dir, exist_ok=True)
    url = _resolve_url(repo_id, filename)
    part_path = cache_path + ".part"

    print(f"Downloading {filename} from Hugging Face ({url})...", file=sys.stderr)
    try:
        _stream_to_part(url, part_path, _get_hf_token(), _ssl_context())
        os.replace(part_path, cache_path)
        return cache_path
    except Exception as e:
        status = getattr(e, "status", None)
        error_cls = ModelNotFoundError if isinstance(e, ModelNotFoundError) else ModelDownloadError
        raise error_cls(
            f"Failed to download model file '{filename}' from Hugging Face. "
            f"Please check your internet connection or ensure the file exists. "
            f"To load offline, you can download the model file and place it in '{cache_dir}' or "
            f"set the OLAVERSE_MODELS_DIR environment variable to its folder. "
            f"If this is an SSL certificate error (common on macOS Python without root "
            f"certificates installed), fix your certificate store or set "
            f"OLAVERSE_INSECURE_SSL=1 to skip verification at your own risk. "
            f"Error: {e}",
            status=status,
        ) from e
