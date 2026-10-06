"""
Tests for the model downloader (olaverse.utils.downloader.get_model_path) and for
how the diactag loader treats a failed download.

The download tests talk to a real HTTP server on localhost that can cut a response
off part-way, ignore ``Range``, or answer with an error, so truncated bodies,
resuming and 416 behave as they do on the wire rather than as a mock says they do.
Nothing leaves the machine.

What they pin down:

  * a download that dies half-way leaves **no file at the cache path** — it used
    to leave a partial or empty one, which later loads picked up and failed on
    (``EOFError: Ran out of input``);
  * a retry resumes from the bytes already on disk and completes the file;
  * a 0-byte cached file counts as missing;
  * only a 404 is "not found" — the diactag loader no longer turns a dropped
    connection into "No ONNX export found".
"""

import os
import re
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from olaverse.nlp import diacritizer as dz
from olaverse.utils import downloader
from olaverse.utils.downloader import (
    ModelDownloadError,
    ModelNotFoundError,
    get_model_path,
)

REPO, NAME = "olaverse/test-repo", "weights.bin"
BODY = bytes(range(256)) * 14_000            # ~3.4 MB: spans several 1 MB chunks


# =========================================================================== #
# A local server that misbehaves on request
# =========================================================================== #

class _Server:
    """Serves ``body``; request N follows ``plan[N]`` (the last step repeats).

    A step is a dict:
      action        "serve" (default) | "truncate" | "error"
      after         truncate: bytes of the payload actually sent before the cut
      code          error: the HTTP status
      honor_range   False answers a Range request with the whole file (a 200)
    """

    def __init__(self, body, plan=None):
        self.body = body
        self.plan = plan or [{}]
        self.requests = []                    # {"range": ...} per request
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                step = server.plan[min(len(server.requests), len(server.plan) - 1)]
                rng = self.headers.get("Range")
                server.requests.append({"range": rng, "auth": self.headers.get("Authorization")})

                if step.get("action") == "error":
                    self.send_response(step["code"])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                body, start, status = server.body, 0, 200
                if rng and step.get("honor_range", True):
                    start = int(re.match(r"bytes=(\d+)-", rng).group(1))
                    if start >= len(body):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(body)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = 206
                payload = body[start:]

                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                self.end_headers()
                if step.get("action") == "truncate":
                    self.wfile.write(payload[:step["after"]])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)      # cut the connection mid-body
                    self.close_connection = True
                    return
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """``served(body, plan)`` starts a server and points the downloader at it.

    Also gives the downloader an empty cache, no token to leak, no proxy, and a
    sleep that records instead of waiting.
    """
    servers, sleeps = [], []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("OLAVERSE_MODELS_DIR", raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(downloader, "_get_hf_token", lambda: None)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: sleeps.append(s))

    def start(body=BODY, plan=None):
        server = _Server(body, plan)
        servers.append(server)
        monkeypatch.setattr(
            downloader, "_resolve_url",
            lambda repo_id, filename: f"http://127.0.0.1:{server.port}/{repo_id}/{filename}")
        server.sleeps = sleeps
        server.cache_path = os.path.join(downloader.get_cache_dir(), REPO, NAME)
        return server

    yield start
    for server in servers:
        server.stop()


def _read(path):
    with open(path, "rb") as f:
        return f.read()


# =========================================================================== #
# The happy path
# =========================================================================== #

def test_a_complete_download_lands_in_the_cache_with_no_part_file(served):
    server = served()
    path = get_model_path(NAME, repo_id=REPO)
    assert path == server.cache_path
    assert _read(path) == BODY
    assert not os.path.exists(path + ".part")
    assert len(server.requests) == 1 and server.requests[0]["range"] is None


def test_a_cached_file_is_returned_without_touching_the_network(served):
    server = served()
    os.makedirs(os.path.dirname(server.cache_path), exist_ok=True)
    with open(server.cache_path, "wb") as f:
        f.write(b"already here")
    assert get_model_path(NAME, repo_id=REPO) == server.cache_path
    assert server.requests == []


def test_the_body_is_streamed_in_one_megabyte_chunks(served):
    served()
    assert downloader._DOWNLOAD_CHUNK == 1 << 20 and downloader._DOWNLOAD_RETRIES == 5

    sizes, real_urlopen = [], urllib.request.urlopen

    class Spy:
        def __init__(self, response):
            self.response = response
            self.status, self.headers = response.status, response.headers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.response.__exit__(*exc)

        def read(self, amt=None):
            sizes.append(amt)
            return self.response.read(amt)

    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: Spy(real_urlopen(*a, **k))):
        path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY
    assert set(sizes) == {1 << 20}
    assert len(sizes) == len(BODY) // (1 << 20) + 2        # full chunks, a remainder, then b""


# =========================================================================== #
# A download that dies half-way
# =========================================================================== #

def test_a_mid_download_failure_leaves_no_file_at_the_cache_path(served):
    """The connection drops after 1,000 bytes, every time, until the retries run out."""
    server = served(plan=[{"action": "truncate", "after": 1000}])
    with pytest.raises(ModelDownloadError, match="received 6000 of 3584000 bytes"):   # 6 attempts x 1000, resumed each time
        get_model_path(NAME, repo_id=REPO)

    assert not os.path.exists(server.cache_path)                 # nothing a later load could pick up
    assert len(server.requests) == 1 + downloader._DOWNLOAD_RETRIES
    assert server.sleeps == [1, 2, 4, 8, 16]                      # backed off between attempts
    assert os.path.getsize(server.cache_path + ".part") == 6 * 1000   # progress kept for resuming


def test_a_later_call_after_a_failed_download_does_not_see_a_broken_file(served):
    server = served(plan=[{"action": "truncate", "after": 1000}] * 6 + [{}])
    with pytest.raises(ModelDownloadError):
        get_model_path(NAME, repo_id=REPO)
    assert not os.path.exists(server.cache_path)

    path = get_model_path(NAME, repo_id=REPO)                    # the network is back
    assert _read(path) == BODY
    assert server.requests[-1]["range"] == "bytes=6000-"          # resumed, not restarted
    assert not os.path.exists(path + ".part")


def test_a_retry_resumes_from_the_bytes_already_on_disk_and_completes_the_file(served):
    cut = 1_500_000
    server = served(plan=[{"action": "truncate", "after": cut}, {}])
    path = get_model_path(NAME, repo_id=REPO)

    assert _read(path) == BODY                                    # byte-identical, no gap, no repeat
    assert len(server.requests) == 2
    assert server.requests[0]["range"] is None
    assert server.requests[1]["range"] == f"bytes={cut}-"
    assert not os.path.exists(path + ".part")
    assert server.sleeps == [1]


def test_it_survives_several_drops_in_a_row(served):
    server = served(plan=[{"action": "truncate", "after": 700_000}] * 3 + [{}])
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY
    assert [r["range"] for r in server.requests] == [
        None, "bytes=700000-", "bytes=1400000-", "bytes=2100000-"]


def test_a_server_that_ignores_range_restarts_the_file_instead_of_appending(served):
    server = served(plan=[{"action": "truncate", "after": 5000}, {"honor_range": False}])
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY                                    # not 5000 bytes + a whole second copy
    assert server.requests[1]["range"] == "bytes=5000-"


def test_the_size_is_checked_against_content_length(served):
    """A short body is an error even though the connection closed 'cleanly'."""
    served(plan=[{"action": "truncate", "after": 10}])
    with pytest.raises(ModelDownloadError, match=r"received \d+ of \d+ bytes"):
        get_model_path(NAME, repo_id=REPO)


def test_an_unfinished_part_file_from_an_earlier_run_is_resumed(served):
    server = served()
    os.makedirs(os.path.dirname(server.cache_path), exist_ok=True)
    with open(server.cache_path + ".part", "wb") as f:
        f.write(BODY[:123_456])
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY
    assert server.requests[0]["range"] == "bytes=123456-"


def test_a_part_file_that_is_already_complete_is_accepted(served):
    """The server answers 416 'bytes */TOTAL' for a Range past the end: nothing left to fetch."""
    server = served()
    os.makedirs(os.path.dirname(server.cache_path), exist_ok=True)
    with open(server.cache_path + ".part", "wb") as f:
        f.write(BODY)
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY and len(server.requests) == 1


def test_a_part_file_that_does_not_belong_to_the_file_is_discarded(served):
    """Longer than the real file: 416 with a different total, so start over."""
    server = served()
    os.makedirs(os.path.dirname(server.cache_path), exist_ok=True)
    with open(server.cache_path + ".part", "wb") as f:
        f.write(BODY + b"leftovers from a different file")
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY
    assert server.requests[1]["range"] is None                    # restarted from byte 0


# =========================================================================== #
# Empty files
# =========================================================================== #

def test_a_zero_byte_cached_file_is_treated_as_missing(served):
    server = served()
    os.makedirs(os.path.dirname(server.cache_path), exist_ok=True)
    open(server.cache_path, "wb").close()                         # what an interrupted download used to leave
    path = get_model_path(NAME, repo_id=REPO)
    assert _read(path) == BODY and len(server.requests) == 1


def test_a_zero_byte_file_in_the_custom_models_dir_is_skipped(served, tmp_path, monkeypatch):
    server = served()
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / NAME).write_bytes(b"")
    monkeypatch.setenv("OLAVERSE_MODELS_DIR", str(custom))
    path = get_model_path(NAME, repo_id=REPO)
    assert path == server.cache_path and _read(path) == BODY


def test_an_empty_response_is_never_installed(served):
    server = served(body=b"")
    with pytest.raises(ModelDownloadError, match="empty file"):
        get_model_path(NAME, repo_id=REPO)
    assert not os.path.exists(server.cache_path)


# =========================================================================== #
# Errors
# =========================================================================== #

def test_a_404_is_reported_as_not_found_and_is_not_retried(served):
    server = served(plan=[{"action": "error", "code": 404}])
    with pytest.raises(ModelNotFoundError) as exc:
        get_model_path(NAME, repo_id=REPO)
    assert exc.value.status == 404 and isinstance(exc.value, RuntimeError)
    assert len(server.requests) == 1 and server.sleeps == []
    assert not os.path.exists(server.cache_path)


@pytest.mark.parametrize("code", [401, 403])
def test_other_client_errors_are_download_errors_not_not_found(served, code):
    server = served(plan=[{"action": "error", "code": code}])
    with pytest.raises(ModelDownloadError) as exc:
        get_model_path(NAME, repo_id=REPO)
    assert not isinstance(exc.value, ModelNotFoundError)
    assert exc.value.status == code
    assert len(server.requests) == 1                              # retrying cannot fix it


@pytest.mark.parametrize("code", [429, 500, 503])
def test_server_side_and_rate_limit_errors_are_retried(served, code):
    server = served(plan=[{"action": "error", "code": code}] * 2 + [{}])
    assert _read(get_model_path(NAME, repo_id=REPO)) == BODY
    assert len(server.requests) == 3 and len(server.sleeps) == 2


def test_persistent_server_errors_give_up_after_the_retries(served):
    server = served(plan=[{"action": "error", "code": 503}])
    with pytest.raises(ModelDownloadError) as exc:
        get_model_path(NAME, repo_id=REPO)
    assert exc.value.status == 503
    assert len(server.requests) == 1 + downloader._DOWNLOAD_RETRIES


def test_a_refused_connection_is_retried_then_reported(served, monkeypatch):
    server = served()
    server.stop()                                                 # nothing is listening any more
    with pytest.raises(ModelDownloadError) as exc:
        get_model_path(NAME, repo_id=REPO)
    assert exc.value.status is None
    assert len(server.sleeps) == downloader._DOWNLOAD_RETRIES
    assert not os.path.exists(server.cache_path)


def test_the_error_message_keeps_the_original_cause_and_the_hints(served):
    served(plan=[{"action": "error", "code": 403}])
    with pytest.raises(ModelDownloadError) as exc:
        get_model_path(NAME, repo_id=REPO)
    message = str(exc.value)
    assert "Failed to download model file 'weights.bin'" in message
    assert "HTTP 403" in message and "OLAVERSE_MODELS_DIR" in message


def test_the_token_is_sent_when_there_is_one(served, monkeypatch):
    server = served()
    monkeypatch.setattr(downloader, "_get_hf_token", lambda: "hf_test_token")
    get_model_path(NAME, repo_id=REPO)
    assert server.requests[0]["auth"] == "Bearer hf_test_token"


# =========================================================================== #
# The diactag loader: only a 404 means "not found"
# =========================================================================== #

def _fail(exc):
    def get(filename, repo_id=None):
        raise exc
    return get


def test_optional_fetch_returns_none_only_for_a_404():
    with patch.object(dz, "get_model_path", _fail(ModelNotFoundError("HTTP 404", status=404))):
        assert dz._diactag_fetch("olaverse/x", "calibration.json", required=False) is None


@pytest.mark.parametrize("error", [
    ModelDownloadError("Failed to download ... Error: HTTP 503", status=503),
    ModelDownloadError("Failed to download ... Error: connection reset"),
    ConnectionResetError("reset by peer"),
    PermissionError("cache not writable"),
])
def test_optional_fetch_reraises_every_other_error_unchanged(error):
    with patch.object(dz, "get_model_path", _fail(error)):
        with pytest.raises(type(error)) as exc:
            dz._diactag_fetch("olaverse/x", "diactag.onnx", required=False)
    assert exc.value is error                                     # the very same exception, message and all


def test_required_fetch_still_adds_the_authentication_hint():
    with patch.object(dz, "get_model_path", _fail(ModelDownloadError("HTTP 401", status=401))):
        with pytest.raises(RuntimeError, match="huggingface-cli") as exc:
            dz._diactag_fetch("olaverse/x", "labels.json")
    assert "HTTP 401" in str(exc.value)
    with patch.object(dz, "get_model_path", _fail(ModelNotFoundError("HTTP 404", status=404))):
        with pytest.raises(RuntimeError, match="Could not fetch 'labels.json'"):
            dz._diactag_fetch("olaverse/x", "labels.json")


def _by_name(table):
    calls = []

    def get(filename, repo_id=None):
        calls.append(filename)
        outcome = table[filename]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    get.calls = calls
    return get


def test_onnx_lookup_reports_not_found_only_when_every_name_is_a_404():
    missing = ModelNotFoundError("HTTP 404", status=404)
    get = _by_name({"diactag.int8.onnx": missing, "diactag.onnx": missing})
    pytest.importorskip("onnxruntime")
    with patch.object(dz, "get_model_path", get):
        with pytest.raises(FileNotFoundError, match="No ONNX export found in 'olaverse/diactag-2.0'"):
            dz.DiacTagDecoder._load_onnx("olaverse/diactag-2.0", labels=MagicMock(n_langs=11))
    assert get.calls == ["diactag.int8.onnx", "diactag.onnx"]


def test_onnx_lookup_does_not_turn_a_download_failure_into_not_found():
    pytest.importorskip("onnxruntime")
    dropped = ModelDownloadError(
        "Failed to download model file 'diactag.int8.onnx' ... Error: received 4096 of 38623044 bytes")
    get = _by_name({"diactag.int8.onnx": dropped, "diactag.onnx": "/never/asked"})
    with patch.object(dz, "get_model_path", get):
        with pytest.raises(ModelDownloadError, match="received 4096 of 38623044 bytes") as exc:
            dz.DiacTagDecoder._load_onnx("olaverse/diactag-2.0", labels=MagicMock(n_langs=11))
    assert exc.value is dropped
    assert get.calls == ["diactag.int8.onnx"]                    # did not quietly fall through to fp32


def test_onnx_lookup_falls_back_to_fp32_on_a_404_and_surfaces_its_failure():
    pytest.importorskip("onnxruntime")
    missing = ModelNotFoundError("HTTP 404", status=404)
    dropped = ModelDownloadError("Failed to download ... Error: HTTP 503", status=503)
    get = _by_name({"diactag.int8.onnx": missing, "diactag.onnx": dropped})
    with patch.object(dz, "get_model_path", get):
        with pytest.raises(ModelDownloadError, match="HTTP 503"):
            dz.DiacTagDecoder._load_onnx("olaverse/diactag-2.0", labels=MagicMock(n_langs=11))
    assert get.calls == ["diactag.int8.onnx", "diactag.onnx"]


def test_onnx_lookup_uses_the_fp32_export_when_there_is_no_int8_one():
    class IO:
        def __init__(self, name):
            self.name = name

    class Session:
        def __init__(self, path, *a, **k):
            self.path = path

        def get_inputs(self):
            return [IO(n) for n in ("ids", "lang", "lang_known", "attn")]

        def get_outputs(self):
            return [IO(n) for n in ("shape_logits", "tone_logits", "lid_logits")]

    fake_ort = MagicMock()
    fake_ort.InferenceSession = Session
    get = _by_name({"diactag.int8.onnx": ModelNotFoundError("HTTP 404", status=404),
                    "diactag.onnx": "/cache/diactag.onnx"})
    with patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
         patch.object(dz, "get_model_path", get):
        adapter = dz.DiacTagDecoder._load_onnx("olaverse/diactag-2.0", labels=MagicMock(n_langs=11))
    assert adapter.sess.path == "/cache/diactag.onnx" and adapter.has_lid
