"""
Vendored diactag inference code.
================================
Upstream: the ``diactag`` research repo (Apache 2.0), the training code behind
``olaverse/diactag-1.0``. Only the modules needed to *run* a checkpoint are
copied here — the training pipeline, data filtering, evaluation harness and
ONNX exporter stay upstream.

Vendored rather than depended on because ``diactag`` is not published to PyPI,
and because the label space is versioned: ``unicode_ops.SPEC_VERSION`` is
checked against ``labels.json`` on load, so a checkpoint and the code that
decodes it have to move together. Pinning that pairing inside olaverse is the
only way a user cannot get it wrong.

Files are kept **verbatim** from upstream (one exception, noted in ``infer.py``,
where ``PROTECTED_RE`` is inlined) so a future diactag release can be dropped in
wholesale. Do not refactor them in place; port upstream and re-copy.

    vendored from  diactag 1.0.0
    SPEC_VERSION   1.2.0

The public wrapper is :class:`olaverse.nlp.diacritizer.DiacTagDecoder`; nothing
here is part of olaverse's public API.
"""

from olaverse.nlp._diactag.unicode_ops import (  # noqa: F401
    LANGS,
    SPEC,
    SPEC_VERSION,
    graphemes,
    normalize_lang,
    strip_diacritics,
)

__all__ = [
    "LANGS",
    "SPEC",
    "SPEC_VERSION",
    "graphemes",
    "normalize_lang",
    "strip_diacritics",
]
