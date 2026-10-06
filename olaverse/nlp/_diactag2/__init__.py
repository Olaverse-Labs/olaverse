"""
Vendored diactag inference code, label-space spec 2.x.
=====================================================
Upstream: the ``diactag`` research repo (Apache 2.0), the code that ships inside
``olaverse/diactag-2.0`` (``code/diactag/``). Only the modules needed to *run* a
checkpoint are copied here — the training pipeline, data filtering, evaluation
harness and label-space extension tooling stay upstream.

This package sits next to :mod:`olaverse.nlp._diactag` (spec 1.x) rather than
replacing it. The label space is versioned, and a spec change can alter how a
character is factorised, so a checkpoint has to be decoded by the code that
matches the ``spec_version`` in its ``labels.json``. Spec 2.0.0 adds Arabic and
changes ``_split_marks`` (Arabic is not NFD-decomposed). Whether that leaves
Latin-script output byte-identical for a 1.x checkpoint is not something the
SDK takes on faith, so diactag-1.0 keeps running on the 1.x code, unchanged.
:mod:`olaverse.nlp._diactag_backend` picks between the two.

The modules are the upstream files, ported rather than rewritten:

    unicode_ops.py   ported  — Arabic marks, ``_is_arabic_letter``, ``script_ok``, ``ara``
    labels.py        ported  — ``LabelSpace.load(path, check_spec=True)``
    infer.py         ported  — ``InferConfig.case_endings``, ``_drop_case_endings``
    model.py         unchanged from the spec 1.x copy (identical upstream)
    lexicon.py       unchanged from the spec 1.x copy (identical upstream)

As in the 1.x package, ``infer.py`` inlines ``PROTECTED_RE`` instead of vendoring
upstream's ``data.py``. Do not refactor these files in place; port upstream and
re-copy.

    vendored from  diactag 2.0.0
    SPEC_VERSION   2.0.0

Nothing here is part of olaverse's public API.
"""

from olaverse.nlp._diactag2.unicode_ops import (  # noqa: F401
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
