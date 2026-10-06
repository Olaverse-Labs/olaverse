"""
Choose the vendored diactag code that matches a checkpoint's label-space spec.

``labels.json`` records the ``spec_version`` the checkpoint was built with, and
the decoding code is only valid for that spec (see :mod:`olaverse.nlp._diactag2`).
The version is read from the file itself rather than inferred from the repo name,
so a fork or a pinned older revision is routed by what it actually is.

    1.x  ->  olaverse.nlp._diactag    (diactag-1.0, SPEC_VERSION 1.2.0)
    2.x  ->  olaverse.nlp._diactag2   (diactag-2.0, SPEC_VERSION 2.0.0)

Routing is by major version only; each package's own ``LabelSpace.load`` then
requires an exact match with the spec it was vendored at, so a 1.1.0 or 2.1.0
file still fails loudly rather than loading against code it was not built for.
The strict check is deliberately not loosened here.
"""

import importlib
import json

_PACKAGES = {
    "1": "olaverse.nlp._diactag",
    "2": "olaverse.nlp._diactag2",
}

SUPPORTED_MAJORS = tuple(_PACKAGES)


class DiacTagBackend:
    """One vendored diactag package (a label-space spec generation)."""

    def __init__(self, package: str):
        self.package = package

    def module(self, name: str):
        """Import ``<package>.<name>`` (``labels``, ``infer``, ``model``, ...)."""
        return importlib.import_module(f"{self.package}.{name}")

    @property
    def spec_version(self) -> str:
        return self.module("unicode_ops").SPEC_VERSION

    @property
    def languages(self) -> tuple:
        """ISO-639-3 codes this generation of the label space covers."""
        return tuple(self.module("unicode_ops").LANGS)

    @property
    def supports_case_endings(self) -> bool:
        """Whether ``InferConfig`` has the Arabic ``case_endings`` switch."""
        return "ara" in self.languages

    def __repr__(self):
        return f"DiacTagBackend({self.package!r}, spec {self.spec_version})"


def read_spec_version(labels_path: str) -> str:
    """The ``spec_version`` recorded in a ``labels.json``."""
    with open(labels_path, encoding="utf-8") as f:
        d = json.load(f)
    spec = d.get("spec_version") if isinstance(d, dict) else None
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            f"{labels_path} has no 'spec_version', so it cannot be matched to a "
            f"decoder. Expected a diactag labels.json."
        )
    return spec.strip()


def backend_for_spec(spec_version: str) -> DiacTagBackend:
    """The vendored package that decodes ``spec_version``, or ``ValueError``."""
    major = str(spec_version).split(".")[0]
    if major not in _PACKAGES:
        supported = ", ".join(f"{m}.x" for m in SUPPORTED_MAJORS)
        raise ValueError(
            f"This checkpoint's label space is unicode spec {spec_version!r}, "
            f"which this olaverse release cannot decode (supported: {supported}). "
            f"A newer checkpoint needs a newer olaverse: pip install -U olaverse."
        )
    return DiacTagBackend(_PACKAGES[major])


def backend_for_labels(labels_path: str) -> DiacTagBackend:
    """Read ``labels.json`` and return the matching backend."""
    return backend_for_spec(read_spec_version(labels_path))
