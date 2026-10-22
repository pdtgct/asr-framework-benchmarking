"""Canonical JSON and sha256 fingerprints.

Fingerprints must not depend on dict order, on whether a path was
absolute, or on which float spelling a runtime prefers. Callers hash the
bytes this module emits, not ``json.dumps`` of a live object.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

#: Joins the four comparison-key components. Canonical JSON escapes control
#: characters, so this byte cannot occur inside a component.
COMPONENT_SEPARATOR = "\x1f"


class CanonicalError(ValueError):
    """The value cannot be hashed reproducibly."""


def format_float(value: float) -> str:
    """Return a stable decimal spelling of one finite float.

    Args:
        value: A finite float. ``-0.0`` is spelled ``0``.

    Returns:
        The canonical spelling.

    Raises:
        CanonicalError: ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        raise CanonicalError("non-finite floats are not canonical")
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def freeze(value: Any) -> Any:
    """Convert tuples and other JSON-ish values into canonical containers.

    Args:
        value: A structure of dicts, lists, tuples, strings, bools, ints,
            floats, and None.

    Returns:
        The same structure with tuples as lists.

    Raises:
        CanonicalError: A type or a float cannot be represented.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError("non-finite floats are not canonical")
        return value
    if isinstance(value, (list, tuple)):
        return [freeze(item) for item in value]
    if isinstance(value, dict):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError("canonical objects have string keys")
            frozen[key] = freeze(item)
        return frozen
    raise CanonicalError(f"cannot canonicalize {type(value).__name__}")


def dumps(value: Any) -> str:
    """Serialize ``value`` to canonical JSON text.

    Object keys are sorted. There is no insignificant whitespace. Floats
    use :func:`format_float`. The result is what a fingerprint hashes.

    Args:
        value: A JSON-ish structure. Tuples are accepted.

    Returns:
        Canonical JSON text, with no trailing newline.

    Raises:
        CanonicalError: ``value`` cannot be represented.
    """
    frozen = freeze(value)
    return _emit(frozen)


def _emit(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, list):
        return "[" + ",".join(_emit(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [
            _emit(key) + ":" + _emit(item)
            for key, item in sorted(value.items())
        ]
        return "{" + ",".join(parts) + "}"
    raise CanonicalError(f"cannot emit {type(value).__name__}")


def sha256_text(text: str) -> str:
    """Return ``sha256:`` plus the hex digest of UTF-8 ``text``.

    Args:
        text: The exact preimage.

    Returns:
        A prefixed digest.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "sha256:" + digest


def sha256_bytes(payload: bytes) -> str:
    """Return ``sha256:`` plus the hex digest of ``payload``.

    Args:
        payload: Raw bytes, such as a PCM clip.

    Returns:
        A prefixed digest.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def fingerprint(value: Any) -> str:
    """Hash the canonical JSON of ``value``.

    Args:
        value: A JSON-ish structure.

    Returns:
        A prefixed sha256 digest.
    """
    return sha256_text(dumps(value))


def join_components(components: list[Any]) -> str:
    """Hash comparison-key components in the order given.

    Each component is canonicalized on its own, then the texts are joined
    with :data:`COMPONENT_SEPARATOR`. That is the concrete reading of the
    design's concatenation: four self-delimiting JSON texts, in design
    order, not one object whose key sort would reorder them.

    Args:
        components: The four identity components, workload fingerprint first.

    Returns:
        A prefixed sha256 digest.
    """
    texts = [dumps(component) for component in components]
    preimage = COMPONENT_SEPARATOR.join(texts)
    return sha256_text(preimage)
