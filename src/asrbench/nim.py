"""NIM identity from ``/v1/metadata``.

A NIM reports its version, the selected model profile, and the model URL
on ``GET /v1/metadata``. This module reads that document from a saved
file or from the NIM itself and maps it onto target fields, so the
receipt records which NIM served the run without hand-copied flags.
Standard library only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from asrbench.config import ConfigError

#: Seconds to wait for ``/v1/metadata`` when reading it over HTTP.
NIM_METADATA_TIMEOUT_S = 10.0


def load_nim_metadata(source: str) -> dict[str, Any]:
    """Read a NIM ``/v1/metadata`` document.

    Args:
        source: An ``http://`` or ``https://`` URL (the NIM's HTTP base
            URL or the full ``/v1/metadata`` URL), or a path to a saved
            copy.

    Returns:
        The decoded JSON object.

    Raises:
        ConfigError: The document cannot be read or is not an object.
    """
    if source.startswith(("http://", "https://")):
        url = source.rstrip("/")
        if not url.endswith("/v1/metadata"):
            url += "/v1/metadata"
        try:
            with urllib.request.urlopen(  # noqa: S310 - http(s) only
                url, timeout=NIM_METADATA_TIMEOUT_S
            ) as response:
                text = response.read().decode("utf-8")
        except (urllib.error.URLError, OSError, UnicodeDecodeError) as error:
            raise ConfigError(
                f"cannot read NIM metadata {url}: {error}"
            ) from None
        where = url
    else:
        try:
            text = Path(source).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ConfigError(
                f"cannot read NIM metadata {source}: {error}"
            ) from None
        where = source
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"NIM metadata {where} is not JSON: {error}"
        ) from None
    if not isinstance(payload, dict):
        raise ConfigError(f"NIM metadata {where} is not a JSON object")
    return payload


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def nim_overrides(metadata: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Map ``/v1/metadata`` onto target fields.

    ``version`` becomes ``engine_version`` and sets ``engine_name`` to
    ``nim``. ``selectedModelProfileId`` becomes ``nim_profile_id``.
    Every ``modelInfo[].modelUrl``, comma-joined in order, becomes
    ``nim_model_url``. A key the document lacks sets nothing. The caller
    lets explicit flags win over these values.

    Args:
        metadata: The decoded ``/v1/metadata`` object.

    Returns:
        Binding path to value.
    """
    found: dict[tuple[str, ...], Any] = {}
    version = _text(metadata.get("version"))
    if version is not None:
        found[("target", "engine_name")] = "nim"
        found[("target", "engine_version")] = version
    profile = _text(metadata.get("selectedModelProfileId"))
    if profile is not None:
        found[("target", "nim_profile_id")] = profile
    info = metadata.get("modelInfo")
    urls: list[str] = []
    if isinstance(info, list):
        for item in info:
            if isinstance(item, dict):
                url = _text(item.get("modelUrl"))
                if url is not None:
                    urls.append(url)
    if urls:
        found[("target", "nim_model_url")] = ",".join(urls)
    return found
