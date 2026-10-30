"""A derived index over a directory of receipts.

The receipt stays the unit of truth. The index is rebuilt from those
files and is byte-identical when rebuilt, including when the input paths
are presented in another order. Deleting the index loses nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from asrbench.canonical import dumps
from asrbench.config import ConfigError, load_json_object


def build_index(paths: list[Path]) -> str:
    """Build a JSONL index from receipt files.

    Args:
        paths: Receipt paths. Only the file name is stored, so the index
            does not depend on the directory's absolute path. A JSON
            object that is not a receipt is an error, not a ``KeyError``.

    Returns:
        Canonical JSONL, one receipt per line, sorted by comparison key
        and file name, with a trailing newline when non-empty.

    Raises:
        ConfigError: A path is not a JSON object, or it has no
            ``schema_version`` and ``comparison_key``.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json_object(path)
        if "schema_version" not in payload or "comparison_key" not in payload:
            raise ConfigError(f"{path}: not an asrbench receipt")
        workload = payload["resolved_config"]["workload"]
        rows.append(
            {
                "comparison_key": payload["comparison_key"],
                "conjunction_passed": payload["summary"]["conjunction_passed"],
                "gpu_name": payload["identity"]["hardware"]["gpu_name"],
                "lookahead_frames": workload["lookahead_frames"],
                "lookahead_ms_label": workload["lookahead_ms_label"],
                "model": payload["identity"]["warn"]["model"],
                "normalization_policy": payload["identity"]["software"][
                    "normalization_policy"
                ],
                "observation_fingerprint": payload["fingerprints"][
                    "observation"
                ],
                "path": path.name,
                "schema_version": payload["schema_version"],
                "transport": payload["identity"]["recorded"]["engine"][
                    "transport"
                ],
                "workload_fingerprint": payload["fingerprints"]["workload"],
            }
        )
    rows.sort(key=lambda row: (str(row["comparison_key"]), str(row["path"])))
    if not rows:
        return ""
    return "\n".join(dumps(row) for row in rows) + "\n"


def write_index(directory: Path, destination: Path) -> str:
    """Rebuild ``destination`` from every ``*.json`` receipt in ``directory``.

    Args:
        directory: Directory of receipt files. Non-JSON names are ignored.
        destination: Index path to write.

    Returns:
        The index text that was written.
    """
    paths = sorted(directory.glob("*.json"))
    text = build_index(paths)
    destination.write_text(text, encoding="utf-8")
    return text
