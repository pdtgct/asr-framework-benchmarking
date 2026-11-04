"""Generate the receipt schema reference from a mock run.

The markdown quotes emitted constants and lists the shape of the mock
preset's receipt. A test fails when this output and
``docs/receipt-schema.md`` disagree.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from asrbench.config import resolve
from asrbench.definitions import SCHEMA_VERSION, emitted_definitions
from asrbench.fingerprint import (
    COMPARISON_EXCLUSIONS,
    RECORDED_COMPARISON_FIELDS,
)
from asrbench.identity import FixedProbe, GitState, collect_hardware
from asrbench.runner import AdvancingClock, run_benchmark


def repo_root() -> Path:
    """Return the repository root for an editable install.

    Returns:
        The directory that contains ``docs`` and ``presets``.
    """
    return Path(__file__).resolve().parents[2]


def example_receipt() -> dict[str, Any]:
    """Run the mock preset on injected identity and a scripted clock.

    Returns:
        One receipt. Wall markers and the run id are fixed. Hardware is
        injected, so the shape does not depend on a GPU or on nvidia-smi.
    """
    root = repo_root()
    config = resolve(
        preset_path=root / "presets" / "mock-streaming.json",
        env={},
        overrides={("run", "output_path"): "receipt.json"},
    )
    hardware = collect_hardware(
        FixedProbe(None),
        cpu_model="injected-cpu",
        logical_cpu_count=2,
        architecture="x86_64",
        system="Linux",
        hostname="injected-host",
        collected_at="2026-01-01T00:00:00Z",
    )
    receipt: dict[str, Any] = asyncio.run(
        run_benchmark(
            config,
            clock=AdvancingClock(1.0),
            hardware=hardware,
            git_state=GitState(commit="unknown", dirty=False, available=False),
            package_version="0.0.1",
            python_version="3.12.0",
            run_id="example",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            materialize_dir=root / ".asrbench-schema-clip",
            env={},
        )
    )
    return receipt


def schema_markdown() -> str:
    """Return the schema reference.

    Returns:
        Markdown quoting every emitted constant and the mock receipt shape.
    """
    receipt = example_receipt()
    lines = [
        "# Receipt schema",
        "",
        "Generated from `asrbench.schema_doc`. Do not edit by hand.",
        "Regenerate with `uv run python -m asrbench.schema_doc`.",
        "",
        f"Schema version: `{SCHEMA_VERSION}`",
        "",
        "Clock and definition strings below are the constants the receipt",
        "writes. Where `docs/metrics.md` and a constant disagree, the",
        "constant is the measurement and this file is what a consumer should",
        "quote.",
        "",
        "A field rendered as `null` is always present. Empty lists are",
        "rendered as `list`. The mock preset carries no GPU, so GPU identity",
        "fields are null in this shape.",
        "",
        "## Comparison key",
        "",
        "Components, in order, joined by U+001F and hashed:",
        "workload fingerprint, target identity, software identity,",
        "hardware identity.",
        "",
        "Not in the hard comparison key.",
        "",
        "Record-only. Shown side by side. A mismatch does not warn",
        "and does not refuse:",
        "",
    ]
    for item in RECORDED_COMPARISON_FIELDS:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Warn-only and other exclusions. Warn-only fields are still",
            "compared and reported; a mismatch does not refuse:",
            "",
        ]
    )
    for item in COMPARISON_EXCLUSIONS:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Emitted constants",
            "",
        ]
    )
    lines.extend(["| Name | Value |", "| --- | --- |"])
    for name, value in emitted_definitions().items():
        cell = value.replace("|", "\\|")
        lines.append(f"| `{name}` | {cell} |")
    lines.extend(
        [
            "",
            "## Mock preset receipt shape",
            "",
            "Types are taken from one run of `presets/mock-streaming.json`",
            "with injected hardware and a scripted cell span. A repeated",
            "object is named once and then cited by path.",
            "",
        ]
    )
    seen: dict[tuple[Any, ...], str] = {}
    lines.extend(_shape(receipt, 0, "", seen))
    lines.append("")
    return "\n".join(lines)


def write_schema_doc(path: Path | None = None) -> Path:
    """Write the schema reference.

    Args:
        path: Destination. The default is ``docs/receipt-schema.md``.

    Returns:
        The path written.
    """
    destination = (
        path if path is not None else repo_root() / "docs" / "receipt-schema.md"
    )
    destination.write_text(schema_markdown(), encoding="utf-8")
    return destination


_ALIAS_KEYS = 6


def _shape(
    value: object,
    indent: int,
    path: str,
    seen: dict[tuple[Any, ...], str],
) -> list[str]:
    if not isinstance(value, dict):
        return []
    pad = "  " * indent
    lines: list[str] = []
    for key, item in value.items():
        child = f"{path}.{key}" if path else str(key)
        lines.extend(_field(pad, str(key), item, indent, child, seen))
    return lines


def _field(
    pad: str,
    key: str,
    item: object,
    indent: int,
    path: str,
    seen: dict[tuple[Any, ...], str],
) -> list[str]:
    if key == "definitions" and isinstance(item, dict):
        return [f"{pad}- `{key}`: object of emitted constants quoted above"]
    if key == "warmup_outcomes":
        return [f"{pad}- `{key}`: list[object] (same item shape as outcomes)"]
    if isinstance(item, dict):
        alias = _alias(item, seen)
        if alias is not None:
            return [f"{pad}- `{key}`: same shape as `{alias}`"]
        lines = [f"{pad}- `{key}`: object"]
        lines.extend(_shape(item, indent + 1, path, seen))
        _remember(item, path, seen)
        return lines
    if isinstance(item, list):
        if not item:
            return [f"{pad}- `{key}`: list"]
        head = item[0]
        if isinstance(head, dict):
            lines = [f"{pad}- `{key}`: list[object]"]
            alias = _alias(head, seen)
            item_path = path + "[]"
            if alias is not None:
                lines.append(f"{pad}  - item: same shape as `{alias}`")
                return lines
            lines.append(f"{pad}  - item: object")
            lines.extend(_shape(head, indent + 2, item_path, seen))
            _remember(head, item_path, seen)
            return lines
        return [f"{pad}- `{key}`: list[{_scalar(head)}]"]
    return [f"{pad}- `{key}`: {_scalar(item)}"]


def _signature(value: object) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return (
            "dict",
            tuple((str(key), _signature(item)) for key, item in value.items()),
        )
    if isinstance(value, list):
        if not value:
            return ("list", "empty")
        return ("list", _signature(value[0]))
    return ("scalar", _scalar(value))


def _alias(
    item: dict[str, Any], seen: dict[tuple[Any, ...], str]
) -> str | None:
    if len(item) < _ALIAS_KEYS:
        return None
    return seen.get(_signature(item))


def _remember(
    item: dict[str, Any], path: str, seen: dict[tuple[Any, ...], str]
) -> None:
    if len(item) >= _ALIAS_KEYS:
        seen[_signature(item)] = path


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


if __name__ == "__main__":
    write_schema_doc()
