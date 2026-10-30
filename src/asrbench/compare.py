"""Refuse a comparison whose hard identity is not the same.

The comparison key is recomputed from the receipt. A stored key that does
not match that recomputation is itself a difference. Warn-only fields are
listed and do not refuse. Audio files are not opened: the stored clip
digest is the workload identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from asrbench.canonical import CanonicalError, leaf_differences
from asrbench.config import (
    ConfigError,
    config_from_mapping,
    validate,
)
from asrbench.definitions import CHECKPOINT_SOURCE_UNVERIFIED
from asrbench.fingerprint import (
    COMPARISON_EXCLUSIONS,
    comparison_key,
    identity_blocks,
    workload_fingerprint,
    workload_material,
)
from asrbench.spec import ResolvedConfig

__all__ = [
    "COMPARISON_EXCLUSIONS",
    "CompareResult",
    "FieldDifference",
    "compare_receipts",
]


@dataclass(frozen=True)
class FieldDifference:
    """One leaf that is not the same on both receipts."""

    component: str
    field: str
    left: Any
    right: Any


@dataclass(frozen=True)
class CompareResult:
    """The machine-readable result of a comparison."""

    status: str
    comparable: bool
    differences: tuple[FieldDifference, ...]
    warnings: tuple[FieldDifference, ...]
    recorded: tuple[FieldDifference, ...]
    summary: tuple[str, ...]
    force_reason: str | None
    observation_fingerprint_match: bool
    comparison_key_left: str
    comparison_key_right: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON object.

        Returns:
            The result, with differences and warnings as objects.
        """
        return {
            "status": self.status,
            "comparable": self.comparable,
            "force_reason": self.force_reason,
            "observation_fingerprint_match": self.observation_fingerprint_match,
            "comparison_key_left": self.comparison_key_left,
            "comparison_key_right": self.comparison_key_right,
            "summary": list(self.summary),
            "differences": [
                _difference_dict(item) for item in self.differences
            ],
            "warnings": [_difference_dict(item) for item in self.warnings],
            "recorded": [_difference_dict(item) for item in self.recorded],
        }


def compare_receipts(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    force: bool = False,
    reason: str | None = None,
) -> CompareResult:
    """Compare two receipts and refuse on a hard-key mismatch.

    Args:
        left: First receipt. The left side of each difference.
        right: Second receipt.
        force: Accept a hard mismatch. Requires a non-blank ``reason``.
        reason: Why a mismatch is being forced. Recorded only when the
            result is ``forced``. A reason with nothing to force is not
            recorded.

    Returns:
        A result whose ``status`` is ``comparable``, ``refused``, or
        ``forced``. Warnings do not change a comparable result.

    Raises:
        ConfigError: ``force`` and ``reason`` are not used together, or a
            receipt has no resolvable config.
    """
    if force and not (reason and reason.strip()):
        raise ConfigError("--force requires --reason")
    if reason is not None and not force:
        raise ConfigError("--reason requires --force")
    left_view = _view(left)
    right_view = _view(right)
    differences = _hard_differences(left_view, right_view)
    warnings = _leaf_diffs("warn", left_view["warn"], right_view["warn"])
    warnings.extend(_unverified_checkpoint(left_view, right_view))
    recorded = _recorded_rows(left_view, right_view)
    left_observation = left_view["observation_fingerprint"]
    right_observation = right_view["observation_fingerprint"]
    observation_match = left_observation == right_observation
    if differences and force:
        status = "forced"
        comparable = False
        force_reason = reason.strip() if reason else None
    elif differences:
        status = "refused"
        comparable = False
        force_reason = None
    else:
        status = "comparable"
        comparable = True
        force_reason = None
    summary = (
        tuple(_summary(item) for item in differences)
        + tuple("warn: " + _summary(item) for item in warnings)
        + tuple("recorded: " + _summary(item) for item in recorded)
    )
    return CompareResult(
        status=status,
        comparable=comparable,
        differences=tuple(differences),
        warnings=tuple(warnings),
        recorded=tuple(recorded),
        summary=summary,
        force_reason=force_reason,
        observation_fingerprint_match=observation_match,
        comparison_key_left=str(left_view["comparison_key"]),
        comparison_key_right=str(right_view["comparison_key"]),
    )


def _view(receipt: dict[str, Any]) -> dict[str, Any]:
    config = _config(receipt)
    blocks = _blocks(config, receipt)
    target = blocks["target"]
    software = blocks["software"]
    hardware = blocks["hardware"]
    recomputed_fp = workload_fingerprint(config)
    key = comparison_key(
        workload_fp=recomputed_fp,
        target=target,
        software=software,
        hardware=hardware,
    )
    return {
        "config": config,
        "workload_fingerprint": recomputed_fp,
        "workload_material": workload_material(config.workload),
        "stored_workload_fingerprint": _get(
            receipt, "fingerprints", "workload"
        ),
        "stored_comparison_key": receipt.get("comparison_key"),
        "comparison_key": key,
        "target": target,
        "software": software,
        "hardware": hardware,
        "stored_target": _mapping(receipt, "identity", "target"),
        "stored_software": _mapping(receipt, "identity", "software"),
        "stored_hardware": _mapping(receipt, "identity", "hardware"),
        "warn": blocks["warn"],
        "recorded": blocks["recorded"],
        "stored_recorded": _mapping(receipt, "identity", "recorded"),
        "observation_fingerprint": _get(receipt, "fingerprints", "observation"),
    }


def _hard_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> list[FieldDifference]:
    found: list[FieldDifference] = []
    found.extend(_integrity(left))
    found.extend(_integrity(right))
    if left["workload_fingerprint"] != right["workload_fingerprint"]:
        found.extend(
            _leaf_diffs(
                "workload_fingerprint",
                left["workload_material"],
                right["workload_material"],
            )
        )
    found.extend(
        _leaf_diffs("target_identity", left["target"], right["target"])
    )
    found.extend(
        _leaf_diffs("software_identity", left["software"], right["software"])
    )
    found.extend(
        _leaf_diffs("hardware_identity", left["hardware"], right["hardware"])
    )
    return found


def _integrity(view: dict[str, Any]) -> list[FieldDifference]:
    found: list[FieldDifference] = []
    if view["stored_workload_fingerprint"] != view["workload_fingerprint"]:
        found.append(
            FieldDifference(
                "workload_fingerprint",
                "stored_vs_recomputed",
                view["stored_workload_fingerprint"],
                view["workload_fingerprint"],
            )
        )
    if view["stored_comparison_key"] != view["comparison_key"]:
        found.append(
            FieldDifference(
                "comparison_key",
                "stored_vs_recomputed",
                view["stored_comparison_key"],
                view["comparison_key"],
            )
        )
    for component, stored_name, fresh_name in (
        ("target_identity", "stored_target", "target"),
        ("software_identity", "stored_software", "software"),
        ("hardware_identity", "stored_hardware", "hardware"),
        ("recorded_identity", "stored_recorded", "recorded"),
    ):
        found.extend(
            _leaf_diffs(
                component,
                view[stored_name],
                view[fresh_name],
                prefix="stored_vs_recomputed",
            )
        )
    return found


def _blocks(
    config: ResolvedConfig, receipt: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Recompute identity from the config and the stored probe.

    Measurement semantics and schema version come from the receipt, not
    from the installed package. A hand edit of ``identity.warn`` that
    does not match the probe is not what is compared.
    """
    probe_hw = _mapping(receipt, "probe", "hardware")
    probe_sw = _mapping(receipt, "probe", "software")
    stored_warn = _mapping(receipt, "identity", "warn")
    stored_software = _mapping(receipt, "identity", "software")

    def pick(primary: dict[str, Any], key: str, default: Any = None) -> Any:
        if key in primary:
            return primary[key]
        return stored_warn.get(key, default)

    adapter = stored_software.get("adapter")
    schema = receipt.get("schema_version")
    semantics = stored_software.get("measurement_semantics_version")
    if isinstance(adapter, str) and adapter:
        adapter_text = adapter
    else:
        adapter_text = config.target.transport
    return identity_blocks(
        config,
        {
            "architecture": str(pick(probe_hw, "architecture", "")),
            "cpu_model": _optional_str(pick(probe_hw, "cpu_model")),
            "driver_version": _optional_str(probe_hw.get("driver_version")),
            "gpu_count": _optional_int(probe_hw.get("gpu_count")),
            "gpu_memory_mib": _optional_int(probe_hw.get("gpu_memory_mib")),
            "gpu_name": _optional_str(probe_hw.get("gpu_name")),
            "hostname": _optional_str(pick(probe_hw, "hostname")),
            "logical_cpu_count": _optional_int(
                pick(probe_hw, "logical_cpu_count")
            ),
            "probe_status": _optional_str(pick(probe_hw, "probe_status")),
            "system": str(pick(probe_hw, "system", "")),
        },
        {
            "adapter": adapter_text,
            "git_commit": str(pick(probe_sw, "git_commit", "unknown")),
            "git_dirty": bool(pick(probe_sw, "git_dirty", False)),
            "measurement_semantics_version": (
                semantics if isinstance(semantics, str) else None
            ),
            "package_version": str(stored_warn.get("package_version", "")),
            "python_version": str(stored_warn.get("python_version", "")),
            "schema_version": schema if isinstance(schema, str) else "",
        },
    )


def _config(receipt: dict[str, Any]) -> ResolvedConfig:
    payload = receipt.get("resolved_config")
    if not isinstance(payload, dict):
        raise ConfigError("receipt has no resolved_config object")
    try:
        config = config_from_mapping(payload, complete=True)
    except ConfigError:
        raise
    except CanonicalError as error:
        raise ConfigError(str(error)) from error
    return validate(config, pin=False)


def _leaf_diffs(
    component: str, left: Any, right: Any, prefix: str = ""
) -> list[FieldDifference]:
    return [
        FieldDifference(component, path or component, item_left, item_right)
        for path, item_left, item_right in leaf_differences(left, right, prefix)
    ]


def _recorded_rows(
    left: dict[str, Any], right: dict[str, Any]
) -> list[FieldDifference]:
    """Engine and precision, shown even when the two sides match."""
    fields = (
        "comparison_axis",
        "engine",
        "server_version",
        "precision",
        "tf32",
    )
    return [
        FieldDifference(
            "recorded", field, left["recorded"][field], right["recorded"][field]
        )
        for field in fields
    ]


def _unverified_checkpoint(
    left: dict[str, Any], right: dict[str, Any]
) -> list[FieldDifference]:
    """Flag a checkpoint that is not a verified weights identity."""
    left_source = left["target"]["checkpoint"]["source"]
    right_source = right["target"]["checkpoint"]["source"]
    if (
        left_source != CHECKPOINT_SOURCE_UNVERIFIED
        and right_source != CHECKPOINT_SOURCE_UNVERIFIED
    ):
        return []
    return [
        FieldDifference("checkpoint", "unverified", left_source, right_source)
    ]


def _summary(difference: FieldDifference) -> str:
    if difference.left == difference.right:
        return (
            f"{difference.component}: {difference.field} "
            f"{_show(difference.left)}"
        )
    return (
        f"{difference.component}: {difference.field} "
        f"{_show(difference.left)} != {_show(difference.right)}"
    )


def _show(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _difference_dict(item: FieldDifference) -> dict[str, Any]:
    return {
        "component": item.component,
        "field": item.field,
        "left": item.left,
        "right": item.right,
    }


def _get(payload: dict[str, Any], *names: str) -> Any:
    current: Any = payload
    for name in names:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def _mapping(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    found = _get(payload, *names)
    if isinstance(found, dict):
        return found
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("identity field is not an int")
    return value
