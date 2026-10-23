"""Refuse a comparison whose hard identity is not the same.

The comparison key is recomputed from the receipt. A stored key that does
not match that recomputation is itself a difference. Warn-only fields are
listed and do not refuse. Audio files are not opened: the stored clip
digest is the workload identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asrbench.canonical import CanonicalError
from asrbench.config import (
    ConfigError,
    config_from_mapping,
    load_json_object,
    validate,
)
from asrbench.fingerprint import (
    COMPARISON_EXCLUSIONS,
    comparison_key,
    driver_for_warn,
    hard_hardware,
    software_identity,
    target_identity,
    warn_identity,
    workload_fingerprint,
    workload_material,
)
from asrbench.spec import ResolvedConfig

__all__ = [
    "COMPARISON_EXCLUSIONS",
    "CompareResult",
    "FieldDifference",
    "compare_receipts",
    "load_receipt",
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
        }


def load_receipt(path: Path) -> dict[str, Any]:
    """Read a receipt object.

    Args:
        path: JSON file.

    Returns:
        The object.

    Raises:
        ConfigError: The file is not a JSON object.
    """
    return load_json_object(path)


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
    left_observation = left_view["observation_fingerprint"]
    right_observation = right_view["observation_fingerprint"]
    observation_match = left_observation == right_observation
    if differences and force:
        status = "forced"
        comparable = False
        recorded = reason.strip() if reason else None
    elif differences:
        status = "refused"
        comparable = False
        recorded = None
    else:
        status = "comparable"
        comparable = True
        recorded = None
    summary = tuple(_summary(item) for item in differences) + tuple(
        "warn: " + _summary(item) for item in warnings
    )
    return CompareResult(
        status=status,
        comparable=comparable,
        differences=tuple(differences),
        warnings=tuple(warnings),
        summary=summary,
        force_reason=recorded,
        observation_fingerprint_match=observation_match,
        comparison_key_left=str(left_view["comparison_key"]),
        comparison_key_right=str(right_view["comparison_key"]),
    )


def _view(receipt: dict[str, Any]) -> dict[str, Any]:
    config = _config(receipt)
    software = _software(config, receipt)
    hardware = _hardware(config, receipt)
    target = target_identity(config.target)
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
        "stored_workload_fingerprint": _nested(
            receipt, "fingerprints", "workload"
        ),
        "stored_comparison_key": receipt.get("comparison_key"),
        "comparison_key": key,
        "target": target,
        "software": software,
        "hardware": hardware,
        "stored_target": _identity_block(receipt, "target"),
        "stored_software": _identity_block(receipt, "software"),
        "stored_hardware": _identity_block(receipt, "hardware"),
        "warn": _warn(config, receipt),
        "observation_fingerprint": _nested(
            receipt, "fingerprints", "observation"
        ),
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


def _software(
    config: ResolvedConfig, receipt: dict[str, Any]
) -> dict[str, Any]:
    stored = _identity_block(receipt, "software")
    schema = receipt.get("schema_version")
    schema_text = schema if isinstance(schema, str) else ""
    adapter = stored.get("adapter")
    if isinstance(adapter, str) and adapter:
        adapter_text = adapter
    else:
        adapter_text = config.target.transport
    return software_identity(
        adapter=adapter_text,
        normalization_policy=config.run.pass_policy.normalization_policy,
        percentile_method=config.run.percentile_method,
        schema_version=schema_text,
    )


def _hardware(
    config: ResolvedConfig, receipt: dict[str, Any]
) -> dict[str, Any]:
    probe = _probe_hardware(receipt)
    return hard_hardware(
        config.target,
        client_gpu_name=_optional_str(probe.get("gpu_name")),
        client_gpu_count=_optional_int(probe.get("gpu_count")),
        client_gpu_memory_mib=_optional_int(probe.get("gpu_memory_mib")),
    )


def _warn(config: ResolvedConfig, receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute warn-only fields from the config and the stored probe.

    A hand edit of ``identity.warn`` that does not match this view is not
    what is compared. The probe and the resolved config are.
    """
    probe = _probe_hardware(receipt)
    software = _probe_software(receipt)
    driver, source = driver_for_warn(
        config.target, _optional_str(probe.get("driver_version"))
    )
    stored = _identity_block(receipt, "warn")

    def pick(primary: dict[str, Any], key: str, default: Any = None) -> Any:
        if key in primary:
            return primary[key]
        return stored.get(key, default)

    return warn_identity(
        endpoint_url=config.target.endpoint_url,
        tls_verify=config.target.tls_verify,
        auth_env=config.target.auth_env,
        lookahead_identity_policy=config.run.lookahead_identity_policy,
        package_version=str(stored.get("package_version", "")),
        python_version=str(stored.get("python_version", "")),
        git_commit=str(pick(software, "git_commit", "unknown")),
        git_dirty=bool(pick(software, "git_dirty", False)),
        driver_version=driver,
        driver_version_source=source,
        system=str(pick(probe, "system", "")),
        architecture=str(pick(probe, "architecture", "")),
        cpu_model=_optional_str(pick(probe, "cpu_model")),
        logical_cpu_count=_optional_int(pick(probe, "logical_cpu_count")),
        hostname=_optional_str(pick(probe, "hostname")),
        probe_status=_optional_str(pick(probe, "probe_status")),
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
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[FieldDifference] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                found.append(
                    FieldDifference(
                        component,
                        path,
                        left.get(key),
                        right.get(key),
                    )
                )
            else:
                found.extend(
                    _leaf_diffs(component, left[key], right[key], path)
                )
        return found
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [
                FieldDifference(component, prefix or component, left, right)
            ]
        found = []
        pairs = zip(left, right, strict=True)
        for index, (item_left, item_right) in enumerate(pairs):
            found.extend(
                _leaf_diffs(
                    component,
                    item_left,
                    item_right,
                    f"{prefix}[{index}]",
                )
            )
        return found
    if left != right:
        return [FieldDifference(component, prefix or component, left, right)]
    return []


def _summary(difference: FieldDifference) -> str:
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


def _nested(payload: dict[str, Any], *names: str) -> Any:
    current: Any = payload
    for name in names:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def _identity_block(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        return {}
    block = identity.get(name)
    if not isinstance(block, dict):
        return {}
    return block


def _probe_hardware(receipt: dict[str, Any]) -> dict[str, Any]:
    probe = receipt.get("probe")
    if not isinstance(probe, dict):
        return {}
    hardware = probe.get("hardware")
    if not isinstance(hardware, dict):
        return {}
    return hardware


def _probe_software(receipt: dict[str, Any]) -> dict[str, Any]:
    probe = receipt.get("probe")
    if not isinstance(probe, dict):
        return {}
    software = probe.get("software")
    if not isinstance(software, dict):
        return {}
    return software


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
