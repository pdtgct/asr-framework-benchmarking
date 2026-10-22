"""Refuse a comparison whose operating point is not the same.

The comparison key is recomputed from the receipt. A stored key that does
not match that recomputation is itself a difference: a hand-edited
identity must not keep a stale hash that says the receipts match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asrbench.canonical import CanonicalError
from asrbench.config import ConfigError, config_from_mapping
from asrbench.fingerprint import (
    COMPARISON_EXCLUSIONS,
    comparison_key,
    hardware_identity,
    software_identity,
    target_identity,
    workload_fingerprint,
)
from asrbench.spec import ResolvedConfig

# Re-export the exclusion list the schema reference quotes.
__all__ = ["COMPARISON_EXCLUSIONS", "CompareResult", "compare_receipts"]


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
    summary: tuple[str, ...]
    force_reason: str | None
    observation_fingerprint_match: bool
    comparison_key_left: str
    comparison_key_right: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON object.

        Returns:
            The result, with differences as objects.
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
                {
                    "component": item.component,
                    "field": item.field,
                    "left": item.left,
                    "right": item.right,
                }
                for item in self.differences
            ],
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: expected a JSON object")
    return payload


def compare_receipts(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    force: bool = False,
    reason: str | None = None,
) -> CompareResult:
    """Compare two receipts and refuse on a key mismatch.

    Args:
        left: First receipt. The left side of each difference.
        right: Second receipt.
        force: Accept a mismatch. Requires ``reason``.
        reason: Why a mismatch is being forced. Recorded on the result.

    Returns:
        A result whose ``status`` is ``comparable``, ``refused``, or
        ``forced``.

    Raises:
        ConfigError: ``force`` and ``reason`` are not used together, or a
            receipt has no resolvable config.
    """
    if force and not (reason and reason.strip()):
        raise ConfigError("--force requires --reason")
    if reason and not force:
        raise ConfigError("--reason requires --force")
    cleaned = reason.strip() if reason else None
    differences = _differences(left, right)
    left_view = _view(left)
    right_view = _view(right)
    observation_match = (
        left_view["observation_fingerprint"]
        == right_view["observation_fingerprint"]
    )
    if differences and force:
        status = "forced"
        comparable = False
    elif differences:
        status = "refused"
        comparable = False
    else:
        status = "comparable"
        comparable = True
    return CompareResult(
        status=status,
        comparable=comparable,
        differences=tuple(differences),
        summary=tuple(_summary(item) for item in differences),
        force_reason=cleaned,
        observation_fingerprint_match=observation_match,
        comparison_key_left=str(left_view["comparison_key"]),
        comparison_key_right=str(right_view["comparison_key"]),
    )


def _view(receipt: dict[str, Any]) -> dict[str, Any]:
    config = _config(receipt)
    stored_workload = _nested(receipt, "fingerprints", "workload")
    recomputed = workload_fingerprint(config)
    target = target_identity(config.target)
    software = _software(config, receipt)
    hardware = _hardware(receipt)
    key = comparison_key(
        workload_fp=recomputed,
        target=target,
        software=software,
        hardware=hardware,
    )
    return {
        "config": config,
        "workload_fingerprint": recomputed,
        "stored_workload_fingerprint": stored_workload,
        "stored_comparison_key": receipt.get("comparison_key"),
        "comparison_key": key,
        "target": target,
        "software": software,
        "hardware": hardware,
        "observation_fingerprint": _nested(
            receipt, "fingerprints", "observation"
        ),
        "workload_material_config": config,
    }


def _differences(
    left: dict[str, Any], right: dict[str, Any]
) -> list[FieldDifference]:
    left_view = _view(left)
    right_view = _view(right)
    found: list[FieldDifference] = []
    found.extend(_integrity("left", left_view))
    found.extend(_integrity("right", right_view))
    if left_view["workload_fingerprint"] != right_view["workload_fingerprint"]:
        found.extend(
            _leaf_diffs(
                "workload_fingerprint",
                _workload_public(left_view["config"]),
                _workload_public(right_view["config"]),
            )
        )
        if not any(item.component == "workload_fingerprint" for item in found):
            found.append(
                FieldDifference(
                    "workload_fingerprint",
                    "digest",
                    left_view["workload_fingerprint"],
                    right_view["workload_fingerprint"],
                )
            )
    found.extend(
        _leaf_diffs(
            "target_identity", left_view["target"], right_view["target"]
        )
    )
    found.extend(
        _leaf_diffs(
            "software_identity", left_view["software"], right_view["software"]
        )
    )
    found.extend(
        _leaf_diffs(
            "hardware_identity", left_view["hardware"], right_view["hardware"]
        )
    )
    return found


def _integrity(side: str, view: dict[str, Any]) -> list[FieldDifference]:
    found: list[FieldDifference] = []
    stored_fp = view["stored_workload_fingerprint"]
    if stored_fp != view["workload_fingerprint"]:
        found.append(
            FieldDifference(
                "workload_fingerprint",
                f"{side}.stored_vs_recomputed",
                stored_fp,
                view["workload_fingerprint"],
            )
        )
    if view["stored_comparison_key"] != view["comparison_key"]:
        found.append(
            FieldDifference(
                "comparison_key",
                f"{side}.stored_vs_recomputed",
                view["stored_comparison_key"],
                view["comparison_key"],
            )
        )
    return found


def _workload_public(config: ResolvedConfig) -> dict[str, Any]:
    """Fields a mismatch report should name.

    The millisecond label is included so a report can show it did not
    change, and excluded from the fingerprint itself. Diffing the public
    view only when fingerprints differ means a label-only edit, which does
    not change the fingerprint, is not reported.
    """
    workload = config.workload
    return {
        "cadence_ms": workload.cadence_ms,
        "chunk_ms": workload.chunk_ms,
        "clip_digest": workload.clip.digest,
        "clip_id": workload.clip.id,
        "iterations": workload.iterations,
        "lanes": workload.lanes,
        "lookahead_frames": workload.lookahead_frames,
        "mode": workload.mode,
        "pacing": workload.pacing,
        "population": [
            {
                "cadence_ms": entry.cadence_ms,
                "lanes": entry.lanes,
                "lookahead_frames": entry.lookahead_frames,
            }
            for entry in workload.population
        ],
        "rate_rps": workload.rate_rps,
        "reference_transcript": workload.reference_transcript,
        "seed": workload.seed,
        "timeout_s": workload.timeout_s,
        "trials": workload.trials,
        "warmup": workload.warmup,
    }


def _software(
    config: ResolvedConfig, receipt: dict[str, Any]
) -> dict[str, Any]:
    identity = _object(receipt.get("identity"), "identity")
    stored = _object(identity.get("software"), "identity.software")
    # Recomputed policy fields come from the resolved config, which is what
    # a reproduction will run. Version, commit, adapter, and interpreter
    # are measurements and stay as the receipt recorded them.
    recomputed = software_identity(
        package_version=str(stored.get("package_version", "")),
        python_version=str(stored.get("python_version", "")),
        git_commit=str(stored.get("git_commit", "unknown")),
        adapter=str(stored.get("adapter", "")),
        lookahead_identity_policy=config.run.lookahead_identity_policy,
        normalization_policy=config.run.pass_policy.normalization_policy,
        percentile_method=config.run.percentile_method,
    )
    return recomputed


def _hardware(receipt: dict[str, Any]) -> dict[str, Any]:
    identity = _object(receipt.get("identity"), "identity")
    stored = _object(identity.get("hardware"), "identity.hardware")
    return hardware_identity(
        system=str(stored.get("system", "")),
        architecture=str(stored.get("architecture", "")),
        cpu_model=_optional_str(stored.get("cpu_model")),
        logical_cpu_count=_optional_int(stored.get("logical_cpu_count")),
        gpu_name=_optional_str(stored.get("gpu_name")),
        gpu_count=_optional_int(stored.get("gpu_count")),
        gpu_memory_mib=_optional_int(stored.get("gpu_memory_mib")),
        driver_version=_optional_str(stored.get("driver_version")),
    )


def _config(receipt: dict[str, Any]) -> ResolvedConfig:
    payload = receipt.get("resolved_config")
    if not isinstance(payload, dict):
        raise ConfigError("receipt has no resolved_config object")
    try:
        config = config_from_mapping(payload)
    except ConfigError:
        raise
    except CanonicalError as error:
        raise ConfigError(str(error)) from error
    # config_from_mapping does not pin. Pin via the same validation the
    # resolver uses, by calling resolve's validator indirectly: applying
    # the mapping through resolve would also re-read the environment.
    from asrbench.config import finalize

    return finalize(config)


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
                        component, path, left.get(key), right.get(key)
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
        for index, (item_left, item_right) in enumerate(
            zip(left, right, strict=True)
        ):
            path = f"{prefix}[{index}]"
            found.extend(_leaf_diffs(component, item_left, item_right, path))
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


def _nested(payload: dict[str, Any], *names: str) -> Any:
    current: Any = payload
    for name in names:
        if not isinstance(current, dict):
            return None
        current = current.get(name)
    return current


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"receipt missing {where}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("hardware identity field is not an int")
    return value
