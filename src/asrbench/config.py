"""Resolve one frozen configuration.

Order is dataclass defaults, then a preset file, then ``ASRBENCH_*``
environment variables, then CLI flags. ``--from-receipt`` replaces the
defaults and the preset with the receipt's resolved config. In that mode
``ASRBENCH_*`` variables are refused, except the named auth variable,
and a flag that changes a field is recorded on the new receipt.

A single-entry population is derived from the top-level lane, cadence,
and lookahead fields after those overrides. A multi-entry population
stays explicit. Secrets are not flags and not preset fields.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from asrbench.canonical import leaves
from asrbench.clips import ClipError, pin_digest
from asrbench.definitions import (
    LATENCY_METRIC_RESPONSE,
    LATENCY_METRIC_TTFS,
    LOOKAHEAD_IDENTITY_CLIENT_DECLARED,
    LOOKAHEAD_IDENTITY_SERVER_REPORTED,
    MODE_OFFLINE,
    MODE_STREAMING,
    NORMALIZATION_CORPUS_REFERENCE,
    NORMALIZATION_ENGINE_VS_OWN_ORACLE,
    SCHEMA_VERSION,
    SUSTAINABILITY_COMPLETION_FRACTION,
    SUSTAINABILITY_STEADY_INTERVAL,
)
from asrbench.percentile import PERCENTILE_METHODS
from asrbench.spec import (
    ClipSpec,
    PassPolicy,
    PopulationEntry,
    ReproductionOverride,
    ResolvedConfig,
    RunSpec,
    TargetSpec,
    WorkloadSpec,
    default_config,
)

_MODES = {MODE_STREAMING, MODE_OFFLINE}
_LOOKAHEAD_POLICIES = {
    LOOKAHEAD_IDENTITY_CLIENT_DECLARED,
    LOOKAHEAD_IDENTITY_SERVER_REPORTED,
}
_NORMALIZATIONS = {
    NORMALIZATION_ENGINE_VS_OWN_ORACLE,
    NORMALIZATION_CORPUS_REFERENCE,
}
_SUSTAINABILITY = {
    SUSTAINABILITY_STEADY_INTERVAL,
    SUSTAINABILITY_COMPLETION_FRACTION,
}
_LATENCY_METRICS = {LATENCY_METRIC_TTFS, LATENCY_METRIC_RESPONSE}
_OWNED_BY_POPULATION = {
    ("workload", "lanes"),
    ("workload", "cadence_ms"),
    ("workload", "lookahead_frames"),
    ("workload", "lookahead_ms_label"),
}
_SECTIONS = {
    "workload": WorkloadSpec,
    "target": TargetSpec,
    "run": RunSpec,
}
_FLAG_TYPE = {"int": int, "float": float, "str": str}
_ENV_BOOL = {
    "1": True,
    "true": True,
    "yes": True,
    "0": False,
    "false": False,
    "no": False,
}


class ConfigError(ValueError):
    """The configuration is incomplete, contradictory, or not reproducible."""


@dataclass(frozen=True)
class Binding:
    """One environment variable and the CLI flag that overrides it."""

    env: str
    flag: str
    path: tuple[str, ...]
    help: str


def _annotation(path: tuple[str, ...]) -> str:
    if path[:2] == ("workload", "clip"):
        model: type[Any] = ClipSpec
        name = path[2]
    elif path[:2] == ("run", "pass_policy"):
        model = PassPolicy
        name = path[2]
    else:
        model = _SECTIONS[path[0]]
        name = path[1]
    found = next(item for item in fields(model) if item.name == name)
    return str(found.type)


def _bindings() -> tuple[Binding, ...]:
    specs: tuple[tuple[tuple[str, ...], str], ...] = (
        (
            ("workload", "chunk_ms"),
            "audio per transmitted piece, in milliseconds",
        ),
        (
            ("workload", "cadence_ms"),
            "requested service cadence, in milliseconds",
        ),
        (
            ("workload", "lookahead_frames"),
            "client-declared lookahead, in frames",
        ),
        (
            ("workload", "lookahead_ms_label"),
            "millisecond label carried beside the frame count; not identity",
        ),
        (("workload", "lanes"), "closed-loop parallel lanes"),
        (("workload", "iterations"), "measured iterations per lane"),
        (
            ("workload", "warmup"),
            "warmup iterations per lane, excluded from denominators",
        ),
        (("workload", "trials"), "independent trials; raw trials are kept"),
        (("workload", "timeout_s"), "per-session timeout, in seconds"),
        (
            ("workload", "seed"),
            "seed recorded for arrival draws; closed-loop lanes do not draw",
        ),
        (("workload", "pacing"), "transmit in real time"),
        (("workload", "mode"), "streaming or offline"),
        (("workload", "language"), "language requested from the server"),
        (("workload", "punctuation"), "ask the server to punctuate"),
        (("workload", "diarization"), "ask the server to diarize"),
        (
            ("workload", "interim_results"),
            "ask the server for interim hypotheses",
        ),
        (
            ("workload", "transcript_policy"),
            "transcript output requested from the server",
        ),
        (
            ("workload", "rate_rps"),
            "offered request rate; rejected until offered-load mode exists",
        ),
        (
            ("workload", "clip", "path"),
            "local path of a file clip; not part of the workload fingerprint",
        ),
        (("target", "endpoint_url"), "server endpoint"),
        (("target", "transport"), "adapter name, such as mock"),
        (("target", "model"), "model name offered at session configuration"),
        (
            ("target", "auth_env"),
            "environment variable NAME holding a secret; never the secret",
        ),
        (("target", "tls_verify"), "verify TLS certificates"),
        (
            ("target", "server_version"),
            "server image or version; hard identity when set",
        ),
        (
            ("target", "server_gpu_name"),
            "server GPU name; hard identity when any server GPU field is set",
        ),
        (("target", "server_gpu_count"), "server GPU count"),
        (("target", "server_gpu_memory_mib"), "server GPU memory, MiB"),
        (
            ("target", "server_driver_version"),
            "server GPU driver; warn-only, not hard identity",
        ),
        (("run", "verbosity"), "observation verbosity; not offered work"),
        (("run", "profiler"), "profiler plugin name; not offered work"),
        (("run", "percentile_method"), "named percentile policy"),
        (
            ("run", "lookahead_identity_policy"),
            "client_declared or server_reported_when_present",
        ),
        (
            ("run", "pass_policy", "normalization_policy"),
            "named WER normalization policy",
        ),
        (
            ("run", "pass_policy", "sustainability_policy"),
            "steady_interval or completion_fraction",
        ),
        (("run", "output_path"), "receipt path; excluded from fingerprints"),
    )
    built: list[Binding] = []
    for path, help_text in specs:
        leaf = path[-1]
        env_name = "ASRBENCH_" + leaf.upper()
        flag = "--" + leaf.replace("_", "-")
        if path == ("run", "output_path"):
            env_name = "ASRBENCH_OUTPUT"
            flag = "--out"
        elif path == ("workload", "clip", "path"):
            env_name = "ASRBENCH_CLIP_PATH"
            flag = "--clip-path"
        built.append(Binding(env_name, flag, path, help_text))
    return tuple(built)


BINDINGS: tuple[Binding, ...] = _bindings()


def load_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from ``path``.

    Args:
        path: Filesystem path.

    Returns:
        The object.

    Raises:
        ConfigError: The file is missing, not JSON, or not an object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: expected a JSON object")
    return payload


def config_to_dict(config: ResolvedConfig) -> dict[str, Any]:
    """Serialize a resolved config to JSON-compatible objects.

    Reproduction overrides are omitted. The receipt records them beside
    the config so a second reproduction is not a different config merely
    because the first one wrote the list down.

    Args:
        config: Frozen config.

    Returns:
        A dict that :func:`json.dumps` can emit and
        :func:`config_from_mapping` can read back.
    """
    workload = asdict(config.workload)
    run = asdict(config.run)
    target = {
        item.name: getattr(config.target, item.name)
        for item in fields(config.target)
    }
    target["transport_options"] = dict(config.target.transport_options)
    target["channel_options"] = dict(config.target.channel_options)
    encoded = {"workload": workload, "target": target, "run": run}
    decoded: dict[str, Any] = json.loads(json.dumps(encoded, allow_nan=False))
    return decoded


def secret_records(
    config: ResolvedConfig, env: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Record auth variable names and whether each was non-empty.

    The value is not read into the result. ``bool(env.get(name))`` is the
    whole use of the variable.

    Args:
        config: Resolved config.
        env: Environment the run resolved against.

    Returns:
        A list of ``{"name", "set"}`` records. Empty when no auth variable
        was configured.
    """
    name = config.target.auth_env
    if name is None:
        return []
    return [{"name": name, "set": bool(env.get(name))}]


def resolve(
    *,
    preset_path: Path | None = None,
    from_receipt: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[tuple[str, ...], Any] | None = None,
) -> ResolvedConfig:
    """Resolve one frozen config.

    Args:
        preset_path: Preset JSON. Mutually exclusive with ``from_receipt``.
        from_receipt: A receipt whose ``resolved_config`` is the base.
            ``ASRBENCH_*`` variables other than the named auth variable
            are refused. Flags still apply, and each change is recorded.
        env: Environment mapping. This function does not read the process
            environment itself.
        overrides: CLI values keyed by binding path. The caller omits
            flags the user did not pass.

    Returns:
        A validated config with the clip digest pinned.

    Raises:
        ConfigError: The inputs contradict each other or a digest pin.
    """
    if preset_path is not None and from_receipt is not None:
        raise ConfigError("pass a preset or --from-receipt, not both")
    supplied = env or {}
    touched: set[tuple[str, ...]] = set()
    base: ResolvedConfig | None = None
    if from_receipt is not None:
        payload = load_json_object(from_receipt)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ConfigError(
                "schema_version "
                f"{payload.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, dict):
            raise ConfigError("receipt has no resolved_config object")
        loaded = config_from_mapping(resolved, complete=True)
        config = replace(loaded, run=replace(loaded.run, output_path=None))
        _reject_receipt_env(supplied, config.target.auth_env)
        base = config
    else:
        config = default_config()
        if preset_path is not None:
            config = apply_preset(config, load_json_object(preset_path))
        if len(config.workload.population) == 1:
            _check_population_agreement(config.workload)
        config, touched = _apply_env(config, supplied)
    seen = set(touched)
    for path, value in (overrides or {}).items():
        config = _set_path(config, path, value)
        seen.add(path)
    config = _derive_population(config, seen)
    config = validate(config, pin=True)
    if base is not None:
        # The receipt already stores the digest. Re-reading the original
        # path would reject a machine where that path is gone and a
        # ``--clip-path`` copy is the audio being pinned below.
        pinned_base = validate(_derive_population(base, set()), pin=False)
        config = replace(
            config, reproduction_overrides=_diff(pinned_base, config)
        )
    return config


def _merge_sections(
    config: ResolvedConfig,
    payload: Mapping[str, Any],
    *,
    complete: bool,
    where: str,
) -> ResolvedConfig:
    _unknown(payload, set(_SECTIONS), where)
    if complete:
        missing = [name for name in _SECTIONS if name not in payload]
        if missing:
            raise ConfigError(f"{where} missing fields: " + ", ".join(missing))
    updates: dict[str, Any] = {}
    for name in _SECTIONS:
        if name not in payload:
            continue
        updates[name] = _merge(
            getattr(config, name),
            _object(payload[name], name),
            name,
            complete=complete,
        )
    return replace(
        config,
        workload=updates.get("workload", config.workload),
        target=updates.get("target", config.target),
        run=updates.get("run", config.run),
    )


def apply_preset(
    config: ResolvedConfig, payload: Mapping[str, Any]
) -> ResolvedConfig:
    """Merge a preset object onto ``config``.

    Args:
        config: Base config, usually the dataclass defaults.
        payload: Preset JSON object.

    Returns:
        The merged config, not yet digest-pinned.

    Raises:
        ConfigError: The preset has an unknown field or a bad type.
    """
    return _merge_sections(config, payload, complete=False, where="preset")


def validate(config: ResolvedConfig, *, pin: bool = True) -> ResolvedConfig:
    """Validate ``config`` and, when ``pin`` is set, pin its clip digest.

    Compare and the reporters pass ``pin=False`` so a receipt from
    another machine is not re-hashed against whatever file now sits at
    ``clip.path``. A run pins, and a declared digest that does not match
    the payload is an error.

    Args:
        config: A config merged from defaults and a preset or receipt.
        pin: When true, read the clip payload and fill or check the digest.

    Returns:
        The validated config.

    Raises:
        ConfigError: The config contradicts itself or its digest pin.
    """
    workload = config.workload
    if workload.mode not in _MODES:
        raise ConfigError("mode must be streaming or offline")
    _require_positive("chunk_ms", workload.chunk_ms, minimum=1)
    _require_positive("lanes", workload.lanes, minimum=1)
    _require_positive("iterations", workload.iterations, minimum=1)
    _require_positive("warmup", workload.warmup, minimum=0)
    _require_positive("trials", workload.trials, minimum=1)
    _require_finite("timeout_s", workload.timeout_s, positive=True)
    if workload.rate_rps is not None:
        _require_finite("rate_rps", workload.rate_rps, positive=True)
    if workload.cadence_ms is not None and workload.cadence_ms < 1:
        raise ConfigError("cadence_ms must be positive when set")
    if workload.lookahead_frames is not None and workload.lookahead_frames < 0:
        raise ConfigError("lookahead_frames must be zero or positive")
    if workload.lookahead_ms_label is not None:
        _require_finite("lookahead_ms_label", workload.lookahead_ms_label)
    if not workload.language or not workload.transcript_policy:
        raise ConfigError("language and transcript_policy must be non-empty")
    if not workload.clip.id or not workload.clip.generator:
        raise ConfigError("clip id and generator must be non-empty")
    if workload.clip.duration_s is not None:
        _require_finite("clip.duration_s", workload.clip.duration_s)
    population = workload.population
    if not population:
        raise ConfigError("population is empty; resolve derives it")
    if sum(entry.lanes for entry in population) != workload.lanes:
        raise ConfigError("population lane counts must sum to workload.lanes")
    for entry in population:
        if entry.lanes < 1:
            raise ConfigError("each population entry needs at least one lane")
        if entry.lookahead_ms_label is not None:
            _require_finite(
                "population lookahead_ms_label", entry.lookahead_ms_label
            )
    _check_population_agreement(workload)
    run = config.run
    if run.percentile_method not in PERCENTILE_METHODS:
        raise ConfigError(
            f"unknown percentile method {run.percentile_method!r}"
        )
    if run.lookahead_identity_policy not in _LOOKAHEAD_POLICIES:
        raise ConfigError(
            "unknown lookahead identity policy "
            f"{run.lookahead_identity_policy!r}"
        )
    if run.verbosity < 0:
        raise ConfigError("verbosity must be zero or positive")
    _validate_policy(run.pass_policy)
    if not config.target.endpoint_url or not config.target.transport:
        raise ConfigError("endpoint_url and transport must be non-empty")
    if not config.target.model:
        raise ConfigError("model must be non-empty")
    _validate_server(config.target)
    if pin:
        try:
            clip = pin_digest(workload.clip)
        except ClipError as error:
            raise ConfigError(str(error)) from error
    elif not workload.clip.digest:
        raise ConfigError("clip digest is required when audio is not re-read")
    else:
        clip = workload.clip
    return replace(config, workload=replace(workload, clip=clip))


def config_from_mapping(
    payload: Mapping[str, Any], *, complete: bool = False
) -> ResolvedConfig:
    """Build a config from a ``resolved_config`` object.

    Args:
        payload: The ``resolved_config`` object.
        complete: When true, every field must be present. Receipt loads
            use this so a missing key cannot pick up a newer default.
            Presets leave it false.

    Returns:
        A config. The caller validates and decides whether to pin.

    Raises:
        ConfigError: The object is not shaped like a resolved config.
    """
    return _merge_sections(
        default_config(),
        payload,
        complete=complete,
        where="resolved_config",
    )


def add_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Register one CLI flag per :data:`BINDINGS` entry.

    Args:
        parser: The ``run`` subparser.
    """
    for binding in BINDINGS:
        dest = _dest(binding.flag)
        base = _base_type(_annotation(binding.path))
        if base == "bool":
            parser.add_argument(
                binding.flag,
                dest=dest,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=binding.help,
            )
        else:
            parser.add_argument(
                binding.flag,
                dest=dest,
                default=None,
                type=_FLAG_TYPE[base],
                help=binding.help,
            )


def overrides_from_args(args: argparse.Namespace) -> dict[tuple[str, ...], Any]:
    """Collect flags the user actually passed.

    Args:
        args: Parsed ``run`` arguments.

    Returns:
        Binding path to value, omitting flags left at ``None``.
    """
    found: dict[tuple[str, ...], Any] = {}
    for binding in BINDINGS:
        value = getattr(args, _dest(binding.flag))
        if value is None:
            continue
        if _annotation(binding.path) == "str | None" and value == "":
            value = None
        found[binding.path] = value
    return found


def _reject_receipt_env(env: Mapping[str, str], auth_name: str | None) -> None:
    for key in env:
        if not key.startswith("ASRBENCH_"):
            continue
        if auth_name is not None and key == auth_name:
            continue
        raise ConfigError(
            f"{key} is set; --from-receipt does not apply ASRBENCH_* "
            "variables. Unset it or pass the setting as a flag."
        )


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _base_type(annotation: str) -> str:
    return annotation.removesuffix(" | None")


def _apply_env(
    config: ResolvedConfig, env: Mapping[str, str]
) -> tuple[ResolvedConfig, set[tuple[str, ...]]]:
    known = {binding.env: binding for binding in BINDINGS}
    auth_binding = known["ASRBENCH_AUTH_ENV"]
    touched: set[tuple[str, ...]] = set()
    if auth_binding.env in env:
        config = _set_path(
            config,
            auth_binding.path,
            _parse_env(
                _annotation(auth_binding.path),
                env[auth_binding.env],
                auth_binding.env,
            ),
        )
        touched.add(auth_binding.path)
    auth_name = config.target.auth_env
    for key, raw in env.items():
        if not key.startswith("ASRBENCH_") or key == auth_binding.env:
            continue
        if auth_name is not None and key == auth_name:
            continue
        binding = known.get(key)
        if binding is None:
            raise ConfigError(
                f"unknown environment variable {key}; "
                "auth secrets are read only when target.auth_env names them"
            )
        parsed = _parse_env(_annotation(binding.path), raw, key)
        config = _set_path(config, binding.path, parsed)
        touched.add(binding.path)
    return config, touched


def _parse_env(annotation: str, raw: str, where: str) -> Any:
    if raw == "":
        raise ConfigError(f"{where} is empty")
    base = _base_type(annotation)
    if base == "bool":
        try:
            return _ENV_BOOL[raw.casefold()]
        except KeyError as error:
            raise ConfigError(f"{where} must be true or false") from error
    if base == "int":
        try:
            parsed: Any = int(raw)
        except ValueError as error:
            raise ConfigError(f"{where} must be an int") from error
    elif base == "float":
        try:
            parsed = float(raw)
        except ValueError as error:
            raise ConfigError(f"{where} must be a number") from error
    else:
        parsed = raw
    return _coerce(parsed, annotation, where)


def _nest(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    cursor = payload
    for key in path[:-1]:
        child: dict[str, Any] = {}
        cursor[key] = child
        cursor = child
    cursor[path[-1]] = value
    return payload


def _set_path(
    config: ResolvedConfig, path: tuple[str, ...], value: Any
) -> ResolvedConfig:
    return apply_preset(config, _nest(path, value))


def _merge(
    instance: Any,
    payload: Mapping[str, Any],
    where: str,
    *,
    complete: bool,
) -> Any:
    known = {item.name: item for item in fields(instance)}
    _unknown(payload, set(known), where)
    if complete:
        missing = sorted(set(known) - set(payload))
        if missing:
            raise ConfigError(f"{where} missing fields: {', '.join(missing)}")
    updates: dict[str, Any] = {}
    for name, value in payload.items():
        if name in {"clip", "pass_policy"}:
            updates[name] = _merge(
                getattr(instance, name),
                _object(value, name),
                name,
                complete=complete,
            )
        elif name == "population":
            updates[name] = _population(value)
        elif name in {"transport_options", "channel_options"}:
            updates[name] = _options(value, name)
        else:
            updates[name] = _coerce(
                value, str(known[name].type), f"{where}.{name}"
            )
    if not updates:
        return instance
    return replace(instance, **updates)


def _population(value: object) -> tuple[PopulationEntry, ...]:
    if not isinstance(value, list):
        raise ConfigError("population must be a list")
    entries: list[PopulationEntry] = []
    for index, item in enumerate(value):
        payload = _object(item, f"population[{index}]")
        _unknown(
            payload,
            {"lanes", "cadence_ms", "lookahead_frames", "lookahead_ms_label"},
            f"population[{index}]",
        )
        if "lanes" not in payload:
            raise ConfigError(f"population[{index}] requires lanes")
        where = f"population[{index}]"
        entries.append(
            PopulationEntry(
                lanes=_coerce(payload["lanes"], "int", f"{where}.lanes"),
                cadence_ms=_coerce(
                    payload.get("cadence_ms"),
                    "int | None",
                    f"{where}.cadence_ms",
                ),
                lookahead_frames=_coerce(
                    payload.get("lookahead_frames"),
                    "int | None",
                    f"{where}.lookahead_frames",
                ),
                lookahead_ms_label=_coerce(
                    payload.get("lookahead_ms_label"),
                    "float | None",
                    f"{where}.lookahead_ms_label",
                ),
            )
        )
    return tuple(entries)


def _options(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise ConfigError(f"{where} keys must be strings")
    return dict(value)


def _derive_population(
    config: ResolvedConfig, touched: set[tuple[str, ...]]
) -> ResolvedConfig:
    workload = config.workload
    if len(workload.population) > 1:
        if touched & _OWNED_BY_POPULATION:
            raise ConfigError(
                "a multi-entry population is explicit; edit its entries "
                "instead of --lanes, --cadence-ms, --lookahead-frames, "
                "or --lookahead-ms-label"
            )
        return config
    entry = PopulationEntry(
        lanes=workload.lanes,
        cadence_ms=workload.cadence_ms,
        lookahead_frames=workload.lookahead_frames,
        lookahead_ms_label=workload.lookahead_ms_label,
    )
    return replace(config, workload=replace(workload, population=(entry,)))


def _check_population_agreement(workload: WorkloadSpec) -> None:
    population = workload.population
    if not population:
        return
    _agree(
        workload.lookahead_frames,
        {entry.lookahead_frames for entry in population},
        "lookahead_frames",
    )
    _agree(
        workload.cadence_ms,
        {entry.cadence_ms for entry in population},
        "cadence_ms",
    )
    _agree(
        workload.lookahead_ms_label,
        {entry.lookahead_ms_label for entry in population},
        "lookahead_ms_label",
    )


def _validate_policy(policy: PassPolicy) -> None:
    if policy.normalization_policy not in _NORMALIZATIONS:
        raise ConfigError(
            f"unknown normalization policy {policy.normalization_policy!r}"
        )
    if policy.sustainability_policy not in _SUSTAINABILITY:
        raise ConfigError(
            f"unknown sustainability policy {policy.sustainability_policy!r}"
        )
    if policy.latency_metric not in _LATENCY_METRICS:
        raise ConfigError(f"unknown latency metric {policy.latency_metric!r}")
    for name in (
        "admission_fraction_min",
        "first_partial_presence_min",
        "sustainability_min",
    ):
        value = getattr(policy, name)
        _require_finite(name, value)
        if value < 0 or value > 1:
            raise ConfigError(f"{name} must be in [0, 1]")
    _require_finite("latency_percentile", policy.latency_percentile)
    if policy.latency_percentile < 0 or policy.latency_percentile > 100:
        raise ConfigError("latency_percentile must be in [0, 100]")
    _require_finite("latency_slo_s", policy.latency_slo_s)
    _require_finite("accuracy_tolerance", policy.accuracy_tolerance)
    if policy.latency_slo_s < 0 or policy.accuracy_tolerance < 0:
        raise ConfigError("latency SLO and accuracy tolerance must be >= 0")


def _validate_server(target: TargetSpec) -> None:
    if target.server_gpu_count is not None and target.server_gpu_count < 0:
        raise ConfigError("server_gpu_count must be zero or positive")
    if (
        target.server_gpu_memory_mib is not None
        and target.server_gpu_memory_mib < 0
    ):
        raise ConfigError("server_gpu_memory_mib must be zero or positive")


def _diff(
    before: ResolvedConfig, after: ResolvedConfig
) -> tuple[ReproductionOverride, ...]:
    left = leaves(config_to_dict(before))
    right = leaves(config_to_dict(after))
    found: list[ReproductionOverride] = []
    for key in sorted(set(left) | set(right)):
        if key == "run.output_path":
            continue
        if left.get(key) != right.get(key):
            found.append(
                ReproductionOverride(
                    key,
                    json.dumps(left.get(key), sort_keys=True),
                    json.dumps(right.get(key), sort_keys=True),
                )
            )
    return tuple(found)


def _agree(top: Any, seen: set[Any], name: str) -> None:
    if len(seen) == 1:
        only = next(iter(seen))
        if top != only:
            raise ConfigError(f"{name} disagrees with the population")
        return
    if top is not None:
        raise ConfigError(f"mixed {name} requires a null top-level {name}")


def _unknown(payload: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ConfigError(f"unknown {where} fields: {', '.join(extra)}")


def _object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be an object")
    return value


def _check_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an int")
    return value


def _check_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{where} must be finite")
    return number


def _check_bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a bool")
    return value


def _check_str(value: object, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{where} must be a non-empty string")
    return value


_CHECKS = {
    "int": _check_int,
    "float": _check_float,
    "bool": _check_bool,
    "str": _check_str,
}


def _coerce(value: object, annotation: str, where: str) -> Any:
    optional = annotation.endswith(" | None")
    base = _base_type(annotation)
    if optional and value is None:
        return None
    if base == "str" and optional:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string or null")
        return value
    try:
        checker = _CHECKS[base]
    except KeyError as error:
        raise ConfigError(
            f"{where} has unsupported type {annotation}"
        ) from error
    return checker(value, where)


def _require_positive(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")


def _require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{name} must be finite")
    if positive and value <= 0:
        raise ConfigError(f"{name} must be positive")
