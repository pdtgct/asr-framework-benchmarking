"""Resolve one frozen configuration.

Order is dataclass defaults, then a preset file, then ``ASRBENCH_*``
environment variables, then CLI flags. ``--from-receipt`` replaces the
defaults and the preset with the receipt's resolved config, and the
environment and flags still apply. The output path stored in a receipt is
dropped before those overrides, so a reproduction does not overwrite the
receipt it was read from.

Secrets are not flags and not preset fields. The target carries a
variable name. :func:`secret_records` records that name and whether the
variable was non-empty. The value is never copied.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from asrbench.clips import ClipError, pin_digest
from asrbench.definitions import (
    LATENCY_METRIC_RESPONSE,
    LATENCY_METRIC_TTFS,
    LOOKAHEAD_IDENTITY_CLIENT_DECLARED,
    LOOKAHEAD_IDENTITY_SERVER_REPORTED,
    MODE_OFFLINE,
    MODE_STREAMING,
    NORMALIZATION_CORPUS_REFERENCE,
    NORMALIZATION_ENGINE_VS_OTHER_SERVER,
    NORMALIZATION_ENGINE_VS_OWN_ORACLE,
    SUSTAINABILITY_COMPLETION_FRACTION,
    SUSTAINABILITY_STEADY_INTERVAL,
)
from asrbench.percentile import PERCENTILE_METHODS
from asrbench.spec import (
    ClipSpec,
    PassPolicy,
    PopulationEntry,
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
    NORMALIZATION_ENGINE_VS_OTHER_SERVER,
}
_SUSTAINABILITY = {
    SUSTAINABILITY_STEADY_INTERVAL,
    SUSTAINABILITY_COMPLETION_FRACTION,
}
_LATENCY_METRICS = {LATENCY_METRIC_TTFS, LATENCY_METRIC_RESPONSE}


class ConfigError(ValueError):
    """The configuration is incomplete, contradictory, or not reproducible."""


@dataclass(frozen=True)
class Binding:
    """One environment variable and the CLI flag that overrides it."""

    env: str
    flag: str
    path: tuple[str, ...]
    kind: str
    help: str


BINDINGS: tuple[Binding, ...] = (
    Binding(
        "ASRBENCH_CHUNK_MS",
        "--chunk-ms",
        ("workload", "chunk_ms"),
        "int",
        "audio per transmitted piece, in milliseconds",
    ),
    Binding(
        "ASRBENCH_CADENCE_MS",
        "--cadence-ms",
        ("workload", "cadence_ms"),
        "optional_int",
        "requested service cadence, in milliseconds",
    ),
    Binding(
        "ASRBENCH_LOOKAHEAD_FRAMES",
        "--lookahead-frames",
        ("workload", "lookahead_frames"),
        "optional_int",
        "client-declared lookahead, in frames",
    ),
    Binding(
        "ASRBENCH_LOOKAHEAD_MS_LABEL",
        "--lookahead-ms-label",
        ("workload", "lookahead_ms_label"),
        "optional_float",
        "millisecond label carried beside the frame count; not identity",
    ),
    Binding(
        "ASRBENCH_LANES",
        "--lanes",
        ("workload", "lanes"),
        "int",
        "closed-loop parallel lanes",
    ),
    Binding(
        "ASRBENCH_ITERATIONS",
        "--iterations",
        ("workload", "iterations"),
        "int",
        "measured iterations per lane",
    ),
    Binding(
        "ASRBENCH_WARMUP",
        "--warmup",
        ("workload", "warmup"),
        "int",
        "warmup iterations per lane, excluded from denominators",
    ),
    Binding(
        "ASRBENCH_TRIALS",
        "--trials",
        ("workload", "trials"),
        "int",
        "independent trials; raw trials are kept",
    ),
    Binding(
        "ASRBENCH_TIMEOUT_S",
        "--timeout-s",
        ("workload", "timeout_s"),
        "float",
        "per-session timeout, in seconds",
    ),
    Binding(
        "ASRBENCH_SEED",
        "--seed",
        ("workload", "seed"),
        "int",
        "seed recorded for arrival draws; closed-loop lanes do not draw",
    ),
    Binding(
        "ASRBENCH_PACING",
        "--pacing",
        ("workload", "pacing"),
        "bool",
        "transmit in real time",
    ),
    Binding(
        "ASRBENCH_MODE",
        "--mode",
        ("workload", "mode"),
        "str",
        "streaming or offline",
    ),
    Binding(
        "ASRBENCH_LANGUAGE",
        "--language",
        ("workload", "language"),
        "str",
        "language requested from the server",
    ),
    Binding(
        "ASRBENCH_PUNCTUATION",
        "--punctuation",
        ("workload", "punctuation"),
        "bool",
        "ask the server to punctuate",
    ),
    Binding(
        "ASRBENCH_DIARIZATION",
        "--diarization",
        ("workload", "diarization"),
        "bool",
        "ask the server to diarize",
    ),
    Binding(
        "ASRBENCH_INTERIM_RESULTS",
        "--interim-results",
        ("workload", "interim_results"),
        "bool",
        "ask the server for interim hypotheses",
    ),
    Binding(
        "ASRBENCH_TRANSCRIPT_POLICY",
        "--transcript-policy",
        ("workload", "transcript_policy"),
        "str",
        "transcript output requested from the server",
    ),
    Binding(
        "ASRBENCH_RATE_RPS",
        "--rate-rps",
        ("workload", "rate_rps"),
        "optional_float",
        "offered request rate; rejected until offered-load mode exists",
    ),
    Binding(
        "ASRBENCH_ENDPOINT_URL",
        "--endpoint-url",
        ("target", "endpoint_url"),
        "str",
        "server endpoint",
    ),
    Binding(
        "ASRBENCH_TRANSPORT",
        "--transport",
        ("target", "transport"),
        "str",
        "adapter name, such as mock",
    ),
    Binding(
        "ASRBENCH_MODEL",
        "--model",
        ("target", "model"),
        "str",
        "model name offered at session configuration",
    ),
    Binding(
        "ASRBENCH_AUTH_ENV",
        "--auth-env",
        ("target", "auth_env"),
        "optional_str",
        "environment variable NAME holding a secret; never the secret",
    ),
    Binding(
        "ASRBENCH_TLS_VERIFY",
        "--tls-verify",
        ("target", "tls_verify"),
        "bool",
        "verify TLS certificates",
    ),
    Binding(
        "ASRBENCH_VERBOSITY",
        "--verbosity",
        ("run", "verbosity"),
        "int",
        "observation verbosity; not offered work",
    ),
    Binding(
        "ASRBENCH_PROFILER",
        "--profiler",
        ("run", "profiler"),
        "optional_str",
        "profiler plugin name; not offered work",
    ),
    Binding(
        "ASRBENCH_PERCENTILE_METHOD",
        "--percentile-method",
        ("run", "percentile_method"),
        "str",
        "named percentile policy",
    ),
    Binding(
        "ASRBENCH_LOOKAHEAD_IDENTITY_POLICY",
        "--lookahead-identity-policy",
        ("run", "lookahead_identity_policy"),
        "str",
        "client_declared or server_reported_when_present",
    ),
    Binding(
        "ASRBENCH_NORMALIZATION_POLICY",
        "--normalization-policy",
        ("run", "pass_policy", "normalization_policy"),
        "str",
        "named WER normalization policy",
    ),
    Binding(
        "ASRBENCH_SUSTAINABILITY_POLICY",
        "--sustainability-policy",
        ("run", "pass_policy", "sustainability_policy"),
        "str",
        "steady_interval or completion_fraction",
    ),
    Binding(
        "ASRBENCH_OUTPUT",
        "--out",
        ("run", "output_path"),
        "optional_str",
        "receipt path; excluded from fingerprints",
    ),
)


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

    Args:
        config: Frozen config.

    Returns:
        A dict that :func:`json.dumps` can emit and
        :func:`config_from_mapping` can read back.
    """
    encoded = json.dumps(asdict(config), allow_nan=False)
    decoded: dict[str, Any] = json.loads(encoded)
    return decoded


def without_output_path(config: ResolvedConfig) -> ResolvedConfig:
    """Drop the output path so two reproductions can be compared.

    Args:
        config: Resolved config.

    Returns:
        The same config with ``run.output_path`` cleared.
    """
    return replace(config, run=replace(config.run, output_path=None))


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
        env: Environment mapping. ``None`` uses ``os.environ`` only when
            the caller passes it; this function does not read the process
            environment itself, so tests stay sealed.
        overrides: CLI values keyed by binding path. The caller omits
            flags the user did not pass.

    Returns:
        A validated config with the clip digest pinned.

    Raises:
        ConfigError: The inputs contradict each other or a digest pin.
    """
    if preset_path is not None and from_receipt is not None:
        raise ConfigError("pass a preset or --from-receipt, not both")
    if from_receipt is not None:
        payload = load_json_object(from_receipt)
        resolved = payload.get("resolved_config")
        if not isinstance(resolved, dict):
            raise ConfigError("receipt has no resolved_config object")
        config = config_from_mapping(resolved)
        config = without_output_path(config)
    else:
        config = default_config()
        if preset_path is not None:
            config = apply_preset(config, load_json_object(preset_path))
    config = _apply_env(config, env or {})
    config = _apply_overrides(config, overrides or {})
    return _validate(config)


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
    _unknown(payload, {"workload", "target", "run"}, "preset")
    if "workload" in payload:
        config = replace(
            config,
            workload=_merge_workload(
                config.workload, _object(payload["workload"], "workload")
            ),
        )
    if "target" in payload:
        config = replace(
            config,
            target=_merge_target(
                config.target, _object(payload["target"], "target")
            ),
        )
    if "run" in payload:
        config = replace(
            config, run=_merge_run(config.run, _object(payload["run"], "run"))
        )
    return config


def finalize(config: ResolvedConfig) -> ResolvedConfig:
    """Validate ``config`` and pin its clip digest.

    Args:
        config: A config merged from defaults and a preset or receipt.

    Returns:
        The validated config.

    Raises:
        ConfigError: The config contradicts itself or its digest pin.
    """
    return _validate(config)


def config_from_mapping(payload: Mapping[str, Any]) -> ResolvedConfig:
    """Build a config from a complete ``resolved_config`` object.

    Missing keys keep dataclass defaults. Unknown keys are an error, so a
    typo cannot hide inside a receipt that is about to be re-run.

    Args:
        payload: The ``resolved_config`` object.

    Returns:
        A config. The caller validates and pins the digest.

    Raises:
        ConfigError: The object is not shaped like a resolved config.
    """
    return apply_preset(default_config(), payload)


def add_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Register one CLI flag per :data:`BINDINGS` entry.

    Args:
        parser: The ``run`` subparser.
    """
    for binding in BINDINGS:
        dest = _dest(binding)
        if binding.kind == "bool":
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
                type=_FLAG_TYPES[binding.kind],
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
        value = getattr(args, _dest(binding))
        if value is None:
            continue
        if binding.kind == "optional_str" and value == "":
            value = None
        found[binding.path] = value
    return found


def _dest(binding: Binding) -> str:
    return binding.flag.lstrip("-").replace("-", "_")


_FLAG_TYPES: dict[str, type[Any]] = {
    "int": int,
    "optional_int": int,
    "float": float,
    "optional_float": float,
    "str": str,
    "optional_str": str,
}


def _apply_env(
    config: ResolvedConfig, env: Mapping[str, str]
) -> ResolvedConfig:
    known = {binding.env: binding for binding in BINDINGS}
    auth_name = config.target.auth_env
    for key, raw in env.items():
        if not key.startswith("ASRBENCH_"):
            continue
        if auth_name is not None and key == auth_name:
            continue
        binding = known.get(key)
        if binding is None:
            raise ConfigError(
                f"unknown environment variable {key}; "
                "auth secrets are read only when target.auth_env names them"
            )
        config = _set_path(
            config, binding.path, _parse_env(binding.kind, raw, key)
        )
    return config


def _apply_overrides(
    config: ResolvedConfig, overrides: Mapping[tuple[str, ...], Any]
) -> ResolvedConfig:
    for path, value in overrides.items():
        config = _set_path(config, path, value)
    return config


def _parse_env(kind: str, raw: str, where: str) -> Any:
    if raw == "":
        raise ConfigError(f"{where} is empty")
    if kind == "bool":
        folded = raw.casefold()
        if folded in {"1", "true", "yes"}:
            return True
        if folded in {"0", "false", "no"}:
            return False
        raise ConfigError(f"{where} must be true or false")
    if kind in {"int", "optional_int"}:
        try:
            return int(raw)
        except ValueError as error:
            raise ConfigError(f"{where} must be an int") from error
    if kind in {"float", "optional_float"}:
        try:
            return float(raw)
        except ValueError as error:
            raise ConfigError(f"{where} must be a number") from error
    if kind in {"str", "optional_str"}:
        return raw
    raise ConfigError(f"unknown binding kind {kind}")


def _set_path(
    config: ResolvedConfig, path: tuple[str, ...], value: Any
) -> ResolvedConfig:
    if path[0] == "workload":
        return replace(
            config, workload=_set_workload(config.workload, path[1], value)
        )
    if path[0] == "target":
        return replace(
            config, target=_set_target(config.target, path[1], value)
        )
    if path == ("run", "pass_policy", "normalization_policy"):
        policy = replace(
            config.run.pass_policy,
            normalization_policy=_as_str(value, "normalization_policy"),
        )
        return replace(config, run=replace(config.run, pass_policy=policy))
    if path == ("run", "pass_policy", "sustainability_policy"):
        policy = replace(
            config.run.pass_policy,
            sustainability_policy=_as_str(value, "sustainability_policy"),
        )
        return replace(config, run=replace(config.run, pass_policy=policy))
    if path[0] == "run":
        return replace(config, run=_set_run(config.run, path[1], value))
    raise ConfigError(f"cannot set {'/'.join(path)}")


def _merge_workload(
    base: WorkloadSpec, payload: Mapping[str, Any]
) -> WorkloadSpec:
    _unknown(
        payload,
        {
            "clip",
            "reference_transcript",
            "mode",
            "pacing",
            "chunk_ms",
            "cadence_ms",
            "lookahead_frames",
            "lookahead_ms_label",
            "lanes",
            "rate_rps",
            "iterations",
            "warmup",
            "trials",
            "timeout_s",
            "seed",
            "language",
            "punctuation",
            "diarization",
            "interim_results",
            "transcript_policy",
            "population",
        },
        "workload",
    )
    workload = base
    if "clip" in payload:
        workload = replace(
            workload,
            clip=_merge_clip(base.clip, _object(payload["clip"], "clip")),
        )
    if "population" in payload:
        workload = replace(
            workload, population=_population(payload["population"])
        )
    for name in (
        "reference_transcript",
        "mode",
        "pacing",
        "chunk_ms",
        "cadence_ms",
        "lookahead_frames",
        "lookahead_ms_label",
        "lanes",
        "rate_rps",
        "iterations",
        "warmup",
        "trials",
        "timeout_s",
        "seed",
        "language",
        "punctuation",
        "diarization",
        "interim_results",
        "transcript_policy",
    ):
        if name in payload:
            workload = _set_workload(workload, name, payload[name])
    return workload


def _merge_clip(base: ClipSpec, payload: Mapping[str, Any]) -> ClipSpec:
    _unknown(
        payload,
        {
            "id",
            "generator",
            "duration_s",
            "sample_rate",
            "digest",
            "path",
            "licence",
            "attribution",
            "note",
        },
        "clip",
    )
    clip = base
    if "id" in payload:
        clip = replace(clip, id=_as_str(payload["id"], "clip.id"))
    if "generator" in payload:
        clip = replace(
            clip, generator=_as_str(payload["generator"], "clip.generator")
        )
    if "duration_s" in payload:
        clip = replace(
            clip,
            duration_s=_as_optional_float(
                payload["duration_s"], "clip.duration_s"
            ),
        )
    if "sample_rate" in payload:
        clip = replace(
            clip,
            sample_rate=_as_int(payload["sample_rate"], "clip.sample_rate"),
        )
    if "digest" in payload:
        clip = replace(
            clip, digest=_as_optional_str(payload["digest"], "clip.digest")
        )
    if "path" in payload:
        clip = replace(
            clip, path=_as_optional_str(payload["path"], "clip.path")
        )
    if "licence" in payload:
        clip = replace(
            clip, licence=_as_str(payload["licence"], "clip.licence")
        )
    if "attribution" in payload:
        clip = replace(
            clip,
            attribution=_as_str(payload["attribution"], "clip.attribution"),
        )
    if "note" in payload:
        clip = replace(clip, note=_as_str(payload["note"], "clip.note"))
    return clip


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
        entries.append(
            PopulationEntry(
                lanes=_as_int(payload["lanes"], f"population[{index}].lanes"),
                cadence_ms=_as_optional_int(
                    payload.get("cadence_ms"),
                    f"population[{index}].cadence_ms",
                ),
                lookahead_frames=_as_optional_int(
                    payload.get("lookahead_frames"),
                    f"population[{index}].lookahead_frames",
                ),
                lookahead_ms_label=_as_optional_float(
                    payload.get("lookahead_ms_label"),
                    f"population[{index}].lookahead_ms_label",
                ),
            )
        )
    return tuple(entries)


def _merge_target(base: TargetSpec, payload: Mapping[str, Any]) -> TargetSpec:
    _unknown(
        payload,
        {
            "endpoint_url",
            "transport",
            "model",
            "auth_env",
            "tls_verify",
            "transport_options",
        },
        "target",
    )
    target = base
    for name in (
        "endpoint_url",
        "transport",
        "model",
        "auth_env",
        "tls_verify",
    ):
        if name in payload:
            target = _set_target(target, name, payload[name])
    if "transport_options" in payload:
        options = payload["transport_options"]
        if not isinstance(options, dict):
            raise ConfigError("transport_options must be an object")
        for key in options:
            if not isinstance(key, str):
                raise ConfigError("transport_options keys must be strings")
        target = replace(target, transport_options=dict(options))
    return target


def _merge_run(base: RunSpec, payload: Mapping[str, Any]) -> RunSpec:
    _unknown(
        payload,
        {
            "output_path",
            "verbosity",
            "profiler",
            "percentile_method",
            "lookahead_identity_policy",
            "pass_policy",
        },
        "run",
    )
    run = base
    for name in (
        "output_path",
        "verbosity",
        "profiler",
        "percentile_method",
        "lookahead_identity_policy",
    ):
        if name in payload:
            run = _set_run(run, name, payload[name])
    if "pass_policy" in payload:
        run = replace(
            run,
            pass_policy=_merge_pass(
                base.pass_policy, _object(payload["pass_policy"], "pass_policy")
            ),
        )
    return run


def _merge_pass(base: PassPolicy, payload: Mapping[str, Any]) -> PassPolicy:
    _unknown(
        payload,
        {
            "admission_fraction_min",
            "first_partial_presence_min",
            "latency_metric",
            "latency_percentile",
            "latency_slo_s",
            "sustainability_policy",
            "sustainability_min",
            "accuracy_tolerance",
            "normalization_policy",
        },
        "pass_policy",
    )
    policy = base
    if "admission_fraction_min" in payload:
        policy = replace(
            policy,
            admission_fraction_min=_as_float(
                payload["admission_fraction_min"], "admission_fraction_min"
            ),
        )
    if "first_partial_presence_min" in payload:
        policy = replace(
            policy,
            first_partial_presence_min=_as_float(
                payload["first_partial_presence_min"],
                "first_partial_presence_min",
            ),
        )
    if "latency_percentile" in payload:
        policy = replace(
            policy,
            latency_percentile=_as_float(
                payload["latency_percentile"], "latency_percentile"
            ),
        )
    if "latency_slo_s" in payload:
        policy = replace(
            policy,
            latency_slo_s=_as_float(payload["latency_slo_s"], "latency_slo_s"),
        )
    if "sustainability_min" in payload:
        policy = replace(
            policy,
            sustainability_min=_as_float(
                payload["sustainability_min"], "sustainability_min"
            ),
        )
    if "accuracy_tolerance" in payload:
        policy = replace(
            policy,
            accuracy_tolerance=_as_float(
                payload["accuracy_tolerance"], "accuracy_tolerance"
            ),
        )
    if "latency_metric" in payload:
        policy = replace(
            policy,
            latency_metric=_as_str(payload["latency_metric"], "latency_metric"),
        )
    if "sustainability_policy" in payload:
        policy = replace(
            policy,
            sustainability_policy=_as_str(
                payload["sustainability_policy"], "sustainability_policy"
            ),
        )
    if "normalization_policy" in payload:
        policy = replace(
            policy,
            normalization_policy=_as_str(
                payload["normalization_policy"], "normalization_policy"
            ),
        )
    return policy


def _set_workload(
    workload: WorkloadSpec, name: str, value: Any
) -> WorkloadSpec:
    if name == "reference_transcript":
        return replace(
            workload,
            reference_transcript=_as_optional_str(value, name),
        )
    if name == "mode":
        return replace(workload, mode=_as_str(value, name))
    if name == "language":
        return replace(workload, language=_as_str(value, name))
    if name == "transcript_policy":
        return replace(workload, transcript_policy=_as_str(value, name))
    if name == "pacing":
        return replace(workload, pacing=_as_bool(value, name))
    if name == "punctuation":
        return replace(workload, punctuation=_as_bool(value, name))
    if name == "diarization":
        return replace(workload, diarization=_as_bool(value, name))
    if name == "interim_results":
        return replace(workload, interim_results=_as_bool(value, name))
    if name == "chunk_ms":
        return replace(workload, chunk_ms=_as_int(value, name))
    if name == "lanes":
        return replace(workload, lanes=_as_int(value, name))
    if name == "iterations":
        return replace(workload, iterations=_as_int(value, name))
    if name == "warmup":
        return replace(workload, warmup=_as_int(value, name))
    if name == "trials":
        return replace(workload, trials=_as_int(value, name))
    if name == "seed":
        return replace(workload, seed=_as_int(value, name))
    if name == "cadence_ms":
        return replace(workload, cadence_ms=_as_optional_int(value, name))
    if name == "lookahead_frames":
        return replace(workload, lookahead_frames=_as_optional_int(value, name))
    if name == "lookahead_ms_label":
        return replace(
            workload, lookahead_ms_label=_as_optional_float(value, name)
        )
    if name == "timeout_s":
        return replace(workload, timeout_s=_as_float(value, name))
    if name == "rate_rps":
        return replace(workload, rate_rps=_as_optional_float(value, name))
    raise ConfigError(f"unknown workload field {name}")


def _set_target(target: TargetSpec, name: str, value: Any) -> TargetSpec:
    if name == "endpoint_url":
        return replace(target, endpoint_url=_as_str(value, name))
    if name == "transport":
        return replace(target, transport=_as_str(value, name))
    if name == "model":
        return replace(target, model=_as_str(value, name))
    if name == "auth_env":
        return replace(target, auth_env=_as_optional_str(value, name))
    if name == "tls_verify":
        return replace(target, tls_verify=_as_bool(value, name))
    raise ConfigError(f"unknown target field {name}")


def _set_run(run: RunSpec, name: str, value: Any) -> RunSpec:
    if name == "output_path":
        return replace(run, output_path=_as_optional_str(value, name))
    if name == "verbosity":
        return replace(run, verbosity=_as_int(value, name))
    if name == "profiler":
        return replace(run, profiler=_as_optional_str(value, name))
    if name == "percentile_method":
        return replace(run, percentile_method=_as_str(value, name))
    if name == "lookahead_identity_policy":
        return replace(run, lookahead_identity_policy=_as_str(value, name))
    raise ConfigError(f"unknown run field {name}")


def _validate(config: ResolvedConfig) -> ResolvedConfig:
    workload = config.workload
    if workload.mode not in _MODES:
        raise ConfigError("mode must be streaming or offline")
    if workload.chunk_ms < 1:
        raise ConfigError("chunk_ms must be positive")
    if workload.lanes < 1:
        raise ConfigError("lanes must be positive")
    if workload.iterations < 1:
        raise ConfigError("iterations must be positive")
    if workload.warmup < 0:
        raise ConfigError("warmup must be zero or positive")
    if workload.trials < 1:
        raise ConfigError("trials must be positive")
    if workload.timeout_s <= 0:
        raise ConfigError("timeout_s must be positive")
    if workload.rate_rps is not None and workload.rate_rps <= 0:
        raise ConfigError("rate_rps must be positive when set")
    if workload.cadence_ms is not None and workload.cadence_ms < 1:
        raise ConfigError("cadence_ms must be positive when set")
    if workload.lookahead_frames is not None and workload.lookahead_frames < 0:
        raise ConfigError("lookahead_frames must be zero or positive")
    if not workload.language or not workload.transcript_policy:
        raise ConfigError("language and transcript_policy must be non-empty")
    if not workload.clip.id or not workload.clip.generator:
        raise ConfigError("clip id and generator must be non-empty")
    population = workload.population
    if not population:
        population = (
            PopulationEntry(
                lanes=workload.lanes,
                cadence_ms=workload.cadence_ms,
                lookahead_frames=workload.lookahead_frames,
                lookahead_ms_label=workload.lookahead_ms_label,
            ),
        )
        workload = replace(workload, population=population)
    if sum(entry.lanes for entry in population) != workload.lanes:
        raise ConfigError("population lane counts must sum to workload.lanes")
    for entry in population:
        if entry.lanes < 1:
            raise ConfigError("each population entry needs at least one lane")
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
    policy = run.pass_policy
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
        if value < 0 or value > 1:
            raise ConfigError(f"{name} must be in [0, 1]")
    if policy.latency_percentile < 0 or policy.latency_percentile > 100:
        raise ConfigError("latency_percentile must be in [0, 100]")
    if policy.latency_slo_s < 0 or policy.accuracy_tolerance < 0:
        raise ConfigError("latency SLO and accuracy tolerance must be >= 0")
    if not config.target.endpoint_url or not config.target.transport:
        raise ConfigError("endpoint_url and transport must be non-empty")
    if not config.target.model:
        raise ConfigError("model must be non-empty")
    try:
        clip = pin_digest(workload.clip)
    except ClipError as error:
        raise ConfigError(str(error)) from error
    return replace(config, workload=replace(workload, clip=clip))


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


def _as_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an int")
    return value


def _as_optional_int(value: object, where: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, where)


def _as_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    return float(value)


def _as_optional_float(value: object, where: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, where)


def _as_str(value: object, where: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _as_optional_str(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string or null")
    return value


def _as_bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be a bool")
    return value
