"""Workload and observation fingerprints, and the identity objects.

The workload fingerprint covers offered work. It includes the reference
transcript's digest: a silent reference swap would otherwise compare as
the same work and report a WER delta that is the reference, not the
server. It excludes the lookahead millisecond label, clip path, licence
text, and every run-observation field.

The comparison key hashes four hard components. The hard target is the
checkpoint. The engine — name, version, transport, and server version —
is the comparison axis and is recorded, not hashed. ``target.model`` and
``transport_options`` are warn-only. ``channel_options`` are warn-only:
keepalive and maximum message size do not refuse. Git commit is warn-only
and is read from this package's repository, not the caller's directory.
Precision is recorded and is not compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asrbench.canonical import fingerprint, join_components, sha256_text
from asrbench.definitions import (
    COMPARISON_AXIS,
    GPU_IDENTITY_CLIENT_PROBE,
    GPU_IDENTITY_SERVER_DECLARED,
)
from asrbench.spec import (
    PopulationEntry,
    ResolvedConfig,
    RunSpec,
    TargetSpec,
    WorkloadSpec,
)

#: Shown beside a comparison. A mismatch does not warn and does not refuse.
RECORDED_COMPARISON_FIELDS: tuple[str, ...] = (
    "identity.recorded.comparison_axis",
    "identity.recorded.engine.name",
    "identity.recorded.engine.version",
    "identity.recorded.engine.transport",
    "identity.recorded.server_version",
    "identity.recorded.precision",
    "identity.recorded.tf32",
)

#: Fields the hard comparison key does not cover. The schema reference
#: quotes this tuple. Warn-only fields are listed here as well: they are
#: compared and reported, and a mismatch does not refuse.
COMPARISON_EXCLUSIONS: tuple[str, ...] = (
    "run_id",
    "started_at",
    "finished_at",
    "run.output_path",
    "dataset.materialized_path",
    "workload.lookahead_ms_label",
    "workload.population.lookahead_ms_label",
    "workload.clip.path",
    "workload.clip.licence",
    "workload.clip.attribution",
    "workload.clip.note",
    "workload.iterations_per_stream",
    "workload.correctness_streams",
    "workload.performance_streams",
    "run.emit_tsv",
    "identity.warn.driver_version",
    "identity.warn.python_version",
    "identity.warn.git_commit",
    "identity.warn.git_dirty",
    "identity.warn.endpoint_url",
    "identity.warn.tls_verify",
    "identity.warn.auth_env",
    "identity.warn.package_version",
    "identity.warn.lookahead_identity_policy",
    "identity.warn.cpu_model",
    "identity.warn.logical_cpu_count",
    "identity.warn.system",
    "identity.warn.architecture",
    "identity.warn.hostname",
    "identity.warn.probe_status",
    "identity.warn.channel_options",
    "identity.warn.model",
    "identity.warn.transport_options",
    "identity.warn.batch_size",
    "identity.warn.max_num_seqs",
    "identity.warn.gpu_memory_utilization",
    "identity.warn.cuda_version",
    "identity.warn.torch_version",
    "identity.warn.client_location",
    "probe.hardware.gpu_uuids",
    "probe.hardware.collected_at",
    "secrets.set",
    "raw outcomes and trial aggregates",
)


def reference_sha256(reference: str | None) -> str | None:
    """Hash a reference transcript, or return ``None`` when there is none.

    Args:
        reference: The transcript, or ``None`` when the run declared no
            reference.

    Returns:
        A prefixed digest, or ``None``. An empty string is a real
        reference and is hashed; it is not treated as missing.
    """
    if reference is None:
        return None
    return sha256_text(reference)


def _population_material(
    entries: tuple[PopulationEntry, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "att_context_right": entry.att_context_right,
            "cadence_ms": entry.cadence_ms,
            "lanes": entry.lanes,
            "lookahead_frames": entry.lookahead_frames,
        }
        for entry in entries
    ]


def workload_material(
    workload: WorkloadSpec,
    *,
    blank_operating_point: bool = False,
    blank_lanes: bool = False,
    blank_rate: bool = False,
) -> dict[str, Any]:
    """Return the structure the workload fingerprint hashes.

    The lookahead-identity policy is not in this structure. It is
    warn-only, so switching the unused policy does not refuse.

    Args:
        workload: Offered work.
        blank_operating_point: When true, frame counts, cadences, and
            ``att_context_right`` are nulled. The frontier groups on that
            fingerprint so one lookahead sweep, including the cadence
            that sets lookahead, shares a series. ``chunk_ms`` stays: it
            is the client's transmit size, not the encoder lookahead.
        blank_lanes: When true, lane counts are nulled. The sustained-
            streams projection groups on that fingerprint.
        blank_rate: When true, ``rate_rps`` is nulled so a later
            offered-load sweep can share a capacity series.

    Returns:
        A JSON-ish dict. Millisecond labels, paths, the per-stream
        iteration factor, and the published stream grids are absent.
        Resolved ``iterations`` and ``lanes`` stay.
    """
    frames = None if blank_operating_point else workload.lookahead_frames
    cadence = None if blank_operating_point else workload.cadence_ms
    right = None if blank_operating_point else workload.att_context_right
    lanes = None if blank_lanes else workload.lanes
    rate = None if blank_rate else workload.rate_rps
    population: list[dict[str, Any]] = []
    for entry in _population_material(workload.population):
        if blank_operating_point:
            entry["lookahead_frames"] = None
            entry["cadence_ms"] = None
            entry["att_context_right"] = None
        if blank_lanes:
            entry["lanes"] = None
        population.append(entry)
    return {
        "att_context_right": right,
        "cadence_ms": cadence,
        "chunk_ms": workload.chunk_ms,
        "decoder": workload.decoder,
        "enable_word_time_offsets": workload.enable_word_time_offsets,
        "clip_digest": workload.clip.digest,
        "clip_duration_s": workload.clip.duration_s,
        "clip_generator": workload.clip.generator,
        "clip_id": workload.clip.id,
        "clip_sample_rate": workload.clip.sample_rate,
        "diarization": workload.diarization,
        "interim_results": workload.interim_results,
        "iterations": workload.iterations,
        "lanes": lanes,
        "language": workload.language,
        "language_model": workload.language_model,
        "lookahead_frames": frames,
        "mode": workload.mode,
        "pacing": workload.pacing,
        "population": population,
        "punctuation": workload.punctuation,
        "rate_rps": rate,
        "reference_sha256": reference_sha256(workload.reference_transcript),
        "seed": workload.seed,
        "timeout_s": workload.timeout_s,
        "transcript_policy": workload.transcript_policy,
        "trials": workload.trials,
        "warmup": workload.warmup,
    }


def workload_fingerprint(config: ResolvedConfig) -> str:
    """Hash offered work, including the client lookahead frame count.

    Args:
        config: A resolved config whose clip digest is already pinned.

    Returns:
        A prefixed sha256 digest.
    """
    return fingerprint(workload_material(config.workload))


def observation_material(run: RunSpec) -> dict[str, Any]:
    """Return the observation fingerprint's preimage.

    Output path is omitted. Percentile method, pass policy, and the
    lookahead-identity policy are included: they change interpretation,
    not offered work, and the workload fingerprint must ignore them.

    Args:
        run: The run spec.

    Returns:
        A JSON-ish dict. ``emit_tsv`` is absent: it chooses an extra
        file and does not change the observation.
    """
    policy = run.pass_policy
    return {
        "lookahead_identity_policy": run.lookahead_identity_policy,
        "pass_policy": {
            "accuracy_tolerance": policy.accuracy_tolerance,
            "admission_fraction_min": policy.admission_fraction_min,
            "first_partial_presence_min": policy.first_partial_presence_min,
            "latency_metric": policy.latency_metric,
            "latency_percentile": policy.latency_percentile,
            "latency_slo_s": policy.latency_slo_s,
            "normalization_policy": policy.normalization_policy,
            "sustainability_min": policy.sustainability_min,
            "sustainability_policy": policy.sustainability_policy,
        },
        "percentile_method": run.percentile_method,
        "profiler": run.profiler,
        "verbosity": run.verbosity,
    }


def observation_fingerprint(config: ResolvedConfig) -> str:
    """Hash observation choices, excluding the output path.

    Args:
        config: A resolved config.

    Returns:
        A prefixed sha256 digest.
    """
    return fingerprint(observation_material(config.run))


def _server_gpu_declared(target: TargetSpec) -> bool:
    """Return whether any server GPU field is set, including a count of zero."""
    return (
        target.server_gpu_name is not None
        or target.server_gpu_count is not None
        or target.server_gpu_memory_mib is not None
    )


def _checkpoint_block(target: TargetSpec) -> dict[str, Any]:
    """Hard weights identity. The routing model name is not this."""
    return {
        "id": target.checkpoint_id,
        "revision": target.checkpoint_revision,
        "source": target.checkpoint_source,
    }


def _engine_version(target: TargetSpec) -> str | None:
    """Engine version, falling back to the declared server version."""
    if target.engine_version is not None:
        return target.engine_version
    return target.server_version


def _target_block(target: TargetSpec) -> dict[str, Any]:
    """Hard target identity. The engine axis is not included."""
    return {"checkpoint": _checkpoint_block(target)}


def _recorded_block(target: TargetSpec) -> dict[str, Any]:
    """Engine axis and precision. Shown beside a comparison, not hashed."""
    return {
        "comparison_axis": COMPARISON_AXIS,
        "engine": {
            "name": target.engine_name,
            "transport": target.transport,
            "version": _engine_version(target),
        },
        "precision": target.precision,
        "server_version": target.server_version,
        "tf32": target.tf32,
    }


def _software_block(
    versions: Mapping[str, Any], policy: str, method: str
) -> dict[str, Any]:
    """Hard software identity. Semantics version is the caller's string."""
    semantics = versions["measurement_semantics_version"]
    return {
        "adapter": versions["adapter"],
        "measurement_semantics_version": semantics,
        "normalization_policy": policy,
        "package": "asrbench",
        "percentile_method": method,
        "schema_version": versions["schema_version"],
    }


def _hardware_block(
    *,
    gpu_name: str | None,
    gpu_count: int | None,
    gpu_memory_mib: int | None,
    gpu_identity_source: str,
) -> dict[str, Any]:
    """Hard hardware identity. Driver version and client CPU are not here."""
    return {
        "gpu_count": gpu_count,
        "gpu_identity_source": gpu_identity_source,
        "gpu_memory_mib": gpu_memory_mib,
        "gpu_name": gpu_name,
    }


def _hard_hardware(
    target: TargetSpec, probe: Mapping[str, Any]
) -> dict[str, Any]:
    """Server-declared GPU identity, or the labelled client probe."""
    if _server_gpu_declared(target):
        return _hardware_block(
            gpu_name=target.server_gpu_name,
            gpu_count=target.server_gpu_count,
            gpu_memory_mib=target.server_gpu_memory_mib,
            gpu_identity_source=GPU_IDENTITY_SERVER_DECLARED,
        )
    return _hardware_block(
        gpu_name=probe.get("gpu_name"),
        gpu_count=probe.get("gpu_count"),
        gpu_memory_mib=probe.get("gpu_memory_mib"),
        gpu_identity_source=GPU_IDENTITY_CLIENT_PROBE,
    )


def _driver(
    target: TargetSpec, client_driver: str | None
) -> tuple[str | None, str]:
    """Warn-only driver string and ``server_declared`` or ``client_probe``."""
    if target.server_driver_version is not None:
        return target.server_driver_version, GPU_IDENTITY_SERVER_DECLARED
    return client_driver, GPU_IDENTITY_CLIENT_PROBE


def _warn_block(
    config: ResolvedConfig,
    probe: Mapping[str, Any],
    versions: Mapping[str, Any],
) -> dict[str, Any]:
    """Fields that are compared and do not refuse."""
    driver, source = _driver(config.target, probe.get("driver_version"))
    return {
        "architecture": probe.get("architecture"),
        "auth_env": config.target.auth_env,
        "cpu_model": probe.get("cpu_model"),
        "driver_version": driver,
        "driver_version_source": source,
        "endpoint_url": config.target.endpoint_url,
        "git_commit": versions["git_commit"],
        "git_dirty": versions["git_dirty"],
        "hostname": probe.get("hostname"),
        "lookahead_identity_policy": config.run.lookahead_identity_policy,
        "logical_cpu_count": probe.get("logical_cpu_count"),
        "package_version": versions["package_version"],
        "batch_size": config.target.batch_size,
        "channel_options": dict(config.target.channel_options),
        "client_location": config.target.client_location,
        "cuda_version": config.target.cuda_version,
        "gpu_memory_utilization": config.target.gpu_memory_utilization,
        "max_num_seqs": config.target.max_num_seqs,
        "model": config.target.model,
        "probe_status": probe.get("probe_status"),
        "torch_version": config.target.torch_version,
        "transport_options": dict(config.target.transport_options),
        "python_version": versions["python_version"],
        "system": probe.get("system"),
        "tls_verify": config.target.tls_verify,
    }


def identity_blocks(
    config: ResolvedConfig,
    probe: Mapping[str, Any],
    versions: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return target, software, hardware, warn, and recorded blocks.

    ``versions["measurement_semantics_version"]`` and
    ``versions["schema_version"]`` are stored as given. Receipt build
    passes the installed constants. Compare passes the strings on the
    receipt, so two old receipts stay comparable after a bump.

    Args:
        config: Resolved config. Clip digest is already pinned or trusted.
        probe: Client probe fields: GPU name, count, memory, driver,
            system, architecture, CPU, hostname, and probe status.
        versions: Adapter name, schema version, measurement semantics
            version, package version, Python version, git commit, and
            git dirty bit.

    Returns:
        A dict with keys ``target``, ``software``, ``hardware``,
        ``warn``, and ``recorded``.
    """
    policy = config.run.pass_policy.normalization_policy
    method = config.run.percentile_method
    return {
        "target": _target_block(config.target),
        "software": _software_block(versions, policy, method),
        "hardware": _hard_hardware(config.target, probe),
        "warn": _warn_block(config, probe, versions),
        "recorded": _recorded_block(config.target),
    }


def comparison_key(
    *,
    workload_fp: str,
    target: dict[str, Any],
    software: dict[str, Any],
    hardware: dict[str, Any],
) -> str:
    """Hash the four hard comparison components in design order.

    Args:
        workload_fp: The workload fingerprint string.
        target: Hard target identity.
        software: Hard software identity.
        hardware: Hard hardware identity.

    Returns:
        A prefixed sha256 digest.
    """
    return join_components([workload_fp, target, software, hardware])


def group_key(
    config: ResolvedConfig,
    software: dict[str, Any],
    hardware: dict[str, Any],
    *,
    blank_operating_point: bool = False,
    blank_lanes: bool = False,
    blank_rate: bool = False,
) -> str:
    """Comparison key with selected workload axes removed.

    Args:
        config: Resolved config.
        software: Hard software identity.
        hardware: Hard hardware identity.
        blank_operating_point: Null lookahead frames and cadence.
        blank_lanes: Null lane counts.
        blank_rate: Null ``rate_rps``.

    Returns:
        A prefixed digest.
    """
    material = workload_material(
        config.workload,
        blank_operating_point=blank_operating_point,
        blank_lanes=blank_lanes,
        blank_rate=blank_rate,
    )
    return join_components(
        [
            fingerprint(material),
            _target_block(config.target),
            software,
            hardware,
        ]
    )


def frontier_group_key(
    config: ResolvedConfig, software: dict[str, Any], hardware: dict[str, Any]
) -> str:
    """Comparison key with lookahead frames and cadence removed.

    ``chunk_ms`` remains. It is the transmit size, not the operating
    point a cadence sweep moves. ``att_context_right`` is removed with
    the lookahead and the cadence.

    Args:
        config: Resolved config.
        software: Hard software identity.
        hardware: Hard hardware identity.

    Returns:
        A prefixed digest shared by one lookahead sweep.
    """
    return group_key(
        config,
        software,
        hardware,
        blank_operating_point=True,
    )


def capacity_group_key(
    config: ResolvedConfig, software: dict[str, Any], hardware: dict[str, Any]
) -> str:
    """Comparison key with lane counts and ``rate_rps`` removed.

    Args:
        config: Resolved config.
        software: Hard software identity.
        hardware: Hard hardware identity.

    Returns:
        A prefixed digest shared by a concurrency sweep of one operating
        point.
    """
    return group_key(
        config,
        software,
        hardware,
        blank_lanes=True,
        blank_rate=True,
    )
