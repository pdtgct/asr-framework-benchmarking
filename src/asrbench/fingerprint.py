"""Workload and observation fingerprints, and the identity objects.

The workload fingerprint covers offered work. It includes the reference
transcript's digest: a silent reference swap would otherwise compare as
the same work and report a WER delta that is the reference, not the
server. It excludes the lookahead millisecond label, clip path, licence
text, and every run-observation field.

The comparison key hashes four hard components. Warn-only fields are
stored beside the key and do not refuse. ``transport_options`` stay hard
because they change what the adapter sends. Git commit is warn-only and
is read from this package's repository, not the caller's directory.
"""

from __future__ import annotations

from typing import Any

from asrbench.canonical import fingerprint, join_components, sha256_text
from asrbench.definitions import (
    GPU_IDENTITY_CLIENT_PROBE,
    GPU_IDENTITY_SERVER_DECLARED,
    MEASUREMENT_SEMANTICS_VERSION,
    SCHEMA_VERSION,
)
from asrbench.spec import (
    PopulationEntry,
    ResolvedConfig,
    RunSpec,
    TargetSpec,
    WorkloadSpec,
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
            "lanes": entry.lanes,
            "cadence_ms": entry.cadence_ms,
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
        blank_operating_point: When true, frame counts and cadences are
            nulled. The frontier groups on that fingerprint so one
            lookahead sweep, including the cadence that sets lookahead,
            shares a series. ``chunk_ms`` stays: it is the client's
            transmit size, not the encoder lookahead.
        blank_lanes: When true, lane counts are nulled. The sustained-
            streams projection groups on that fingerprint.
        blank_rate: When true, ``rate_rps`` is nulled so a later
            offered-load sweep can share a capacity series.

    Returns:
        A JSON-ish dict. Millisecond labels and paths are absent.
    """
    frames = None if blank_operating_point else workload.lookahead_frames
    cadence = None if blank_operating_point else workload.cadence_ms
    lanes = None if blank_lanes else workload.lanes
    rate = None if blank_rate else workload.rate_rps
    population: list[dict[str, Any]] = []
    for entry in _population_material(workload.population):
        if blank_operating_point:
            entry["lookahead_frames"] = None
            entry["cadence_ms"] = None
        if blank_lanes:
            entry["lanes"] = None
        population.append(entry)
    return {
        "cadence_ms": cadence,
        "chunk_ms": workload.chunk_ms,
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
        A JSON-ish dict.
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


def server_gpu_declared(target: TargetSpec) -> bool:
    """Return whether the target spec names a server GPU.

    Args:
        target: The target spec.

    Returns:
        True when any server GPU field is set. Those fields are then
        the hard GPU identity. A count of zero is a declaration.
    """
    return (
        target.server_gpu_name is not None
        or target.server_gpu_count is not None
        or target.server_gpu_memory_mib is not None
    )


def target_identity(target: TargetSpec) -> dict[str, Any]:
    """Return the hard target identity.

    Endpoint URL, TLS, and the auth variable name are warn-only.
    ``transport_options`` stay here because they change adapter behavior.

    Args:
        target: The target spec.

    Returns:
        A JSON-ish dict.
    """
    return {
        "model": target.model,
        "server_version": target.server_version,
        "transport": target.transport,
        "transport_options": dict(target.transport_options),
    }


def software_identity(
    *,
    adapter: str,
    normalization_policy: str,
    percentile_method: str,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return the hard software identity.

    Git commit, the Python version, the package version, and the
    lookahead-identity policy are warn-only. ``schema_version`` and
    :data:`~asrbench.definitions.MEASUREMENT_SEMANTICS_VERSION` are what
    a comparison refuses on when the measurement itself changed.

    Args:
        adapter: Adapter ``name``.
        normalization_policy: Named WER policy.
        percentile_method: Named percentile policy.
        schema_version: Receipt schema string. Receipt build uses the
            package constant. Compare uses the string the receipt stored.

    Returns:
        A JSON-ish dict.
    """
    return {
        "adapter": adapter,
        "measurement_semantics_version": MEASUREMENT_SEMANTICS_VERSION,
        "normalization_policy": normalization_policy,
        "package": "asrbench",
        "percentile_method": percentile_method,
        "schema_version": schema_version,
    }


def hardware_identity(
    *,
    gpu_name: str | None,
    gpu_count: int | None,
    gpu_memory_mib: int | None,
    gpu_identity_source: str,
) -> dict[str, Any]:
    """Return the hard hardware identity.

    Args:
        gpu_name: Server GPU name, or the client probe's name.
        gpu_count: How many GPUs that source reported.
        gpu_memory_mib: Minimum total memory in MiB, or ``None``.
        gpu_identity_source: ``server_declared`` or ``client_probe``.

    Returns:
        A JSON-ish dict. Driver version and client CPU are not here.
    """
    return {
        "gpu_count": gpu_count,
        "gpu_identity_source": gpu_identity_source,
        "gpu_memory_mib": gpu_memory_mib,
        "gpu_name": gpu_name,
    }


def hard_hardware(
    target: TargetSpec,
    *,
    client_gpu_name: str | None,
    client_gpu_count: int | None,
    client_gpu_memory_mib: int | None,
) -> dict[str, Any]:
    """Choose server-declared GPU identity, or the labelled client probe.

    Args:
        target: Target spec. Server GPU fields win when any is set.
        client_gpu_name: Client probe GPU name.
        client_gpu_count: Client probe GPU count.
        client_gpu_memory_mib: Client probe minimum memory.

    Returns:
        The hard hardware identity.
    """
    if server_gpu_declared(target):
        return hardware_identity(
            gpu_name=target.server_gpu_name,
            gpu_count=target.server_gpu_count,
            gpu_memory_mib=target.server_gpu_memory_mib,
            gpu_identity_source=GPU_IDENTITY_SERVER_DECLARED,
        )
    return hardware_identity(
        gpu_name=client_gpu_name,
        gpu_count=client_gpu_count,
        gpu_memory_mib=client_gpu_memory_mib,
        gpu_identity_source=GPU_IDENTITY_CLIENT_PROBE,
    )


def warn_identity(
    *,
    endpoint_url: str,
    tls_verify: bool,
    auth_env: str | None,
    lookahead_identity_policy: str,
    package_version: str,
    python_version: str,
    git_commit: str,
    git_dirty: bool,
    driver_version: str | None,
    driver_version_source: str,
    system: str,
    architecture: str,
    cpu_model: str | None,
    logical_cpu_count: int | None,
    hostname: str | None,
    probe_status: str | None,
) -> dict[str, Any]:
    """Return fields that are compared and do not refuse.

    Args:
        endpoint_url: Server URL. A redeploy at the same URL is not
            caught here; server version is the hard field for that.
        tls_verify: Whether the client verified TLS.
        auth_env: Auth variable name. The value is never included.
        lookahead_identity_policy: Named lookahead policy.
        package_version: ``asrbench`` version string.
        python_version: Interpreter ``major.minor.micro``.
        git_commit: Hex commit of this package's repository, or
            ``unknown``.
        git_dirty: Whether that worktree had uncommitted changes.
        driver_version: Server driver when declared, else the client
            probe's driver.
        driver_version_source: Which of those two was used.
        system: Client ``platform.system()`` value.
        architecture: Client ``platform.machine()`` value.
        cpu_model: Client CPU model, or ``None``.
        logical_cpu_count: Client logical CPUs, or ``None``.
        hostname: Client hostname. Warn-only, and still not a secret.
        probe_status: Client GPU probe status.

    Returns:
        A JSON-ish dict.
    """
    return {
        "architecture": architecture,
        "auth_env": auth_env,
        "cpu_model": cpu_model,
        "driver_version": driver_version,
        "driver_version_source": driver_version_source,
        "endpoint_url": endpoint_url,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hostname": hostname,
        "lookahead_identity_policy": lookahead_identity_policy,
        "logical_cpu_count": logical_cpu_count,
        "package_version": package_version,
        "probe_status": probe_status,
        "python_version": python_version,
        "system": system,
        "tls_verify": tls_verify,
    }


def driver_for_warn(
    target: TargetSpec, client_driver: str | None
) -> tuple[str | None, str]:
    """Return the warn-only driver string and which source it came from.

    Args:
        target: Target spec. ``server_driver_version`` wins when set.
        client_driver: Driver string from the client probe.

    Returns:
        The version and ``server_declared`` or ``client_probe``.
    """
    if target.server_driver_version is not None:
        return target.server_driver_version, GPU_IDENTITY_SERVER_DECLARED
    return client_driver, GPU_IDENTITY_CLIENT_PROBE


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
        target: :func:`target_identity`.
        software: :func:`software_identity`.
        hardware: :func:`hardware_identity`.

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
            target_identity(config.target),
            software,
            hardware,
        ]
    )


def frontier_group_key(
    config: ResolvedConfig, software: dict[str, Any], hardware: dict[str, Any]
) -> str:
    """Comparison key with lookahead frames and cadence removed.

    ``chunk_ms`` remains. It is the transmit size, not the operating
    point a cadence sweep moves.

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
