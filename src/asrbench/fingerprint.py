"""Workload and observation fingerprints, and the identity objects.

The workload fingerprint covers offered work. It includes the reference
transcript's digest: a silent reference swap would otherwise compare as
the same work and report a WER delta that is the reference, not the
server. It excludes the lookahead millisecond label, clip path, licence
text, and every run-observation field.

The comparison key is the four-component hash from the design. Software
identity carries the normalization policy, the percentile method, and the
lookahead-identity policy so those are not pooled silently. They are not
part of the workload fingerprint: toggling them does not change the work
offered to the server.
"""

from __future__ import annotations

from typing import Any

from asrbench.canonical import fingerprint, join_components, sha256_text
from asrbench.spec import (
    PopulationEntry,
    ResolvedConfig,
    RunSpec,
    TargetSpec,
    WorkloadSpec,
)

#: Fields the comparison key does not cover. The schema reference quotes
#: this tuple. Tests lock the behavior; this list is what a reader sees.
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
    "probe.hardware.hostname",
    "probe.hardware.gpu_uuids",
    "probe.hardware.collected_at",
    "probe.software.git_dirty",
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
    blank_lookahead: bool = False,
    blank_lanes: bool = False,
) -> dict[str, Any]:
    """Return the structure the workload fingerprint hashes.

    The lookahead-identity policy is not in this structure. It lives on
    software identity, so switching policies does not pretend the offered
    work changed, and compare still refuses to pool the two policies.

    Args:
        workload: Offered work.
        blank_lookahead: When true, frame counts are nulled. The frontier
            groups on that fingerprint so two lookaheads of one setup
            share a series.
        blank_lanes: When true, lane counts are nulled. The sustained-
            streams projection groups on that fingerprint.

    Returns:
        A JSON-ish dict. Millisecond labels and paths are absent.
    """
    frames = None if blank_lookahead else workload.lookahead_frames
    lanes = None if blank_lanes else workload.lanes
    population: list[dict[str, Any]] = []
    for entry in _population_material(workload.population):
        if blank_lookahead:
            entry["lookahead_frames"] = None
        if blank_lanes:
            entry["lanes"] = None
        population.append(entry)
    return {
        "cadence_ms": workload.cadence_ms,
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
        "lookahead_frames_source": "client_declared",
        "mode": workload.mode,
        "pacing": workload.pacing,
        "population": population,
        "punctuation": workload.punctuation,
        "rate_rps": workload.rate_rps,
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


def target_identity(target: TargetSpec) -> dict[str, Any]:
    """Return the target half of the comparison key.

    The auth variable's name is included. Whether it was set, and the
    value, are not.

    Args:
        target: The target spec.

    Returns:
        A JSON-ish dict.
    """
    return {
        "auth_env": target.auth_env,
        "endpoint_url": target.endpoint_url,
        "model": target.model,
        "tls_verify": target.tls_verify,
        "transport": target.transport,
        "transport_options": target.transport_options,
    }


def software_identity(
    *,
    package_version: str,
    python_version: str,
    git_commit: str,
    adapter: str,
    lookahead_identity_policy: str,
    normalization_policy: str,
    percentile_method: str,
) -> dict[str, Any]:
    """Return the software half of the comparison key.

    Git dirty state is not included: an uncommitted edit does not by
    itself change the commit the key names, and the probe records the
    dirty bit separately. The normalization policy and the percentile
    method are included so a WER or a percentile from a different named
    policy is not compared as the same software result.

    Args:
        package_version: ``asrbench`` version string.
        python_version: Interpreter ``major.minor.micro``.
        git_commit: Hex commit, or ``unknown``.
        adapter: Adapter ``name``.
        lookahead_identity_policy: Named lookahead policy.
        normalization_policy: Named WER policy.
        percentile_method: Named percentile policy.

    Returns:
        A JSON-ish dict.
    """
    return {
        "adapter": adapter,
        "git_commit": git_commit,
        "lookahead_identity_policy": lookahead_identity_policy,
        "normalization_policy": normalization_policy,
        "package": "asrbench",
        "package_version": package_version,
        "percentile_method": percentile_method,
        "python_version": python_version,
    }


def hardware_identity(
    *,
    system: str,
    architecture: str,
    cpu_model: str | None,
    logical_cpu_count: int | None,
    gpu_name: str | None,
    gpu_count: int | None,
    gpu_memory_mib: int | None,
    driver_version: str | None,
) -> dict[str, Any]:
    """Return the hardware half of the comparison key.

    Args:
        system: ``platform.system()`` value.
        architecture: ``platform.machine()`` value.
        cpu_model: A model string, or ``None`` when it could not be read.
        logical_cpu_count: Logical CPUs, or ``None``.
        gpu_name: Common GPU model from nvidia-smi, or ``None``.
        gpu_count: How many GPUs the probe saw, or ``None``.
        gpu_memory_mib: Minimum total memory in MiB, or ``None``.
        driver_version: Driver version, or ``None``.

    Returns:
        A JSON-ish dict. Hostname and GPU UUIDs are not here.
    """
    return {
        "architecture": architecture,
        "cpu_model": cpu_model,
        "driver_version": driver_version,
        "gpu_count": gpu_count,
        "gpu_memory_mib": gpu_memory_mib,
        "gpu_name": gpu_name,
        "logical_cpu_count": logical_cpu_count,
        "system": system,
    }


def comparison_key(
    *,
    workload_fp: str,
    target: dict[str, Any],
    software: dict[str, Any],
    hardware: dict[str, Any],
) -> str:
    """Hash the four comparison components in design order.

    Args:
        workload_fp: The workload fingerprint string.
        target: :func:`target_identity`.
        software: :func:`software_identity`.
        hardware: :func:`hardware_identity`.

    Returns:
        A prefixed sha256 digest.
    """
    return join_components([workload_fp, target, software, hardware])


def frontier_group_key(
    config: ResolvedConfig, software: dict[str, Any], hardware: dict[str, Any]
) -> str:
    """Comparison key with lookahead frame counts removed.

    Args:
        config: Resolved config.
        software: Software identity.
        hardware: Hardware identity.

    Returns:
        A prefixed digest shared by receipts that differ only in lookahead.
    """
    material = workload_material(config.workload, blank_lookahead=True)
    return join_components(
        [
            fingerprint(material),
            target_identity(config.target),
            software,
            hardware,
        ]
    )


def capacity_group_key(
    config: ResolvedConfig, software: dict[str, Any], hardware: dict[str, Any]
) -> str:
    """Comparison key with lane counts removed.

    Args:
        config: Resolved config.
        software: Software identity.
        hardware: Hardware identity.

    Returns:
        A prefixed digest shared by a concurrency sweep of one operating
        point.
    """
    material = workload_material(config.workload, blank_lanes=True)
    return join_components(
        [
            fingerprint(material),
            target_identity(config.target),
            software,
            hardware,
        ]
    )
