"""Assemble one receipt.

The receipt is the unit of truth. Reporters read it and do not recompute
from anywhere else. Raw outcomes and raw trial reductions both stay.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, cast

from asrbench.canonical import sha256_text
from asrbench.config import config_to_dict
from asrbench.definitions import (
    AUDIO_CONDITION_DEFINITION,
    AUDIO_CONDITION_NATIVE,
    DIGEST_COVERS_PCM16,
    LOOKAHEAD_IDENTITY_DEFINITION,
    MANIFEST_DIGEST_DEFINITION,
    MEASUREMENT_SEMANTICS_VERSION,
    SCHEMA_VERSION,
    emitted_definitions,
)
from asrbench.fingerprint import (
    comparison_key,
    identity_blocks,
    observation_fingerprint,
    sent_sample_rate,
    workload_fingerprint,
)
from asrbench.identity import GitState, HardwareRecord
from asrbench.manifest import Manifest
from asrbench.outcome import SessionOutcome
from asrbench.reducers import is_productive, reduce_trial, summarize_trials
from asrbench.resample import RESAMPLER
from asrbench.spec import ResolvedConfig


def json_ready(value: Any) -> Any:
    """Round-trip ``value`` through JSON so the receipt has JSON types.

    Args:
        value: A structure of dicts, lists, strings, numbers, bools, and
            None.

    Returns:
        The same structure after ``json.dumps`` and ``json.loads``.

    Raises:
        ValueError: The structure contains a non-finite float.
    """
    return json.loads(json.dumps(value, allow_nan=False))


def outcome_dict(outcome: SessionOutcome) -> dict[str, Any]:
    """Copy one outcome into a JSON object.

    Args:
        outcome: A session outcome.

    Returns:
        Every dataclass field, in field order.
    """
    encoded: dict[str, Any] = json_ready(asdict(outcome))
    return encoded


def build_receipt(
    config: ResolvedConfig,
    trials: list[dict[str, Any]],
    *,
    secrets: list[dict[str, Any]],
    hardware: HardwareRecord,
    git_state: GitState,
    package_version: str,
    python_version: str,
    adapter_name: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    materialized_path: str,
    corpus: Manifest | None = None,
) -> dict[str, Any]:
    """Build the authoritative receipt object.

    Args:
        config: Resolved configuration, digest already pinned.
        trials: Each item has ``trial_index``, ``wall_span_s``,
            ``outcomes``, and ``warmup_outcomes``.
        secrets: Name and set-flag records. No values.
        hardware: Collected or injected hardware record.
        git_state: Collected or injected commit.
        package_version: ``asrbench`` version.
        python_version: Interpreter version.
        adapter_name: ``TransportAdapter.name``.
        run_id: Per-run id. Not part of the comparison key.
        started_at: RFC 3339 start marker.
        finished_at: RFC 3339 finish marker.
        materialized_path: Where the clip WAV was written, or the
            directory of a manifest's clips. Not identity.
        corpus: The loaded manifest of a corpus workload. Each session is
            scored against its own clip's reference.

    Returns:
        A JSON-ready receipt dict.
    """
    workload_fp = workload_fingerprint(config)
    observation_fp = observation_fingerprint(config)
    blocks = identity_blocks(
        config,
        {
            "architecture": hardware.architecture,
            "cpu_model": hardware.cpu_model,
            "driver_version": hardware.driver_version,
            "gpu_count": hardware.gpu_count,
            "gpu_memory_mib": hardware.gpu_memory_mib,
            "gpu_name": hardware.gpu_name,
            "hostname": hardware.hostname,
            "logical_cpu_count": hardware.logical_cpu_count,
            "probe_status": hardware.probe_status,
            "system": hardware.system,
        },
        {
            "adapter": adapter_name,
            "git_commit": git_state.commit,
            "git_dirty": git_state.dirty,
            "measurement_semantics_version": MEASUREMENT_SEMANTICS_VERSION,
            "package_version": package_version,
            "python_version": python_version,
            "schema_version": SCHEMA_VERSION,
        },
    )
    target = blocks["target"]
    software = blocks["software"]
    hardware_id = blocks["hardware"]
    warn = blocks["warn"]
    reference: str | dict[str, str] | None = (
        config.workload.reference_transcript
        if corpus is None
        else corpus.references()
    )
    rendered_trials: list[dict[str, Any]] = []
    measured: list[SessionOutcome] = []
    spans: list[float] = []
    for trial in trials:
        outcomes: list[SessionOutcome] = trial["outcomes"]
        warmup: list[SessionOutcome] = trial["warmup_outcomes"]
        measured.extend(outcomes)
        spans.append(float(trial["wall_span_s"]))
        reduction = reduce_trial(
            outcomes,
            wall_span_s=float(trial["wall_span_s"]),
            mode=config.workload.mode,
            pacing=config.workload.pacing,
            lanes=config.workload.lanes,
            reference=reference,
            pass_policy=config.run.pass_policy,
            percentile_method=config.run.percentile_method,
            warmup_excluded=len(warmup),
            language=config.workload.language,
        )
        rendered_trials.append(
            {
                "trial_index": trial["trial_index"],
                "warmup_excluded": len(warmup),
                "cell_wall_span_s": float(trial["wall_span_s"]),
                "outcomes": [outcome_dict(item) for item in outcomes],
                "warmup_outcomes": [outcome_dict(item) for item in warmup],
                "reduction": reduction,
            }
        )
    summary = summarize_trials(
        [item["reduction"] for item in rendered_trials],
        measured,
        wall_span_s=sum(spans),
        mode=config.workload.mode,
        pacing=config.workload.pacing,
        lanes=config.workload.lanes,
        reference=reference,
        pass_policy=config.run.pass_policy,
        percentile_method=config.run.percentile_method,
        warmup_excluded=sum(
            item["warmup_excluded"] for item in rendered_trials
        ),
        language=config.workload.language,
    )
    clip = config.workload.clip
    attempted = sum(item.audio_duration_s for item in measured)
    successful = sum(
        item.audio_duration_s for item in measured if is_productive(item)
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "definitions": emitted_definitions(),
        "resolved_config": config_to_dict(config),
        "reproduction": {
            "overrides": [
                asdict(item) for item in config.reproduction_overrides
            ],
        },
        "secrets": secrets,
        "fingerprints": {
            "workload": workload_fp,
            "observation": observation_fp,
        },
        "identity": {
            "target": target,
            "software": software,
            "hardware": hardware_id,
            "warn": warn,
            "recorded": blocks["recorded"],
        },
        "probe": {
            "hardware": {
                "architecture": hardware.architecture,
                "collected_at": hardware.collected_at,
                "cpu_model": hardware.cpu_model,
                "driver_version": hardware.driver_version,
                "gpu_count": hardware.gpu_count,
                "gpu_memory_mib": hardware.gpu_memory_mib,
                "gpu_name": hardware.gpu_name,
                "gpu_uuids": list(hardware.gpu_uuids),
                "hostname": hardware.hostname,
                "logical_cpu_count": hardware.logical_cpu_count,
                "probe_name": hardware.probe_name,
                "probe_status": hardware.probe_status,
                "system": hardware.system,
            },
            "software": {
                "git_available": git_state.available,
                "git_commit": git_state.commit,
                "git_dirty": git_state.dirty,
            },
        },
        "comparison_key": comparison_key(
            workload_fp=workload_fp,
            target=target,
            software=software,
            hardware=hardware_id,
        ),
        "dataset": {
            "clip_id": clip.id,
            "clip_digest": clip.digest,
            "digest_covers": DIGEST_COVERS_PCM16,
            "generator": clip.generator,
            "duration_s": clip.duration_s,
            "sample_rate": clip.sample_rate,
            "licence": clip.licence,
            "attribution": clip.attribution,
            "note": clip.note,
            "materialized_path": materialized_path,
            "reference_transcript": config.workload.reference_transcript,
            **_manifest_dataset(config, corpus),
            **_audio_condition_dataset(config),
        },
        "audio_denominators": {
            "definition": emitted_definitions()["AUDIO_DENOMINATOR_DEFINITION"],
            "attempted_audio_s": attempted,
            "successful_audio_s": successful,
            "warmup_excluded_sessions": sum(
                item["warmup_excluded"] for item in rendered_trials
            ),
        },
        "lookahead": {
            "policy": config.run.lookahead_identity_policy,
            "frames": config.workload.lookahead_frames,
            "att_context_right": config.workload.att_context_right,
            "frames_source": "client_declared",
            "millisecond_label": config.workload.lookahead_ms_label,
            "server_reported_frames": None,
            "definition": LOOKAHEAD_IDENTITY_DEFINITION,
        },
        "trials": rendered_trials,
        "summary": summary,
    }
    return cast(dict[str, Any], json_ready(receipt))


def _manifest_dataset(
    config: ResolvedConfig, corpus: Manifest | None
) -> dict[str, Any]:
    """Return ``{"manifest": ...}`` for a corpus run, else nothing.

    Single-clip receipts keep their old ``dataset`` shape. A corpus
    receipt carries every clip's digest and reference text, so it can be
    re-scored under another normalizer without the staged audio.
    """
    if corpus is None:
        return {}
    spec = config.workload.manifest
    return {
        "manifest": {
            "id": corpus.id,
            "digest": corpus.digest,
            "digest_definition": MANIFEST_DIGEST_DEFINITION,
            "language": corpus.language,
            "corpus": corpus.corpus,
            "licence": corpus.licence,
            "attribution": corpus.attribution,
            "assignment": spec.assignment,
            "clip_count": len(corpus.clips),
            "total_duration_s": corpus.total_duration_s,
            "path": spec.path,
            "clips": [
                {
                    "id": clip.id,
                    "pcm_sha256": clip.pcm_sha256,
                    "duration_s": clip.duration_s,
                    "reference_sha256": sha256_text(clip.reference),
                    "reference": clip.reference,
                }
                for clip in corpus.clips
            ],
        }
    }


def _audio_condition_dataset(config: ResolvedConfig) -> dict[str, Any]:
    """Return ``{"audio_condition": ...}`` unless the audio is native."""
    workload = config.workload
    if workload.audio_condition == AUDIO_CONDITION_NATIVE:
        return {}
    return {
        "audio_condition": {
            "name": workload.audio_condition,
            "definition": AUDIO_CONDITION_DEFINITION,
            "source_sample_rate_hertz": workload.clip.sample_rate,
            "sent_sample_rate_hertz": sent_sample_rate(workload),
            "resampler": RESAMPLER,
        }
    }
