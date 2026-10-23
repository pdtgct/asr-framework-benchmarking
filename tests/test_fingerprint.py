"""Fingerprints cover offered work and ignore observation choices."""

from pathlib import Path

from asrbench.canonical import CanonicalError, fingerprint, format_float
from asrbench.config import resolve
from asrbench.fingerprint import (
    comparison_key,
    hardware_identity,
    observation_fingerprint,
    reference_sha256,
    software_identity,
    target_identity,
    workload_fingerprint,
    workload_material,
)
from asrbench.spec import ResolvedConfig

_PRESET = Path("presets/mock-streaming.json")


def _key(config: ResolvedConfig) -> str:
    software = software_identity(
        adapter="mock",
        normalization_policy=config.run.pass_policy.normalization_policy,
        percentile_method=config.run.percentile_method,
    )
    hardware = hardware_identity(
        gpu_name=None,
        gpu_count=None,
        gpu_memory_mib=None,
        gpu_identity_source="client_probe",
    )
    return comparison_key(
        workload_fp=workload_fingerprint(config),
        target=target_identity(config.target),
        software=software,
        hardware=hardware,
    )


def test_canonical_json_ignores_dict_order_and_signed_zero() -> None:
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    assert format_float(-0.0) == "0"
    assert format_float(0.0) == "0"
    assert fingerprint({"x": 0.1}) == fingerprint({"x": 0.1})
    try:
        format_float(float("nan"))
    except CanonicalError:
        return
    raise AssertionError("non-finite floats must be rejected")


def test_observation_choices_do_not_change_the_workload_fingerprint() -> None:
    base = resolve(env={})
    noisy = resolve(
        env={
            "ASRBENCH_OUTPUT": "/tmp/one.json",
            "ASRBENCH_PROFILER": "off",
        },
        overrides={("run", "verbosity"): 3},
    )
    assert workload_fingerprint(base) == workload_fingerprint(noisy)
    assert observation_fingerprint(base) != observation_fingerprint(noisy)
    same_observation = resolve(env={"ASRBENCH_OUTPUT": "/tmp/other.json"})
    assert observation_fingerprint(base) == observation_fingerprint(
        same_observation
    )
    assert _key(base) == _key(same_observation)


def test_chunk_and_lookahead_frames_change_the_workload_fingerprint() -> None:
    base = resolve(env={})
    chunk = resolve(env={"ASRBENCH_CHUNK_MS": "80"})
    frames = resolve(env={"ASRBENCH_LOOKAHEAD_FRAMES": "8"})
    label = resolve(env={"ASRBENCH_LOOKAHEAD_MS_LABEL": "999"})
    assert workload_fingerprint(base) != workload_fingerprint(chunk)
    assert workload_fingerprint(base) != workload_fingerprint(frames)
    assert workload_fingerprint(base) == workload_fingerprint(label)
    assert _key(base) == _key(label)
    assert _key(base) != _key(frames)
    material = workload_material(label.workload)
    assert "lookahead_ms_label" not in material
    assert material["lookahead_frames"] == 4


def test_percentile_method_is_software_identity_not_offered_work() -> None:
    base = resolve(env={})
    other = resolve(env={"ASRBENCH_PERCENTILE_METHOD": "nearest_rank"})
    assert workload_fingerprint(base) == workload_fingerprint(other)
    assert observation_fingerprint(base) != observation_fingerprint(other)
    assert _key(base) != _key(other)


def test_reference_text_is_in_the_workload_fingerprint() -> None:
    base = resolve(env={})
    swapped = resolve(
        env={},
        overrides={
            ("workload", "reference_transcript"): "a different sentence"
        },
    )
    assert workload_fingerprint(base) != workload_fingerprint(swapped)
    assert reference_sha256(None) is None
    assert reference_sha256("") is not None
    assert reference_sha256("") != reference_sha256("a")


def test_lookahead_policy_does_not_change_offered_work() -> None:
    base = resolve(env={})
    other = resolve(
        env={
            "ASRBENCH_LOOKAHEAD_IDENTITY_POLICY": "server_reported_when_present"
        }
    )
    assert workload_fingerprint(base) == workload_fingerprint(other)
    assert observation_fingerprint(base) != observation_fingerprint(other)
    assert _key(base) == _key(other)
    material = workload_material(base.workload)
    assert "path" not in material
    assert "licence" not in material
    assert "lookahead_frames_source" not in material


def test_preset_path_and_licence_are_not_offered_work() -> None:
    config = resolve(preset_path=_PRESET, env={})
    material = workload_material(config.workload)
    assert material["clip_digest"].startswith("sha256:")
    for absent in (
        "path",
        "licence",
        "attribution",
        "note",
        "lookahead_ms_label",
        "lookahead_frames_source",
    ):
        assert absent not in material
