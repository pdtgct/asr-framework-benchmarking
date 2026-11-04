"""Reproduction, refusal, and a recorded force reason."""

import contextlib
import json
import os
from io import StringIO
from pathlib import Path

import pytest

from asrbench.cli import main
from asrbench.compare import compare_receipts

_PRESET = Path("presets/mock-streaming.json")


@pytest.fixture
def sealed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("ASRBENCH_"):
            monkeypatch.delenv(key, raising=False)


def _run(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with (
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def test_from_receipt_reproduces_the_resolved_config(
    tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    first = tmp_path / "r1.json"
    second = tmp_path / "r2.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(first)])
    assert code == 0, err
    code, _, err = _run(
        ["run", "--from-receipt", str(first), "--out", str(second)]
    )
    assert code == 0, err
    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))
    assert left["comparison_key"] == right["comparison_key"]
    assert left["fingerprints"] == right["fingerprints"]
    left_config = left["resolved_config"]
    right_config = right["resolved_config"]
    assert left_config["run"]["output_path"] == str(first)
    assert right_config["run"]["output_path"] == str(second)
    left_config["run"]["output_path"] = None
    right_config["run"]["output_path"] = None
    assert left_config == right_config
    code, out, err = _run(["compare", str(first), str(second)])
    assert code == 0, err
    assert json.loads(out)["status"] == "comparable"
    encoded = first.read_text(encoding="utf-8")
    assert "ASRBENCH_TOKEN" in encoded
    assert '"set": false' in encoded or '"set": true' in encoded


def test_compare_refuses_another_gpu_and_another_lookahead(
    tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    source = tmp_path / "r1.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(source)])
    assert code == 0, err
    base = json.loads(source.read_text(encoding="utf-8"))
    a40 = tmp_path / "a40.json"
    h100 = tmp_path / "h100.json"
    left = json.loads(json.dumps(base))
    right = json.loads(json.dumps(base))
    left["identity"]["hardware"]["gpu_name"] = "A40"
    right["identity"]["hardware"]["gpu_name"] = "H100"
    a40.write_text(json.dumps(left), encoding="utf-8")
    h100.write_text(json.dumps(right), encoding="utf-8")
    code, out, err = _run(["compare", str(a40), str(h100)])
    assert code == 2, out
    assert err == ""
    assert "hardware_identity" in out
    assert "gpu_name" in out
    assert "A40" in out
    assert "H100" in out

    other = json.loads(json.dumps(base))
    other["resolved_config"]["workload"]["lookahead_frames"] = 8
    other["resolved_config"]["workload"]["population"][0][
        "lookahead_frames"
    ] = 8
    other["lookahead"]["frames"] = 8
    shifted = tmp_path / "lookahead.json"
    shifted.write_text(json.dumps(other), encoding="utf-8")
    code, out, _err = _run(["compare", str(source), str(shifted)])
    assert code == 2
    assert "workload_fingerprint" in out
    assert "lookahead_frames" in out

    code, out, err = _run(
        [
            "compare",
            str(a40),
            str(h100),
            "--force",
            "--reason",
            "different machine",
        ]
    )
    assert code == 0, err
    forced = json.loads(out)
    assert forced["status"] == "forced"
    assert forced["force_reason"] == "different machine"
    code, _out, err = _run(["compare", str(a40), str(h100), "--force"])
    assert code == 2
    assert "reason" in err


def test_observation_fingerprint_mismatch_does_not_refuse(
    tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    source = tmp_path / "r1.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(source)])
    assert code == 0, err
    left = json.loads(source.read_text(encoding="utf-8"))
    right = json.loads(json.dumps(left))
    right["fingerprints"]["observation"] = "sha256:" + ("ab" * 32)
    other = tmp_path / "observed.json"
    other.write_text(json.dumps(right), encoding="utf-8")
    result = compare_receipts(left, right)
    assert result.status == "comparable"
    assert result.observation_fingerprint_match is False
    code, out, err = _run(["compare", str(source), str(other)])
    assert code == 0, err
    assert json.loads(out)["observation_fingerprint_match"] is False


def test_reason_without_force_is_refused(
    tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    source = tmp_path / "r1.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(source)])
    assert code == 0, err
    code, _out, err = _run(
        ["compare", str(source), str(source), "--reason", "because"]
    )
    assert code == 2
    assert "--force" in err


def test_force_without_a_hard_mismatch_does_not_record_the_reason(
    tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    source = tmp_path / "r1.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(source)])
    assert code == 0, err
    code, out, err = _run(
        ["compare", str(source), str(source), "--force", "--reason", "unused"]
    )
    assert code == 0, err
    parsed = json.loads(out)
    assert parsed["status"] == "comparable"
    assert parsed["force_reason"] is None
    code, _out, err = _run(
        ["compare", str(source), str(source), "--force", "--reason", "   "]
    )
    assert code == 2
    assert "reason" in err


def test_a_reproduction_override_is_printed_and_not_a_secret(
    tmp_path: Path, sealed_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del sealed_env
    first = tmp_path / "r1.json"
    second = tmp_path / "r2.json"
    code, _, err = _run(["run", "--preset", str(_PRESET), "--out", str(first)])
    assert code == 0, err
    monkeypatch.setenv("ASRBENCH_TOKEN", "should-stay-out")
    code, _, err = _run(
        [
            "run",
            "--from-receipt",
            str(first),
            "--out",
            str(second),
            "--warmup",
            "9",
        ]
    )
    assert code == 0, err
    assert "reproduction workload.warmup:" in err
    assert "should-stay-out" not in err
    right = json.loads(second.read_text(encoding="utf-8"))
    assert right["resolved_config"]["workload"]["warmup"] == 9
    assert right["reproduction"]["overrides"]
    assert "should-stay-out" not in second.read_text(encoding="utf-8")


def test_measurement_error_is_a_clean_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sealed_env: None
) -> None:
    del sealed_env
    from asrbench.reducers import MeasurementError

    def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise MeasurementError("ttfs_s is not finite")

    monkeypatch.setattr("asrbench.cli.run_benchmark", boom)
    code, _out, err = _run(
        ["run", "--preset", str(_PRESET), "--out", str(tmp_path / "r.json")]
    )
    assert code == 2
    assert "not finite" in err
