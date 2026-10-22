"""Resolution order, secret handling, and digest pins."""

import json
from pathlib import Path

import pytest

from asrbench.config import (
    ConfigError,
    apply_preset,
    config_to_dict,
    resolve,
    secret_records,
)
from asrbench.spec import default_config

_PRESET = Path("presets/mock-streaming.json")


def test_resolution_order_is_defaults_then_preset_then_env_then_cli() -> None:
    defaults = resolve(env={})
    assert defaults.workload.warmup == 0
    preset = resolve(preset_path=_PRESET, env={})
    assert preset.workload.warmup == 1
    from_env = resolve(preset_path=_PRESET, env={"ASRBENCH_WARMUP": "2"})
    assert from_env.workload.warmup == 2
    from_cli = resolve(
        preset_path=_PRESET,
        env={"ASRBENCH_WARMUP": "2"},
        overrides={("workload", "warmup"): 4},
    )
    assert from_cli.workload.warmup == 4


def test_unknown_preset_field_and_unknown_env_are_errors() -> None:
    with pytest.raises(ConfigError, match="unknown preset"):
        apply_preset(default_config(), {"extra": 1})
    with pytest.raises(ConfigError, match="unknown environment variable"):
        resolve(env={"ASRBENCH_NOT_A_SETTING": "1"})


def test_auth_variable_is_recorded_by_name_and_never_by_value() -> None:
    secret = "super-secret-value"
    config = resolve(
        preset_path=_PRESET,
        env={"ASRBENCH_TOKEN": secret, "ASRBENCH_WARMUP": "1"},
    )
    encoded = json.dumps(config_to_dict(config))
    assert secret not in encoded
    assert config.target.auth_env == "ASRBENCH_TOKEN"
    assert secret_records(config, {"ASRBENCH_TOKEN": secret}) == [
        {"name": "ASRBENCH_TOKEN", "set": True}
    ]
    assert secret_records(config, {}) == [
        {"name": "ASRBENCH_TOKEN", "set": False}
    ]


def test_wrong_digest_pin_fails_closed() -> None:
    payload = json.loads(_PRESET.read_text(encoding="utf-8"))
    payload["workload"]["clip"]["digest"] = "sha256:" + ("ab" * 32)
    config = apply_preset(default_config(), payload)
    from asrbench.config import finalize

    with pytest.raises(ConfigError):
        finalize(config)


def test_preset_and_from_receipt_together_are_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "r.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="not both"):
        resolve(preset_path=_PRESET, from_receipt=receipt, env={})


def test_bool_env_and_bool_json_are_strict() -> None:
    with pytest.raises(ConfigError):
        resolve(env={"ASRBENCH_PACING": "maybe"})
    payload = json.loads(_PRESET.read_text(encoding="utf-8"))
    payload["workload"]["chunk_ms"] = True
    with pytest.raises(ConfigError):
        apply_preset(default_config(), payload)


def test_json_int_timeout_is_a_float() -> None:
    config = resolve(preset_path=_PRESET, env={})
    assert config.workload.timeout_s == 5.0
    assert isinstance(config.workload.timeout_s, float)
