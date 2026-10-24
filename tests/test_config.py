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
from asrbench.definitions import SCHEMA_VERSION
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
    from asrbench.config import validate

    with pytest.raises(ConfigError):
        validate(config)


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


def test_single_entry_population_follows_lane_override() -> None:
    config = resolve(
        preset_path=_PRESET,
        env={},
        overrides={("workload", "lanes"): 4},
    )
    assert config.workload.lanes == 4
    assert len(config.workload.population) == 1
    assert config.workload.population[0].lanes == 4


def test_multi_entry_population_rejects_a_lane_override(tmp_path: Path) -> None:
    payload = json.loads(_PRESET.read_text(encoding="utf-8"))
    payload["workload"]["lanes"] = 3
    payload["workload"]["cadence_ms"] = None
    payload["workload"]["lookahead_frames"] = None
    payload["workload"]["lookahead_ms_label"] = None
    payload["workload"]["population"] = [
        {
            "lanes": 1,
            "cadence_ms": 160,
            "lookahead_frames": 2,
            "lookahead_ms_label": 160,
        },
        {
            "lanes": 2,
            "cadence_ms": 320,
            "lookahead_frames": 4,
            "lookahead_ms_label": 320,
        },
    ]
    preset = tmp_path / "mixed.json"
    preset.write_text(json.dumps(payload), encoding="utf-8")
    config = resolve(preset_path=preset, env={})
    assert config.workload.cadence_ms is None
    cadences = [entry.cadence_ms for entry in config.workload.population]
    assert cadences == [160, 320]
    with pytest.raises(ConfigError, match="edit its entries"):
        resolve(
            preset_path=preset,
            env={},
            overrides={("workload", "lanes"): 4},
        )


def test_contradictory_single_entry_preset_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(_PRESET.read_text(encoding="utf-8"))
    payload["workload"]["population"][0]["lookahead_frames"] = 9
    preset = tmp_path / "disagree.json"
    preset.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="lookahead_frames"):
        resolve(preset_path=preset, env={})


def test_non_finite_timeout_is_rejected() -> None:
    with pytest.raises(ConfigError, match="finite"):
        resolve(env={"ASRBENCH_TIMEOUT_S": "nan"})
    with pytest.raises(ConfigError, match="finite"):
        resolve(env={"ASRBENCH_TIMEOUT_S": "inf"})


def test_auth_env_is_applied_before_the_named_secret() -> None:
    secret = "named-secret"
    config = resolve(
        env={"ASRBENCH_AUTH_ENV": "ASRBENCH_KEY", "ASRBENCH_KEY": secret}
    )
    assert config.target.auth_env == "ASRBENCH_KEY"
    assert secret not in json.dumps(config_to_dict(config))


def test_from_receipt_rejects_stray_env_and_a_foreign_schema(
    tmp_path: Path,
) -> None:
    from asrbench.config import config_from_mapping

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "asrbench-receipt/0",
                "resolved_config": config_to_dict(
                    resolve(preset_path=_PRESET, env={})
                ),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="schema_version"):
        resolve(from_receipt=source, env={})
    good = resolve(preset_path=_PRESET, env={})
    receipt = tmp_path / "good.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "resolved_config": config_to_dict(good),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="ASRBENCH_WARMUP"):
        resolve(from_receipt=receipt, env={"ASRBENCH_WARMUP": "9"})
    reproduced = resolve(
        from_receipt=receipt,
        env={"ASRBENCH_TOKEN": "allowed-secret"},
        overrides={("workload", "warmup"): 9},
    )
    assert reproduced.workload.warmup == 9
    assert reproduced.reproduction_overrides
    assert "allowed-secret" not in json.dumps(config_to_dict(reproduced))
    with pytest.raises(ConfigError, match="missing"):
        config_from_mapping({"workload": {}}, complete=True)
