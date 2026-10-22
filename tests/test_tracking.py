"""The index is derived and rebuilds byte-identically."""

import json
from pathlib import Path

from asrbench.tracking import build_index, write_index


def _receipt(path: Path, key: str, gpu: str) -> None:
    payload = {
        "comparison_key": key,
        "schema_version": "asrbench-receipt/1",
        "summary": {"conjunction_passed": True},
        "identity": {
            "hardware": {"gpu_name": gpu},
            "target": {"model": "mock-asr", "transport": "mock"},
            "software": {"normalization_policy": "engine_vs_own_oracle"},
        },
        "fingerprints": {
            "observation": "sha256:" + "ab" * 32,
            "workload": "sha256:" + "cd" * 32,
        },
        "resolved_config": {
            "workload": {"lookahead_frames": 4, "lookahead_ms_label": 320.0}
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_index_is_order_independent_and_rebuildable(tmp_path: Path) -> None:
    first = tmp_path / "b.json"
    second = tmp_path / "a.json"
    _receipt(first, "sha256:" + "11" * 32, "A40")
    _receipt(second, "sha256:" + "22" * 32, "H100")
    forward = build_index([first, second])
    backward = build_index([second, first])
    assert forward == backward
    assert forward.endswith("\n")
    assert "A40" in forward
    assert str(tmp_path) not in forward
    destination = tmp_path / "index.jsonl"
    assert write_index(tmp_path, destination) == write_index(
        tmp_path, destination
    )
    assert destination.read_text(encoding="utf-8") == build_index(
        sorted(tmp_path.glob("*.json"))
    )


def test_empty_directory_indexes_to_empty_text(tmp_path: Path) -> None:
    assert build_index([]) == ""
    destination = tmp_path / "index.jsonl"
    assert write_index(tmp_path, destination) == ""
