"""NIM identity from /v1/metadata on the receipt (issue #14)."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from asrbench.cli import main
from asrbench.compare import compare_receipts
from asrbench.config import ConfigError
from asrbench.nim import load_nim_metadata, nim_overrides
from asrbench.reporters import nvidia_streaming_row

METADATA: dict[str, Any] = {
    "version": "1.4.0",
    "selectedModelProfileId": "a1b2c3d4",
    "modelInfo": [
        {"modelUrl": "ngc://nim/nvidia/parakeet-1-1b-ctc-en-us:1.4.0"},
        {"modelUrl": "ngc://nim/nvidia/punctuation:1.0"},
        {"shortName": "no url"},
    ],
}


def test_overrides_map_version_profile_and_model_urls() -> None:
    assert nim_overrides(METADATA) == {
        ("target", "engine_name"): "nim",
        ("target", "engine_version"): "1.4.0",
        ("target", "nim_profile_id"): "a1b2c3d4",
        ("target", "nim_model_url"): (
            "ngc://nim/nvidia/parakeet-1-1b-ctc-en-us:1.4.0,"
            "ngc://nim/nvidia/punctuation:1.0"
        ),
    }
    assert nim_overrides({"version": " ", "modelInfo": "x"}) == {}


def test_load_from_file_rejects_bad_documents(tmp_path: Path) -> None:
    good = tmp_path / "meta.json"
    good.write_text(json.dumps(METADATA), encoding="utf-8")
    assert load_nim_metadata(str(good)) == METADATA
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not JSON"):
        load_nim_metadata(str(bad))
    bad.write_text("[1]", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a JSON object"):
        load_nim_metadata(str(bad))
    with pytest.raises(ConfigError, match="cannot read"):
        load_nim_metadata(str(tmp_path / "missing.json"))


@pytest.fixture
def metadata_server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/v1/metadata":
                self.send_error(404)
                return
            body = json.dumps(METADATA).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_load_over_http_from_base_or_full_url(metadata_server: str) -> None:
    assert load_nim_metadata(metadata_server) == METADATA
    assert load_nim_metadata(metadata_server + "/v1/metadata/") == METADATA
    with pytest.raises(ConfigError, match="cannot read NIM metadata"):
        load_nim_metadata(metadata_server + "/nope")


def _run(tmp_path: Path, name: str, extra: list[str]) -> dict[str, Any]:
    out = tmp_path / f"{name}.json"
    code = main(
        [
            "run",
            "--preset",
            "presets/mock-streaming.json",
            "--no-pacing",
            "--out",
            str(out),
            *extra,
        ]
    )
    assert code == 0
    receipt: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return receipt


def test_run_records_nim_identity(tmp_path: Path, metadata_server: str) -> None:
    receipt = _run(
        tmp_path,
        "nim",
        [
            "--nim-metadata",
            metadata_server,
            "--nim-tags-selector",
            "name=parakeet-1-1b-ctc-en-us,mode=str",
            "--engine-version",
            "explicit-wins",
        ],
    )
    target = receipt["resolved_config"]["target"]
    assert target["engine_name"] == "nim"
    assert target["engine_version"] == "explicit-wins"
    assert target["nim_profile_id"] == "a1b2c3d4"
    assert (
        target["nim_tags_selector"] == "name=parakeet-1-1b-ctc-en-us,mode=str"
    )
    recorded = receipt["identity"]["recorded"]
    assert recorded["nim"] == {
        "model_url": target["nim_model_url"],
        "profile_id": "a1b2c3d4",
        "tags_selector": "name=parakeet-1-1b-ctc-en-us,mode=str",
    }
    assert recorded["engine"]["version"] == "explicit-wins"
    row = nvidia_streaming_row(receipt, "nim.json")
    assert row["nim_profile_id"] == "a1b2c3d4"
    assert row["engine_version"] == "explicit-wins"
    assert row["nim_tags_selector"].startswith("name=parakeet")


def test_nim_identity_is_recorded_not_hashed(
    tmp_path: Path, metadata_server: str
) -> None:
    plain = _run(tmp_path, "plain", [])
    nim = _run(tmp_path, "nim", ["--nim-metadata", metadata_server])
    result = compare_receipts(plain, nim)
    assert result.status == "comparable"
    rows = {item.field: item for item in result.recorded}
    assert rows["nim"].left["profile_id"] is None
    assert rows["nim"].right["profile_id"] == "a1b2c3d4"


def test_receipt_without_nim_fields_still_compares(tmp_path: Path) -> None:
    left = _run(tmp_path, "left", [])
    right = json.loads(json.dumps(left))
    # A receipt written before the NIM fields existed has neither key.
    del right["identity"]["recorded"]["nim"]
    for key in ("nim_profile_id", "nim_model_url", "nim_tags_selector"):
        del right["resolved_config"]["target"][key]
    result = compare_receipts(left, right)
    assert result.status == "comparable", result.differences
    assert nvidia_streaming_row(right, "old.json")["nim_profile_id"] is None


def test_blank_tags_selector_is_refused(tmp_path: Path) -> None:
    preset = json.loads(
        Path("presets/mock-streaming.json").read_text(encoding="utf-8")
    )
    preset["target"]["nim_tags_selector"] = " "
    path = tmp_path / "p.json"
    path.write_text(json.dumps(preset), encoding="utf-8")
    code = main(
        ["run", "--preset", str(path), "--out", str(tmp_path / "x.json")]
    )
    assert code == 2


def test_bad_metadata_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run",
            "--preset",
            "presets/mock-streaming.json",
            "--nim-metadata",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "x.json"),
        ]
    )
    assert code == 2
    assert "cannot read NIM metadata" in capsys.readouterr().err
