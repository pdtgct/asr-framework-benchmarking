"""riva-grpc TLS options and the shared channel, against the stand-in."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("grpc")
pytest.importorskip("riva.client.proto.riva_asr_pb2")
pytest.importorskip("riva.client.proto.riva_asr_pb2_grpc")

from asrbench.adapters.riva_grpc import RivaGrpcAdapter
from asrbench.cli import main
from asrbench.clips import pcm16_wav, silence_pcm16
from asrbench.config import ConfigError
from asrbench.outcome import SessionRequest
from asrbench.spec import TargetSpec
from tests import tls_fixture
from tests.riva_fake import FakeRiva, RunningFake

needs_openssl = pytest.mark.skipif(
    not tls_fixture.available(), reason="openssl is not on PATH"
)


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "clip.wav"
    path.write_bytes(pcm16_wav(silence_pcm16(0.2)))
    return path


def _request(path: Path, timeout_s: float = 5.0) -> SessionRequest:
    return SessionRequest(
        endpoint_url="unused",
        model="fake-model",
        clip_path=path,
        label="s",
        chunk_ms=100,
        cadence_ms=100,
        timeout_s=timeout_s,
        lifecycle="full",
        pace=False,
        language="en-US",
        punctuation=False,
        diarization=False,
        interim_results=False,
        workload_mode="streaming",
    )


def _adapter(
    endpoint: str, *, tls_verify: bool = True, **options: Any
) -> RivaGrpcAdapter:
    return RivaGrpcAdapter.from_target(
        TargetSpec(
            endpoint_url=endpoint,
            transport="riva-grpc",
            model="fake-model",
            tls_verify=tls_verify,
            transport_options=options,
        )
    )


def test_tls_options_are_validated(tmp_path: Path) -> None:
    pem = tmp_path / "x.pem"
    pem.write_text("not really", encoding="utf-8")
    with pytest.raises(ConfigError, match="needs use_ssl"):
        _adapter("h:1", tls_verify=False, use_ssl=False, ssl_root_cert=str(pem))
    with pytest.raises(ConfigError, match="needs use_ssl"):
        _adapter(
            "h:1",
            tls_verify=False,
            use_ssl=False,
            ssl_target_name_override="nim",
        )
    with pytest.raises(ConfigError, match="set together"):
        _adapter("h:1", use_ssl=True, ssl_client_cert=str(pem))
    with pytest.raises(ConfigError, match="cannot read"):
        _adapter("h:1", use_ssl=True, ssl_root_cert=str(tmp_path / "nope"))
    with pytest.raises(ConfigError, match="share_channel must be a bool"):
        _adapter("h:1", tls_verify=False, share_channel="yes")
    adapter = _adapter(
        "h:1",
        use_ssl=True,
        ssl_root_cert=str(pem),
        ssl_target_name_override="nim.site.internal",
    )
    assert adapter._tls_material == {"ssl_root_cert": b"not really"}
    assert (
        "grpc.ssl_target_name_override",
        "nim.site.internal",
    ) in adapter._channel_args
    assert "not really" not in repr(adapter._channel_args)


@needs_openssl
def test_private_ca_and_target_name_override(tmp_path: Path) -> None:
    site = tls_fixture.make_site_tls(tmp_path)
    tls = (site.server_key.read_bytes(), site.server_cert.read_bytes())
    server = RunningFake(FakeRiva(mode="final"), tls=tls)
    try:
        # The server certificate names nim.site.internal, not 127.0.0.1.
        good = asyncio.run(
            _adapter(
                server.endpoint,
                use_ssl=True,
                ssl_root_cert=str(site.ca),
                ssl_target_name_override=tls_fixture.SERVER_NAME,
            ).run(_request(_wav(tmp_path)))
        )
        no_override = asyncio.run(
            _adapter(
                server.endpoint, use_ssl=True, ssl_root_cert=str(site.ca)
            ).run(_request(_wav(tmp_path), timeout_s=1.0))
        )
    finally:
        server.stop()
    assert good.terminal_status == "completed"
    assert good.transcript == "HELLO"
    assert no_override.terminal_status == "lane_unavailable"


@needs_openssl
def test_mutual_tls_needs_the_client_certificate(tmp_path: Path) -> None:
    site = tls_fixture.make_site_tls(tmp_path)
    tls = (site.server_key.read_bytes(), site.server_cert.read_bytes())
    server = RunningFake(
        FakeRiva(mode="final"), tls=tls, client_ca=site.ca.read_bytes()
    )
    base: dict[str, Any] = {
        "use_ssl": True,
        "ssl_root_cert": str(site.ca),
        "ssl_target_name_override": tls_fixture.SERVER_NAME,
    }
    try:
        with_cert = asyncio.run(
            _adapter(
                server.endpoint,
                ssl_client_cert=str(site.client_cert),
                ssl_client_key=str(site.client_key),
                **base,
            ).run(_request(_wav(tmp_path)))
        )
        without = asyncio.run(
            _adapter(server.endpoint, **base).run(
                _request(_wav(tmp_path), timeout_s=1.0)
            )
        )
    finally:
        server.stop()
    assert with_cert.terminal_status == "completed"
    assert without.terminal_status == "lane_unavailable"
    key_text = site.client_key.read_text(encoding="utf-8")
    assert key_text.splitlines()[1] not in json.dumps(with_cert.protocol_extras)


def test_shared_channel_is_reused_across_sessions(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="final"))
    opened: list[object] = []
    adapter = _adapter(server.endpoint, tls_verify=False, share_channel=True)
    original = adapter._open_channel

    def counting(grpc: Any) -> Any:
        channel = original(grpc)
        opened.append(channel)
        return channel

    adapter._open_channel = counting  # type: ignore[method-assign]
    path = _wav(tmp_path)

    async def three() -> list[Any]:
        outcomes = await asyncio.gather(
            *(adapter.run(_request(path)) for _ in range(3))
        )
        await adapter.aclose()
        return list(outcomes)

    try:
        outcomes = asyncio.run(three())
    finally:
        server.stop()
    assert len(opened) == 1
    assert all(item.terminal_status == "completed" for item in outcomes)
    assert all(item.protocol_extras["channel"] == "shared" for item in outcomes)
    assert adapter._shared is None


def test_per_session_channel_is_the_default(tmp_path: Path) -> None:
    server = RunningFake(FakeRiva(mode="final"))
    try:
        outcome = asyncio.run(
            _adapter(server.endpoint, tls_verify=False).run(
                _request(_wav(tmp_path))
            )
        )
    finally:
        server.stop()
    assert outcome.protocol_extras["channel"] == "per_session"


def test_shared_channel_through_the_runner(tmp_path: Path) -> None:
    payload = json.loads(
        Path("presets/nim-compat-streaming.json").read_text(encoding="utf-8")
    )
    server = RunningFake(FakeRiva(mode="per-chunk-final-last"))
    payload["workload"]["clip"] = {
        "id": "shared-channel-silence",
        "generator": "pcm16-silence",
        "duration_s": 0.32,
        "sample_rate": 16000,
        "licence": "CC0-1.0",
        "attribution": "Synthetic silence generated by asrbench; no corpus.",
        "note": "In-process stand-in.",
    }
    payload["workload"]["timeout_s"] = 5
    payload["target"]["endpoint_url"] = server.endpoint
    payload["target"]["auth_env"] = None
    payload["target"]["tls_verify"] = False
    payload["target"]["transport_options"]["use_ssl"] = False
    payload["target"]["transport_options"]["share_channel"] = True
    preset = tmp_path / "p.json"
    preset.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "r.json"
    try:
        code = main(
            [
                "run",
                "--preset",
                str(preset),
                "--out",
                str(out),
                "--lanes",
                "4",
                "--no-pacing",
            ]
        )
    finally:
        server.stop()
    assert code == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    counts = receipt["summary"]["pooled"]["counts"]
    assert counts["completed"] == counts["attempted"] == 36
    channels = {
        outcome["protocol_extras"]["channel"]
        for trial in receipt["trials"]
        for outcome in trial["outcomes"]
    }
    assert channels == {"shared"}
    options = receipt["resolved_config"]["target"]["transport_options"]
    assert options["share_channel"] is True
