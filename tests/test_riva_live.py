"""Hosted NIM check. Skipped unless NVIDIA_API_KEY is set.

Numbers from the hosted lane are not assertions. An unopened lane is a
successful result. The key is never written into an assertion message.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("grpc")
pytest.importorskip("riva.client.proto.riva_asr_pb2")

from asrbench.adapters.riva_grpc import RivaGrpcAdapter
from asrbench.clips import pcm16_wav, silence_pcm16
from asrbench.outcome import SessionRequest
from asrbench.spec import TargetSpec

_LIST_URLS = (
    "https://api.ngc.nvidia.com/v2/nvcf/functions?visibility=public",
    "https://api.nvcf.nvidia.com/v2/nvcf/functions?visibility=public",
)


def _function_id(token: str) -> str | None:
    """Resolve a public nemotron ASR function id. Bodies are not logged."""
    for url in _LIST_URLS:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
        ):
            continue
        found = _pick(payload)
        if found:
            return found
    return None


def _pick(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    functions = payload.get("functions")
    if not isinstance(functions, list):
        return None
    ranked: list[tuple[int, str]] = []
    for item in functions:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).casefold()
        identifier = item.get("id")
        if "nemotron" not in name or "asr" not in name:
            continue
        if not isinstance(identifier, str) or not identifier:
            continue
        score = 0 if "stream" in name else 1
        ranked.append((score, identifier))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][1]


def _refuses(token: str, text: str) -> None:
    if token and token in text:
        raise AssertionError("outcome contains the auth token")


@pytest.mark.live
def test_hosted_nim_validates_or_is_lane_unavailable(tmp_path: Path) -> None:
    token = os.environ.get("NVIDIA_API_KEY")
    if not token:
        pytest.skip("NVIDIA_API_KEY is absent")
    function_id = _function_id(token)
    if function_id is None:
        pytest.skip("NVCF function list did not name a nemotron ASR model")
    wav = tmp_path / "clip.wav"
    wav.write_bytes(pcm16_wav(silence_pcm16(0.32)))
    target = TargetSpec(
        endpoint_url="grpc.nvcf.nvidia.com:443",
        transport="riva-grpc",
        model="nemotron-asr-streaming",
        auth_env="NVIDIA_API_KEY",
        tls_verify=True,
        transport_options={
            "use_ssl": True,
            "function_id": function_id,
            "max_alternatives": 1,
            "enable_word_time_offsets": False,
            "custom_configuration": {},
        },
    )
    adapter = RivaGrpcAdapter.from_target(target)
    request = SessionRequest(
        endpoint_url=target.endpoint_url,
        model=target.model,
        clip_path=wav,
        label="live",
        chunk_ms=160,
        cadence_ms=160,
        timeout_s=5,
        lifecycle="full",
        pace=True,
        language="en-US",
        punctuation=True,
        interim_results=False,
    )

    import asyncio

    outcome = asyncio.run(adapter.run(request))
    blob = json.dumps(outcome.protocol_extras, default=str)
    _refuses(token, blob)
    if outcome.error_detail:
        _refuses(token, outcome.error_detail)
    allowed = {"lane_unavailable", "completed", "timed_out"}
    if outcome.terminal_status not in allowed:
        status = outcome.error_code or outcome.terminal_status
        raise AssertionError("hosted NIM status " + status)
    if outcome.terminal_status == "lane_unavailable":
        assert outcome.configured is False
        assert outcome.ttfs_s is None
        assert outcome.final_latency_s is None
        assert outcome.response_latencies_s == []
