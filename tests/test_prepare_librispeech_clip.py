"""The LibriSpeech FLAC recipe, without the audio in the tree."""

import importlib.util
import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from asrbench.clips import pcm16_wav, silence_pcm16

_SCRIPT = Path("scripts/prepare_librispeech_clip.py")
_FLAC_SHA = (
    "sha256:26cbebbdd8954104c936dc38fbb8a80958f2169ada0fa6fe46a89cfa9e3d824c"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "prepare_librispeech_clip", _SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_entry_is_the_pinned_flac(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe = _load()
    entry = recipe.load_entry(Path("datasets.lock"), recipe.LOCK_NAME)
    assert entry["dest"] == "clips/1272-135031-0000.flac"
    assert recipe.entry_sha256(entry) == _FLAC_SHA
    assert recipe.DEFAULT_WAV.as_posix() == "clips/1272-135031-0000.wav"
    assert recipe.main(["--check-lock"]) == 0
    captured = capsys.readouterr()
    assert _FLAC_SHA in captured.out
    assert "model_name" in captured.out
    assert "GetRivaSpeechRecognitionConfig" in captured.out


def test_digest_mismatch_names_both_digests(tmp_path: Path) -> None:
    recipe = _load()
    source = tmp_path / "clip.flac"
    source.write_bytes(b"not-the-locked-flac")
    lock = tmp_path / "datasets.lock"
    lock.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": recipe.LOCK_NAME,
                        "dest": "clips/1272-135031-0000.flac",
                        "sha256": _FLAC_SHA,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(recipe.PrepareError, match="digest mismatch"):
        recipe.prepare(
            lock,
            recipe.LOCK_NAME,
            source,
            tmp_path / "out.wav",
            "ffmpeg",
        )


def test_transcode_writes_16k_pcm16_mono(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not on PATH")
    recipe = _load()
    source = tmp_path / "source.wav"
    source.write_bytes(pcm16_wav(silence_pcm16(0.02)))
    flac = tmp_path / "clip.flac"
    encoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-c:a",
            "flac",
            str(flac),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert encoded.returncode == 0, encoded.stderr
    digest = recipe.sha256_file(flac)
    lock = tmp_path / "datasets.lock"
    lock.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": recipe.LOCK_NAME,
                        "dest": "clips/1272-135031-0000.flac",
                        "sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    wav = tmp_path / "clips" / "1272-135031-0000.wav"
    found = recipe.prepare(lock, recipe.LOCK_NAME, flac, wav, ffmpeg)
    assert found == recipe.sha256_file(wav)
    assert found.startswith("sha256:")
    with wave.open(str(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getsampwidth() == 2
