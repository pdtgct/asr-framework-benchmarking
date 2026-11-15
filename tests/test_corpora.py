"""AMI and LibriSpeech manifest builders over tiny synthetic trees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asrbench.corpora import ami, librispeech
from asrbench.manifest import load_manifest, verify_manifest
from tests.corpus_fixtures import (
    MEETING,
    speech_like_pcm,
    stage_ami,
    stage_librispeech,
)


@pytest.mark.parametrize("zipped", [False, True])
def test_ami_segments_carry_corpus_words(tmp_path: Path, zipped: bool) -> None:
    audio_root, notes = stage_ami(tmp_path, zipped=zipped)
    result = ami.build_segments(
        meetings=[MEETING],
        audio_root=audio_root,
        annotations=notes,
        out_dir=tmp_path / "out",
        manifest_id="ami-tiny",
    )
    manifest = result.manifest
    assert [clip.reference for clip in manifest.clips] == [
        "hello there.",
        "next one?",
        "okay",
    ]
    assert manifest.clips[0].id == "ami-ts9999a-h0-sync.A0"
    assert manifest.clips[0].duration_s == pytest.approx(0.9)
    assert manifest.licence == ami.LICENCE
    assert any("no words" in line for line in result.warnings)
    assert verify_manifest(load_manifest(manifest.path)) == []
    provenance = json.loads(manifest.path.read_text())["provenance"]
    assert (provenance["annotations_sha256"] is not None) is zipped


def test_ami_segments_one_channel_and_cap(tmp_path: Path) -> None:
    audio_root, notes = stage_ami(tmp_path, zipped=False)
    result = ami.build_segments(
        meetings=[f"{MEETING}:1"],
        audio_root=audio_root,
        annotations=notes,
        out_dir=tmp_path / "out",
        manifest_id="ami-b",
        max_clips=1,
    )
    assert [clip.reference for clip in result.manifest.clips] == ["okay"]


def test_ami_unknown_meeting_is_an_error(tmp_path: Path) -> None:
    audio_root, notes = stage_ami(tmp_path, zipped=False)
    with pytest.raises(ami.AmiError, match="meetings.xml"):
        ami.build_segments(
            meetings=["XX0000a"],
            audio_root=audio_root,
            annotations=notes,
            out_dir=tmp_path / "out",
            manifest_id="x",
        )


def _recipe(tmp_path: Path, **pin: str) -> Path:
    recipe = {
        "id": "ami-tiny-excerpts",
        "excerpts": [
            {
                "name": "tiny-a",
                "meeting": MEETING,
                "channel": 0,
                "start_s": 0.0,
                "duration_s": 2.0,
                "role": "short",
                **pin,
            }
        ],
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


def test_ami_excerpts_window_words_and_pins(tmp_path: Path) -> None:
    audio_root, notes = stage_ami(tmp_path, zipped=True)
    result = ami.build_excerpts(
        recipe_spec=str(_recipe(tmp_path)),
        audio_root=audio_root,
        annotations=notes,
        out_dir=tmp_path / "out",
    )
    clip = result.manifest.clips[0]
    assert result.manifest.id == "ami-tiny-excerpts"
    assert clip.reference == "hello there."
    # A steady tone has no pauses: the builder says so instead of failing.
    assert any("pause" in line for line in result.warnings)
    source = json.loads(result.manifest.path.read_text())["clips"][0]["source"]

    pinned = _recipe(tmp_path, file_sha256=source["file_sha256"])
    ami.build_excerpts(
        recipe_spec=str(pinned),
        audio_root=audio_root,
        annotations=notes,
        out_dir=tmp_path / "again",
    )
    wrong = _recipe(tmp_path, file_sha256="sha256:" + "0" * 64)
    with pytest.raises(ami.AmiError, match="file_sha256"):
        ami.build_excerpts(
            recipe_spec=str(wrong),
            audio_root=audio_root,
            annotations=notes,
            out_dir=tmp_path / "bad",
        )


def test_builtin_recipe_is_the_omni_port_set() -> None:
    recipe, digest = ami.load_recipe("ami-ihm-excerpts-v1")
    assert digest.startswith("sha256:")
    assert len(recipe["excerpts"]) == 11
    assert all(item["file_sha256"] for item in recipe["excerpts"])
    urls = ami.mirror_urls(ami.recipe_channels(recipe))
    assert urls[0] == ami.ANNOTATIONS_URL
    assert all(url.startswith(ami.MIRROR) for url in urls[1:])


def test_join_words_attaches_punctuation() -> None:
    words = [
        ami.Word("1", 0.0, 0.1, "so", False),
        ami.Word("2", 0.1, 0.1, ",", True),
        ami.Word("3", 0.2, 0.3, "yes", False),
    ]
    assert ami.join_words(words) == "so, yes"


def test_window_words_assigns_by_midpoint() -> None:
    words = [
        ami.Word("1", 0.0, 1.0, "in", False),
        ami.Word("2", 1.8, 2.4, "edge", False),
        ami.Word("3", 2.2, 3.0, "out", False),
    ]
    kept = ami.window_words(words, 0.0, 2.0)
    assert [word.text for word in kept.words] == ["in"]
    assert kept.boundary_words == 1


def test_propose_offsets_finds_pause_rich_windows() -> None:
    pcm = speech_like_pcm(
        [(1.0, True), (0.9, False), (1.0, True), (0.9, False), (1.0, True)] * 3
    )
    quiet, runs = ami.silence_runs(pcm)
    assert sum(1 for _, length in runs if length >= 800) >= 4
    offsets = ami.propose_offsets(
        quiet, runs, window_s=4.8, min_silence_ms=800, want=2
    )
    assert offsets


def test_librispeech_references_are_verbatim(tmp_path: Path) -> None:
    root = stage_librispeech(tmp_path / "ls")
    result = librispeech.build(
        root=root, split="test-clean", out_dir=tmp_path / "out"
    )
    manifest = result.manifest
    assert manifest.id == "librispeech-test-clean"
    assert [clip.id for clip in manifest.clips] == [
        "1089-134686-0000",
        "1089-134686-0001",
    ]
    assert manifest.clips[0].reference == (
        "HE HOPED THERE WOULD BE STEW FOR DINNER"
    )
    assert result.warnings == []

    first = librispeech.build(
        root=root, split="test-clean", out_dir=tmp_path / "one", limit=1
    )
    assert first.manifest.id == "librispeech-test-clean-first1"
    assert first.warnings


def test_librispeech_needs_decoded_wavs(tmp_path: Path) -> None:
    root = stage_librispeech(tmp_path / "ls")
    next(root.rglob("*-0000.wav")).unlink()
    with pytest.raises(
        librispeech.LibriSpeechError, match="decode_librispeech"
    ):
        librispeech.build(root=root, split="test-clean", out_dir=tmp_path / "o")
    with pytest.raises(librispeech.LibriSpeechError, match="not extracted"):
        librispeech.build(root=root, split="dev-other", out_dir=tmp_path / "o")


def test_cli_builds_and_verifies_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from asrbench.cli import main

    audio_root, notes = stage_ami(tmp_path, zipped=True)
    out = tmp_path / "out"
    assert (
        main(
            [
                "corpus",
                "ami-segments",
                "--audio-root",
                str(audio_root),
                "--annotations",
                str(notes),
                "--out-dir",
                str(out),
                "--meetings",
                MEETING,
            ]
        )
        == 0
    )
    assert "ami-ihm-custom-segments" in capsys.readouterr().out
    assert main(["corpus", "verify", str(out / "manifest.json")]) == 0
    (out / "clips" / "ami-ts9999a-h1-sync.B0.wav").write_bytes(b"RIFF")
    assert main(["corpus", "verify", str(out / "manifest.json")]) == 1
