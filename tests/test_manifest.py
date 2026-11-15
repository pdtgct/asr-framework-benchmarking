"""Manifest format, content digest, verification, and clip assignment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asrbench.canonical import sha256_bytes
from asrbench.manifest import (
    MANIFEST_SCHEMA,
    ManifestError,
    assign_clip,
    load_manifest,
    materialize_manifest,
    verify_manifest,
    write_manifest,
)
from tests.corpus_fixtures import tone_pcm, write_wav


def _two_clips(root: Path, *, reference: str = "hello world") -> Path:
    clips = []
    for name, seconds in (("a", 0.25), ("b", 0.5)):
        pcm = tone_pcm(seconds)
        write_wav(root / "clips" / f"{name}.wav", pcm)
        clips.append(
            {
                "id": name,
                "audio": f"clips/{name}.wav",
                "reference": reference if name == "a" else "second clip",
                "pcm_sha256": sha256_bytes(pcm),
                "duration_s": seconds,
            }
        )
    write_manifest(
        root / "manifest.json",
        manifest_id="tiny",
        language="en-US",
        clips=clips,
        note="first",
        provenance={"builder": "test"},
    )
    return root / "manifest.json"


def test_round_trip_and_summary(tmp_path: Path) -> None:
    manifest = load_manifest(_two_clips(tmp_path))
    assert manifest.id == "tiny"
    assert [clip.id for clip in manifest.clips] == ["a", "b"]
    assert manifest.total_duration_s == pytest.approx(0.75)
    assert manifest.references() == {
        "a.wav": "hello world",
        "b.wav": "second clip",
    }
    assert manifest.digest.startswith("sha256:")
    assert json.loads((tmp_path / "manifest.json").read_text())["schema"] == (
        MANIFEST_SCHEMA
    )
    assert verify_manifest(manifest) == []


def test_digest_ignores_paths_and_notes_but_not_references(
    tmp_path: Path,
) -> None:
    first = load_manifest(_two_clips(tmp_path / "one"))
    moved = tmp_path / "two"
    _two_clips(moved)
    raw = json.loads((moved / "manifest.json").read_text())
    raw["note"] = "different"
    raw["provenance"] = {"builder": "elsewhere"}
    (moved / "clips" / "a.wav").rename(moved / "a.wav")
    raw["clips"][0]["audio"] = "a.wav"
    (moved / "manifest.json").write_text(json.dumps(raw))
    assert load_manifest(moved / "manifest.json").digest == first.digest

    other = load_manifest(_two_clips(tmp_path / "three", reference="hello"))
    assert other.digest != first.digest


def test_verify_and_materialize_catch_a_changed_clip(tmp_path: Path) -> None:
    path = _two_clips(tmp_path)
    write_wav(tmp_path / "clips" / "b.wav", tone_pcm(0.5, hz=880.0))
    manifest = load_manifest(path)
    problems = verify_manifest(manifest)
    assert len(problems) == 1 and "b" in problems[0]
    with pytest.raises(ManifestError):
        materialize_manifest(manifest, tmp_path / "out")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(schema="asrbench-manifest/0"), "schema"),
        (lambda raw: raw["clips"].append(dict(raw["clips"][0])), "duplicate"),
        (lambda raw: raw.update(clips=[]), "clip"),
        (lambda raw: raw["clips"][0].update(id="a/b"), "safe name"),
        (lambda raw: raw["clips"][0].update(pcm_sha256="abc"), "sha256"),
        (lambda raw: raw["clips"][0].update(language="de-DE"), "one language"),
        (lambda raw: raw.update(extra=1), "unknown fields"),
    ],
)
def test_malformed_manifests_are_refused(
    tmp_path: Path, mutate: object, message: str
) -> None:
    path = _two_clips(tmp_path)
    raw = json.loads(path.read_text())
    assert callable(mutate)
    mutate(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


def test_round_robin_covers_every_clip_once_per_pass() -> None:
    lanes, clips = 3, 7
    seen = [
        assign_clip(
            lane=lane, iteration=iteration, lanes=lanes, clip_count=clips
        )
        for iteration in range(3)
        for lane in range(lanes)
    ]
    assert sorted(seen[:clips]) == list(range(clips))
    # Concurrent sessions in one iteration never share a clip.
    assert len(set(seen[:lanes])) == lanes
