"""Corpus manifests: many clips, each with its own reference transcript.

A manifest is a JSON file naming clips by local path, PCM16 digest, and
reference text. It turns one receipt into a corpus run: every measured
session plays one manifest clip and is scored against that clip's own
reference, so corpus WER comes out of one run instead of hand-pooled
receipts.

Identity is content, not location. The manifest digest covers the
schema, the language, and each clip's id, PCM16 digest, and reference
digest, in order. Paths, notes, and provenance are not covered: moving a
staged corpus to another disk does not change what was offered, and an
edited reference does. The digest is part of the workload fingerprint,
so two receipts whose manifests differ by one word refuse to compare.

Everything here is stdlib and reads local files only. Nothing fetches.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asrbench.canonical import fingerprint, sha256_bytes, sha256_text
from asrbench.clips import pcm16_wav

#: Manifest file format written by :mod:`asrbench.corpora` builders.
MANIFEST_SCHEMA = "asrbench-manifest/1"

#: Clip ids become file names and receipt labels.
_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_TOP_FIELDS = {
    "schema",
    "id",
    "language",
    "corpus",
    "licence",
    "attribution",
    "note",
    "provenance",
    "clips",
}
_CLIP_FIELDS = {
    "id",
    "audio",
    "reference",
    "pcm_sha256",
    "duration_s",
    "language",
    "source",
}


class ManifestError(ValueError):
    """The manifest is malformed, or its audio does not match its pins."""


@dataclass(frozen=True)
class ManifestClip:
    """One clip: where its audio is, what it should hash to, what was said."""

    id: str
    audio: Path
    reference: str
    pcm_sha256: str
    duration_s: float

    @property
    def wav_name(self) -> str:
        """File name the runner materializes, and outcomes carry."""
        return f"{self.id}.wav"


@dataclass(frozen=True)
class Manifest:
    """A loaded manifest. Clip order is significant: it is the assignment."""

    path: Path
    id: str
    language: str
    corpus: str | None
    licence: str | None
    attribution: str | None
    clips: tuple[ManifestClip, ...]

    @property
    def digest(self) -> str:
        """Content digest; see :func:`manifest_digest`."""
        return manifest_digest(self.language, self.clips)

    @property
    def total_duration_s(self) -> float:
        """Sum of clip durations, in seconds."""
        return sum(clip.duration_s for clip in self.clips)

    def references(self) -> dict[str, str]:
        """Return materialized WAV name to reference text."""
        return {clip.wav_name: clip.reference for clip in self.clips}


def manifest_digest(language: str, clips: tuple[ManifestClip, ...]) -> str:
    """Hash what a manifest offers and what it scores against.

    Args:
        language: The manifest language tag.
        clips: Clips in manifest order.

    Returns:
        A prefixed sha256 over canonical JSON of the schema, language,
        and per clip ``id``, ``pcm_sha256``, and ``reference_sha256``.
    """
    return fingerprint(
        {
            "schema": MANIFEST_SCHEMA,
            "language": language,
            "clips": [
                {
                    "id": clip.id,
                    "pcm_sha256": clip.pcm_sha256,
                    "reference_sha256": sha256_text(clip.reference),
                }
                for clip in clips
            ],
        }
    )


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def _optional_string(value: object, where: str) -> str | None:
    if value is None:
        return None
    return _string(value, where)


def load_manifest(path: Path) -> Manifest:
    """Read and structurally validate a manifest. Audio is not read.

    Args:
        path: The manifest JSON. Relative clip paths resolve against its
            directory.

    Returns:
        The manifest.

    Raises:
        ManifestError: The file is missing, malformed, or has duplicate,
            unsafe, or unpinned clips.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"{path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"{path}: expected a JSON object")
    extra = sorted(set(payload) - _TOP_FIELDS)
    if extra:
        raise ManifestError(f"{path}: unknown fields {', '.join(extra)}")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"{path}: schema {payload.get('schema')!r} != {MANIFEST_SCHEMA}"
        )
    manifest_id = _string(payload.get("id"), "manifest id")
    if not _CLIP_ID.match(manifest_id):
        raise ManifestError(f"manifest id {manifest_id!r} is not a safe name")
    language = _string(payload.get("language"), "manifest language")
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ManifestError(f"{path}: clips must be a non-empty list")
    base = path.parent
    clips: list[ManifestClip] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_clips):
        where = f"clips[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{where} must be an object")
        extra = sorted(set(item) - _CLIP_FIELDS)
        if extra:
            raise ManifestError(f"{where}: unknown fields {', '.join(extra)}")
        clip_id = _string(item.get("id"), f"{where}.id")
        if not _CLIP_ID.match(clip_id):
            raise ManifestError(f"{where}.id {clip_id!r} is not a safe name")
        if clip_id in seen:
            raise ManifestError(f"duplicate clip id {clip_id!r}")
        seen.add(clip_id)
        clip_language = item.get("language")
        if clip_language is not None and clip_language != language:
            raise ManifestError(
                f"{where}.language {clip_language!r} != manifest language "
                f"{language!r}; one manifest is one language"
            )
        reference = item.get("reference")
        if not isinstance(reference, str):
            raise ManifestError(f"{where}.reference must be a string")
        digest = _string(item.get("pcm_sha256"), f"{where}.pcm_sha256")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ManifestError(f"{where}.pcm_sha256 must be sha256:<hex>")
        duration = item.get("duration_s")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            raise ManifestError(f"{where}.duration_s must be positive")
        audio = Path(_string(item.get("audio"), f"{where}.audio"))
        clips.append(
            ManifestClip(
                id=clip_id,
                audio=audio if audio.is_absolute() else base / audio,
                reference=reference,
                pcm_sha256=digest,
                duration_s=float(duration),
            )
        )
    return Manifest(
        path=path,
        id=manifest_id,
        language=language,
        corpus=_optional_string(payload.get("corpus"), "corpus"),
        licence=_optional_string(payload.get("licence"), "licence"),
        attribution=_optional_string(payload.get("attribution"), "attribution"),
        clips=tuple(clips),
    )


def read_clip_pcm(clip: ManifestClip) -> bytes:
    """Read one clip and check it against its pinned digest and duration.

    Args:
        clip: A manifest clip.

    Returns:
        PCM16 mono 16 kHz payload.

    Raises:
        ManifestError: The file is unreadable, not 16 kHz mono PCM16, or
            does not hash to ``pcm_sha256``.
    """
    import wave

    try:
        with wave.open(str(clip.audio), "rb") as handle:
            shape = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
            )
            pcm = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise ManifestError(f"{clip.id}: {clip.audio}: {error}") from error
    if shape != (1, BYTES_PER_SAMPLE, SAMPLE_RATE):
        raise ManifestError(
            f"{clip.id}: need mono PCM16 {SAMPLE_RATE} Hz, got "
            f"{shape[0]}ch/{shape[1] * 8}bit/{shape[2]}Hz"
        )
    digest = sha256_bytes(pcm)
    if digest != clip.pcm_sha256:
        raise ManifestError(
            f"{clip.id}: payload {digest} != pinned {clip.pcm_sha256}"
        )
    seconds = len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
    if abs(seconds - clip.duration_s) > 1.0 / SAMPLE_RATE:
        raise ManifestError(
            f"{clip.id}: audio is {seconds:.6f}s, manifest says "
            f"{clip.duration_s:.6f}s"
        )
    return pcm


def verify_manifest(manifest: Manifest) -> list[str]:
    """Check every clip's audio. Collects problems instead of stopping.

    Args:
        manifest: A loaded manifest.

    Returns:
        One message per clip that failed. Empty means every pin holds.
    """
    problems: list[str] = []
    for clip in manifest.clips:
        try:
            read_clip_pcm(clip)
        except ManifestError as error:
            problems.append(str(error))
    return problems


def materialize_manifest(manifest: Manifest, dest_dir: Path) -> dict[str, Path]:
    """Write every clip as ``<id>.wav`` under ``dest_dir``.

    The runner sends the materialized copy, never the staged file, so a
    clip is re-read and re-checked against its digest on every run.

    Args:
        manifest: A loaded manifest.
        dest_dir: Directory created if needed.

    Returns:
        Clip id to materialized WAV path.

    Raises:
        ManifestError: A clip fails its pin.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for clip in manifest.clips:
        target = dest_dir / clip.wav_name
        target.write_bytes(pcm16_wav(read_clip_pcm(clip)))
        written[clip.id] = target
    return written


def assign_clip(
    *, lane: int, iteration: int, lanes: int, clip_count: int
) -> int:
    """Return the manifest index one session plays.

    Round robin over the global session index ``iteration * lanes +
    lane``. Sessions running at the same moment play different clips.
    Every clip is played when ``lanes * iterations`` is at least the clip
    count; the corpus WER block records whether that held (``complete``)
    and whether every clip was played equally often (``balanced``). Each
    trial replays the same assignment.

    Args:
        lane: Lane index, from zero.
        iteration: Iteration index within its phase, from zero.
        lanes: Lanes in the trial.
        clip_count: Clips in the manifest.

    Returns:
        A clip index in ``range(clip_count)``.
    """
    return (iteration * lanes + lane) % clip_count


def write_manifest(
    path: Path,
    *,
    manifest_id: str,
    language: str,
    clips: list[dict[str, Any]],
    corpus: str | None = None,
    licence: str | None = None,
    attribution: str | None = None,
    note: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> Manifest:
    """Write a manifest, then load it back so the caller gets its digest.

    Args:
        path: Destination JSON.
        manifest_id: Safe name.
        language: Language tag, such as ``en-US``.
        clips: Clip objects in :data:`MANIFEST_SCHEMA` shape.
        corpus: Corpus name, for humans.
        licence: SPDX-style licence of the derived audio.
        attribution: Text a published result must carry.
        note: Free text. Not identity.
        provenance: Builder inputs. Not identity.

    Returns:
        The manifest as :func:`load_manifest` reads it.
    """
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "id": manifest_id,
        "language": language,
        "corpus": corpus,
        "licence": licence,
        "attribution": attribution,
        "note": note,
        "provenance": provenance or {},
        "clips": clips,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return load_manifest(path)
