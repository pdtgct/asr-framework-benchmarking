"""LibriSpeech: utterances and their ``.trans.txt`` references as a manifest.

Ported in shape from the author's LibriSpeech staging (nemotron-omni-port
work), reading
an extracted tree that is already on disk. Nothing fetches, so this runs
air-gapped once ``<split>.tar.gz`` from <https://www.openslr.org/12> is
staged and extracted.

The reference is the corpus transcript, verbatim from
``<spk>/<chap>/<spk>-<chap>.trans.txt``: uppercase, unpunctuated, numbers
spelled out. That is the text the Open ASR Leaderboard and NVIDIA's
LibriSpeech rows score against, so WER here is comparable to theirs
within one normalizer.

Audio: LibriSpeech ships FLAC, which the standard library cannot decode,
and the package does not shell out. ``scripts/decode_librispeech.py``
writes ``<utt>.wav`` (16 kHz mono PCM16) beside each FLAC with ffmpeg,
the same transcode ``scripts/prepare_librispeech_clip.py`` uses; this
builder reads those WAVs. The manifest pins the PCM16 digest, so two
hosts that decode differently disagree loudly instead of silently.

LibriSpeech is CC BY 4.0; a published result must carry
:data:`ATTRIBUTION`.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asrbench.canonical import sha256_bytes
from asrbench.clips import pcm16_wav
from asrbench.corpora.ami import BuildResult
from asrbench.manifest import ManifestError, write_manifest

SPLITS = ("dev-clean", "dev-other", "test-clean", "test-other")
SOURCE_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "LibriSpeech ASR corpus, CC BY 4.0. V. Panayotov, G. Chen, D. Povey, "
    "S. Khudanpur, 'Librispeech: An ASR corpus based on public domain "
    "audio books,' ICASSP 2015. https://www.openslr.org/12"
)


class LibriSpeechError(ManifestError):
    """The staged tree is incomplete or a file cannot be decoded."""


def parse_transcripts(path: Path) -> dict[str, str]:
    """Read one ``.trans.txt``: ``<utt_id> <TEXT>`` per line.

    Raises:
        LibriSpeechError: A line has no id or no text.
    """
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, _, text = line.strip().partition(" ")
        if not utt or not text.strip():
            raise LibriSpeechError(f"{path}: malformed line {line!r}")
        found[utt] = text.strip()
    return found


def _split_dir(root: Path, split: str) -> Path:
    for candidate in (root / split, root / "LibriSpeech" / split):
        if candidate.is_dir():
            return candidate
    raise LibriSpeechError(
        f"{split} is not extracted under {root} "
        f"(stage {SOURCE_URL.format(split=split)} on a connected host)"
    )


def _decoded(flac: Path) -> bytes:
    wav = flac.with_suffix(".wav")
    if not wav.is_file():
        raise LibriSpeechError(
            f"{flac.name}: no {wav.name} beside it; run "
            "scripts/decode_librispeech.py on the split first"
        )
    return _read_wav(wav)


def _read_wav(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as handle:
            shape = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
            )
            pcm = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise LibriSpeechError(f"{path}: {error}") from error
    if shape != (1, BYTES_PER_SAMPLE, SAMPLE_RATE):
        raise LibriSpeechError(f"{path}: need mono PCM16 {SAMPLE_RATE} Hz")
    return pcm


def build(
    *,
    root: Path,
    split: str,
    out_dir: Path,
    manifest_id: str | None = None,
    limit: int | None = None,
    speakers: list[str] | None = None,
) -> BuildResult:
    """Write one clip per utterance of ``split`` and a manifest.

    Utterances are ordered by speaker, chapter, and utterance id, as the
    corpus lists them, so the same tree always yields the same digest.

    Args:
        root: Directory holding ``<split>/`` or ``LibriSpeech/<split>/``.
        split: One of :data:`SPLITS`.
        out_dir: Receives ``clips/*.wav`` and ``manifest.json``.
        manifest_id: Defaults to ``librispeech-<split>``, with a
            ``-first<N>`` suffix when ``limit`` truncates.
        limit: Keep the first ``N`` utterances. Smoke runs only; the
            published WER is over the whole split.
        speakers: Keep only these speaker ids.

    Returns:
        The manifest and warnings.

    Raises:
        LibriSpeechError: Missing split, transcript, or audio.
    """
    if split not in SPLITS:
        raise LibriSpeechError(f"unknown split {split!r}; one of {SPLITS}")
    split_dir = _split_dir(root, split)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    wanted = set(speakers or [])
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        speaker = trans.parent.parent.name
        if wanted and speaker not in wanted:
            continue
        for utt, text in sorted(parse_transcripts(trans).items()):
            if limit is not None and len(clips) >= limit:
                break
            flac = trans.parent / f"{utt}.flac"
            if not flac.is_file() and not flac.with_suffix(".wav").is_file():
                raise LibriSpeechError(f"{utt}: transcript without audio")
            pcm = _decoded(flac)
            (clips_dir / f"{utt}.wav").write_bytes(pcm16_wav(pcm))
            clips.append(
                {
                    "id": utt,
                    "audio": f"clips/{utt}.wav",
                    "reference": text,
                    "pcm_sha256": sha256_bytes(pcm),
                    "duration_s": len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE,
                    "source": {
                        "split": split,
                        "speaker": speaker,
                        "chapter": trans.parent.name,
                    },
                }
            )
    if not clips:
        raise LibriSpeechError(f"no utterances found under {split_dir}")
    if limit is not None:
        warnings.append(
            f"kept the first {len(clips)} utterances; not the full split"
        )
    default_id = f"librispeech-{split}"
    if limit is not None:
        default_id += f"-first{len(clips)}"
    manifest = write_manifest(
        out_dir / "manifest.json",
        manifest_id=manifest_id or default_id,
        language="en-US",
        clips=clips,
        corpus="LibriSpeech ASR corpus",
        licence=LICENCE,
        attribution=ATTRIBUTION,
        note=(
            "Corpus transcripts verbatim. Utterances are pre-trimmed read "
            "speech; do not use them for endpointing."
        ),
        provenance={
            "builder": "asrbench.corpora.librispeech.build",
            "split": split,
            "source": SOURCE_URL.format(split=split),
            "limit": limit,
            "speakers": sorted(wanted) or None,
        },
    )
    return BuildResult(manifest=manifest, warnings=warnings)
