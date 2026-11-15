"""Tiny synthetic corpora: no network, no corpus audio."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

from asrbench.clips import pcm16_wav

RATE = 16_000


def tone_pcm(seconds: float, *, hz: float = 440.0, amp: float = 0.3) -> bytes:
    """Return PCM16 mono 16 kHz of a sine, or silence when ``amp`` is 0."""
    count = int(round(seconds * RATE))
    out = bytearray()
    for index in range(count):
        value = int(amp * 32767 * math.sin(2 * math.pi * hz * index / RATE))
        out += value.to_bytes(2, "little", signed=True)
    return bytes(out)


def speech_like_pcm(pattern: list[tuple[float, bool]]) -> bytes:
    """Concatenate tone (True) and silence (False) spans."""
    return b"".join(
        tone_pcm(seconds, amp=0.3 if loud else 0.0) for seconds, loud in pattern
    )


def write_wav(path: Path, pcm: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm16_wav(pcm))
    return path


MEETING = "TS9999a"

_MEETINGS = f"""<?xml version="1.0" encoding="UTF-8"?>
<meetings>
  <meeting observation="{MEETING}">
    <speaker nxt_agent="A" channel="0"/>
    <speaker nxt_agent="B" channel="1"/>
  </meeting>
</meetings>
"""

_NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def _words(agent: str, words: list[tuple[str, float, float, bool]]) -> str:
    body = []
    for index, (text, start, end, punc) in enumerate(words):
        extra = ' punc="true"' if punc else ""
        body.append(
            f'<w nite:id="{MEETING}.{agent}.words{index}" '
            f'starttime="{start}" endtime="{end}"{extra}>{text}</w>'
        )
    body.append(
        f'<vocalsound nite:id="{MEETING}.{agent}.words99" '
        'starttime="0.1" endtime="0.2" type="laugh"/>'
    )
    return (
        f'<?xml version="1.0"?>\n<nite:root {_NS}>{"".join(body)}</nite:root>\n'
    )


def _segments(
    agent: str, spans: list[tuple[float, float, int, int | None]]
) -> str:
    body = []
    for index, (start, end, first, last) in enumerate(spans):
        ref = f"{MEETING}.{agent}.words.xml#id({MEETING}.{agent}.words{first})"
        if last is not None:
            ref += f"..id({MEETING}.{agent}.words{last})"
        body.append(
            f'<segment nite:id="{MEETING}.sync.{agent}{index}" '
            f'transcriber_start="{start}" transcriber_end="{end}" '
            f'channel="{0 if agent == "A" else 1}">'
            f'<nite:child href="{ref}"/></segment>'
        )
    return (
        f'<?xml version="1.0"?>\n<nite:root {_NS}>{"".join(body)}</nite:root>\n'
    )


def annotation_members() -> dict[str, str]:
    """Return annotation members keyed by their path inside the zip."""
    words_a = [
        ("hello", 0.2, 0.5, False),
        ("there", 0.5, 0.9, False),
        (".", 0.9, 0.9, True),
        ("next", 2.2, 2.6, False),
        ("one", 2.6, 3.0, False),
        ("?", 3.0, 3.0, True),
    ]
    words_b = [("okay", 1.2, 1.6, False)]
    return {
        "corpusResources/meetings.xml": _MEETINGS,
        f"words/{MEETING}.A.words.xml": _words("A", words_a),
        f"words/{MEETING}.B.words.xml": _words("B", words_b),
        f"segments/{MEETING}.A.segments.xml": _segments(
            "A", [(0.1, 1.0, 0, 2), (2.1, 3.1, 3, 5), (3.5, 3.9, 99, None)]
        ),
        f"segments/{MEETING}.B.segments.xml": _segments(
            "B", [(1.1, 1.7, 0, None)]
        ),
    }


def stage_ami(root: Path, *, zipped: bool) -> tuple[Path, Path]:
    """Stage annotations and two 4 s headsets; return (audio_root, notes)."""
    audio_root = root / "audio"
    for channel in (0, 1):
        write_wav(
            audio_root / MEETING / "audio" / f"{MEETING}.Headset-{channel}.wav",
            tone_pcm(4.0, hz=300.0 + 100 * channel),
        )
    members = annotation_members()
    if zipped:
        notes = root / "ami_public_manual_1.6.2.zip"
        with zipfile.ZipFile(notes, "w") as archive:
            for name, text in members.items():
                archive.writestr(name, text)
    else:
        notes = root / "annotations"
        for name, text in members.items():
            target = notes / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    return audio_root, notes


def stage_librispeech(root: Path) -> Path:
    """Stage a two-utterance test-clean tree with decoded WAVs."""
    chapter = root / "LibriSpeech" / "test-clean" / "1089" / "134686"
    chapter.mkdir(parents=True)
    (chapter / "1089-134686.trans.txt").write_text(
        "1089-134686-0001 STUFF IT INTO YOU\n"
        "1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER\n",
        encoding="utf-8",
    )
    for utt, seconds in (("0000", 0.5), ("0001", 0.25)):
        (chapter / f"1089-134686-{utt}.flac").write_bytes(b"fLaC")
        write_wav(chapter / f"1089-134686-{utt}.wav", tone_pcm(seconds))
    return root
