#!/usr/bin/env python3
"""Decode an extracted LibriSpeech split to WAV beside each FLAC.

``asrbench corpus librispeech`` reads ``<utt>.wav`` (16 kHz mono PCM16)
next to each ``<utt>.flac``; the package itself does not shell out. Run
this once per split on the staging host, with ffmpeg on PATH:

    uv run python scripts/decode_librispeech.py LibriSpeech/test-clean

Existing WAVs are kept, so a re-run only fills gaps. The decoded audio is
derived from LibriSpeech (CC BY 4.0) and is never committed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def decode(ffmpeg: str, flac: Path) -> str | None:
    """Write ``flac``'s WAV; return an error line or ``None``."""
    wav = flac.with_suffix(".wav")
    if wav.is_file():
        return None
    partial = wav.with_suffix(".partial.wav")
    command = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(flac),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(partial),
    ]
    done = subprocess.run(command, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        return f"{flac}: {done.stderr.strip() or 'ffmpeg failed'}"
    partial.replace(wav)
    return None


def main() -> int:
    """Decode every FLAC under the given split directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", nargs="+", type=Path)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not args.ffmpeg:
        print("ffmpeg is required (pass --ffmpeg PATH)", file=sys.stderr)
        return 2
    flacs = sorted(f for split in args.splits for f in split.rglob("*.flac"))
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        errors = [
            e for e in pool.map(lambda f: decode(args.ffmpeg, f), flacs) if e
        ]
    for line in errors:
        print(line, file=sys.stderr)
    print(f"decoded {len(flacs) - len(errors)} of {len(flacs)} FLAC files")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
