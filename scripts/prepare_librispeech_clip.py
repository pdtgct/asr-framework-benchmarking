#!/usr/bin/env python3
"""Transcode the locked LibriSpeech FLAC to 16 kHz PCM16 mono WAV.

``datasets.lock`` entry ``librispeech-1272-135031-0000`` pins the FLAC.
This recipe checks that digest, writes the WAV the NIM presets name, and
prints the WAV sha256. The WAV is derived audio and is not committed.
Copy the printed digest into the run notes.

``target.model`` is the server's ``model_name``. Read it from
``GetRivaSpeechRecognitionConfig`` (``model_config[].model_name``) and
pass it as ``--model``. The adapter copies that string into
``RecognitionConfig.model``. An unknown name fails on Riva.

Run from the repository root:

    uv run python scripts/prepare_librispeech_clip.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

LOCK_NAME = "librispeech-1272-135031-0000"
DEFAULT_FLAC = Path("clips/1272-135031-0000.flac")
DEFAULT_WAV = Path("clips/1272-135031-0000.wav")
DEFAULT_LOCK = Path("datasets.lock")

_MODEL_HELP = (
    "Set target.model or --model to the model_name from "
    "GetRivaSpeechRecognitionConfig. RecognitionConfig.model is that "
    "string."
)


class PrepareError(Exception):
    """The FLAC pin failed or the WAV is not 16 kHz PCM16 mono."""


def sha256_file(path: Path) -> str:
    """Return ``sha256:<hex>`` for the bytes at ``path``.

    Args:
        path: File to hash.

    Returns:
        The digest in the same form as ``datasets.lock``.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_entry(lock_path: Path, name: str) -> dict[str, Any]:
    """Return the lock object named ``name``.

    Args:
        lock_path: ``datasets.lock`` path.
        name: Entry name.

    Returns:
        The entry object.

    Raises:
        PrepareError: The file has no such entry.
    """
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise PrepareError(f"{lock_path} has no entries list")
    for item in entries:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise PrepareError(f"{lock_path} has no entry {name}")


def entry_sha256(entry: dict[str, Any]) -> str:
    """Return a lock digest after checking its shape.

    Args:
        entry: One ``datasets.lock`` object.

    Returns:
        ``sha256:`` plus 64 hex characters.

    Raises:
        PrepareError: The field is missing or not that shape.
    """
    value = entry.get("sha256")
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise PrepareError("lock sha256 must be sha256:<hex>")
    hex_part = value.removeprefix("sha256:")
    if len(hex_part) != 64:
        raise PrepareError("lock sha256 must be 64 hex characters")
    try:
        int(hex_part, 16)
    except ValueError as error:
        raise PrepareError("lock sha256 is not hex") from error
    return value


def verify_file(path: Path, expected: str) -> str:
    """Check ``path`` against a ``sha256:`` digest.

    Args:
        path: File on disk.
        expected: Digest from the lock, including the prefix.

    Returns:
        The digest that matched.

    Raises:
        PrepareError: The file is missing or the digest differs.
    """
    if not path.is_file():
        raise PrepareError(f"missing {path}")
    found = sha256_file(path)
    if found != expected:
        raise PrepareError(
            f"digest mismatch for {path}: payload {found}, lock {expected}"
        )
    return found


def transcode(ffmpeg: str, source: Path, dest: Path) -> None:
    """Decode ``source`` to 16 kHz mono PCM16 WAV at ``dest``.

    Args:
        ffmpeg: ffmpeg executable.
        source: FLAC or another format ffmpeg can read.
        dest: WAV path. Parents are created. An existing file is
            replaced.

    Raises:
        PrepareError: ffmpeg failed, or the WAV is not 16 kHz PCM16
            mono.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(dest),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = detail[-1] if detail else "ffmpeg failed"
        raise PrepareError(tail)
    _require_pcm16(dest)


def wav_lock_digest(lock_path: Path, wav: Path) -> str | None:
    """Return a lock digest whose ``dest`` is this WAV, if one exists.

    Args:
        lock_path: ``datasets.lock`` path.
        wav: Output path. Compared with the lock's ``dest`` string.

    Returns:
        The pinned digest, or ``None`` when the WAV is not in the lock.
        The FLAC entry is the pin this recipe checks. A later WAV entry
        is checked when present.
    """
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    wanted = {wav.as_posix(), str(wav)}
    for item in entries:
        if not isinstance(item, dict):
            continue
        dest = item.get("dest")
        if isinstance(dest, str) and dest in wanted:
            return entry_sha256(item)
    return None


def prepare(lock: Path, name: str, flac: Path, wav: Path, ffmpeg: str) -> str:
    """Verify the FLAC, transcode it, and return the WAV digest.

    Args:
        lock: ``datasets.lock`` path.
        name: FLAC entry name.
        flac: Source file. Its bytes must match the lock.
        wav: Destination WAV.
        ffmpeg: ffmpeg executable.

    Returns:
        ``sha256:`` digest of the WAV bytes.

    Raises:
        PrepareError: The pin failed, ffmpeg failed, or a WAV pin in
            the lock does not match the file just written.
    """
    entry = load_entry(lock, name)
    verify_file(flac, entry_sha256(entry))
    transcode(ffmpeg, flac, wav)
    digest = sha256_file(wav)
    pinned = wav_lock_digest(lock, wav)
    if pinned is not None:
        verify_file(wav, pinned)
    return digest


def _require_pcm16(path: Path) -> None:
    """Raise unless ``path`` is 16 kHz, mono, 16-bit PCM."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
    if channels == 1 and rate == 16000 and width == 2:
        return
    raise PrepareError(
        f"{path} is {channels} ch, {rate} Hz, {width * 8}-bit; "
        "required 1 ch, 16000 Hz, 16-bit"
    )


def main(argv: list[str] | None = None) -> int:
    """Verify the locked FLAC and write the preset WAV.

    Args:
        argv: Arguments, excluding the program name.

    Returns:
        ``0`` when the WAV digest was printed or ``--check-lock``
        succeeded. ``1`` when preparation failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--name", default=LOCK_NAME)
    parser.add_argument("--flac", type=Path, default=DEFAULT_FLAC)
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument(
        "--check-lock",
        action="store_true",
        help="print the FLAC pin and exit",
    )
    parser.add_argument("--ffmpeg", default="")
    args = parser.parse_args(argv)
    try:
        entry = load_entry(args.lock, args.name)
        digest = entry_sha256(entry)
        if args.check_lock:
            name = entry.get("name")
            dest = entry.get("dest")
            print(f"name {name}")
            print(f"dest {dest}")
            print(f"sha256 {digest}")
            print(_MODEL_HELP)
            return 0
        ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or ""
        if not ffmpeg:
            raise PrepareError("ffmpeg is not on PATH")
        wav_digest = prepare(args.lock, args.name, args.flac, args.wav, ffmpeg)
    except PrepareError as error:
        print(f"prepare_librispeech_clip: {error}", file=sys.stderr)
        return 1
    print(f"flac {args.flac} verified")
    print(f"wav {args.wav}")
    print(f"sha256 {wav_digest}")
    print(_MODEL_HELP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
