"""AMI Meeting Corpus: headset excerpts and segments as a manifest.

Ported from the author's ``ami_excerpts.py`` (nemotron-omni-port work),
with three changes so it suits a benchmark that must run air-gapped:

1. **Local files only.** The port fetched byte ranges from the Edinburgh
   mirror. This reads headset WAVs and the manual annotations from disk;
   :func:`mirror_urls` prints what to stage on a connected host.
2. **References.** The port cut audio for endpointing and carried no
   transcript. Here each clip carries the channel speaker's words from
   ``ami_public_manual_1.6.2``, so a run scores corpus WER.
3. **Stdlib.** The silence survey no longer needs numpy.

Two clip shapes, for two questions:

- ``excerpts``: fixed windows from a recipe (the ported recipe ships
  as :data:`BUILTIN_RECIPES`). Durations follow encoder geometry, so
  they exercise saturated streaming state and endpointing. A headset
  window also carries crosstalk from the other participants that the
  channel's reference does not transcribe, so its WER is inflated by
  insertions. Read it as a streaming-condition number.
- ``segments``: one clip per manually segmented utterance on an
  individual headset. This is the IHM condition the Open ASR Leaderboard
  AMI set derives from, so it is the shape to put beside a published AMI
  WER, with the caveat that the leaderboard's "cleaned" references and
  segmentation are not byte-identical to the manual annotations.

The AMI corpus is CC BY 4.0. Excerpts and segments are derived works;
any published result must carry :data:`ATTRIBUTION`. Nothing here is
committed audio: recipes and digests are.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import re
import sys
import wave
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from asrbench.audio import BYTES_PER_SAMPLE, SAMPLE_RATE
from asrbench.canonical import sha256_bytes
from asrbench.clips import pcm16_wav
from asrbench.manifest import Manifest, ManifestError, write_manifest

MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "AMI Meeting Corpus, CC BY 4.0. J. Carletta et al., 'The AMI Meeting "
    "Corpus: A Pre-announcement,' MLMI 2005. "
    "https://groups.inf.ed.ac.uk/ami/corpus/"
)

#: The full-corpus evaluation split, as Kaldi ``egs/ami/s5b`` defines it
#: in ``local/split_eval.orig`` and ``local/split_dev.orig``.
SPLITS: dict[str, tuple[str, ...]] = {
    "eval": (
        "EN2002a",
        "EN2002b",
        "EN2002c",
        "EN2002d",
        "ES2004a",
        "ES2004b",
        "ES2004c",
        "ES2004d",
        "IS1009a",
        "IS1009b",
        "IS1009c",
        "IS1009d",
        "TS3003a",
        "TS3003b",
        "TS3003c",
        "TS3003d",
    ),
    "dev": (
        "ES2011a",
        "ES2011b",
        "ES2011c",
        "ES2011d",
        "IB4001",
        "IB4002",
        "IB4003",
        "IB4004",
        "IB4010",
        "IB4011",
        "IS1008a",
        "IS1008b",
        "IS1008c",
        "IS1008d",
        "TS3004a",
        "TS3004b",
        "TS3004c",
        "TS3004d",
    ),
}

#: Absolute floor; headset channels carry crosstalk, so this means "this
#: speaker is not talking", not digital silence. As in the original survey.
SILENCE_DBFS = -50.0
#: A window below this speech fraction transcribes to almost nothing.
MIN_SPEECH_FRACTION = 0.30
FRAME_MS = 20

_RECIPES_DIR = Path(__file__).with_name("recipes")
#: Recipes shipped in the wheel, usable by name with no network.
BUILTIN_RECIPES = {
    "ami-ihm-excerpts-v1": _RECIPES_DIR / "ami-ihm-excerpts-v1.json",
}

_NITE = "{http://nite.sourceforge.net/}"
_HREF_RANGE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")


class AmiError(ManifestError):
    """A staged AMI file is missing or not what the recipe expects."""


@dataclass(frozen=True)
class Word:
    """One ``<w>`` element of an AMI words file."""

    id: str
    start_s: float | None
    end_s: float | None
    text: str
    punctuation: bool


@dataclass(frozen=True)
class Segment:
    """One ``<segment>`` of an AMI segments file."""

    id: str
    channel: int
    start_s: float
    end_s: float
    first_word: str | None
    last_word: str | None


class Annotations:
    """``ami_public_manual_1.6.2`` read from its zip or an extracted tree."""

    def __init__(self, root: Path) -> None:
        """Open the annotations.

        Args:
            root: The zip as downloaded, or the directory it extracts to.

        Raises:
            AmiError: ``root`` is neither.
        """
        self.root = root
        self._zip: zipfile.ZipFile | None = None
        if root.is_file():
            try:
                self._zip = zipfile.ZipFile(root)
            except zipfile.BadZipFile as error:
                raise AmiError(f"{root}: {error}") from error
        elif not root.is_dir():
            raise AmiError(f"annotations not found: {root}")
        self._agents: dict[str, dict[int, str]] | None = None

    def _read(self, member: str) -> bytes | None:
        if self._zip is not None:
            try:
                return self._zip.read(member)
            except KeyError:
                return None
        path = self.root / member
        return path.read_bytes() if path.is_file() else None

    def digest(self) -> str | None:
        """Return the sha256 of the zip, or ``None`` for a directory."""
        if self._zip is None:
            return None
        return _file_sha256(self.root)

    def _load_agents(self) -> dict[str, dict[int, str]]:
        if self._agents is None:
            raw = self._read("corpusResources/meetings.xml")
            if raw is None:
                raise AmiError("corpusResources/meetings.xml is missing")
            agents: dict[str, dict[int, str]] = {}
            for meeting_el in ElementTree.fromstring(raw).iter("meeting"):
                name = meeting_el.get("observation")
                if not name:
                    continue
                for speaker in meeting_el.iter("speaker"):
                    letter = speaker.get("nxt_agent")
                    chan = speaker.get("channel")
                    if letter and chan is not None and chan.isdigit():
                        agents.setdefault(name, {})[int(chan)] = letter
            self._agents = agents
        return self._agents

    def agent(self, meeting: str, channel: int) -> str:
        """Return the participant letter recorded on ``Headset-<channel>``.

        Args:
            meeting: Observation id, such as ``IS1009a``.
            channel: Headset channel number.

        Raises:
            AmiError: The meeting or channel is not in ``meetings.xml``.
        """
        try:
            return self._load_agents()[meeting][channel]
        except KeyError as error:
            raise AmiError(
                f"{meeting} Headset-{channel} is not in meetings.xml"
            ) from error

    def channels(self, meeting: str) -> list[int]:
        """Return the headset channels ``meetings.xml`` lists for a meeting.

        Raises:
            AmiError: The meeting is not in ``meetings.xml``.
        """
        found = self._load_agents().get(meeting)
        if not found:
            raise AmiError(f"{meeting} is not in meetings.xml")
        return sorted(found)

    def element_ids(self, meeting: str, agent: str) -> list[tuple[str, bool]]:
        """Return every element's id in the words file, in file order.

        ``<vocalsound>``, ``<disfmarker>``, ``<gap>`` and the like share the
        ``wordsN`` id sequence with ``<w>``, and a segment's href range may
        start or end on one. The flag says whether the element is a ``<w>``.

        Raises:
            AmiError: The words file is missing.
        """
        member = f"words/{meeting}.{agent}.words.xml"
        raw = self._read(member)
        if raw is None:
            raise AmiError(f"{member} is missing")
        return [
            (element.get(f"{_NITE}id", ""), element.tag == "w")
            for element in ElementTree.fromstring(raw)
        ]

    def words(self, meeting: str, agent: str) -> list[Word]:
        """Return one participant's words in file order.

        Raises:
            AmiError: The words file is missing.
        """
        member = f"words/{meeting}.{agent}.words.xml"
        raw = self._read(member)
        if raw is None:
            raise AmiError(f"{member} is missing")
        found: list[Word] = []
        for element in ElementTree.fromstring(raw):
            if element.tag != "w":
                continue
            found.append(
                Word(
                    id=element.get(f"{_NITE}id", ""),
                    start_s=_maybe_float(element.get("starttime")),
                    end_s=_maybe_float(element.get("endtime")),
                    text=(element.text or "").strip(),
                    punctuation=element.get("punc") == "true",
                )
            )
        return found

    def segments(self, meeting: str, agent: str) -> list[Segment]:
        """Return one participant's segments in file order.

        Raises:
            AmiError: The segments file is missing.
        """
        member = f"segments/{meeting}.{agent}.segments.xml"
        raw = self._read(member)
        if raw is None:
            raise AmiError(f"{member} is missing")
        found: list[Segment] = []
        for element in ElementTree.fromstring(raw):
            if element.tag != "segment":
                continue
            start = _maybe_float(element.get("transcriber_start"))
            end = _maybe_float(element.get("transcriber_end"))
            channel = element.get("channel")
            if start is None or end is None or channel is None:
                continue
            first = last = None
            for child in element:
                match = _HREF_RANGE.search(child.get("href", ""))
                if match:
                    first = match.group(1)
                    last = match.group(2) or match.group(1)
                    break
            found.append(
                Segment(
                    id=element.get(f"{_NITE}id", ""),
                    channel=int(channel),
                    start_s=start,
                    end_s=end,
                    first_word=first,
                    last_word=last,
                )
            )
        return found


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def join_words(words: Iterable[Word]) -> str:
    """Join AMI tokens into one reference string.

    Punctuation tokens attach to the token before them. Vocal sounds,
    disfluency markers, and gaps are separate elements and never reach
    here, so the reference is what the participant said in words.
    """
    out: list[str] = []
    for word in words:
        if not word.text:
            continue
        if word.punctuation and out:
            out[-1] += word.text
        elif not word.punctuation:
            out.append(word.text)
    return " ".join(out)


@dataclass(frozen=True)
class WindowWords:
    """Words a window keeps, and how many it cut through."""

    words: list[Word]
    boundary_words: int


def window_words(
    words: list[Word], start_s: float, end_s: float
) -> WindowWords:
    """Keep timed words whose midpoint lies in ``[start_s, end_s)``.

    A word whose span crosses an edge is still assigned by its midpoint,
    and counted in ``boundary_words`` so a caller can see how much of the
    reference rests on a cut through speech.
    """
    kept: list[Word] = []
    boundary = 0
    for word in words:
        if word.start_s is None or word.end_s is None:
            continue
        middle = (word.start_s + word.end_s) / 2
        if start_s <= middle < end_s:
            kept.append(word)
        if not word.punctuation and (
            word.start_s < start_s < word.end_s
            or word.start_s < end_s < word.end_s
        ):
            boundary += 1
    return WindowWords(words=kept, boundary_words=boundary)


def headset_path(audio_root: Path, meeting: str, channel: int) -> Path:
    """Find a staged headset WAV.

    Accepts the mirror layout ``<root>/<M>/audio/<M>.Headset-<N>.wav`` and
    a flat ``<root>/<M>.Headset-<N>.wav``.

    Raises:
        AmiError: Neither exists.
    """
    name = f"{meeting}.Headset-{channel}.wav"
    for candidate in (audio_root / meeting / "audio" / name, audio_root / name):
        if candidate.is_file():
            return candidate
    raise AmiError(
        f"{name} is not staged under {audio_root} "
        f"(fetch {channel_url(meeting, channel)} on a connected host)"
    )


def channel_url(meeting: str, channel: int) -> str:
    """Mirror URL for one individual-headset channel."""
    return f"{MIRROR}/{meeting}/audio/{meeting}.Headset-{channel}.wav"


class HeadsetAudio:
    """One staged headset channel, read by sample range."""

    def __init__(self, path: Path) -> None:
        """Open and check the format.

        Raises:
            AmiError: The file is not 16 kHz mono PCM16.
        """
        self.path = path
        try:
            with wave.open(str(path), "rb") as handle:
                shape = (
                    handle.getnchannels(),
                    handle.getsampwidth(),
                    handle.getframerate(),
                )
                self.frames = handle.getnframes()
        except (OSError, EOFError, wave.Error) as error:
            raise AmiError(f"{path}: {error}") from error
        if shape != (1, BYTES_PER_SAMPLE, SAMPLE_RATE):
            raise AmiError(
                f"{path}: AMI headsets are mono PCM16 {SAMPLE_RATE} Hz, got "
                f"{shape[0]}ch/{shape[1] * 8}bit/{shape[2]}Hz"
            )

    @property
    def duration_s(self) -> float:
        """Channel length in seconds."""
        return self.frames / SAMPLE_RATE

    def read(self, start: int, count: int) -> bytes:
        """Return PCM16 for frames ``[start, start + count)``.

        Raises:
            AmiError: The range runs past the end of the channel.
        """
        if start < 0 or count < 1 or start + count > self.frames:
            raise AmiError(
                f"{self.path.name}: frames {start}+{count} exceed "
                f"{self.frames} ({self.duration_s:.1f}s)"
            )
        with wave.open(str(self.path), "rb") as handle:
            handle.setpos(start)
            return handle.readframes(count)

    def read_all(self) -> bytes:
        """Return the whole channel's PCM16."""
        return self.read(0, self.frames)


def silence_runs(
    pcm: bytes,
    frame_ms: int = FRAME_MS,
    floor_dbfs: float = SILENCE_DBFS,
) -> tuple[list[bool], list[tuple[int, int]]]:
    """Return per-frame quiet flags and ``(start_ms, length_ms)`` runs.

    The floor is absolute dBFS, never relative to the clip's peak: a
    peak-relative floor is not comparable between a whole channel and an
    excerpt cut from it (a 20 s window of the public IS1009a headset
    measures 0.0% quiet peak-relative and 47.7% at -50 dBFS).
    """
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    width = SAMPLE_RATE * frame_ms // 1000
    count = len(samples) // width
    threshold = (10.0 ** (floor_dbfs / 20.0) * 32768.0) ** 2 * width
    quiet: list[bool] = []
    for index in range(count):
        block = samples[index * width : (index + 1) * width]
        energy = sum(value * value for value in block)
        quiet.append(energy < threshold)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            runs.append((start * frame_ms, (index - start) * frame_ms))
            start = None
    if start is not None:
        runs.append((start * frame_ms, (len(quiet) - start) * frame_ms))
    return quiet, runs


def propose_offsets(
    quiet: list[bool],
    runs: list[tuple[int, int]],
    window_s: float,
    min_silence_ms: int,
    want: int,
    min_speech: float = MIN_SPEECH_FRACTION,
    frame_ms: int = FRAME_MS,
) -> list[float]:
    """Pick window starts that are both pause-bearing and speech-bearing.

    A window with fewer than two qualifying pauses cannot exercise re-arm
    after an endpoint; a window that is nearly all silence transcribes to
    nothing (a single AMI channel can be quiet for minutes). Offsets are
    duration-specific. Same rule as the original survey.
    """
    qualifying = [start for start, length in runs if length >= min_silence_ms]
    if len(qualifying) < 2:
        return []
    window_ms = window_s * 1000
    span = int(window_ms / frame_ms)
    offsets: list[float] = []
    for index in range(len(qualifying) - 1):
        begin = max(0.0, qualifying[index] - 500)
        inside = [q for q in qualifying if begin <= q < begin + window_ms]
        if len(inside) < 2:
            continue
        first = int(begin / frame_ms)
        window = quiet[first : first + span]
        if len(window) < span:
            continue
        if 1.0 - sum(window) / len(window) < min_speech:
            continue
        candidate = round(begin / 1000, 3)
        if all(abs(candidate - other) > window_s for other in offsets):
            offsets.append(candidate)
        if len(offsets) >= want:
            break
    return offsets


def load_recipe(spec: str) -> tuple[dict[str, Any], str]:
    """Load an excerpt recipe by built-in name or path.

    Args:
        spec: A :data:`BUILTIN_RECIPES` name or a JSON path.

    Returns:
        The recipe and the sha256 of its bytes.

    Raises:
        AmiError: Missing, malformed, or without excerpts.
    """
    path = BUILTIN_RECIPES.get(spec, Path(spec))
    try:
        raw = path.read_bytes()
        recipe = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise AmiError(f"recipe {spec}: {error}") from error
    if not isinstance(recipe, dict) or not isinstance(
        recipe.get("excerpts"), list
    ):
        raise AmiError(f"recipe {spec}: expected an object with excerpts")
    return recipe, sha256_bytes(raw)


@dataclass(frozen=True)
class BuildResult:
    """A written manifest and what the builder noticed on the way."""

    manifest: Manifest
    warnings: list[str]


def build_excerpts(
    *,
    recipe_spec: str,
    audio_root: Path,
    annotations: Path,
    out_dir: Path,
    manifest_id: str | None = None,
    language: str = "en-US",
    min_silence_ms: int = 800,
) -> BuildResult:
    """Cut every recipe excerpt from local headsets and write a manifest.

    Args:
        recipe_spec: Built-in recipe name or recipe path.
        audio_root: Where headset WAVs are staged.
        annotations: ``ami_public_manual_1.6.2.zip`` or its extraction.
        out_dir: Receives ``clips/*.wav`` and ``manifest.json``.
        manifest_id: Defaults to the recipe's ``id`` or the recipe name.
        language: Manifest language tag.
        min_silence_ms: Pause threshold re-measured on every cut.

    Returns:
        The manifest and warnings. A recipe ``file_sha256`` that does not
        match the cut is an error, not a warning: the offset or the
        staged channel is not the one the recipe was measured on.

    Raises:
        AmiError: A staged file is missing or a pin fails.
    """
    recipe, recipe_digest = load_recipe(recipe_spec)
    notes = Annotations(annotations)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    sources: dict[str, str] = {}
    for item in recipe["excerpts"]:
        name = str(item["name"])
        meeting = str(item["meeting"])
        channel = int(item["channel"])
        start_s = float(item["start_s"])
        duration_s = float(item["duration_s"])
        source = headset_path(audio_root, meeting, channel)
        audio = HeadsetAudio(source)
        start = int(round(start_s * SAMPLE_RATE))
        count = int(round(duration_s * SAMPLE_RATE))
        pcm = audio.read(start, count)
        wav = pcm16_wav(pcm)
        file_digest = sha256_bytes(wav)
        expected = item.get("file_sha256")
        if expected is not None and expected != file_digest:
            raise AmiError(
                f"{name}: cut {file_digest} != recipe file_sha256 {expected}; "
                f"is {source.name} the corpus release the recipe measured?"
            )
        target = clips_dir / f"{name}.wav"
        target.write_bytes(wav)
        quiet, runs = silence_runs(pcm)
        pauses = sum(1 for _, length in runs if length >= min_silence_ms)
        if pauses < 2:
            warnings.append(
                f"{name}: {pauses} pause(s) >= {min_silence_ms}ms; cannot "
                "exercise endpoint re-arm"
            )
        agent = notes.agent(meeting, channel)
        kept = window_words(
            notes.words(meeting, agent), start_s, start_s + duration_s
        )
        if kept.boundary_words:
            warnings.append(
                f"{name}: {kept.boundary_words} word(s) straddle a window "
                "edge; assigned by midpoint"
            )
        sources.setdefault(source.name, _file_sha256(source))
        clips.append(
            {
                "id": name,
                "audio": f"clips/{name}.wav",
                "reference": join_words(kept.words),
                "pcm_sha256": sha256_bytes(pcm),
                "duration_s": count / SAMPLE_RATE,
                "source": {
                    "meeting": meeting,
                    "channel": channel,
                    "agent": agent,
                    "start_s": start_s,
                    "file_sha256": file_digest,
                    "pauses": pauses,
                    "speech_fraction": round(
                        1.0 - sum(quiet) / max(len(quiet), 1), 4
                    ),
                    "boundary_words": kept.boundary_words,
                    "role": item.get("role"),
                },
            }
        )
    default_id = str(recipe.get("id") or Path(recipe_spec).stem)
    manifest = write_manifest(
        out_dir / "manifest.json",
        manifest_id=manifest_id or default_id,
        language=language,
        clips=clips,
        corpus="AMI Meeting Corpus",
        licence=LICENCE,
        attribution=ATTRIBUTION,
        note=(
            "Headset windows. Crosstalk from other participants is audible "
            "and not in the reference; read WER as a streaming condition."
        ),
        provenance={
            "builder": "asrbench.corpora.ami.build_excerpts",
            "recipe": recipe_spec,
            "recipe_sha256": recipe_digest,
            "annotations": annotations.name,
            "annotations_sha256": notes.digest(),
            "sources_sha256": dict(sorted(sources.items())),
            "min_silence_ms": min_silence_ms,
            "silence_floor_dbfs": SILENCE_DBFS,
        },
    )
    return BuildResult(manifest=manifest, warnings=warnings)


def build_segments(
    *,
    meetings: list[str],
    audio_root: Path,
    annotations: Path,
    out_dir: Path,
    manifest_id: str,
    language: str = "en-US",
    min_duration_s: float = 0.0,
    max_clips: int | None = None,
) -> BuildResult:
    """Write one clip per annotated headset segment of ``meetings``.

    Segments whose reference has no words (only vocal sounds) are
    skipped, as the leaderboard skips empty references. Clip order is
    meeting order, then channel, then segment order in the file.

    Args:
        meetings: Observation ids, such as :data:`SPLITS` values. An
            entry ``MEETING:N`` takes only headset ``N`` of that meeting.
        audio_root: Where headset WAVs are staged.
        annotations: ``ami_public_manual_1.6.2.zip`` or its extraction.
        out_dir: Receives ``clips/*.wav`` and ``manifest.json``.
        manifest_id: Safe name.
        language: Manifest language tag.
        min_duration_s: Skip shorter segments.
        max_clips: Stop after this many clips. For smoke runs only; the
            provenance records it so a truncated set is visible.

    Returns:
        The manifest and warnings.

    Raises:
        AmiError: A staged file is missing.
    """
    notes = Annotations(annotations)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    warnings: list[str] = []
    sources: dict[str, str] = {}
    skipped_empty = skipped_short = 0
    for spec in meetings:
        meeting, _, only = spec.partition(":")
        chosen = [int(only)] if only else notes.channels(meeting)
        for channel in chosen:
            agent = notes.agent(meeting, channel)
            words = notes.words(meeting, agent)
            # Position of every element (not only <w>) in file order, and
            # how many <w> precede each position, so a range that starts or
            # ends on a vocalsound still selects the words inside it.
            elements = notes.element_ids(meeting, agent)
            index = {
                element_id: position
                for position, (element_id, _) in enumerate(elements)
            }
            words_before = [0]
            for _, is_word in elements:
                words_before.append(words_before[-1] + int(is_word))
            source = headset_path(audio_root, meeting, channel)
            audio = HeadsetAudio(source)
            for segment in notes.segments(meeting, agent):
                if max_clips is not None and len(clips) >= max_clips:
                    break
                if segment.first_word is None or segment.last_word is None:
                    skipped_empty += 1
                    continue
                first = index.get(segment.first_word)
                last = index.get(segment.last_word)
                if first is None or last is None:
                    skipped_empty += 1
                    continue
                if last < first:
                    skipped_empty += 1
                    continue
                reference = join_words(
                    words[words_before[first] : words_before[last + 1]]
                )
                if not reference:
                    skipped_empty += 1
                    continue
                duration = segment.end_s - segment.start_s
                if duration < max(min_duration_s, 1.0 / SAMPLE_RATE):
                    skipped_short += 1
                    continue
                start = int(round(segment.start_s * SAMPLE_RATE))
                count = int(round(duration * SAMPLE_RATE))
                count = min(count, audio.frames - start)
                if count < 1:
                    skipped_short += 1
                    continue
                pcm = audio.read(start, count)
                tail = segment.id.removeprefix(f"{meeting}.")
                clip_id = f"ami-{meeting.lower()}-h{channel}-{_safe(tail)}"
                (clips_dir / f"{clip_id}.wav").write_bytes(pcm16_wav(pcm))
                sources.setdefault(source.name, _file_sha256(source))
                clips.append(
                    {
                        "id": clip_id,
                        "audio": f"clips/{clip_id}.wav",
                        "reference": reference,
                        "pcm_sha256": sha256_bytes(pcm),
                        "duration_s": count / SAMPLE_RATE,
                        "source": {
                            "meeting": meeting,
                            "channel": channel,
                            "agent": agent,
                            "segment": segment.id,
                            "start_s": segment.start_s,
                            "end_s": segment.end_s,
                        },
                    }
                )
    if not clips:
        raise AmiError("no segment produced a clip")
    if skipped_empty:
        warnings.append(f"skipped {skipped_empty} segment(s) with no words")
    if skipped_short:
        warnings.append(f"skipped {skipped_short} segment(s) below duration")
    manifest = write_manifest(
        out_dir / "manifest.json",
        manifest_id=manifest_id,
        language=language,
        clips=clips,
        corpus="AMI Meeting Corpus",
        licence=LICENCE,
        attribution=ATTRIBUTION,
        note=(
            "Individual-headset (IHM) segments from the manual annotations. "
            "Most are shorter than one encoder context."
        ),
        provenance={
            "builder": "asrbench.corpora.ami.build_segments",
            "meetings": list(meetings),
            "annotations": annotations.name,
            "annotations_sha256": notes.digest(),
            "sources_sha256": dict(sorted(sources.items())),
            "min_duration_s": min_duration_s,
            "max_clips": max_clips,
            "skipped_empty": skipped_empty,
            "skipped_short": skipped_short,
        },
    )
    return BuildResult(manifest=manifest, warnings=warnings)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


@dataclass(frozen=True)
class SurveyRow:
    """One channel's pause statistics and proposed window starts."""

    meeting: str
    channel: int
    duration_s: float
    qualifying_pauses: int
    longest_gap_ms: int
    proposals: dict[float, list[float]]


def survey(
    *,
    channels: list[tuple[str, int]],
    audio_root: Path,
    windows: list[float],
    min_silence_ms: int = 800,
    propose: int = 3,
) -> list[SurveyRow]:
    """Measure each staged channel's pauses and propose excerpt offsets.

    Run this before writing a recipe: endpointing is only measurable on
    audio that contains qualifying pauses.
    """
    rows: list[SurveyRow] = []
    for meeting, channel in channels:
        audio = HeadsetAudio(headset_path(audio_root, meeting, channel))
        quiet, runs = silence_runs(audio.read_all())
        qualifying = [length for _, length in runs if length >= min_silence_ms]
        rows.append(
            SurveyRow(
                meeting=meeting,
                channel=channel,
                duration_s=audio.duration_s,
                qualifying_pauses=len(qualifying),
                longest_gap_ms=max((length for _, length in runs), default=0),
                proposals={
                    window: propose_offsets(
                        quiet, runs, window, min_silence_ms, propose
                    )
                    for window in windows
                },
            )
        )
    return rows


def mirror_urls(meetings: Iterable[tuple[str, int]]) -> list[str]:
    """Return what to fetch on a connected host: annotations, then WAVs."""
    urls = [ANNOTATIONS_URL]
    urls.extend(channel_url(meeting, channel) for meeting, channel in meetings)
    return urls


def recipe_channels(recipe: dict[str, Any]) -> list[tuple[str, int]]:
    """Return the distinct ``(meeting, channel)`` pairs a recipe cuts from."""
    seen: dict[tuple[str, int], None] = {}
    for item in recipe["excerpts"]:
        seen[(str(item["meeting"]), int(item["channel"]))] = None
    return list(seen)
