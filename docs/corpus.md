# Corpus policy

Which audio answers which question, why, and how each artifact is prepared.

Everything here is reproducible from public, permissively licensed sources. No corpus
requires registration, and none carries a non-commercial or no-derivatives
restriction. Both are CC BY 4.0; see [`THIRD_PARTY.md`](../THIRD_PARTY.md) for the
attribution obligation, which derived excerpts inherit.

## Which corpus for which question

| Question | Corpus | Why this one |
|---|---|---|
| Does it match a vendor's published numbers? | LibriSpeech `dev-clean`, file `1272-135031-0000` | The exact file NVIDIA benchmarks with. Same audio, same chunk size, same client shape. |
| Is transcription accurate? | LibriSpeech `dev-clean`, or per-chapter concatenations | Reference transcripts; clean read speech. |
| Does long-running state stay bounded? | LibriSpeech per-chapter concatenations, 1–4 min | Continuous speech over thousands of chunks. |
| Do concurrent streams stay isolated? | Four files, different speakers, different lengths | Different speakers make contamination legible. |
| Does end-of-utterance detection work? | **AMI Meeting Corpus**, individual headset channels | Real conversational pauses. LibriSpeech has almost none. |
| Threshold-exact endpointing edges | Constructed fixtures from `dev-clean` utterances | Natural audio cannot land on a threshold boundary. |

## Do not evaluate endpointing on LibriSpeech

This is the single most important corpus caveat, and it is measured rather than
assumed. LibriSpeech utterances are distributed pre-trimmed, so concatenating them
yields near-continuous speech. Across four concatenations totalling 523 s and 87
utterances, **only three silences reached 800 ms**. An endpointing suite built on it
is vacuously green: it will pass a server that never detects an endpoint at all.

## Clip duration is dictated by model geometry

Streaming encoders carry a bounded left context. Until that context is full, the
model is running its partial-history path, not the saturated steady state it occupies
during real use — so short audio cannot support a claim about streaming behavior.

The thresholds are multiples of one full context window. For a model with an 80 ms
encoder frame retaining 56 frames of left context, one window is **4.48 s**:

| Duration | Contexts | What it exercises |
|---|---|---|
| 4.48 s | 1.0 | First moment the left context is full |
| 20.16 s | 4.5 | Initial fill plus 3.5 complete replacements |
| 49.28 s | 11.0 | Initial fill plus 10 complete replacements |
| 60 s | 13.4 | Above, plus clears a 30 s accepted-audio queue |

Use at least **20.16 s** for any claim about streaming behavior, and at least
**60 s** for claims about long-running stability. Recompute these for a model with a
different frame rate or context length — the multiples are what matters, not the
seconds.

This also frames how to read published vendor figures. NVIDIA's benchmark file
`1272-135031-0000.wav` is **10.885 s** — 2.4 contexts. Long enough to fill the
window, short enough that it says little about sustained operation.

## LibriSpeech

<https://www.openslr.org/12> — `dev-clean` (337 MB).

Per-utterance FLAC, 16 kHz mono. Transcripts are uppercase, unpunctuated, with
numbers spelled as words — which is why word error rate is only comparable within a
declared normalization policy.

**Concatenations.** Joining each chapter into one WAV plus a line-per-utterance
transcript gives 40 s to 8 min of continuous speech per file. These are the
long-stream and concurrency fixtures.

**Choosing a concurrency set.** Pick **four different speakers with different
durations** — for example 42 s, 95 s, 153 s, 233 s, two female and two male. Two
reasons, both load-bearing:

- Equal-length streams all finish together, so they never exercise release-and-reuse
  while other sessions are still running — precisely the path where cross-session
  state leaks.
- Different speakers make a leak show up as recognizably foreign words rather than as
  an opaque diff.

**Canonical comparison fixtures**, lossless 16 kHz mono PCM16 decodes of
`dev-clean/1272/135031/1272-135031-0000.flac`:

- `1272-135031-0000.wav` — 10.885 s, for real-time streaming cells.
- `1272-135031-0000x5.wav` — the first WAV concatenated exactly five times, for the
  unpaced offline-throughput cell.

These reproduce a published comparison. They do not replace the longer fixtures for
state-boundedness or endpointing.

## AMI Meeting Corpus

<https://groups.inf.ed.ac.uk/ami/corpus/> — for endpointing and conversational
realism.

Distributed as 16 kHz mono 16-bit WAV, so no resampler enters the signal path for a
16 kHz model.

Download **individual headset channels only**:

```
https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<MEETING>/audio/<MEETING>.Headset-<N>.wav
```

- `Headset-0` … `Headset-3` — close-talking mics, one per participant. **Use these.**
- `Array1-*`, `Array2-*` — far-field arrays. Reverberant and noisy; that is a
  noise-robustness condition, not what this measures, and it is most of the corpus's
  185 GB.
- `Mix-Headset` — all speakers mixed into one track, so utterances overlap.
- `*.rm` — RealMedia streaming copies, not WAV.

Each meeting has four headset channels; sizes seen are 26–46 MB per channel
(≈14–25 min).

**Take one channel from each of four different meetings, not four channels from one
meeting.** The byte cost is identical, but four channels of one meeting are four
microphones on one conversation: every channel carries audible crosstalk of the other
speakers, so a genuine cross-stream leak could hide behind crosstalk that belongs
there. Four separate meetings give independent speakers, rooms, and topics.

Annotations — transcripts plus a per-speaker speech/silence segmentation usable as an
independent pause reference:

```
https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip   (22 MB)
```

### Survey before excerpting

Endpointing evaluation is only meaningful on audio that contains qualifying pauses,
and the survey is what proves it does. It reports pauses per minute and the longest
gap per channel, and proposes window offsets containing **at least two** qualifying
pauses — one pause cannot exercise re-arming after a detected endpoint.

Use the same silence measurement across both corpora so their pause distributions are
directly comparable; that comparability is what makes the LibriSpeech finding above a
measurement rather than an impression.

Excerpting over HTTP range requests transfers about 630 kB for a 20.16 s excerpt
rather than the 26 MB source, and is byte-identical for a given recipe.

## Constructed fixtures

Natural audio cannot land on a threshold boundary, so threshold-exact endpointing
edges — at, just below, just above, re-arm after two boundaries — are built by
inserting silence gaps of an exact frame count between real utterances.

Constructed fixtures have a property no natural recording has: **the ground truth is
known by construction**, not by inference. The gap length, the boundary positions,
and the reference segments are recorded when the file is written. That makes them the
right source for emission-lag measurement, which otherwise needs a forced aligner.

One trap worth naming: general-purpose voice-activity detectors default to a minimum
silence duration around 100 ms and pad speech regions by tens of milliseconds. Those
defaults erase sub-100 ms gaps — exactly the single-frame boundaries these fixtures
exist to probe. Configure the detector to the frame grid, or use the construction-time
ground truth instead.

## Pinning

`datasets.lock` records every artifact by SHA-256 with the source it came from:

```json
{
  "entries": [
    {
      "name": "librispeech-1272-135031-0000",
      "dest": "clips/1272-135031-0000.flac",
      "sha256": "sha256:...",
      "source": "https://www.openslr.org/resources/12/dev-clean.tar.gz#..."
    }
  ]
}
```

`source` is either a fetchable URL or a `gen:<generator>:<name>` identifier for a
derived artifact, in which case the generator and its recipe are committed.

**Commit the recipe, not the audio.** Derived audio inherits CC BY 4.0, and a
digest-pinned recipe is a stronger reproducibility claim than a checked-in WAV.
