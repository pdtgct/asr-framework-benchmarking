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

Prepare `clips/1272-135031-0000.wav` on the machine that runs the cell.
`datasets.lock` pins the FLAC (`librispeech-1272-135031-0000`).
`scripts/prepare_librispeech_clip.py` checks that digest, decodes 16 kHz mono
PCM16 to the WAV, and prints the WAV sha256. Record that digest in the run
notes. Do not commit the audio.

Set `target.model` to the `model_name` reported by
`GetRivaSpeechRecognitionConfig` (`model_config[].model_name`), via `--model`
or the preset field. The adapter sends that string as `RecognitionConfig.model`.
An unknown name fails on Riva. `presets/nim-compat-smoke.json` uses
`nemotron-asr-streaming` for `target.model` and `checkpoint_id`, the same
declared name as the streaming preset, with `checkpoint_source` unverified.
The literal `server-reported-model-name` is rejected for a non-mock transport.

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

## Manifests: corpus WER over a test set

A manifest (`asrbench-manifest/1`) names a set of clips and the corpus's own
reference transcript for each. `asrbench run --manifest manifest.json` plays the
clips across lanes and scores every session against its clip's corpus transcript,
so one receipt carries the test-set WER NVIDIA and the Open ASR Leaderboard
publish. Builders write the manifest from locally staged corpus files; nothing is
fetched at run time.

```json
{
  "schema": "asrbench-manifest/1",
  "id": "librispeech-test-clean",
  "language": "en-US",
  "corpus": "LibriSpeech ASR corpus",
  "licence": "CC-BY-4.0",
  "attribution": "...",
  "note": "free text, not identity",
  "provenance": {"builder": "...", "...": "builder inputs, not identity"},
  "clips": [
    {
      "id": "1089-134686-0000",
      "audio": "clips/1089-134686-0000.wav",
      "reference": "HE HOPED THERE WOULD BE STEW FOR DINNER ...",
      "pcm_sha256": "sha256:...",
      "duration_s": 10.435,
      "source": {"split": "test-clean", "speaker": "1089", "chapter": "134686"}
    }
  ]
}
```

- `audio` is relative to the manifest's directory. Audio is 16 kHz mono PCM16.
- One manifest is one language. A clip may repeat `language` but not differ.
- **The manifest digest is workload identity.** It hashes the schema, the language,
  and each clip's `id`, `pcm_sha256`, and the sha256 of its `reference`, in order.
  Paths, `note`, `provenance`, and `source` are excluded, so moving the tree or
  re-describing it does not change the digest; changing one sample or one
  reference word does. It appears in the workload fingerprint and therefore the
  comparison key, so receipts over different clip sets or references never compare.
- The runner re-hashes every clip when it materializes the run copy, and refuses a
  clip whose PCM no longer matches. `asrbench corpus verify manifest.json` does the
  same check without a run.
- A manifest needs `normalization_policy: corpus_reference`. It supersedes the
  preset's single `clip` and `reference_transcript`; the receipt records
  `reference_transcript: null` and the clip set under `dataset.manifest`, including
  every reference.

### Clip to lane assignment

Session `i` of lane `l` plays clip `(i * lanes + l) mod N`. Sessions running at the
same moment play different clips, and every trial (and warmup) replays the same
assignment, so trials differ only in timing. With `lanes * iterations >= N` every
clip is played; choose `--iterations` as `ceil(N / lanes)` for one pass. A shorter
run is legal (a smoke run) and is recorded, not refused: the corpus block's
`complete` is false and `clips_scored < clip_count`. `balanced` is true when every
scored clip was played the same number of times.

### The reduction

`summary.*.wer.corpus` holds:

- `corpus_wer`: edits summed over every session divided by reference words summed
  over every session (`pooled_errors_over_reference_words`). This is the leaderboard
  and NVIDIA definition. When `complete` and `balanced` hold it equals the test-set
  WER; with uneven plays, clips played more often weigh more.
- `per_clip_mean_wer`, and a `per_clip_wer` distribution: each clip's own pooled WER,
  averaged with equal weight. A secondary view for short-utterance sets.
- `clips`: per clip, sessions, edits, reference words, deletions from missing
  transcripts, and WER. A failed session stays in the denominator as deletions of
  its whole reference, as everywhere else in asrbench.

`micro_wer` beside it is the same pooled number.

## Corpus builders

All builders read local files only, so they run the same way on an air-gapped host.

```bash
asrbench corpus ami-urls [--recipe ami-ihm-excerpts-v1 | --split eval]   # what to stage
asrbench corpus ami-survey IS1009a:0 --audio-root ami/                   # pauses, offsets
asrbench corpus ami-excerpts --recipe ami-ihm-excerpts-v1 \
  --audio-root ami/ --annotations ami_public_manual_1.6.2.zip --out-dir corpora/ami-excerpts
asrbench corpus ami-segments --split eval \
  --audio-root ami/ --annotations ami_public_manual_1.6.2.zip --out-dir corpora/ami-eval
python scripts/decode_librispeech.py LibriSpeech/test-clean               # needs ffmpeg
asrbench corpus librispeech --root . --split test-clean --out-dir corpora/ls-test-clean
asrbench corpus verify corpora/ls-test-clean/manifest.json
```

`--audio-root` accepts the mirror layout (`<M>/audio/<M>.Headset-<N>.wav`) or a flat
directory of `<M>.Headset-<N>.wav`. `--annotations` accepts the zip as downloaded or
its extracted directory.

### AMI: excerpts and segments are different questions

**`ami-segments` is the WER path.** One clip per manual-annotation segment of each
headset (IHM), with the reference built from that segment's words
(`words/<M>.<agent>.words.xml`; punctuation tokens attach to the previous word, vocal
sounds and disfluency markers are not words). Segments with no words are skipped, as
the leaderboard skips empty references. `--split eval` is the Kaldi s5b evaluation
split (EN2002a–d, ES2004a–d, IS1009a–d, TS3003a–d); `--split dev` the dev split.
`--meetings IS1009a:0 ...` takes chosen headsets. Most segments are shorter than one
encoder context, so this is an accuracy set, not a streaming-stability set.

Not identical to the leaderboard's AMI row: that row uses the ESB "cleaned" AMI
references, re-segmented and text-normalized by its curators. asrbench builds
references from the original NXT annotations, so expect a small, systematic
difference; do not put an asrbench AMI number in the same column as a published one
without saying so.

**`ami-excerpts` is the streaming path.** It cuts fixed windows (20.16 s, 60 s) from
whole headset channels per a committed recipe. The built-in recipe
`ami-ihm-excerpts-v1` is the eleven-excerpt set from the author's
nemotron-omni-port project (its `examples/ami-streaming-eval.recipe.json`), with the per-excerpt WAV sha256s it
published; the builder refuses a cut whose bytes differ, which catches a different
corpus release or channel. Each excerpt records its qualifying pauses and speech
fraction (same -50 dBFS absolute floor and 20 ms frames as the omni-port survey). Its
reference is the headset's own speaker's words whose midpoint lies in the window.
**Crosstalk from other participants is audible on every headset and is not in that
reference**, so excerpt WER mixes recognition errors with insertions of other
speakers' words. Read it as a streaming condition, not as an accuracy figure.

### LibriSpeech

`librispeech` writes one clip per utterance of a split (`dev-clean`, `dev-other`,
`test-clean`, `test-other`), in speaker, chapter, utterance order, with the
`.trans.txt` line verbatim as the reference. FLAC decoding is not in the package (it
would need a subprocess); `scripts/decode_librispeech.py` writes a 16 kHz PCM16 WAV
beside each FLAC with `ffmpeg`, on either side of the air gap. `--limit N` keeps the
first N utterances and names the manifest `librispeech-<split>-first<N>` so a
truncated set is never mistaken for the split.

## 8 kHz and 16 kHz

`--audio-condition` plays the same manifest (or clip) three ways:

| Condition | Content | Sent at | Question |
|---|---|---|---|
| `native` | as staged, 16 kHz | 16 kHz | wideband accuracy |
| `telephony_8k` | low-passed at 3.8 kHz, decimated | 8 kHz, `sample_rate_hertz: 8000` | does the server's 8 kHz path work, and at what WER |
| `telephony_8k_in_16k` | the same narrowband signal, interpolated back | 16 kHz | narrowband content on a 16 kHz-only server |

The filter is a 127-tap Blackman-windowed sinc, unity DC gain
(`fir_blackman_sinc_127tap_fc3800_v1`), in pure Python so the core stays
stdlib-only. Source digests do not change; the condition, the filter name, and the
sent rate are added to the workload fingerprint, so an 8 kHz and a 16 kHz receipt of
the same clips share `dataset.manifest.digest` but never a comparison key.
`dataset.audio_condition` records both rates.

Adapters declare the rates they can label on the wire. `riva-grpc` and
`nim-realtime` send the rate in `RecognitionConfig.sample_rate_hertz` and
`input_audio_params.sample_rate_hz`; `realtime-ws` is 16 kHz only and refuses
`telephony_8k`. Whether a given NIM profile *accepts* 8 kHz is the server's choice:
many streaming English profiles are deployed for 16 kHz only and either reject 8 kHz
or resample internally. Run `telephony_8k` once as a smoke test; if the server
refuses it, the failure is on the receipt, and `telephony_8k_in_16k` is the fallback
that still measures narrowband content.

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
