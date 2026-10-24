# asrbench

A benchmark for **streaming speech recognition servers**, measured through their
public protocols.

Point it at a container that speaks a streaming ASR protocol — an OpenAI-style
realtime WebSocket, NVIDIA Riva gRPC, an offline HTTP transcription endpoint — and
it reports what a caller actually experiences: how fast the first words come back,
how many concurrent real-time streams the deployment sustains, how accurate the
transcript is, and how much of that survives under load.

It imports no model code, no inference engine, and no orchestration system. If the
server speaks the protocol, it can be measured.

## Status

**Pre-1.0. Schemas are unstable and will change without a migration path.**

The measurement core is in the tree: a frozen workload, one receipt per run, reducers
that keep failures in the denominator, and reporters that read only that receipt.
`presets/mock-streaming.json` runs with no server and no network. `riva-grpc` speaks
NVIDIA's public StreamingRecognize protocol, and `presets/nim-compat-streaming.json`
is the vendor-compatibility cell. `presets/nim-compat-smoke.json` is the single-GPU
smoke: interims on, three explicit iterations, TTFS p99 under 1 s, and
corpus-reference WER within 0.10. Finalization is written beside that receipt and
is not a pass gate. `scripts/prepare_librispeech_clip.py` turns the locked FLAC
into the WAV that preset names and prints the WAV sha256. Set `--model` to the
`model_name` the server reports. `nim-realtime` speaks the public NIM transcription WebSocket
(`/v1/realtime?intent=transcription`). That is a different dialect from
`realtime-ws`. Offline Recognize is not implemented, so roadmap step 3 is not closed.
See [`docs/roadmap.md`](docs/roadmap.md).

Version 1.0 is when the receipt schema and the adapter interface take semantic
versioning.

## Documentation

| | |
|---|---|
| [`docs/design.md`](docs/design.md) | Architecture: what each piece owns, and which boundaries are load-bearing |
| [`docs/metrics.md`](docs/metrics.md) | What is measured, from which clock, over which denominator |
| [`docs/corpus.md`](docs/corpus.md) | Which audio answers which question, and how each artifact is prepared |
| [`docs/roadmap.md`](docs/roadmap.md) | Current position and the remaining steps |
| [`docs/receipt-schema.md`](docs/receipt-schema.md) | Generated receipt shape and the emitted definition strings |
| [`AGENTS.md`](AGENTS.md) | Working rules — canonical for both humans and agents |

## What it measures

Four families, kept separate because they fail independently.

| Family | Answers | Examples |
|---|---|---|
| **Accuracy** | Did it hear the words? | WER with substitutions / deletions / insertions broken out, CER, per-utterance distribution |
| **Streaming latency** | How fast does a caller see output? | Time to first speech (TTFS), per-response latency, finalization latency, emission lag |
| **Stability** | Is the stream well-behaved? | Prefix violations, revision rate, interim cadence conformance |
| **Capacity** | How much load holds up? | Sustained real-time streams at an SLO, streaming RTFX, offline RTFX, goodput |

### Rules that keep the numbers honest

These are not style preferences. Each one is a way benchmark results get misread,
and the framework enforces them rather than documenting them.

- **Failures stay in every denominator.** Percentiles are conditional on outcomes
  that possess the measure, so a cell passes only through an explicit conjunction —
  admission fraction, first-partial presence, latency SLO, and sustainability
  together. Otherwise survivorship makes an overloaded server look fast.
- **Streaming RTFX is a concurrency measure, not a speedup.** Under real-time
  pacing it tracks stream count almost exactly. Read it as "this many real-time
  streams sustained," and read the latency percentiles for responsiveness.
- **Client and server clocks are different clocks.** Correlate by request; never
  subtract one from the other.
- **WER is comparable only within one named normalization policy.**
  Engine-vs-own-oracle is raw. Engine-vs-corpus-reference is full normalization.
  Another accuracy policy needs its own hypothesis source.
- **Two receipts are comparable only when their comparison key matches.** A silent
  cross-hardware or cross-model delta is worse than no delta, so the tool refuses
  rather than guessing.

### Cadence is an accuracy dial, not just a latency knob

For cache-aware streaming models, the service cadence sets how many future frames
the encoder may attend to before committing a representation. A longer cadence
therefore produces a **different and generally more accurate** transcript — not the
same transcript, later.

So a cadence sweep is a latency/accuracy Pareto frontier, not a consistency check.
Cross-cadence transcript agreement is not an invariant and must never be asserted as
a pass condition. The invariant that does hold is **streaming ≡ offline at a fixed
lookahead**.

Identify an operating point by its **lookahead frame count**, and carry the
millisecond label beside it. Vendors do not agree on the millisecond convention —
some count lookahead alone, some count lookahead plus the emitted frame — so two
tables can describe the same configuration with labels one frame apart.

## Corpora

Everything is reproducible from public, permissively licensed sources. No corpus used
here requires registration, and none carries a non-commercial or no-derivatives
restriction. Selection, geometry, and the preparation recipes are in
[`docs/corpus.md`](docs/corpus.md).

Two findings from that document are worth stating up front, because both are
measured rather than assumed:

- **Do not evaluate endpointing on LibriSpeech.** Its utterances ship pre-trimmed, so
  concatenating them yields near-continuous speech. Across 523 s and 87 utterances,
  only three silences reached 800 ms. An endpointing suite built on it is vacuously
  green.
- **Clip duration is dictated by model geometry, not convenience.** Audio shorter
  than one full encoder context never reaches the saturated steady state the model
  runs in during real use, so it cannot support a claim about streaming behavior.

Corpus contents are pinned by digest in [`datasets.lock`](datasets.lock). Recipes are
committed; audio is not.

## Installing

Requires Python 3.10 or newer.

```bash
uv venv
uv pip install -e ".[dev]"
```

## Quickstart

The mock preset needs no server, no GPU, and no network. Re-running from the receipt
reproduces the resolved configuration, and the two receipts compare.

<!-- asrbench:doc-test -->
```bash
uv run asrbench run --preset presets/mock-streaming.json --out receipt.json
uv run asrbench run --from-receipt receipt.json --out receipt-again.json
uv run asrbench compare receipt.json receipt-again.json
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The core must remain runnable with no GPU and
no server — if a change makes the mock path unrunnable, the framework is no longer
publicly reproducible.

## Licence and attribution

Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

Apache-2.0 matches vLLM and vLLM-Omni, which is deliberate: the adapter and metric
shapes here are modeled on vLLM's serving benchmark, and an adapter written against
this interface should be transplantable upstream without a licence conversation.

Corpora are CC BY 4.0 and require attribution wherever results are published —
excerpts and concatenations inherit that requirement. Details in
[THIRD_PARTY.md](THIRD_PARTY.md).
