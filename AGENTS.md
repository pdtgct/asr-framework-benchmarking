# AGENTS.md

Working instructions for AI coding agents in this repository. This is the canonical
file; `CLAUDE.md` imports it.

## What this repository is

`asrbench` is a benchmark for **streaming speech recognition servers**, measured
through their public protocols. It is not an inference engine, not a model port, and
not a deployment tool. It sends audio to something already running and reports what a
caller experiences.

Distribution name and import package are `asrbench`; the repository is named
`asr-framework-benchmarking`. That mismatch is intentional and not a bug to fix.

**Status: pre-1.0. Schemas are unstable.** Changing one is allowed; changing one
silently is not.

## Layout

```
src/asrbench/            the framework — stdlib only
src/asrbench/adapters/   transport adapters — the only place a client library lives
tests/                   default tier: no GPU, no server, no network
docs/                    corpus policy, methodology, generated schema reference
datasets.lock            audio pinned by digest; recipes committed, audio never
```

## Setup and gates

```bash
uv venv
uv pip install -e ".[dev]"
```

Run all four before proposing a change. CI runs the same on Python 3.10 and 3.12.

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy          # strict, over src and tests
uv run pytest -q
```

Never use system `python3` or bare `pip`. If `VIRTUAL_ENV` points somewhere else, an
install will land in the wrong environment without saying so — check before you
install, and check `uv pip list` after.

## Invariants

Four rules. Each protects something a reviewer cannot easily re-derive by reading a
diff, so each is enforced by a test rather than by review.

**The core stays stdlib-only.** Schemas, runner, reducers, and reporters import
nothing outside the standard library. A transport's client library is confined to the
adapter that speaks that protocol, declared as an extra in `pyproject.toml`. If you
need a library in the core, that is a decision to surface, not a dependency to add:
say so and stop. Enforced by `tests/test_core_boundary.py`, which reads its permitted
set out of `pyproject.toml`, so declaring the extra is the single act that widens the
boundary.

**The default test tier needs no GPU and no server.** A benchmark others cannot run
is a claim, not a measurement. Tests requiring a live endpoint carry
`@pytest.mark.live` and are excluded by default. If a change makes the mock path
unrunnable, stop — the framework has stopped being publicly reproducible.

**Metric definitions live in code, not prose.** Every metric carries its clock and
its definition as an emitted constant; the receipt writes it; the documentation
quotes the constant. A definition that exists only in Markdown will drift from the
number it describes. Never restate a clock definition in a doc as free text.

**Failures stay in the denominator.** No reducer may silently drop a session that
lacks a measure. Percentiles are conditional on outcomes that possess the measure,
and the population count travels beside them. Dropping failures is how an overloaded
server is made to look fast.

## Measurement honesty

These are not style preferences — each is a specific way results get misread. Do not
soften, merge, or "simplify" any of them.

- **Streaming RTFX is a concurrency measure, not a speedup.** Under real-time pacing
  it tracks stream count almost exactly. It is not an inference-speed number.
- **A vendor's response latency and this project's TTFS are different metrics**, with
  different origins and different sample counts. Compatibility output may carry both;
  it must never relabel one as the other.
- **Client and server clocks are different clocks.** Correlate by request; never
  subtract one from the other.
- **WER is comparable only within one named normalization policy.**
  Engine-vs-own-oracle is compared raw — normalizing there lets a real regression
  hide behind text cleanup. Engine-vs-corpus-reference is the full-normalization
  policy. Another accuracy policy needs its own hypothesis source; a second name
  on the corpus-reference normalizer is a relabel.
- **Cadence is an accuracy dial, not just a latency knob.** For cache-aware streaming
  models the service cadence sets the encoder's lookahead, so a longer cadence
  produces a *different and generally more accurate* transcript. Cross-cadence
  transcript agreement is **not** an invariant and must never become a pass
  condition. The invariant that holds is streaming ≡ offline at a fixed lookahead.
  Identify an operating point by **lookahead frame count**; carry the millisecond
  label beside it, because vendors disagree on the millisecond convention by one
  frame.
- **Two receipts are comparable only when their comparison key matches.** The tool
  refuses rather than guessing. A silent cross-hardware or cross-model delta is worse
  than no delta.

## Adding a transport adapter

An adapter owns protocol mechanics and timestamps — nothing else. It takes a
`SessionRequest`, returns a `SessionOutcome`, and never learns what workload it is
part of or how its result will be reduced. Diagnostics are bounded lifecycle edges,
never per-chunk logging.

1. Add the client library as its own extra in `pyproject.toml`.
2. Register the extra in `_TRANSPORT_EXTRAS` in `tests/test_core_boundary.py`.
3. Implement `TransportAdapter` under `src/asrbench/adapters/`.
4. Add a test that drives it without the network, so the interface is exercised on
   the default tier.

## Corpora

Commit recipes and digests, never audio. Every `datasets.lock` entry carries its
licence and the attribution a published result must reproduce; derived audio inherits
the source corpus's licence. Do not add a corpus requiring registration or carrying a
non-commercial or no-derivatives restriction.

Two corpus facts are measured, not assumed, and must survive any edit to
`docs/corpus.md`: endpointing cannot be evaluated on LibriSpeech, and clip duration
thresholds derive from the model's context geometry rather than from convenience.

## Sources

Sources of record are **public and reachable**: published papers, public model cards,
public vendor documentation, and public client repositories. Do not introduce
material from any non-public source, and do not cite a source a reader cannot reach.
When a fact is available from more than one place, cite the public one.

## Commits

Sign off with `git commit -s`. Contributions are under Apache-2.0, per that licence's
§5 and this project's `LICENSE`.

Write commit messages that say what changed and why it is safe. When a change alters
the receipt schema, bump the schema version string in the same commit and state what
a consumer of the old version must do.
