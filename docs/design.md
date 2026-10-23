# Design

The architecture of record. What the pieces are, what each owns, and which
boundaries are load-bearing.

## Purpose

Compare containerized ASR servers through their **public protocols**. The framework
depends on no model code, no inference engine, no Kubernetes, no cloud provider, and
no profiler. If a server speaks the protocol, it can be measured; if it does not, no
amount of privileged access to its internals would make the comparison fair.

That constraint is the design. Everything below follows from it.

## Source-grounded conventions

The structure is modeled on vLLM's serving benchmark, checked against vLLM
`702f4814fe54fabff350d43cb753ae3e47c0c276` — particularly
`vllm/benchmarks/serve.py` and `vllm/benchmarks/lib/endpoint_request_func.py`, where
transport adapters return a common request result and the runner owns offered load,
concurrency, warmup, percentiles, raw-result persistence, and SLO-based goodput — and
against vLLM-Omni `00f1abd192643ed0878b190f4ffb26d16c02d2e7`, where modality-specific
metrics extend the common serving benchmark rather than creating a second
orchestration system. `RequestFuncOutput` there already carries an audio-duration
field, so upstream has begun extending that shape for audio; mirroring it is what
makes this read as native to a vLLM maintainer.

Streaming methodology follows
[NVIDIA's published ASR benchmark](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html):
real-time pacing with equal client and server chunk duration, three iterations per
stream over a public LibriSpeech clip, a fivefold concatenation for unpaced offline
maximum throughput, throughput as audio duration over computation time, and
three-trial averaging with variability-aware rounding.

A vendor's reported response latency and this project's TTFS are **different
metrics**. Compatibility output may carry both; it must never relabel one as the
other.

## Architecture

```text
deployment provider (optional, out of process)
        |
        v
workload spec -> runner -> transport adapter -> public server protocol
                    |              |
                    v              v
              raw outcomes <- protocol timestamps
                    |
                    v
              metric reducers -> receipt -> JSON / TSV / console reporters
                    |
                    +---- optional profiler evidence attachment
```

Four owners, and the separations between them are what keep results comparable:

- The **workload** owns what work is offered. It is serializable and fingerprinted.
- The **adapter** owns protocol mechanics and timestamps. Nothing else.
- The **runner** owns concurrency, arrivals, warmup, and trials.
- The **reducers and reporters** own interpretation, and are pure projections of the
  raw outcomes the receipt already contains.

### Workload specification

One serializable workload owns streaming or offline mode; public clip identity and
digest; real-time pacing, chunk duration, and requested service cadence as three
distinct values; fixed parallel streams, a finite request rate, or an explicit
arrival trace; per-lane iterations, warmup, trial count, timeout, and random seed;
language, punctuation, diarization, interim-result, and transcript-output policy; and
a homogeneous or mixed-cadence population vector.

The workload fingerprint **excludes observation tools and includes everything that
can change the work offered to the server**. That line is the whole point of the
fingerprint: attaching a profiler must not change the identity of the workload, and
changing the chunk size must.

### Transport adapters

An adapter takes a `SessionRequest` and returns a `SessionOutcome`. It never learns
what workload it is part of, how many lanes are running, or how its result will be
reduced.

Every adapter returns the same outcome shape: connection, configuration, and
admission states; first-sample send time; first non-empty partial time; response
latencies; final time; audio duration; transcript; public error code; timeout; and
terminal status. Protocol-specific residue travels in a named extras field rather
than widening the common shape — a WebSocket dialect must not become the framework's
vocabulary.

Adapter diagnostics are bounded lifecycle edges, never per-chunk log spam.

Planned adapters: an OpenAI-style realtime WebSocket, NVIDIA Riva public gRPC, and
offline HTTP or gRPC transcription for maximum-throughput cells.

### Runner

The runner owns lane concurrency and arrivals: closed-loop lanes with three
sequential iterations per stream for vendor-comparable cells; request-rate and
maximum-concurrency sweeps for vLLM-comparable cells; a start barrier so paired
control and profile runs begin together; warmup excluded from measured denominators;
three-trial orchestration with the raw trials retained; and fail-closed timeouts with
complete failure accounting.

Deployment providers — local process, Docker, Kubernetes, a cloud GPU host — may
start, update, probe, and stop targets, but **the measurement core does not import
them**. A benchmark can always target an already-running endpoint.

### Metrics and pass semantics

See [`metrics.md`](metrics.md) for the definitions. Two structural rules live here:

**Percentiles are conditional on outcomes that possess the measure.** A session that
never produced a first partial has no TTFS; it is not a fast session and it is not an
absent one. So a cell passes only through an explicit **conjunction** — admission
fraction, first-partial presence, latency SLO, and sustainability together — and
failures stay in every population denominator. Without this, survivorship makes an
overloaded server look fast.

**Accuracy is orthogonal to performance.** Transcript quality is a reducer over the
same outcomes, never inferred from timing and never traded against it in a single
score.

### Receipt and reporters

The receipt is JSON, authoritative, and one per run. It carries the schema version;
the complete raw per-session outcomes; workload and observation fingerprints; the
exact clock definitions as emitted constants; server, model, software, and hardware
identity; attempted and successful audio denominators; trial identity with raw trial
aggregates beside the cross-trial summary; and public dataset attribution with clip
digests.

Reporters are projections only — a vLLM-shaped JSON and console summary, a
vendor-comparable TSV for homogeneous cells, a compact evidence card. A reporter that
cannot be regenerated from the receipt is a bug in the receipt.

Profiler integrations are plugins. Missing profiler support must not prevent a
protocol benchmark, and profiler-derived values never replace client-visible metrics.

## Comparability

Two receipts are comparable only when their **comparison key** matches. The key
hashes four hard components:

```
comparison_key = sha256(workload_fingerprint ‖ target_identity
                        ‖ software_identity ‖ hardware_identity)
```

Hard target identity is the model, the transport, `transport_options`, and the
server version. Hard software identity is the adapter, the normalization policy,
the percentile method, `SCHEMA_VERSION`, and `MEASUREMENT_SEMANTICS_VERSION`.
Hard hardware identity is GPU name, count, and memory, plus
`gpu_identity_source`: `server_declared` when the target spec names a server
GPU, otherwise `client_probe` from nvidia-smi. Those two sources are not mixed.

Warn-only fields are stored and listed on a mismatch, and they do not refuse:
driver version, Python version, this package's git commit and dirty bit,
endpoint URL, TLS verification, the auth variable name, package version, the
lookahead-identity policy, and the client CPU and host. Git commit is read
from asrbench's own checkout. `docs/receipt-schema.md` quotes
`COMPARISON_KEY_DEFINITION` and `WARN_ONLY_COMPARISON_DEFINITION`.

Comparison refuses when a hard field differs and names that field. Forcing a
comparison requires an explicit reason, recorded only when the result is
forced. A reason with no hard mismatch is not recorded. A silent
cross-hardware or cross-model delta is worse than no delta.

## Configuration

Three specs, separated by their fingerprint semantics rather than by subject matter:

| Spec | Owns | Fingerprinted | In receipt |
|---|---|---|---|
| Workload | Everything that changes offered work | Yes | Yes |
| Target | Endpoint, transport kind, model name, auth variable *name*, TLS | No — changes no offered work | Yes, as identity |
| Run | Output paths, verbosity, plugin toggles | No | Observation fingerprint only |

Resolution order is dataclass defaults, then preset file, then `ASRBENCH_*`
environment, then CLI flags, resolved once into a frozen configuration.

**Secrets are environment-only.** Configuration carries a variable *name*; the receipt
records the name and whether it was set, never a value.

The acceptance property is reproduction: re-running from a receipt must reconstruct a
resolved configuration equal to the one that receipt embeds, modulo output paths. If
that holds, the resolved configuration is necessarily complete and there is no hidden
ambient state. It is a test, not a style rule.

## Decisions and alternatives

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Adapter interface shape | Mirror vLLM's request-in / result-out protocol | Bespoke streaming-first interface | Transplantability upstream, and a maintainer recognizes it on sight |
| Core dependencies | None; transports are extras | Depend on one HTTP/WS client | A benchmark that cannot install cannot be reproduced |
| Result storage | Receipt file, derived index | Database | No service to stand up before a first measurement |
| Cross-run comparison | Refuse on key mismatch | Warn and proceed | A silent invalid delta is the failure mode that matters |
| Protocol residue | Named extras field | Widen the common outcome | Prevents one protocol's dialect becoming the vocabulary |
| Trials | Raw trials retained beside the summary | Summary only | Dispersion is a result, not a rounding detail |
