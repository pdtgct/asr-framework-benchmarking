# Metrics

What is measured, from which clock, over which denominator.

This is the design; the definitions become emitted constants in code, and the receipt
writes them, so a number and its definition cannot drift apart. Where this document
and a receipt disagree, the receipt is right and this document is stale.

Those constants live in `asrbench.definitions`. [`receipt-schema.md`](receipt-schema.md)
quotes each one. A clock or a definition copied into prose here would drift from the
number it describes, so this file keeps the design and the schema reference keeps the
spelling.

Four families. They are kept separate because they fail independently: a server can
be accurate and slow, fast and unstable, or stable and unable to hold concurrency.
Collapsing them into one score hides exactly the trade a reader needs to see.

## 1. Accuracy

| Metric | Definition |
|---|---|
| **WER** | Word error rate against a reference, under a **named** normalization policy |
| **Substitutions / deletions / insertions** | The three WER components, reported separately |
| **CER** | Character error rate, for scripts and conditions where word boundaries mislead |
| **Per-utterance distribution** | Not just the corpus aggregate — a mean hides a handful of catastrophic streams |

Breaking WER into S/D/I is not decoration. Deletions rising under load means the
server is dropping audio; insertions rising means it is hallucinating through
silence; substitutions rising means it is simply mishearing. One aggregate number
cannot distinguish a capacity bug from an acoustic one.

**Normalization is part of the metric, not a preprocessing step.** Three comparisons
exist and they must not share a normalizer:

1. **Engine vs its own reference oracle** — compare **raw**, no normalization.
   Normalizing here lets a real regression hide behind text cleanup.
2. **Engine vs corpus reference** — full normalization: case-fold, strip punctuation,
   strip control tags, normalize numbers, collapse whitespace.
3. **Engine vs another server** — full normalization plus a declared tolerance.

On read speech, number-formatting differences dominate WER and will swamp the
streaming behavior being measured. A word error rate published without its policy
name is not a measurement.

## 2. Streaming latency

| Metric | Clock and origin |
|---|---|
| **TTFS** | First audio sample transmitted → first **non-empty** partial received, one client monotonic clock |
| **Response latency** | Per response, relating each to the audio position it covers; report mean and p50/p90/p95/p99 |
| **Finalization latency** | Last audio sample sent → final result received |
| **Emission lag** | A word's audio ending → that word's first appearance in any output |
| **Lag excess** | Emission lag minus the cadence's theoretical floor — what the *implementation* costs, over what the *configuration* costs |
| **Pacing fidelity** | Deviation between intended and actual chunk send times, from the client's send trace |

Four rules:

- **The TTFS origin is immediately before transmission of the first audio sample can
  yield.** Connection setup, configuration, and admission wait are separate
  measurements and are never silently folded in.
- **A session with no non-empty partial is a failed outcome, not a missing sample.**
  It stays in the denominator.
- **TTFS and per-response latency are different metrics** — one sample per stream
  versus one per response. Vendor tables typically publish the latter.
- **Pacing fidelity qualifies everything else.** If the client failed to deliver audio
  on schedule, every latency in that cell is measuring the client. It is a validity
  check, not a performance number.

**Emission lag needs ground truth**, and the honest source is construction rather
than inference: fixtures built by inserting silence of an exact frame count between
known utterances carry their boundary positions from the moment they are written. No
forced aligner, and no dependence on a voice-activity detector whose defaults would
erase the sub-100 ms gaps such fixtures exist to probe.

**Lag excess is the number that survives a cadence change.** Raw emission lag is
mostly the cadence's own lookahead, so it cannot compare two cadences. Subtracting
the theoretical floor leaves what the implementation added.

## 3. Stability

| Metric | Definition |
|---|---|
| **Prefix violations** | Count of interims that are not a prefix of the following interim or of the final |
| **Revision rate** | Fraction of emitted tokens later retracted or rewritten |
| **Interim cadence conformance** | Distribution of inter-interim intervals against the configured cadence |

For an append-only streaming decoder, **every interim must be a prefix of the next
and of the final**. This is the strongest correctness property a streaming ASR client
can check without a reference transcript: it needs no ground truth, it is exact
rather than statistical, and a violation is unambiguously a server defect.

A caption that rewrites itself is a bad caption even at a perfect WER, which is why
revision rate is reported rather than averaged away.

## 4. Capacity

| Metric | Definition |
|---|---|
| **Sustained streams at SLO** | Largest concurrency at which the full pass conjunction still holds |
| **Streaming RTFX** | Successfully transcribed audio seconds ÷ wall-clock span of the cell |
| **Offline RTFX** | The same ratio for an unpaced batch — the maximum-throughput measure |
| **Goodput** | Streams meeting every SLO simultaneously, per unit time |
| **Admission and productivity counts** | Attempted, admitted, completed, productive, errored, timed-out, no-first-partial |

**Streaming RTFX is a concurrency measure, not a speedup.** Under real-time pacing
audio arrives in real time, so RTFX tracks stream count almost exactly — 1.0 at one
stream, 8.0 at eight. Read it as "this many real-time streams sustained," and read
the latency percentiles for responsiveness. Offline RTFX is a genuine throughput
ratio and may be far larger; the two are not interchangeable, and a paced run with
pacing merely disabled is not an offline substitute because it still exercises
realtime admission and buffering.

**Sustained streams at SLO is the headline capacity number**, because it is the only
one that cannot be improved by degrading service. RTFX rises when a server admits
more streams and serves them all badly; the sustained figure does not.

## The pass conjunction

A cell passes when **all** of the following hold on the same run. Any one alone is a
partial view that can be gamed:

1. Admission fraction ≥ threshold
2. First-partial presence ≥ threshold
3. Latency SLO met at the declared percentile
4. Sustainability condition met over the declared steady interval
5. Accuracy within tolerance of the declared reference

Failures remain in every denominator throughout.

## Reading a cadence sweep

For cache-aware streaming models, the service cadence sets how many future frames the
encoder may attend to before committing a representation. A longer cadence therefore
produces a **different and generally more accurate** transcript — not the same
transcript, later.

So a cadence sweep is a **latency/accuracy Pareto frontier**, not a consistency
check. Cross-cadence transcript agreement is not an invariant and must never become a
pass condition. The invariant that does hold is **streaming ≡ offline at a fixed
lookahead**.

Two consequences for reporting:

- **The lookahead belongs inside the comparison key.** Two receipts at different
  lookaheads describe different operating points of the model, so pooling their WER
  is as wrong as pooling across hardware.
- **Identify an operating point by lookahead frame count**, and carry the millisecond
  label beside it. Vendors disagree on the millisecond convention — some count
  lookahead alone, some count lookahead plus the emitted frame — so two tables can
  describe the same configuration with labels one frame apart.

Across cadences, expect only directional properties: WER non-increasing in lookahead,
and no dominated operating point. A configuration that is both slower and less
accurate than another is a misconfiguration, and that is a real finding.

## Presentation

Numeric first. A dashboard for this data is an instrument, not an infographic:
tabular figures with aligned digits, distributions rather than bare means, and every
number carrying its denominator and its unit.

The primary panel is the **frontier**: accuracy against measured latency, one point
per lookahead, with the Pareto edge marked. It answers "which operating point should
I run?" directly, where a cadence-by-concurrency matrix leaves the reader to
cross-reference two tables. Concurrency scaling, latency distributions, and the
outcome-accounting breakdown are the supporting panels.

State encodes in form as well as number — a cell that fails the conjunction should
read as failed at a glance, and it should be visible *which* clause it failed.
