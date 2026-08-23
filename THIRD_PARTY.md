# Third-party attributions

This file records everything this project derives from, what the obligation is, and
what satisfies it. `NOTICE` is the short form; this is the detail.

## Software

### vLLM — Apache License 2.0

<https://github.com/vllm-project/vllm>

**What is derived:** interface shape and naming, not code.

vLLM's serving benchmark separates a transport function from the runner that owns
offered load, concurrency, warmup, percentiles, and goodput. A transport is a
callable taking a request-input record and returning a request-output record; the
runner never learns which protocol ran. This project adopts that separation, and
names its equivalents to match:

| vLLM | asrbench |
|---|---|
| `RequestFuncInput` | `SessionRequest` |
| `RequestFuncOutput` | `SessionOutcome` |
| `RequestFunc` (Protocol) | `TransportAdapter` (Protocol) |
| `calculate_metrics` | metric reducers over outcomes |
| `goodput_config_dict` | SLO conjunction in the pass semantics |

The correspondence is deliberate. A vLLM maintainer reading this codebase should
recognize the structure immediately, and an adapter written here should be
transplantable there.

**What is not derived:** no vLLM source file is copied, vendored, or mechanically
translated into this repository. The implementations are written from this project's
own contracts and tests.

**Obligation:** Apache-2.0 §4 attribution applies to distributed copies of the
licensed work. Since no vLLM source is distributed here, the requirement is not
triggered — this entry is recorded because acknowledging structural debt is correct
practice, not because a licence compels it. Should any vLLM source ever be vendored,
this entry must be upgraded to carry that file's own copyright and NOTICE content,
and the vendored file must retain its licence header.

This project is itself Apache-2.0, which is the same licence, chosen for that
reason: an adapter written against this interface can move upstream into vLLM or
vLLM-Omni without a licence conversation, and vice versa.

### NVIDIA Riva / Speech NIM — public documentation and clients

- Benchmark methodology:
  <https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html>
- Public client library and sample data:
  <https://github.com/nvidia-riva/python-clients>

**What is derived:** measurement methodology, so that results are comparable to
NVIDIA's published tables — real-time pacing with equal client and server chunk
duration, three iterations per stream, a fivefold-concatenated clip for the unpaced
offline throughput cell, throughput reported as audio duration over computation
time, and three-trial averaging.

**Boundary:** NVIDIA's reported response latency and this project's TTFS are
**different metrics**, measured from different origins with different sample counts.
Compatibility output may carry both; it must never relabel one as the other.

**Obligation:** none beyond accurate citation. Only public documentation, the public
`nvidia-riva/python-clients` repository, published papers, and public model cards are
used as sources.

## Corpora

Both corpora are licensed **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>), which permits commercial use and
derivative works and requires attribution.

### LibriSpeech ASR corpus

<https://www.openslr.org/12>

> V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: An ASR corpus
> based on public domain audio books," in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210.

Used for accuracy, long-stream stability, concurrency isolation, and reproducing
NVIDIA's published streaming figures.

### AMI Meeting Corpus

<https://groups.inf.ed.ac.uk/ami/corpus/>

> J. Carletta et al., "The AMI Meeting Corpus: A Pre-announcement," in *Proc. MLMI*,
> 2005, pp. 28–39.

Used for endpointing and conversational realism, because LibriSpeech cannot support
either — see `docs/corpus.md`.

### Derived audio

Every decode, excerpt, concatenation, and constructed fixture built from these
corpora is a derivative work and carries the same attribution requirement. This
repository therefore commits **recipes and digests, never audio**: `datasets.lock`
pins each artifact by SHA-256 and records the source it was derived from, so a
published result names exactly what it ran on.

## Obligations when publishing results

A result produced with this benchmark and published anywhere must:

1. Credit each corpus it used, per CC BY 4.0.
2. State the GPU class, the server build, and the model identity — the receipt's
   comparison key carries all three.
3. Report the normalization policy by name alongside any word error rate.
4. Report the lookahead frame count alongside any cadence expressed in milliseconds.
