# Roadmap

Where the project is, and what each remaining step must deliver. The design is in
[`design.md`](design.md); the measurement definitions are in [`metrics.md`](metrics.md).

## Origin

This framework is an extraction, not a greenfield design. A working streaming ASR
benchmark existed first — fused to one server, one protocol, and one model — and was
used in anger before any of it was generalized. Extraction started only after the
receipt schema had survived real runs, so the public interface is shaped by what
measurement actually required rather than by what looked tidy in advance.

Two things follow. The interfaces here have already been driven by a real transport,
so they are not speculative. And the private harness remains the proving ground:
material moves out to this repository once it is target-neutral, and moving it is a
deliberate diff rather than a copy, so the two do not silently diverge.

## The arc

| Step | Deliverable | Done when |
|---|---|---|
| **0** ✅ | Public repo scaffold | Licence, NOTICE, attribution, corpus policy, CI matrix, boundary test, agent instructions |
| **1** ✅ | Adapter seam behind an existing transport | Both existing runners untouched at their call sites; receipt byte-identical on replay; existing tests green |
| **2** ✅ | The `asrbench` package: workload, outcome, reducers, reporters, config, tracking | Golden-schema and mock-adapter tests define the contract; percentile, identity, and commit helpers exist exactly once |
| **3** | Riva public gRPC adapter and a vendor-compatibility preset | Published streaming and offline commands reproduced through the common runner, three trials, TSV emitted |
| **4** | vLLM-style offered load | Request rate, burstiness, max concurrency, SLO goodput, and mixed-cadence vectors compose with TTFS and outcome accounting without weakening the homogeneous preset |
| **5** | 1.0 | Semantic versioning on the receipt schema and the adapter interface; deployment recipes demoted to optional examples |

Steps 0 and 1 are complete. Step 1 was *built* in the private harness, because its
entire acceptance was that an existing regression net still passed and that net lived
there; doing it here would have changed two repositories at once and weakened the
claim. It has since been migrated out, leaving the harness as it was before the
extraction.

## Current position

**Step 2's measurement core is in this repository.** Workload, target, and run specs,
the receipt, reducers, reporters, comparison, the mock preset, and the generated
schema reference are in the tree. Step 3 — a second public transport — is next.

**Step 0 is in this repository.** Scaffold, licence and attribution, corpus policy,
the stdlib-only boundary test, CI on Python 3.10 and 3.12, and the agent instruction
files.

**Step 1's seam has landed here.** It was built and proven in the private harness,
where the acceptance bar was that an existing regression net still passed — proven by
differential equivalence rather than asserted: pre-change and post-change
implementations were run side by side against byte-identical scripted servers across
fourteen cases, and key order, key counts, types, and every non-timing value came out
identical. It has since been migrated and the private harness restored to its
pre-seam state, so there is now exactly one copy.

| Module | Role |
|---|---|
| `outcome.py` | `SessionRequest` (frozen) and `SessionOutcome` — the two neutral shapes |
| `adapter.py` | The `TransportAdapter` protocol, plus admission-barrier and progress-callback protocols |
| `adapters/realtime_ws.py` | The realtime WebSocket adapter, the first implementation |
| `audio.py` | Sample-rate constants, clip duration, PCM16 read and chunk |
| `clock.py` | Timestamp helper |

One module was deliberately **not** migrated: a compatibility shim that translated an
outcome back into the private harness's pre-existing record shape. It existed to keep
one caller byte-compatible during the extraction, and shipping it would have exported
a dialect this project does not speak.

The measurement core sits on that seam:

| Module | Role |
|---|---|
| `spec.py`, `config.py` | Workload, target, and run specs, resolved once and frozen |
| `fingerprint.py`, `canonical.py`, `compare.py` | Workload fingerprint and the comparison key |
| `reducers.py`, `percentile.py`, `wer.py` | Metric families; one percentile implementation |
| `receipt.py`, `tracking.py`, `reporters.py` | The receipt, a rebuildable index, projections |
| `runner.py`, `adapters/mock.py`, `cli.py` | Closed-loop lanes and the mock path |
| `identity.py` | Hardware probe and the one git-commit helper |
| `definitions.py`, `schema_doc.py` | Emitted constants and the generated schema reference |

## Step 2 — the framework package

Build the measurement core on top of the migrated seam.

### Then build

**Reducers and reporters.** The metric families in [`metrics.md`](metrics.md), each
carrying its clock and definition as an emitted constant that the receipt writes.
Reporters are pure projections of the receipt.

**Configuration.** The three specs from [`design.md`](design.md), the resolution
chain, environment-only secrets, and the reproduction property — re-running from a
receipt reconstructs an equal resolved configuration.

**Tracking.** The receipt as the unit of truth, a derived and rebuildable index, and
a comparison that refuses on comparison-key mismatch naming the component that
diverged. Trials stay raw beside their summary. Exporters to external experiment
trackers are optional plugins, never core dependencies.

**Documentation.** A generated receipt-schema reference checked in CI against a fresh
generation, and documentation code blocks that execute against the mock adapter — a
quickstart that no longer runs should be a test failure, not a bug report.

### Shape decisions now in the measurement core

These were settled after step 1 and are implemented in the comparison key and
the frontier reporter:

1. **The lookahead frame count is inside the comparison key**, not merely
   reported beside it. Two receipts at different lookaheads measure different
   operating points of the model, so pooling them is as wrong as pooling across
   hardware. The cadence that sets that lookahead is in the key as well. The
   frontier removes both when it groups a sweep. `chunk_ms` stays: it is
   transmit size.
2. **The frontier is the primary panel** — accuracy against measured latency,
   one point per lookahead and cadence, Pareto edge marked — rather than a
   cadence-by-concurrency matrix. The matrix was designed on the assumption that
   cadence varies load at a fixed result; it varies the result too.

### Acceptance

```bash
asrbench run --preset presets/<name>.json --out r1.json
asrbench run --from-receipt r1.json --out r2.json
# resolved configuration in r1 equals r2; comparison keys identical
asrbench compare r1.json r2.json                     # succeeds
asrbench compare r1.json <receipt-from-other-gpu>    # must refuse, naming hardware identity
```

**The duplication canary:** the private harness currently has percentile, GPU
identity, and git-commit helpers duplicated across two runner scripts. If they still
exist twice after step 2, the runner split did not actually happen.

## Steps 3 through 5

**Step 3** adds the Riva public gRPC adapter and the vendor-compatibility preset,
reproducing published streaming and offline figures through the common runner. This
is where the adapter interface earns the name — an interface with one implementation
has not been shown to be an interface, and the mock adapter is a proof of concept,
not a second protocol. An adapter-authoring guide is written after this, not before,
so it describes an interface that has survived a second transport.

StreamingRecognize, the NIM compatibility preset, and the NIM transcription
WebSocket are in the tree. Offline Recognize is not implemented, so this step
is not done.

Vendor lanes that require entitlement must degrade to "lane unavailable" rather than
failing the run. Public users will not have that access, and a benchmark that refuses
to run for them is not public.

**Step 4** adds offered-load modes: request rate, burstiness, maximum concurrency,
SLO goodput, and mixed-cadence population vectors. The constraint is that these
compose with the streaming metrics without weakening the homogeneous
vendor-comparable preset, which is what makes cross-vendor numbers meaningful.

**Step 5** is 1.0: semantic versioning on the receipt schema and the adapter
interface, and deployment recipes demoted to optional examples.

## Standing constraints

These do not expire with a step.

- **The core stays stdlib-only**, enforced by the boundary test rather than by review.
- **The default test tier needs no GPU, no server, and no network.**
- **Metric definitions live in code as emitted constants**, quoted by the docs.
- **Failures stay in every denominator.**
- **Sources of record are public and reachable.**

See [`AGENTS.md`](../AGENTS.md) for the working rules these imply.
