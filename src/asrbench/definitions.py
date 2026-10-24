"""Emitted clocks, definitions, and named measurement policies.

Every public string constant here is copied onto the receipt and quoted by
the generated schema reference. A definition that exists only in prose will
drift from the number it describes; these constants are the measurement.

Clocks that already had a name on the outcome are re-exported, not
restated, so the outcome and the receipt cannot disagree about the spelling.
"""

from __future__ import annotations

from asrbench.outcome import (
    FINALIZATION_CLOCK as FINALIZATION_CLOCK,
)
from asrbench.outcome import (
    NATIVE_LATENCY_CLOCK as NATIVE_LATENCY_CLOCK,
)
from asrbench.outcome import (
    RESPONSE_LATENCY_CLOCK as RESPONSE_LATENCY_CLOCK,
)

# Same clock the outcome already names. The receipt key is TTFS so a
# reader does not have to know the outcome field's historical name.
TTFS_CLOCK = NATIVE_LATENCY_CLOCK

SCHEMA_VERSION = "asrbench-receipt/2"

#: Bumped when reducer or comparison-key semantics change. Replaces the
#: git commit in the hard comparison key, so a docs-only commit does not
#: refuse a comparison. Git commit stays on the receipt as warn-only.
MEASUREMENT_SEMANTICS_VERSION = "1"

GPU_IDENTITY_SERVER_DECLARED = "server_declared"
GPU_IDENTITY_CLIENT_PROBE = "client_probe"

#: Seconds of slack after the adapter's own waits and the barrier budget.
#: The adapter owns each ``timeout_s``. This grace is only the hang guard.
RUNNER_HANG_GRACE_S = 0.25

MODE_STREAMING = "streaming"
MODE_OFFLINE = "offline"

PERCENTILE_METHOD_NUMPY_LINEAR = "numpy_linear"
PERCENTILE_METHOD_NEAREST_RANK = "nearest_rank"
DEFAULT_PERCENTILE_METHOD = PERCENTILE_METHOD_NUMPY_LINEAR

LOOKAHEAD_IDENTITY_CLIENT_DECLARED = "client_declared"
LOOKAHEAD_IDENTITY_SERVER_REPORTED = "server_reported_when_present"
DEFAULT_LOOKAHEAD_IDENTITY_POLICY = LOOKAHEAD_IDENTITY_CLIENT_DECLARED

NORMALIZATION_ENGINE_VS_OWN_ORACLE = "engine_vs_own_oracle"
NORMALIZATION_CORPUS_REFERENCE = "corpus_reference"
DEFAULT_NORMALIZATION_POLICY = NORMALIZATION_ENGINE_VS_OWN_ORACLE

SUSTAINABILITY_STEADY_INTERVAL = "steady_interval"
SUSTAINABILITY_COMPLETION_FRACTION = "completion_fraction"
DEFAULT_SUSTAINABILITY_POLICY = SUSTAINABILITY_STEADY_INTERVAL

HARDWARE_PROBE_NVIDIA_SMI = "nvidia-smi"

TTFS_DEFINITION = (
    "Seconds from the first audio sample transmitted to the first "
    "non-empty partial, on one client monotonic clock. Connection, "
    "configuration, and admission wait are not included. A session with "
    "no non-empty partial stays in n_total and is absent from "
    "n_with_measure. This is not a vendor response latency and not vLLM "
    "TTFT."
)
RESPONSE_LATENCY_DEFINITION = (
    "Seconds from the most recent audio transmitted to each non-empty "
    "interim, on one client monotonic clock. One sample per response, "
    "not per stream. Not TTFS. Client and server clocks are never "
    "subtracted."
)
FINALIZATION_DEFINITION = (
    "Seconds from the last audio sample sent to the final result "
    "received, on the client monotonic clock."
)
POPULATION_DEFINITION = (
    "n_with_measure counts outcomes, or samples where the metric says "
    "so, that possess the measure. n_total counts the measured "
    "population, including failures. Percentiles are conditional on "
    "n_with_measure. An empty sample is an absent percentile, not zero. "
    "Neither count is omitted."
)
OUTCOME_ACCOUNTING_DEFINITION = (
    "Counts over every measured session. attempted is the population. "
    "admitted is configured. completed is a terminal transcript without "
    "timeout or protocol error, including an empty transcript. "
    "productive is completed with a non-empty transcript. timed-out and "
    "errored are mutually exclusive, and a timeout wins when both "
    "signals are present. no-first-partial overlaps those buckets. "
    "lane_unavailable counts sessions the adapter labeled as an entitled "
    "lane this process could not open. Warmup sessions are excluded from "
    "attempted and recorded as warmup_excluded."
)
PRODUCTIVE_DEFINITION = (
    "A measured session that was configured, received a non-empty final "
    "transcript, and was not timed out and not errored."
)
AUDIO_DENOMINATOR_DEFINITION = (
    "attempted_audio_s sums clip duration over every measured session, "
    "including failures. successful_audio_s sums clip duration over "
    "productive sessions only. Both are reported."
)
PER_UTTERANCE_WER_DEFINITION = (
    "Distribution of per-utterance word error rates under the named "
    "policy. The micro-average is a separate number. A mean of rates "
    "hides a catastrophic utterance, so both are reported. Utterances "
    "without a defined rate stay in n_total."
)
WER_DEFINITION = (
    "Word error rate (substitutions + alignment deletions + insertions + "
    "deletions_from_missing_transcript) / reference words, "
    "micro-averaged, under the named normalization policy. A session "
    "with a reference and no transcript stays in the denominator as "
    "deletions_from_missing_transcript and is not an alignment deletion. "
    "n_failed counts those sessions. A received empty transcript is an "
    "empty hypothesis and its deletions are alignment deletions. A "
    "session with no reference stays in n_total and is absent from "
    "n_with_measure. Undefined when the reference-word denominator is "
    "zero and any edit remains."
)
CER_DEFINITION = (
    "Character error rate under the same named normalization policy as "
    "the accompanying WER, over the normalized utterance text. Not "
    "interchangeable across policies."
)
SDI_DEFINITION = (
    "Alignment substitutions, deletions, and insertions counted "
    "separately from deletions_from_missing_transcript. The backtrace "
    "is deterministic: on a cost tie it prefers a substitution, then a "
    "deletion, then an insertion."
)
ORACLE_NORMALIZATION_DEFINITION = (
    "engine_vs_own_oracle compares raw text after whitespace "
    "tokenization only. No case-folding, no punctuation stripping, no "
    "tag stripping, and no number normalization."
)
CORPUS_NORMALIZATION_DEFINITION = (
    "corpus_reference applies full normalization: case-fold, strip "
    "control tags, strip punctuation, normalize English cardinals "
    "(en_cardinals_v1, 0 through 999, including the 'and' form), and "
    "collapse whitespace."
)
PREFIX_VIOLATION_DEFINITION = (
    "Count of prefix checks that fail, with no normalization. Each "
    "recorded interim is checked against the next interim, and each "
    "recorded interim is checked against the final. One interim can "
    "contribute twice. An interim count without interim text is not "
    "this measure."
)
REVISION_RATE_DEFINITION = (
    "Revised whitespace tokens divided by newly emitted whitespace "
    "tokens. The first interim contributes every token; each later "
    "interim contributes only tokens past the common token prefix of "
    "the previous interim. A revised token falls outside the common "
    "token prefix of the following interim, or of the final for the "
    "last interim. Not a word error rate. Cross-cadence transcript "
    "agreement is not this metric and is not a pass condition."
)
INTERIM_CADENCE_CLOCK = "client_monotonic/inter_interim_arrival_interval"
INTERIM_CADENCE_DEFINITION = (
    "Distribution of inter-interim arrival intervals against the "
    "configured cadence. Unavailable until outcomes carry interim "
    "arrival timestamps."
)
INTERIM_CADENCE_UNAVAILABLE_REASON = "interim_arrival_timestamps_not_in_outcome"
EMISSION_LAG_CLOCK = "client_monotonic/word_audio_end_to_first_appearance"
EMISSION_LAG_DEFINITION = (
    "A word's audio ending to that word's first appearance in any "
    "output. Requires construction-time ground truth: silence of an "
    "exact frame count between known utterances. Not inferred by a "
    "forced aligner or a voice-activity detector."
)
EMISSION_LAG_UNAVAILABLE_REASON = "no_construction_ground_truth"
LAG_EXCESS_DEFINITION = (
    "Emission lag minus the cadence's theoretical floor: what the "
    "implementation costs above what the configuration costs. "
    "Unavailable when emission lag is unavailable. Raw emission lag is "
    "not compared across cadences."
)
LAG_EXCESS_UNAVAILABLE_REASON = "emission_lag_unavailable"
PACING_FIDELITY_CLOCK = "client_monotonic/intended_vs_actual_chunk_send"
PACING_FIDELITY_DEFINITION = (
    "Deviation between intended and actual chunk send times, from the "
    "client send trace. A validity check on every other latency in the "
    "cell, not a performance number."
)
PACING_FIDELITY_UNAVAILABLE_REASON = "client_send_trace_not_in_outcome"
STREAMING_RTFX_CLOCK = "client_monotonic/cell_wall_span"
STREAMING_RTFX_DEFINITION = (
    "Successfully transcribed audio seconds divided by the wall-clock "
    "span of the measured cell, warmup excluded from the span. Under "
    "real-time pacing this tracks stream count: it is a concurrency "
    "measure, not a speedup and not an inference-speed number. The "
    "numerator is productive audio only. Without a client send trace the "
    "ratio is unqualified, because pacing fidelity is what says the "
    "client actually paced."
)
OFFLINE_RTFX_DEFINITION = (
    "Successfully transcribed audio seconds divided by the wall-clock "
    "span of an unpaced offline cell. A throughput ratio. Not "
    "interchangeable with streaming RTFX. A streaming cell with pacing "
    "disabled is not this measure."
)
GOODPUT_DEFINITION = (
    "Measured sessions that simultaneously were admitted, produced a "
    "first partial, met the named latency SLO on that session's own "
    "samples, completed, and — when a reference was declared — scored "
    "within the accuracy tolerance, divided by the measured cell wall "
    "span in seconds. Not computed while the sustainability policy is "
    "unevaluable, so a missing steady interval cannot become a goodput."
)
SUSTAINED_STREAMS_DEFINITION = (
    "Largest offered concurrency at which the full pass conjunction "
    "still holds. One cell reports its offered lanes and whether its "
    "conjunction passed. It does not invent a sweep maximum. A sweep "
    "projection may take the maximum passing lanes across receipts that "
    "share an operating point aside from lane count."
)
SUSTAINED_STREAMS_UNAVAILABLE_REASON = (
    "single_cell_cannot_identify_the_largest_passing_concurrency"
)
PASS_CONJUNCTION_DEFINITION = (
    "A cell passes only when every clause passes on that cell: admission "
    "fraction, first-partial presence, the latency SLO at the declared "
    "percentile of the named latency metric, the named sustainability "
    "policy, and accuracy within tolerance under the named normalization "
    "policy. An unevaluable clause fails the conjunction. The report "
    "names the clause. Cross-cadence transcript agreement is not a "
    "clause."
)
STEADY_INTERVAL_DEFINITION = (
    "Sustainability over a declared steady interval of the cell. "
    "Unevaluable until outcomes carry that interval. Fail-closed: the "
    "clause does not pass."
)
STEADY_INTERVAL_UNAVAILABLE_REASON = "steady_interval_not_in_outcome"
COMPLETION_FRACTION_DEFINITION = (
    "Fraction of measured sessions that completed. A named policy a "
    "preset may select in place of the steady-interval condition. It is "
    "not that condition."
)
PERCENTILE_LINEAR_DEFINITION = (
    "numpy_linear, Hyndman-Fan type 7, the NumPy default percentile: "
    "rank = (n - 1) * (q / 100), linearly interpolated between the "
    "closest order statistics. q is in [0, 100]. Empty input is absent, "
    "not zero. This is the only percentile implementation in the "
    "package; nearest_rank is the named alternative."
)
PERCENTILE_NEAREST_RANK_DEFINITION = (
    "nearest_rank: 1-based rank = max(1, ceil(q / 100 * n)), clamped to "
    "n. Empty input is absent, not zero. Named alternative to "
    "numpy_linear. Not the default."
)
LOOKAHEAD_IDENTITY_DEFINITION = (
    "An operating point is identified by lookahead frame count. The "
    "millisecond label is carried beside it and is not identity, because "
    "vendors disagree on that convention by one frame. Policy "
    "client_declared uses the frame count the workload requested. Policy "
    "server_reported_when_present uses a server-reported frame count "
    "when an outcome carries one; none do yet, so the client count is "
    "used and server_reported_lookahead_frames is null. The policy name "
    "is warn-only until an adapter reports a frame count, so naming the "
    "unused policy does not refuse a comparison."
)
HARDWARE_IDENTITY_DEFINITION = (
    "The hard hardware identity is gpu_name, gpu_count, "
    "gpu_memory_mib, and gpu_identity_source. Those GPU fields are the "
    "server declaration when the target spec carries one, otherwise "
    "the client nvidia-smi probe, and gpu_identity_source says which. "
    "Driver version, client CPU, host, and probe status are warn-only "
    "and are not this identity. The run does not require a GPU."
)
COMPARISON_KEY_DEFINITION = (
    "sha256 of the canonical JSON texts of the workload fingerprint, "
    "the hard target identity, the hard software identity, and the hard "
    "hardware identity, in that order, joined by U+001F. Lookahead "
    "frame count is inside the workload fingerprint. The millisecond "
    "label is not. Hard software identity includes normalization "
    "policy, percentile method, SCHEMA_VERSION, and "
    "MEASUREMENT_SEMANTICS_VERSION. Warn-only mismatches are listed "
    "and do not refuse."
)
WARN_ONLY_COMPARISON_DEFINITION = (
    "Warn-only mismatches are recorded and do not refuse: GPU driver "
    "version, Python version, asrbench git commit and dirty bit, "
    "endpoint_url, tls_verify, auth_env, package version, lookahead "
    "identity policy, client CPU, host, probe status, and "
    "channel_options. transport_options stay in the hard target identity "
    "because they change adapter behavior. channel_options are keepalive "
    "and maximum message size; a channel tweak does not refuse."
)
FRONTIER_DEFINITION = (
    "Accuracy against measured latency, one point per receipt, identified "
    "by lookahead frame count and the cadence that sets it. The "
    "millisecond label is carried beside the point and is not the "
    "identity. Latency is pooled TTFS p50 and accuracy is pooled micro "
    "WER, each under the clock and normalization policy the receipt "
    "already names. The Pareto edge is the set of points with no other "
    "point both faster and more accurate. A dominated point is a "
    "finding, not a pass failure. Cross-cadence transcript agreement is "
    "not computed. Points are grouped by comparison identity with "
    "lookahead frame count and cadence removed, so a lookahead sweep "
    "that also changes cadence is one series, and different hardware "
    "or models are not drawn on one edge."
)
RUNNER_HANG_GUARD_DEFINITION = (
    "The runner hang guard waits twice timeout_s, plus one timeout_s "
    "for the cohort barrier, plus the clip duration when pacing, plus "
    f"{RUNNER_HANG_GRACE_S} seconds. realtime-ws may spend timeout_s "
    "waiting for session.updated and timeout_s again waiting for the "
    "final, and a peer may spend timeout_s at the barrier on the "
    "slowest handshake. The adapter owns each timeout_s and may return "
    "a partial outcome. The guard replaces the outcome only when the "
    "adapter does not return."
)
VLLM_FORMAT_NOTE = (
    "Shaped like a vLLM serving-benchmark summary so a serving engineer "
    "can read the counts. ttfs_s is this project's time to first "
    "non-empty partial. response_latency_s is per response. Neither "
    "field is vLLM TTFT, and neither is a vendor-reported response "
    "latency."
)
LANE_UNAVAILABLE_DEFINITION = (
    "A session whose adapter set terminal_status to lane_unavailable: an "
    "entitled lane this process could not open. The session stays in the "
    "population. It is not a crash of the run."
)
LATENCY_METRIC_TTFS = "ttfs"
LATENCY_METRIC_RESPONSE = "response_latency"
RTFX_VALIDITY_UNQUALIFIED = "pacing_fidelity_unavailable"
RTFX_VALIDITY_PACING_DISABLED = "pacing_disabled"
RTFX_VALIDITY_NOT_APPLICABLE = "not_this_mode"
RTFX_VALIDITY_SIMULATED = "simulated"
DIGEST_COVERS_PCM16 = "pcm16_payload"


def emitted_definitions() -> dict[str, str]:
    """Return every public definition string, keyed by constant name.

    The receipt writes this mapping. The schema reference quotes it.
    Key order is sorted so two processes emit the same object.

    Returns:
        Constant name to emitted text.
    """
    values = {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str)
    }
    return dict(sorted(values.items()))
