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

SCHEMA_VERSION = "asrbench-receipt/6"

#: Bumped when reducer or comparison-key semantics change. Replaces the
#: git commit in the hard comparison key, so a docs-only commit does not
#: refuse a comparison. Git commit stays on the receipt as warn-only.
#: ``2`` adds the engine comparison axis, checkpoint identity, and the
#: #177 headline ratios. ``3`` takes the adapter out of the hard key,
#: keeps a failed or timed-out session out of E2E RTFx, and reads
#: finalization as final-receive minus last-audio-sent when both stamps
#: exist. ``4`` keeps a session that did not complete out of the
#: finalization percentiles. ``5`` reads frontier lag against
#: audio_processed, with the same decrease and absence gates as
#: frontier RTFx, counts a frontier session only when it contributes a
#: ratio, pairs Riva compat only at equal send and response counts, and
#: hashes word_time_offsets_sent so a transport that does not send
#: offsets does not match one that does. A consumer of ``1`` through
#: ``4`` must re-run. ``6`` pairs Riva compat at a response count of
#: chunks or chunks plus one and ignores the trailing extra response, as
#: riva_streaming_asr_client does; splits that statistic into
#: intermediate and final by the response's is_final flag; measures
#: pacing fidelity from the client send trace and qualifies streaming
#: RTFx with it; and, on riva-grpc, paces chunks against an absolute
#: schedule from stream start. A consumer of ``5`` must re-run: a
#: chunks-plus-one session that read invalid now reads measured, and a
#: paced riva-grpc cell no longer accumulates sleep drift.
MEASUREMENT_SEMANTICS_VERSION = "6"

GPU_IDENTITY_SERVER_DECLARED = "server_declared"
GPU_IDENTITY_CLIENT_PROBE = "client_probe"

#: Seconds of slack after the adapter's own waits and the barrier budget.
#: The adapter owns each ``timeout_s``. This grace is only the hang guard.
RUNNER_HANG_GRACE_S = 0.25

MODE_STREAMING = "streaming"
MODE_OFFLINE = "offline"

PERCENTILE_METHOD_NUMPY_LINEAR = "numpy_linear"
PERCENTILE_METHOD_NEAREST_RANK = "nearest_rank"
PERCENTILE_METHOD_FLOOR_INDEX = "floor_index"
DEFAULT_PERCENTILE_METHOD = PERCENTILE_METHOD_NUMPY_LINEAR

#: The one comparison axis. Fields on it are recorded and are not the key.
COMPARISON_AXIS = "engine"

CHECKPOINT_SOURCE_HUGGINGFACE = "huggingface"
CHECKPOINT_SOURCE_NGC = "ngc"
CHECKPOINT_SOURCE_UNVERIFIED = "unverified"

CLIENT_LOCATION_LOOPBACK = "loopback"
CLIENT_LOCATION_SAME_REGION = "same-region"
CLIENT_LOCATION_WAN = "wan"

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
    "configuration, and admission wait are not included. Under riva-grpc "
    "real-time pacing the first chunk is sent one chunk after the "
    "schedule origin, once that chunk could have been captured, so TTFS "
    "starts after that wait, as riva_streaming_asr_client's first send "
    "does. A "
    "session with "
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
    "received, on the client monotonic clock. A session that did not "
    "complete stays in n_total and contributes no sample. A negative "
    'on a completed session is kept. Not NVIDIA\'s "final latency" '
    "column: riva_streaming_asr_client pairs response i with chunk i and "
    "drops the trailing final, so its final column is "
    "riva_client_compat_final_s, never this metric."
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
CORPUS_NORMALIZER_EN_CARDINALS = "en_cardinals_v1"
CORPUS_NORMALIZER_OPEN_ASR_ENGLISH = "open_asr_english_v1"
CORPUS_NORMALIZER_OPEN_ASR_MULTILINGUAL = "open_asr_multilingual_v1"
DEFAULT_CORPUS_NORMALIZER = CORPUS_NORMALIZER_EN_CARDINALS
OPEN_ASR_NORMALIZER_COMMIT = "c21a412f2dd7cbe3af658d69fbf3d0dd536a264e"
CORPUS_NORMALIZER_DEFINITION = (
    "Under corpus_reference, pass_policy.corpus_normalizer names the text "
    "normalizer applied to reference and hypothesis alike. "
    "en_cardinals_v1 is the built-in stdlib normalizer described by "
    "CORPUS_NORMALIZATION_DEFINITION; English only. open_asr_english_v1 "
    "is the Open ASR Leaderboard EnglishTextNormalizer (Whisper-derived: "
    "fillers, spelling variants, numbers, currencies, ordinals), vendored "
    "at OPEN_ASR_NORMALIZER_COMMIT; English only. "
    "open_asr_multilingual_v1 dispatches as the leaderboard's scorer "
    "does: English to that normalizer, any other language to "
    "MultilingualNormalizer(remove_diacritics=False) with lang set to the "
    "workload language's primary subtag (digits to words via num2words), "
    "then normalize_compound_pairs on each reference/hypothesis pair. "
    "Angle and square bracket tags are stripped first, as "
    "en_cardinals_v1 does. kaldialign merge_compounds is not replicated, "
    "and the leaderboard's Chinese and Armenian normalizers are not "
    "vendored, so zh and hy take the basic multilingual path; "
    "wer.normalizer.parity says which. The open_asr normalizers need the "
    "asrbench[wer-multilingual] extra. WER is comparable only between "
    "receipts with the same policy and the same normalizer."
)
AUDIO_CONDITION_NATIVE = "native"
AUDIO_CONDITION_TELEPHONY_8K = "telephony_8k"
AUDIO_CONDITION_TELEPHONY_8K_IN_16K = "telephony_8k_in_16k"
AUDIO_CONDITION_DEFINITION = (
    "native sends each clip as staged, 16 kHz PCM16. telephony_8k "
    "low-passes at 3.8 kHz, decimates to 8 kHz, and sends 8 kHz PCM16 "
    "with sample_rate_hertz 8000. telephony_8k_in_16k applies the same "
    "decimation, interpolates back to 16 kHz with the same filter, and "
    "sends 16 kHz PCM16: narrowband content at the wideband rate, for a "
    "server that accepts only 16 kHz. The filter is named by "
    "dataset.audio_condition.resampler. Clip and manifest digests name "
    "the source audio; the condition and the sent rate are separate "
    "workload identity, so 8 kHz and 16 kHz receipts of the same clips "
    "share a source digest and differ in fingerprint."
)
CLIP_ASSIGNMENT_ROUND_ROBIN = "round_robin"
CLIP_ASSIGNMENT_DEFINITION = (
    "round_robin: with a corpus manifest, the session on lane l at "
    "iteration i of its phase plays manifest clip (i * lanes + l) mod "
    "clip_count. Concurrent sessions play different clips, every trial "
    "replays the same assignment, and every clip is played only when "
    "lanes * iterations >= clip_count; wer.corpus.complete records it."
)
MANIFEST_DIGEST_DEFINITION = (
    "sha256 over canonical JSON of the manifest schema, its language, and "
    "per clip in manifest order the clip id, the sha256 of its PCM16 "
    "payload, and the sha256 of its UTF-8 reference text. Paths, notes, "
    "and provenance are not covered. Part of the workload fingerprint: a "
    "changed clip, a changed reference, or a reordered manifest is "
    "different offered work."
)
CORPUS_WER_DEFINITION = (
    "Corpus WER: total edits (substitutions + alignment deletions + "
    "insertions + deletions_from_missing_transcript) over total reference "
    "words, pooled over every measured session of every clip. A long clip "
    "weighs by its words, as the Open ASR Leaderboard and NVIDIA's "
    "published accuracy tables reduce a test set. Equal to micro_wer. "
    "Only a complete run (every clip scored) and a balanced one (every "
    "clip played equally often) is the test-set number."
)
PER_CLIP_MEAN_WER_DEFINITION = (
    "Secondary. Each clip's WER pooled over its own sessions, then the "
    "unweighted mean over clips with a defined rate. A one-word clip "
    "weighs as much as a thousand-word clip, so this is not the published "
    "corpus number; it shows whether a few clips carry the errors."
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
    "Per paced session with a client send trace, the lag of chunk i is "
    "its send time minus (audio clock origin + E_i), where E_i is the "
    "cumulative audio end of chunk i, min(audio duration, (i + 1) * "
    "chunk_ms). Every chunk is scored, the shorter last chunk included. "
    "The origin is the recorded schedule origin, so a late first chunk "
    "is visible; see AUDIO_CLOCK_ORIGIN_DEFINITION for outcomes that did "
    "not record one. Positive is late. Lags from every measured session "
    "are pooled. max_session_lag_s is the largest lag in any session. A "
    "validity check on every other latency in the cell, not a "
    "performance number. A session without a send trace stays in "
    "n_total and contributes no lag."
)
PACING_FIDELITY_TOLERANCE_DEFINITION = (
    "A paced cell is paced when every measured lag is at most "
    "max(5 ms, 5% of chunk_ms), against latencies of a few to a few "
    "hundred milliseconds. Up to one chunk_ms it is pacing_lagged. "
    "Beyond one chunk it is unpaced: the client fell behind real time, "
    "so throughput and latency describe the client as well as the "
    "server. Only paced feeds a comparable NVIDIA row."
)
PACING_FIDELITY_TOLERANCE_FLOOR_S = "0.005"
PACING_FIDELITY_TOLERANCE_FRACTION = "0.05"
PACING_FIDELITY_UNAVAILABLE_REASON = "client_send_trace_not_in_outcome"
PACING_FIDELITY_NOT_PACED_REASON = "pacing_disabled"
RIVA_PACING_SCHEDULE = "absolute_from_stream_start"
RIVA_PACING_SCHEDULE_DEFINITION = (
    "riva-grpc paces against an absolute schedule from stream start, as "
    "riva_streaming_asr_client does: chunk i is sent no earlier than "
    "stream start plus i times chunk_ms plus the duration of chunk i, "
    "so audio is never sent before it would have been recorded and a "
    "late wake-up does not delay every later chunk. Here stream start is "
    "taken after the config write and the cohort barrier release, not "
    "before the config write as in the NVIDIA client, so chunk 0 waits "
    "its own duration from that point."
)
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
    "policy. The named latency metric is ttfs, response_latency, "
    "finalization, or frontier_lag. TTFS stays selectable and is "
    "secondary. The headline latency measures are finalization and "
    "frontier lag. An unevaluable clause fails the conjunction. The "
    "report names the clause. Cross-cadence transcript agreement is "
    "not a clause."
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
    "not zero. nearest_rank and floor_index are the named alternatives."
)
PERCENTILE_NEAREST_RANK_DEFINITION = (
    "nearest_rank: 1-based rank = max(1, ceil(q / 100 * n)), clamped to "
    "n. Empty input is absent, not zero. Named alternative to "
    "numpy_linear. Not the default."
)
PERCENTILE_FLOOR_INDEX_DEFINITION = (
    "floor_index: zero-based index = min(floor(q / 100 * n), n - 1) "
    "into the sorted sample. Empty input is absent, not zero. Named "
    "for a table placed next to Riva's published percentiles. Not the "
    "default."
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
    "hardware identity, in that order, joined by U+001F. The hard "
    "target identity is the checkpoint id, revision, and source. It is "
    "not target.model and not server_version. The comparison axis is "
    "engine: engine name, engine version, transport, and server_version "
    "are recorded beside the key and a mismatch does not refuse. "
    "The adapter name is recorded with the transport and is not hard "
    "software identity. "
    "chunk_ms, att_context_right, lookahead frame count, and cadence "
    "are inside the workload fingerprint. The millisecond label is not. "
    "Decode features in that fingerprint are language, punctuation, "
    "diarization, interim_results, decoder, language_model, and "
    "enable_word_time_offsets. word_time_offsets_sent is true only when "
    "that flag is set and the transport sends word offsets, so a "
    "declaration the adapter does not send does not match a run that "
    "sent them. Hard software identity includes "
    "normalization policy, percentile method, SCHEMA_VERSION, and "
    "MEASUREMENT_SEMANTICS_VERSION. Precision is recorded and is not "
    "this key. Warn-only mismatches are listed and do not refuse."
)
WARN_ONLY_COMPARISON_DEFINITION = (
    "Warn-only mismatches are recorded and do not refuse: target.model, "
    "transport_options, batch_size, max_num_seqs, gpu_memory_utilization, "
    "CUDA version, torch version, GPU driver version, client_location, "
    "Python version, asrbench git commit and dirty bit, endpoint_url, "
    "tls_verify, auth_env, package version, lookahead identity policy, "
    "client CPU, host, probe status, and channel_options. "
    "channel_options are keepalive and maximum message size. A checkpoint "
    "whose source is unverified is flagged and does not by itself refuse "
    "when the declared checkpoint matches."
)
PRECISION_RECORD_DEFINITION = (
    "precision is recorded under the NVIDIA names FP32, FP16, FP8, and "
    "INT8, joined with + when mixed, plus an optional tf32 flag. It is "
    "shown beside a comparison and is not a comparison key. No further "
    "precision rule is applied."
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
    "or checkpoints are not drawn on one edge."
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
    "non-empty partial. response_latency_s is per response. e2e_rtfx is "
    "offered audio divided by first-audio-to-final. None of these is "
    "vLLM TTFT, and none is a vendor per-chunk latency. "
    "riva_client_compat_s is the labelled chunk-i/response-i Riva client "
    "compat statistic and is not a headline."
)
LANE_UNAVAILABLE_DEFINITION = (
    "A session whose adapter set terminal_status to lane_unavailable: an "
    "entitled lane this process could not open. The session stays in the "
    "population. It is not a crash of the run."
)
LATENCY_METRIC_TTFS = "ttfs"
LATENCY_METRIC_RESPONSE = "response_latency"
LATENCY_METRIC_FINALIZATION = "finalization"
LATENCY_METRIC_FRONTIER_LAG = "frontier_lag"
RTFX_VALIDITY_UNQUALIFIED = "pacing_fidelity_unavailable"
RTFX_VALIDITY_PACING_DISABLED = "pacing_disabled"
RTFX_VALIDITY_NOT_APPLICABLE = "not_this_mode"
RTFX_VALIDITY_SIMULATED = "simulated"
RTFX_VALIDITY_PACED = "paced"
RTFX_VALIDITY_PACING_LAGGED = "pacing_lagged"
RTFX_VALIDITY_UNPACED = "unpaced"
DIGEST_COVERS_PCM16 = "pcm16_payload"

E2E_RTFX_CLOCK = "client_monotonic/first_audio_sent_to_final_received"
E2E_RTFX_DEFINITION = (
    "Per measured session, offered audio seconds divided by the "
    "client-monotonic interval from the first audio sample sent to the "
    "final result received. Real-time pacing caps this near 1. Read it "
    "as kept up, not as speed. A session that did not complete, including "
    "a timeout, a protocol error, or an early final on a failed stream, "
    "stays in n_total and contributes no ratio. A completed session "
    "missing either timestamp stays in n_total and is absent from "
    "n_with_measure. Not streaming_rtfx and not a vendor per-chunk latency."
)
FRONTIER_RTFX_CLOCK = (
    "client_monotonic/audio_processed_delta_over_receive_delta"
)
FRONTIER_RTFX_DEFINITION = (
    "Per consecutive audio_processed samples on one stream, the increase "
    "in audio_processed divided by the client-monotonic receive interval. "
    "Independent of how the transcript is segmented. Invalid when "
    "audio_processed decreases on that stream. Unavailable when the "
    "transport did not report audio_processed. An invalid or absent "
    "session stays in n_total and contributes no ratio."
)
FRONTIER_RTFX_INVALID_REASON = "audio_processed_decreased"
FRONTIER_RTFX_UNAVAILABLE_REASON = "audio_processed_absent"
FRONTIER_LAG_CLOCK = "client_monotonic/response_receive_minus_audio_processed"
FRONTIER_LAG_DEFINITION = (
    "For each audio_processed sample, client-monotonic receive time "
    "minus the sum of the audio clock origin and audio_processed at that "
    "sample. The origin is audio t=0 on the client clock "
    "(AUDIO_CLOCK_ORIGIN_DEFINITION), not first-audio-sent, which under "
    "real-time pacing is one chunk later and would bias every sample by "
    "minus chunk_ms. Negatives are kept. Invalid when audio_processed "
    "decreases on that stream. Unavailable when audio_processed or the "
    "audio clock origin is absent. An invalid or absent session stays "
    "in n_total and contributes no sample. Not response latency and not "
    "the Riva client compat statistic."
)
FRONTIER_LAG_INVALID_REASON = "audio_processed_decreased"
FRONTIER_LAG_UNAVAILABLE_REASON = "audio_processed_or_audio_clock_origin_absent"
WORD_LATENCY_CLOCK = "client_monotonic/final_receive_minus_word_end"
WORD_LATENCY_DEFINITION = (
    "Client-monotonic time of the final that carried the word, minus "
    "the audio clock origin (AUDIO_CLOCK_ORIGIN_DEFINITION) plus "
    "word.end_time in seconds on the paced audio clock. Requires "
    "enable_word_time_offsets. Negatives are kept. "
    "Unavailable when word end times were not recorded. Not applicable "
    "in offline mode: there is no paced audio clock."
)
WORD_LATENCY_UNAVAILABLE_REASON = "word_time_offsets_not_recorded"
AGGREGATE_RTFX_CLOCK = "client_monotonic/benchmark_wall_span"
AGGREGATE_RTFX_DEFINITION = (
    "Sum of offered audio seconds over every measured session, including "
    "sessions that produced no transcript, divided by the benchmark wall "
    "span from the first stream start to the last stream finish. Under "
    "real-time pacing this tracks stream count: a concurrency measure, "
    "not a speedup. successful_audio_s is reported beside the ratio. "
    "streaming_rtfx is the productive-audio ratio on the cell span and "
    "is not this number."
)
RIVA_CLIENT_COMPAT_CLOCK = "client_monotonic/response_i_minus_chunk_i"
RIVA_CLIENT_COMPAT_DEFINITION = (
    "Labelled Riva client compat only. Receive time of response i minus "
    "send time of chunk i, for every chunk i. Valid when the response "
    "count equals the chunk count or the chunk count plus one; the "
    "trailing extra response is ignored, as riva_streaming_asr_client "
    "does. Any other count makes the session, and so the statistic, "
    "invalid. An adapter may omit a protocol event from those lists "
    "before pairing. Not applicable when the workload is not paced: "
    "riva_streaming_asr_client prints latency only under "
    "--simulate_realtime. Not a headline. Not frontier lag, not "
    "finalization, and not TTFS."
)
RIVA_CLIENT_COMPAT_NOT_PACED_REASON = "pacing_disabled"
RIVA_CLIENT_COMPAT_LABEL = "Riva client compat"
RIVA_CLIENT_COMPAT_INVALID_REASON = (
    "response_count_not_chunks_or_chunks_plus_one"
)
RIVA_CLIENT_COMPAT_UNAVAILABLE_REASON = "chunk_or_response_times_absent"
RIVA_CLIENT_COMPAT_SPLIT_DEFINITION = (
    "The riva_client_compat_s pairs, split by whether response i carried "
    "a result with is_final true (final) or not (intermediate), as "
    "riva_streaming_asr_client splits its intermediate and final "
    "latency lists. Same validity as the pooled statistic. Unavailable "
    "when the adapter did not record one final flag per response. The "
    'final side is NVIDIA\'s "final latency" by code, not '
    "finalization_s: when the final rides the trailing extra response it "
    "is ignored, as the client ignores it, so this side often holds no "
    "real final. finalization_s (last audio sent to final) is the "
    "asrbench measure of finalization and is never substituted here."
)
RIVA_CLIENT_COMPAT_SPLIT_UNAVAILABLE_REASON = "response_final_flags_absent"
RIVA_OFFLINE_RECOGNIZE_DEFINITION = (
    "In offline mode riva-grpc sends the whole clip in one unary "
    "Recognize, as riva_asr_client does. First and last audio sent are "
    "both the call start and the final is the response, so "
    "finalization_s is that client's per-request latency and TTFS is "
    "empty. Lanes are requests kept in flight. aggregate_rtfx is offered "
    "audio over the trial span, the analogue of the client's throughput "
    "RTFX, biased low: the asrbench span also covers each session's "
    "channel open and GetRivaSpeechRecognitionConfig call and the lane "
    "barrier, which riva_asr_client does once before its timer starts. "
    "offline_rtfx counts productive audio only. word_latency_s is "
    "not_applicable offline. The client sums the server's "
    "audio_processed; asrbench uses the clip duration, keeps the server "
    "value as server_audio_processed_s residue, and reports "
    "server_audio_rtfx beside it (SERVER_AUDIO_RTFX_DEFINITION)."
)
SERVER_AUDIO_RTFX_CLOCK = "client_monotonic/benchmark_wall_span"
SERVER_AUDIO_RTFX_DEFINITION = (
    "Offline only. Sum over completed requests of the server's "
    "audio_processed on the last result of each response, divided by "
    "the benchmark wall span: riva_asr_client's throughput numerator. "
    "gap_vs_clip is server audio over completed clip audio minus one. "
    "mismatch is true when the absolute gap exceeds "
    "SERVER_AUDIO_RTFX_MISMATCH_TOLERANCE, meaning the server counted a "
    "different amount of audio than the clip holds. Unavailable when a "
    "completed request carried no audio_processed. Not applicable in "
    "streaming mode."
)
SERVER_AUDIO_RTFX_MISMATCH_TOLERANCE = "0.005"
SERVER_AUDIO_RTFX_UNAVAILABLE_REASON = "server_audio_processed_absent"
NVIDIA_OFFLINE_ROW_DEFINITION = (
    "One row per offline receipt in NVIDIA's offline table layout: "
    "parallel requests (lanes) and throughput, the trial mean of "
    "aggregate_rtfx. throughput_rtfx_server is the trial mean of "
    "server_audio_rtfx, riva_asr_client's own numerator, and "
    "throughput_server_gap flags a mismatch above 0.5%. Latency columns "
    "are finalization_s in milliseconds, the request latency "
    "riva_asr_client prints; latency_source says asrbench_only because "
    "NVIDIA's offline tables have no latency columns. Comparable "
    "(throughput only) when every request completed in every trial and "
    "percentile_method is floor_index; riva_asr_client prints no "
    "throughput when any request failed."
)
NVIDIA_STREAMING_ROW_DEFINITION = (
    "One row per receipt in NVIDIA's ASR NIM streaming table layout. "
    "Latency avg, p50, p90, p95, and p99 are milliseconds, each the mean "
    "over trials of the per-trial value, as NVIDIA averages three runs; "
    "the *_std columns are the sample standard deviation over trials. "
    "latency_source names what filled them: riva_compat "
    "(riva_client_compat_s, NVIDIA's recv[i] - send[i]) when every "
    "trial measured it; otherwise chunk_coverage/audio_processed or "
    "chunk_coverage/send_watermark (chunk_latency_s, a lower bound) "
    "when the server is not one response per chunk; n/a when the server "
    "sent finals only. Throughput is the trial mean of aggregate_rtfx, "
    "offered audio over the trial wall span. comparable is true only "
    "when latency_source is riva_compat or chunk_coverage/audio_processed "
    "on every trial, percentile_method is floor_index, pacing_validity "
    "is paced, interim_results and enable_word_time_offsets are off, "
    "and iterations per lane is at least 3 (NVIDIA runs exactly 3; more "
    "changes the sample size); not_comparable lists each gate that "
    "failed. chunk_coverage/audio_processed counts as comparable though "
    "NVIDIA's own client would print nothing for such a server; it "
    "equals recv[i] - send[i] where that exists. A trial where any "
    "session sent chunks and got no response is not measured. "
    "final_avg_ms is NVIDIA's final split "
    "(riva_client_compat_final_s); finalization_p50_ms beside it is "
    "asrbench's last-audio-to-final and is not that column."
)
NVIDIA_TRIAL_STD_DEFINITION = (
    "Sample standard deviation (n - 1) of the per-trial values a column "
    "averages. Null with fewer than two trials. NVIDIA rounds by it, so "
    "it is needed to judge whether two rows match."
)
AUDIO_CLOCK_ORIGIN_DEFINITION = (
    "Audio t=0 on the client monotonic clock: audio offset a was "
    "capturable at origin + a. riva-grpc records its pacing schedule "
    "origin (schedule_origin). An outcome without one derives it: a "
    "paced session uses first-audio-sent minus the first chunk's "
    "duration, since a paced chunk is sent once it could have been "
    "captured (first_send_minus_first_chunk); an unpaced session uses "
    "first-audio-sent (first_send). Frontier lag, word latency, and "
    "pacing fidelity all use it, and each block counts which source "
    "every session used."
)
CHUNK_LATENCY_CLOCK = (
    "client_monotonic/first_covering_response_minus_chunk_send"
)
CHUNK_LATENCY_DEFINITION = (
    "Per-chunk coverage latency. For chunk i with send time s_i and "
    "cumulative audio end E_i, the receive time of the first response j "
    "with r_j >= s_i whose covered audio a_j >= E_i, minus s_i. "
    "Population is chunks, as NVIDIA's. a_j is the response's "
    "audio_processed when the session's audio_processed is trusted: "
    "present on every non-final response, never decreasing, and never "
    "more than CHUNK_LATENCY_AUDIO_PROCESSED_SLACK_S beyond the audio "
    "sent by then (coverage_source audio_processed). Otherwise a_j is "
    "the send watermark, E_k of the last chunk sent by r_j "
    "(coverage_source send_watermark), a lower bound. On a server that "
    "answers chunk i before chunk i + 1 is sent it equals "
    "riva_client_compat_s pair for pair; on other servers it shows the "
    "server's own buffering, and a duplicate or back-to-back response "
    "cannot move it because the first covering response wins. "
    "availability is measured for audio_processed coverage, "
    "measured_lower_bound when any session used the watermark, and "
    "final_only when every measured session sent finals only "
    "(interim_results off on a server that does not answer per chunk): "
    "then no number is published, because the degenerate value would be "
    "about half the clip. A chunk no response covered is counted, not "
    "sampled. A session that sent chunks and got no response is counted "
    "as no_responses, its chunks as uncovered, and is not a coverage "
    "source. Unpaced, chunk_latency_s is not_applicable "
    "(pacing_disabled), as riva_client_compat_s is. n_responses, "
    "n_bursts (consecutive responses under "
    "CHUNK_LATENCY_BURST_GAP_S apart), and n_text_repeats (consecutive "
    "identical interim texts) are diagnostics. Not finalization_s."
)
CHUNK_LATENCY_AUDIO_PROCESSED_SLACK_S = "0.01"
CHUNK_LATENCY_BURST_GAP_S = "0.002"
CHUNK_LATENCY_UNAVAILABLE_REASON = "chunk_or_response_times_absent"
CHUNK_LATENCY_FINAL_ONLY_REASON = "server_sent_finals_only"
CHUNK_LATENCY_NO_RESPONSES_REASON = "sessions_without_responses"
MODE_IS_OFFLINE_REASON = "mode_is_offline"
CHUNK_LATENCY_FINAL_FLAGS_INFERRED = (
    "When an outcome carries no response_final_flags (receipts before "
    "schema 6), a session with no interim at all is taken to have sent "
    "finals only; otherwise every response, the real final included, is "
    "taken as non-final, so n_chunks_covered_by_final reads 0 for those "
    "sessions."
)
CHUNK_LATENCY_LABEL = "per-chunk coverage latency"


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
