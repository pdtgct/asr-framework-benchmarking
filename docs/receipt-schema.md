# Receipt schema

Generated from `asrbench.schema_doc`. Do not edit by hand.
Regenerate with `uv run python -m asrbench.schema_doc`.

Schema version: `asrbench-receipt/5`

Clock and definition strings below are the constants the receipt
writes. Where `docs/metrics.md` and a constant disagree, the
constant is the measurement and this file is what a consumer should
quote.

A field rendered as `null` is always present. Empty lists are
rendered as `list`. The mock preset carries no GPU, so GPU identity
fields are null in this shape.

## Comparison key

Components, in order, joined by U+001F and hashed:
workload fingerprint, target identity, software identity,
hardware identity.

Not in the hard comparison key.

Record-only. Shown side by side. A mismatch does not warn
and does not refuse:

- `identity.recorded.adapter`
- `identity.recorded.comparison_axis`
- `identity.recorded.engine.name`
- `identity.recorded.engine.version`
- `identity.recorded.engine.transport`
- `identity.recorded.server_version`
- `identity.recorded.precision`
- `identity.recorded.tf32`

Warn-only and other exclusions. Warn-only fields are still
compared and reported; a mismatch does not refuse:

- `run_id`
- `started_at`
- `finished_at`
- `run.output_path`
- `dataset.materialized_path`
- `workload.lookahead_ms_label`
- `workload.population.lookahead_ms_label`
- `workload.clip.path`
- `workload.clip.licence`
- `workload.clip.attribution`
- `workload.clip.note`
- `workload.iterations_per_stream`
- `workload.correctness_streams`
- `workload.performance_streams`
- `run.emit_tsv`
- `identity.warn.driver_version`
- `identity.warn.python_version`
- `identity.warn.git_commit`
- `identity.warn.git_dirty`
- `identity.warn.endpoint_url`
- `identity.warn.tls_verify`
- `identity.warn.auth_env`
- `identity.warn.package_version`
- `identity.warn.lookahead_identity_policy`
- `identity.warn.cpu_model`
- `identity.warn.logical_cpu_count`
- `identity.warn.system`
- `identity.warn.architecture`
- `identity.warn.hostname`
- `identity.warn.probe_status`
- `identity.warn.channel_options`
- `identity.warn.model`
- `identity.warn.transport_options`
- `identity.warn.batch_size`
- `identity.warn.max_num_seqs`
- `identity.warn.gpu_memory_utilization`
- `identity.warn.cuda_version`
- `identity.warn.torch_version`
- `identity.warn.client_location`
- `probe.hardware.gpu_uuids`
- `probe.hardware.collected_at`
- `secrets.set`
- `raw outcomes and trial aggregates`

## Emitted constants

| Name | Value |
| --- | --- |
| `AGGREGATE_RTFX_CLOCK` | client_monotonic/benchmark_wall_span |
| `AGGREGATE_RTFX_DEFINITION` | Sum of offered audio seconds over every measured session, including sessions that produced no transcript, divided by the benchmark wall span from the first stream start to the last stream finish. Under real-time pacing this tracks stream count: a concurrency measure, not a speedup. successful_audio_s is reported beside the ratio. streaming_rtfx is the productive-audio ratio on the cell span and is not this number. |
| `AUDIO_DENOMINATOR_DEFINITION` | attempted_audio_s sums clip duration over every measured session, including failures. successful_audio_s sums clip duration over productive sessions only. Both are reported. |
| `CER_DEFINITION` | Character error rate under the same named normalization policy as the accompanying WER, over the normalized utterance text. Not interchangeable across policies. |
| `CHECKPOINT_SOURCE_HUGGINGFACE` | huggingface |
| `CHECKPOINT_SOURCE_NGC` | ngc |
| `CHECKPOINT_SOURCE_UNVERIFIED` | unverified |
| `CLIENT_LOCATION_LOOPBACK` | loopback |
| `CLIENT_LOCATION_SAME_REGION` | same-region |
| `CLIENT_LOCATION_WAN` | wan |
| `COMPARISON_AXIS` | engine |
| `COMPARISON_KEY_DEFINITION` | sha256 of the canonical JSON texts of the workload fingerprint, the hard target identity, the hard software identity, and the hard hardware identity, in that order, joined by U+001F. The hard target identity is the checkpoint id, revision, and source. It is not target.model and not server_version. The comparison axis is engine: engine name, engine version, transport, and server_version are recorded beside the key and a mismatch does not refuse. The adapter name is recorded with the transport and is not hard software identity. chunk_ms, att_context_right, lookahead frame count, and cadence are inside the workload fingerprint. The millisecond label is not. Decode features in that fingerprint are language, punctuation, diarization, interim_results, decoder, language_model, and enable_word_time_offsets. word_time_offsets_sent is true only when that flag is set and the transport sends word offsets, so a declaration the adapter does not send does not match a run that sent them. Hard software identity includes normalization policy, percentile method, SCHEMA_VERSION, and MEASUREMENT_SEMANTICS_VERSION. Precision is recorded and is not this key. Warn-only mismatches are listed and do not refuse. |
| `COMPLETION_FRACTION_DEFINITION` | Fraction of measured sessions that completed. A named policy a preset may select in place of the steady-interval condition. It is not that condition. |
| `CORPUS_NORMALIZATION_DEFINITION` | corpus_reference applies full normalization: case-fold, strip control tags, strip punctuation, normalize English cardinals (en_cardinals_v1, 0 through 999, including the 'and' form), and collapse whitespace. |
| `DEFAULT_LOOKAHEAD_IDENTITY_POLICY` | client_declared |
| `DEFAULT_NORMALIZATION_POLICY` | engine_vs_own_oracle |
| `DEFAULT_PERCENTILE_METHOD` | numpy_linear |
| `DEFAULT_SUSTAINABILITY_POLICY` | steady_interval |
| `DIGEST_COVERS_PCM16` | pcm16_payload |
| `E2E_RTFX_CLOCK` | client_monotonic/first_audio_sent_to_final_received |
| `E2E_RTFX_DEFINITION` | Per measured session, offered audio seconds divided by the client-monotonic interval from the first audio sample sent to the final result received. Real-time pacing caps this near 1. Read it as kept up, not as speed. A session that did not complete, including a timeout, a protocol error, or an early final on a failed stream, stays in n_total and contributes no ratio. A completed session missing either timestamp stays in n_total and is absent from n_with_measure. Not streaming_rtfx and not a vendor per-chunk latency. |
| `EMISSION_LAG_CLOCK` | client_monotonic/word_audio_end_to_first_appearance |
| `EMISSION_LAG_DEFINITION` | A word's audio ending to that word's first appearance in any output. Requires construction-time ground truth: silence of an exact frame count between known utterances. Not inferred by a forced aligner or a voice-activity detector. |
| `EMISSION_LAG_UNAVAILABLE_REASON` | no_construction_ground_truth |
| `FINALIZATION_CLOCK` | client_monotonic/last_audio_sample_sent_to_final_result_received |
| `FINALIZATION_DEFINITION` | Seconds from the last audio sample sent to the final result received, on the client monotonic clock. A session that did not complete stays in n_total and contributes no sample. A negative on a completed session is kept. |
| `FRONTIER_DEFINITION` | Accuracy against measured latency, one point per receipt, identified by lookahead frame count and the cadence that sets it. The millisecond label is carried beside the point and is not the identity. Latency is pooled TTFS p50 and accuracy is pooled micro WER, each under the clock and normalization policy the receipt already names. The Pareto edge is the set of points with no other point both faster and more accurate. A dominated point is a finding, not a pass failure. Cross-cadence transcript agreement is not computed. Points are grouped by comparison identity with lookahead frame count and cadence removed, so a lookahead sweep that also changes cadence is one series, and different hardware or checkpoints are not drawn on one edge. |
| `FRONTIER_LAG_CLOCK` | client_monotonic/response_receive_minus_audio_processed |
| `FRONTIER_LAG_DEFINITION` | For each audio_processed sample, client-monotonic receive time minus the sum of first-audio-sent and audio_processed at that sample. Negatives are kept. Invalid when audio_processed decreases on that stream. Unavailable when audio_processed or the first-audio timestamp is absent. An invalid or absent session stays in n_total and contributes no sample. Not response latency and not the Riva 1:1 statistic. |
| `FRONTIER_LAG_INVALID_REASON` | audio_processed_decreased |
| `FRONTIER_LAG_UNAVAILABLE_REASON` | audio_processed_or_first_audio_absent |
| `FRONTIER_RTFX_CLOCK` | client_monotonic/audio_processed_delta_over_receive_delta |
| `FRONTIER_RTFX_DEFINITION` | Per consecutive audio_processed samples on one stream, the increase in audio_processed divided by the client-monotonic receive interval. Independent of how the transcript is segmented. Invalid when audio_processed decreases on that stream. Unavailable when the transport did not report audio_processed. An invalid or absent session stays in n_total and contributes no ratio. |
| `FRONTIER_RTFX_INVALID_REASON` | audio_processed_decreased |
| `FRONTIER_RTFX_UNAVAILABLE_REASON` | audio_processed_absent |
| `GOODPUT_DEFINITION` | Measured sessions that simultaneously were admitted, produced a first partial, met the named latency SLO on that session's own samples, completed, and — when a reference was declared — scored within the accuracy tolerance, divided by the measured cell wall span in seconds. Not computed while the sustainability policy is unevaluable, so a missing steady interval cannot become a goodput. |
| `GPU_IDENTITY_CLIENT_PROBE` | client_probe |
| `GPU_IDENTITY_SERVER_DECLARED` | server_declared |
| `HARDWARE_IDENTITY_DEFINITION` | The hard hardware identity is gpu_name, gpu_count, gpu_memory_mib, and gpu_identity_source. Those GPU fields are the server declaration when the target spec carries one, otherwise the client nvidia-smi probe, and gpu_identity_source says which. Driver version, client CPU, host, and probe status are warn-only and are not this identity. The run does not require a GPU. |
| `HARDWARE_PROBE_NVIDIA_SMI` | nvidia-smi |
| `INTERIM_CADENCE_CLOCK` | client_monotonic/inter_interim_arrival_interval |
| `INTERIM_CADENCE_DEFINITION` | Distribution of inter-interim arrival intervals against the configured cadence. Unavailable until outcomes carry interim arrival timestamps. |
| `INTERIM_CADENCE_UNAVAILABLE_REASON` | interim_arrival_timestamps_not_in_outcome |
| `LAG_EXCESS_DEFINITION` | Emission lag minus the cadence's theoretical floor: what the implementation costs above what the configuration costs. Unavailable when emission lag is unavailable. Raw emission lag is not compared across cadences. |
| `LAG_EXCESS_UNAVAILABLE_REASON` | emission_lag_unavailable |
| `LANE_UNAVAILABLE_DEFINITION` | A session whose adapter set terminal_status to lane_unavailable: an entitled lane this process could not open. The session stays in the population. It is not a crash of the run. |
| `LATENCY_METRIC_FINALIZATION` | finalization |
| `LATENCY_METRIC_FRONTIER_LAG` | frontier_lag |
| `LATENCY_METRIC_RESPONSE` | response_latency |
| `LATENCY_METRIC_TTFS` | ttfs |
| `LOOKAHEAD_IDENTITY_CLIENT_DECLARED` | client_declared |
| `LOOKAHEAD_IDENTITY_DEFINITION` | An operating point is identified by lookahead frame count. The millisecond label is carried beside it and is not identity, because vendors disagree on that convention by one frame. Policy client_declared uses the frame count the workload requested. Policy server_reported_when_present uses a server-reported frame count when an outcome carries one; none do yet, so the client count is used and server_reported_lookahead_frames is null. The policy name is warn-only until an adapter reports a frame count, so naming the unused policy does not refuse a comparison. |
| `LOOKAHEAD_IDENTITY_SERVER_REPORTED` | server_reported_when_present |
| `MEASUREMENT_SEMANTICS_VERSION` | 5 |
| `MODE_OFFLINE` | offline |
| `MODE_STREAMING` | streaming |
| `NATIVE_LATENCY_CLOCK` | client_monotonic/first_audio_append_to_nonempty_transcription_delta |
| `NORMALIZATION_CORPUS_REFERENCE` | corpus_reference |
| `NORMALIZATION_ENGINE_VS_OWN_ORACLE` | engine_vs_own_oracle |
| `OFFLINE_RTFX_DEFINITION` | Successfully transcribed audio seconds divided by the wall-clock span of an unpaced offline cell. A throughput ratio. Not interchangeable with streaming RTFX. A streaming cell with pacing disabled is not this measure. |
| `ORACLE_NORMALIZATION_DEFINITION` | engine_vs_own_oracle compares raw text after whitespace tokenization only. No case-folding, no punctuation stripping, no tag stripping, and no number normalization. |
| `OUTCOME_ACCOUNTING_DEFINITION` | Counts over every measured session. attempted is the population. admitted is configured. completed is a terminal transcript without timeout or protocol error, including an empty transcript. productive is completed with a non-empty transcript. timed-out and errored are mutually exclusive, and a timeout wins when both signals are present. no-first-partial overlaps those buckets. lane_unavailable counts sessions the adapter labeled as an entitled lane this process could not open. Warmup sessions are excluded from attempted and recorded as warmup_excluded. |
| `PACING_FIDELITY_CLOCK` | client_monotonic/intended_vs_actual_chunk_send |
| `PACING_FIDELITY_DEFINITION` | Deviation between intended and actual chunk send times, from the client send trace. A validity check on every other latency in the cell, not a performance number. |
| `PACING_FIDELITY_UNAVAILABLE_REASON` | client_send_trace_not_in_outcome |
| `PASS_CONJUNCTION_DEFINITION` | A cell passes only when every clause passes on that cell: admission fraction, first-partial presence, the latency SLO at the declared percentile of the named latency metric, the named sustainability policy, and accuracy within tolerance under the named normalization policy. The named latency metric is ttfs, response_latency, finalization, or frontier_lag. TTFS stays selectable and is secondary. The headline latency measures are finalization and frontier lag. An unevaluable clause fails the conjunction. The report names the clause. Cross-cadence transcript agreement is not a clause. |
| `PERCENTILE_FLOOR_INDEX_DEFINITION` | floor_index: zero-based index = min(floor(q / 100 * n), n - 1) into the sorted sample. Empty input is absent, not zero. Named for a table placed next to Riva's published percentiles. Not the default. |
| `PERCENTILE_LINEAR_DEFINITION` | numpy_linear, Hyndman-Fan type 7, the NumPy default percentile: rank = (n - 1) * (q / 100), linearly interpolated between the closest order statistics. q is in [0, 100]. Empty input is absent, not zero. nearest_rank and floor_index are the named alternatives. |
| `PERCENTILE_METHOD_FLOOR_INDEX` | floor_index |
| `PERCENTILE_METHOD_NEAREST_RANK` | nearest_rank |
| `PERCENTILE_METHOD_NUMPY_LINEAR` | numpy_linear |
| `PERCENTILE_NEAREST_RANK_DEFINITION` | nearest_rank: 1-based rank = max(1, ceil(q / 100 * n)), clamped to n. Empty input is absent, not zero. Named alternative to numpy_linear. Not the default. |
| `PER_UTTERANCE_WER_DEFINITION` | Distribution of per-utterance word error rates under the named policy. The micro-average is a separate number. A mean of rates hides a catastrophic utterance, so both are reported. Utterances without a defined rate stay in n_total. |
| `POPULATION_DEFINITION` | n_with_measure counts outcomes, or samples where the metric says so, that possess the measure. n_total counts the measured population, including failures. Percentiles are conditional on n_with_measure. An empty sample is an absent percentile, not zero. Neither count is omitted. |
| `PRECISION_RECORD_DEFINITION` | precision is recorded under the NVIDIA names FP32, FP16, FP8, and INT8, joined with + when mixed, plus an optional tf32 flag. It is shown beside a comparison and is not a comparison key. No further precision rule is applied. |
| `PREFIX_VIOLATION_DEFINITION` | Count of prefix checks that fail, with no normalization. Each recorded interim is checked against the next interim, and each recorded interim is checked against the final. One interim can contribute twice. An interim count without interim text is not this measure. |
| `PRODUCTIVE_DEFINITION` | A measured session that was configured, received a non-empty final transcript, and was not timed out and not errored. |
| `RESPONSE_LATENCY_CLOCK` | client_monotonic/latest_audio_append_to_nonempty_transcription_delta |
| `RESPONSE_LATENCY_DEFINITION` | Seconds from the most recent audio transmitted to each non-empty interim, on one client monotonic clock. One sample per response, not per stream. Not TTFS. Client and server clocks are never subtracted. |
| `REVISION_RATE_DEFINITION` | Revised whitespace tokens divided by newly emitted whitespace tokens. The first interim contributes every token; each later interim contributes only tokens past the common token prefix of the previous interim. A revised token falls outside the common token prefix of the following interim, or of the final for the last interim. Not a word error rate. Cross-cadence transcript agreement is not this metric and is not a pass condition. |
| `RIVA_CLIENT_COMPAT_CLOCK` | client_monotonic/response_i_minus_chunk_i |
| `RIVA_CLIENT_COMPAT_DEFINITION` | Labelled Riva client compat only. Receive time of response i minus send time of chunk i. Valid only when the response count equals the chunk count. An adapter may omit a protocol event from those lists before pairing; the reducer does not drop a trailing response. Not a headline. Not frontier lag, not finalization, and not TTFS. |
| `RIVA_CLIENT_COMPAT_INVALID_REASON` | response_count_not_one_to_one |
| `RIVA_CLIENT_COMPAT_LABEL` | Riva client compat |
| `RIVA_CLIENT_COMPAT_UNAVAILABLE_REASON` | chunk_or_response_times_absent |
| `RTFX_VALIDITY_NOT_APPLICABLE` | not_this_mode |
| `RTFX_VALIDITY_PACING_DISABLED` | pacing_disabled |
| `RTFX_VALIDITY_SIMULATED` | simulated |
| `RTFX_VALIDITY_UNQUALIFIED` | pacing_fidelity_unavailable |
| `RUNNER_HANG_GUARD_DEFINITION` | The runner hang guard waits twice timeout_s, plus one timeout_s for the cohort barrier, plus the clip duration when pacing, plus 0.25 seconds. realtime-ws may spend timeout_s waiting for session.updated and timeout_s again waiting for the final, and a peer may spend timeout_s at the barrier on the slowest handshake. The adapter owns each timeout_s and may return a partial outcome. The guard replaces the outcome only when the adapter does not return. |
| `SCHEMA_VERSION` | asrbench-receipt/5 |
| `SDI_DEFINITION` | Alignment substitutions, deletions, and insertions counted separately from deletions_from_missing_transcript. The backtrace is deterministic: on a cost tie it prefers a substitution, then a deletion, then an insertion. |
| `STEADY_INTERVAL_DEFINITION` | Sustainability over a declared steady interval of the cell. Unevaluable until outcomes carry that interval. Fail-closed: the clause does not pass. |
| `STEADY_INTERVAL_UNAVAILABLE_REASON` | steady_interval_not_in_outcome |
| `STREAMING_RTFX_CLOCK` | client_monotonic/cell_wall_span |
| `STREAMING_RTFX_DEFINITION` | Successfully transcribed audio seconds divided by the wall-clock span of the measured cell, warmup excluded from the span. Under real-time pacing this tracks stream count: it is a concurrency measure, not a speedup and not an inference-speed number. The numerator is productive audio only. Without a client send trace the ratio is unqualified, because pacing fidelity is what says the client actually paced. |
| `SUSTAINABILITY_COMPLETION_FRACTION` | completion_fraction |
| `SUSTAINABILITY_STEADY_INTERVAL` | steady_interval |
| `SUSTAINED_STREAMS_DEFINITION` | Largest offered concurrency at which the full pass conjunction still holds. One cell reports its offered lanes and whether its conjunction passed. It does not invent a sweep maximum. A sweep projection may take the maximum passing lanes across receipts that share an operating point aside from lane count. |
| `SUSTAINED_STREAMS_UNAVAILABLE_REASON` | single_cell_cannot_identify_the_largest_passing_concurrency |
| `TTFS_CLOCK` | client_monotonic/first_audio_append_to_nonempty_transcription_delta |
| `TTFS_DEFINITION` | Seconds from the first audio sample transmitted to the first non-empty partial, on one client monotonic clock. Connection, configuration, and admission wait are not included. A session with no non-empty partial stays in n_total and is absent from n_with_measure. This is not a vendor response latency and not vLLM TTFT. |
| `VLLM_FORMAT_NOTE` | Shaped like a vLLM serving-benchmark summary so a serving engineer can read the counts. ttfs_s is this project's time to first non-empty partial. response_latency_s is per response. e2e_rtfx is offered audio divided by first-audio-to-final. None of these is vLLM TTFT, and none is a vendor per-chunk latency. riva_client_compat_s is the labelled 1:1 statistic and is not a headline. |
| `WARN_ONLY_COMPARISON_DEFINITION` | Warn-only mismatches are recorded and do not refuse: target.model, transport_options, batch_size, max_num_seqs, gpu_memory_utilization, CUDA version, torch version, GPU driver version, client_location, Python version, asrbench git commit and dirty bit, endpoint_url, tls_verify, auth_env, package version, lookahead identity policy, client CPU, host, probe status, and channel_options. channel_options are keepalive and maximum message size. A checkpoint whose source is unverified is flagged and does not by itself refuse when the declared checkpoint matches. |
| `WER_DEFINITION` | Word error rate (substitutions + alignment deletions + insertions + deletions_from_missing_transcript) / reference words, micro-averaged, under the named normalization policy. A session with a reference and no transcript stays in the denominator as deletions_from_missing_transcript and is not an alignment deletion. n_failed counts those sessions. A received empty transcript is an empty hypothesis and its deletions are alignment deletions. A session with no reference stays in n_total and is absent from n_with_measure. Undefined when the reference-word denominator is zero and any edit remains. |
| `WORD_LATENCY_CLOCK` | client_monotonic/final_receive_minus_word_end |
| `WORD_LATENCY_DEFINITION` | Client-monotonic time of the final that carried the word, minus first-audio-sent plus word.end_time in seconds on the paced audio clock. Requires enable_word_time_offsets. Negatives are kept. Unavailable when word end times were not recorded. |
| `WORD_LATENCY_UNAVAILABLE_REASON` | word_time_offsets_not_recorded |

## Mock preset receipt shape

Types are taken from one run of `presets/mock-streaming.json`
with injected hardware and a scripted cell span. A repeated
object is named once and then cited by path.

- `schema_version`: string
- `run_id`: string
- `started_at`: string
- `finished_at`: string
- `definitions`: object of emitted constants quoted above
- `resolved_config`: object
  - `workload`: object
    - `clip`: object
      - `id`: string
      - `generator`: string
      - `duration_s`: float
      - `sample_rate`: int
      - `digest`: string
      - `path`: null
      - `licence`: string
      - `attribution`: string
      - `note`: string
    - `reference_transcript`: string
    - `mode`: string
    - `pacing`: bool
    - `chunk_ms`: int
    - `cadence_ms`: int
    - `lookahead_frames`: int
    - `lookahead_ms_label`: float
    - `att_context_right`: null
    - `lanes`: int
    - `rate_rps`: null
    - `iterations`: int
    - `warmup`: int
    - `trials`: int
    - `timeout_s`: float
    - `seed`: int
    - `language`: string
    - `punctuation`: bool
    - `diarization`: bool
    - `interim_results`: bool
    - `decoder`: null
    - `language_model`: null
    - `enable_word_time_offsets`: bool
    - `transcript_policy`: string
    - `population`: list[object]
      - item: object
        - `lanes`: int
        - `cadence_ms`: int
        - `lookahead_frames`: int
        - `lookahead_ms_label`: float
        - `att_context_right`: null
    - `iterations_per_stream`: null
    - `correctness_streams`: list
    - `performance_streams`: list
  - `target`: object
    - `endpoint_url`: string
    - `transport`: string
    - `model`: string
    - `auth_env`: string
    - `tls_verify`: bool
    - `transport_options`: object
      - `interim_every_chunks`: int
      - `transcript`: string
    - `channel_options`: object
    - `server_version`: null
    - `server_gpu_name`: null
    - `server_gpu_count`: null
    - `server_gpu_memory_mib`: null
    - `server_driver_version`: null
    - `checkpoint_id`: null
    - `checkpoint_revision`: null
    - `checkpoint_source`: string
    - `engine_name`: null
    - `engine_version`: null
    - `precision`: null
    - `tf32`: null
    - `batch_size`: null
    - `max_num_seqs`: null
    - `gpu_memory_utilization`: null
    - `cuda_version`: null
    - `torch_version`: null
    - `client_location`: null
  - `run`: object
    - `output_path`: string
    - `verbosity`: int
    - `profiler`: null
    - `percentile_method`: string
    - `lookahead_identity_policy`: string
    - `pass_policy`: object
      - `admission_fraction_min`: float
      - `first_partial_presence_min`: float
      - `latency_metric`: string
      - `latency_percentile`: float
      - `latency_slo_s`: float
      - `sustainability_policy`: string
      - `sustainability_min`: float
      - `accuracy_tolerance`: float
      - `normalization_policy`: string
    - `emit_tsv`: bool
- `reproduction`: object
  - `overrides`: list
- `secrets`: list[object]
  - item: object
    - `name`: string
    - `set`: bool
- `fingerprints`: object
  - `workload`: string
  - `observation`: string
- `identity`: object
  - `target`: object
    - `checkpoint`: object
      - `id`: null
      - `revision`: null
      - `source`: string
  - `software`: object
    - `measurement_semantics_version`: string
    - `normalization_policy`: string
    - `package`: string
    - `percentile_method`: string
    - `schema_version`: string
  - `hardware`: object
    - `gpu_count`: null
    - `gpu_identity_source`: string
    - `gpu_memory_mib`: null
    - `gpu_name`: null
  - `warn`: object
    - `architecture`: string
    - `auth_env`: string
    - `cpu_model`: string
    - `driver_version`: null
    - `driver_version_source`: string
    - `endpoint_url`: string
    - `git_commit`: string
    - `git_dirty`: bool
    - `hostname`: string
    - `lookahead_identity_policy`: string
    - `logical_cpu_count`: int
    - `package_version`: string
    - `batch_size`: null
    - `channel_options`: object
    - `client_location`: null
    - `cuda_version`: null
    - `gpu_memory_utilization`: null
    - `max_num_seqs`: null
    - `model`: string
    - `probe_status`: string
    - `torch_version`: null
    - `transport_options`: object
      - `interim_every_chunks`: int
      - `transcript`: string
    - `python_version`: string
    - `system`: string
    - `tls_verify`: bool
  - `recorded`: object
    - `adapter`: string
    - `comparison_axis`: string
    - `engine`: object
      - `name`: null
      - `transport`: string
      - `version`: null
    - `precision`: null
    - `server_version`: null
    - `tf32`: null
- `probe`: object
  - `hardware`: object
    - `architecture`: string
    - `collected_at`: string
    - `cpu_model`: string
    - `driver_version`: null
    - `gpu_count`: null
    - `gpu_memory_mib`: null
    - `gpu_name`: null
    - `gpu_uuids`: list
    - `hostname`: string
    - `logical_cpu_count`: int
    - `probe_name`: string
    - `probe_status`: string
    - `system`: string
  - `software`: object
    - `git_available`: bool
    - `git_commit`: string
    - `git_dirty`: bool
- `comparison_key`: string
- `dataset`: object
  - `clip_id`: string
  - `clip_digest`: string
  - `digest_covers`: string
  - `generator`: string
  - `duration_s`: float
  - `sample_rate`: int
  - `licence`: string
  - `attribution`: string
  - `note`: string
  - `materialized_path`: string
  - `reference_transcript`: string
- `audio_denominators`: object
  - `definition`: string
  - `attempted_audio_s`: float
  - `successful_audio_s`: float
  - `warmup_excluded_sessions`: int
- `lookahead`: object
  - `policy`: string
  - `frames`: int
  - `att_context_right`: null
  - `frames_source`: string
  - `millisecond_label`: float
  - `server_reported_frames`: null
  - `definition`: string
- `trials`: list[object]
  - item: object
    - `trial_index`: int
    - `warmup_excluded`: int
    - `cell_wall_span_s`: float
    - `outcomes`: list[object]
      - item: object
        - `label`: string
        - `lifecycle`: string
        - `clip_name`: string
        - `chunk_ms`: int
        - `audio_duration_s`: float
        - `connected`: bool
        - `configured`: bool
        - `interim_count`: int
        - `transcript`: string
        - `protocol_errors`: list
        - `ttfs_s`: float
        - `final_latency_s`: float
        - `response_latencies_s`: list[float]
        - `latency_clock`: string
        - `response_latency_clock`: string
        - `protocol_extras`: object
          - `adapter`: string
          - `markers`: object
            - `cohort_label`: string
        - `interim_texts`: list[string]
        - `timed_out`: bool
        - `error_code`: null
        - `terminal_status`: string
        - `error_detail`: null
        - `final_latency_clock`: string
        - `entry_index`: int
        - `cadence_ms`: int
        - `lookahead_frames`: int
        - `first_audio_sent_s`: float
        - `last_audio_sent_s`: float
        - `final_received_s`: float
        - `audio_progress`: null
        - `word_timings`: null
        - `chunk_send_times_s`: null
        - `response_receive_times_s`: null
    - `warmup_outcomes`: list[object] (same item shape as outcomes)
    - `reduction`: object
      - `counts`: object
        - `definition`: string
        - `attempted`: int
        - `admitted`: int
        - `completed`: int
        - `productive`: int
        - `errored`: int
        - `timed_out`: int
        - `no_first_partial`: int
        - `lane_unavailable`: int
        - `warmup_excluded`: int
      - `ttfs_s`: object
        - `name`: string
        - `clock`: string
        - `clock_attribution`: string
        - `clock_matches_native`: bool
        - `definition`: string
        - `unit`: string
        - `percentile_method`: string
        - `availability`: string
        - `mean`: float
        - `minimum`: float
        - `maximum`: float
        - `p50`: float
        - `p90`: float
        - `p95`: float
        - `p99`: float
        - `n_with_measure`: int
        - `n_total`: int
        - `n_samples`: int
        - `population_definition`: string
      - `response_latency_s`: same shape as `trials[].reduction.ttfs_s`
      - `finalization_s`: same shape as `trials[].reduction.ttfs_s`
      - `e2e_rtfx`: same shape as `trials[].reduction.ttfs_s`
      - `frontier_rtfx`: object
        - `name`: string
        - `clock`: string
        - `definition`: string
        - `percentile_method`: string
        - `availability`: string
        - `reason`: string
        - `unit`: string
        - `mean`: null
        - `minimum`: null
        - `maximum`: null
        - `p50`: null
        - `p90`: null
        - `p95`: null
        - `p99`: null
        - `n_with_measure`: int
        - `n_total`: int
        - `n_samples`: int
        - `n_invalid_sessions`: int
        - `n_unavailable_sessions`: int
        - `population_definition`: string
        - `published`: bool
      - `frontier_lag_s`: object
        - `name`: string
        - `clock`: string
        - `clock_attribution`: string
        - `clock_matches_native`: bool
        - `definition`: string
        - `unit`: string
        - `percentile_method`: string
        - `availability`: string
        - `reason`: string
        - `mean`: null
        - `minimum`: null
        - `maximum`: null
        - `p50`: null
        - `p90`: null
        - `p95`: null
        - `p99`: null
        - `n_with_measure`: int
        - `n_total`: int
        - `n_samples`: int
        - `n_unavailable_sessions`: int
        - `population_definition`: string
      - `word_latency_s`: same shape as `trials[].reduction.frontier_lag_s`
      - `aggregate_rtfx`: object
        - `name`: string
        - `availability`: string
        - `reason`: null
        - `value`: float
        - `offered_audio_s`: float
        - `successful_audio_s`: float
        - `benchmark_span_s`: float
        - `clock`: string
        - `definition`: string
      - `riva_client_compat_s`: object
        - `name`: string
        - `label`: string
        - `clock`: string
        - `definition`: string
        - `percentile_method`: string
        - `availability`: string
        - `reason`: string
        - `unit`: string
        - `mean`: null
        - `minimum`: null
        - `maximum`: null
        - `p50`: null
        - `p90`: null
        - `p95`: null
        - `p99`: null
        - `n_with_measure`: int
        - `n_total`: int
        - `n_samples`: int
        - `n_invalid_sessions`: int
        - `n_unavailable_sessions`: int
        - `population_definition`: string
      - `wer`: object
        - `policy`: string
        - `definition`: string
        - `normalization_definition`: string
        - `availability`: string
        - `reason`: null
        - `micro_wer`: float
        - `micro_cer`: float
        - `cer_definition`: string
        - `sdi_definition`: string
        - `substitutions`: int
        - `deletions`: int
        - `deletions_from_missing_transcript`: int
        - `n_failed`: int
        - `chars_from_missing_transcript`: int
        - `insertions`: int
        - `reference_tokens`: int
        - `hypothesis_tokens`: int
        - `reference_chars`: int
        - `n_with_measure`: int
        - `n_total`: int
        - `per_utterance_wer`: object
          - `name`: string
          - `definition`: string
          - `percentile_method`: string
          - `availability`: string
          - `mean`: float
          - `p50`: float
          - `p90`: float
          - `p95`: float
          - `p99`: float
          - `n_with_measure`: int
          - `n_total`: int
          - `population_definition`: string
        - `utterances`: list[object]
          - item: object
            - `label`: string
            - `scored`: bool
            - `reason`: null
            - `wer`: float
            - `cer`: float
            - `substitutions`: int
            - `deletions`: int
            - `insertions`: int
            - `reference_tokens`: int
            - `deletions_from_missing_transcript`: int
      - `streaming_rtfx`: object
        - `name`: string
        - `availability`: string
        - `reason`: null
        - `value`: float
        - `successful_audio_s`: float
        - `attempted_audio_s`: float
        - `cell_wall_span_s`: float
        - `span_pooling`: string
        - `clock`: string
        - `definition`: string
        - `pacing`: bool
        - `validity`: string
      - `offline_rtfx`: object
        - `name`: string
        - `availability`: string
        - `reason`: string
        - `value`: null
        - `successful_audio_s`: float
        - `attempted_audio_s`: float
        - `cell_wall_span_s`: float
        - `span_pooling`: string
        - `clock`: string
        - `definition`: string
        - `pacing`: bool
        - `validity`: string
      - `stability`: object
        - `prefix_violations`: object
          - `availability`: string
          - `reason`: null
          - `count`: int
          - `pairs_checked`: int
          - `final_checked`: bool
          - `definition`: string
        - `revision_rate`: object
          - `availability`: string
          - `reason`: null
          - `rate`: float
          - `revised_tokens`: int
          - `emitted_tokens`: int
          - `definition`: string
      - `goodput`: object
        - `name`: string
        - `definition`: string
        - `n_total`: int
        - `cell_wall_span_s`: float
        - `availability`: string
        - `reason`: null
        - `streams_meeting_slo`: int
        - `per_second`: float
      - `sustained_streams_at_slo`: object
        - `name`: string
        - `availability`: string
        - `reason`: string
        - `definition`: string
        - `offered_lanes`: int
        - `cell_conjunction_passed`: bool
        - `clock`: string
      - `unavailable`: list[object]
        - item: object
          - `name`: string
          - `availability`: string
          - `reason`: string
          - `clock`: string
          - `definition`: string
      - `conjunction`: object
        - `definition`: string
        - `passed`: bool
        - `failed_clauses`: list
        - `unevaluable_clauses`: list
        - `clauses`: list[object]
          - item: object
            - `name`: string
            - `status`: string
            - `observed`: float
            - `threshold`: float
            - `detail`: string
      - `audio`: object
        - `definition`: string
        - `attempted_audio_s`: float
        - `successful_audio_s`: float
        - `productive_definition`: string
- `summary`: object
  - `n_trials`: int
  - `conjunction_passed`: bool
  - `failed_clauses`: list
  - `unevaluable_clauses`: list
  - `across_trials`: object
    - `ttfs_p50_s`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `ttfs_p99_s`: same shape as `summary.across_trials.ttfs_p50_s`
    - `response_latency_p50_s`: same shape as `summary.across_trials.ttfs_p50_s`
    - `response_latency_p99_s`: same shape as `summary.across_trials.ttfs_p50_s`
    - `finalization_p50_s`: same shape as `summary.across_trials.ttfs_p50_s`
    - `e2e_rtfx_mean`: same shape as `summary.across_trials.ttfs_p50_s`
    - `aggregate_rtfx`: same shape as `summary.across_trials.ttfs_p50_s`
    - `micro_wer`: same shape as `summary.across_trials.ttfs_p50_s`
    - `streaming_rtfx`: same shape as `summary.across_trials.ttfs_p50_s`
    - `offline_rtfx`: object
      - `values`: list[null]
      - `mean`: null
      - `minimum`: null
      - `maximum`: null
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `goodput_per_second`: same shape as `summary.across_trials.ttfs_p50_s`
  - `pooled`: same shape as `trials[].reduction`
