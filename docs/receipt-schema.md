# Receipt schema

Generated from `asrbench.schema_doc`. Do not edit by hand.
Regenerate with `uv run python -m asrbench.schema_doc`.

Schema version: `asrbench-receipt/1`

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

Excluded:

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
- `probe.hardware.hostname`
- `probe.hardware.gpu_uuids`
- `probe.hardware.collected_at`
- `probe.software.git_dirty`
- `secrets.set`
- `raw outcomes and trial aggregates`

## Emitted constants

### `AUDIO_DENOMINATOR_DEFINITION`

> attempted_audio_s sums clip duration over every measured session, including failures. successful_audio_s sums clip duration over productive sessions only. Both are reported.

### `CER_DEFINITION`

> Character error rate under the same named normalization policy as the accompanying WER, over the normalized utterance text. Not interchangeable across policies.

### `COMPARISON_KEY_DEFINITION`

> sha256 of the canonical JSON texts of workload_fingerprint, target_identity, software_identity, and hardware_identity, in that order, joined by U+001F. Lookahead frame count is inside the workload fingerprint. The lookahead millisecond label, output paths, run id, timestamps, hostname, GPU UUIDs, git dirty bit, and whether a secret variable was set are not.

### `COMPLETION_FRACTION_DEFINITION`

> Fraction of measured sessions that completed. A named policy a preset may select in place of the steady-interval condition. It is not that condition.

### `CORPUS_NORMALIZATION_DEFINITION`

> corpus_reference applies full normalization: case-fold, strip control tags, strip punctuation, normalize English cardinals (en_cardinals_v1, 0 through 999, including the 'and' form), and collapse whitespace.

### `DEFAULT_LOOKAHEAD_IDENTITY_POLICY`

> client_declared

### `DEFAULT_NORMALIZATION_POLICY`

> engine_vs_own_oracle

### `DEFAULT_PERCENTILE_METHOD`

> numpy_linear

### `DEFAULT_SUSTAINABILITY_POLICY`

> steady_interval

### `DIGEST_COVERS_PCM16`

> pcm16_payload

### `EMISSION_LAG_CLOCK`

> client_monotonic/word_audio_end_to_first_appearance

### `EMISSION_LAG_DEFINITION`

> A word's audio ending to that word's first appearance in any output. Requires construction-time ground truth: silence of an exact frame count between known utterances. Not inferred by a forced aligner or a voice-activity detector.

### `EMISSION_LAG_UNAVAILABLE_REASON`

> no_construction_ground_truth

### `FINALIZATION_CLOCK`

> client_monotonic/last_audio_sample_sent_to_final_result_received

### `FINALIZATION_DEFINITION`

> Seconds from the last audio sample sent to the final result received, on the client monotonic clock.

### `FRONTIER_DEFINITION`

> Accuracy against measured latency, one point per receipt, identified by lookahead frame count. The millisecond label is carried beside the point and is not the identity. Latency is pooled TTFS p50 and accuracy is pooled micro WER, each under the clock and normalization policy the receipt already names. The Pareto edge is the set of points with no other point both faster and more accurate. A dominated point is a finding, not a pass failure. Cross-cadence transcript agreement is not computed. Points are grouped by comparison identity with lookahead frame count removed, so different hardware or models are not drawn on one edge.

### `GOODPUT_DEFINITION`

> Measured sessions that simultaneously were admitted, produced a first partial, met the named latency SLO on that session's own samples, completed, and — when a reference was declared — scored within the accuracy tolerance, divided by the measured cell wall span in seconds. Not computed while the sustainability policy is unevaluable, so a missing steady interval cannot become a goodput.

### `HARDWARE_IDENTITY_DEFINITION`

> Hardware identity is system, architecture, cpu_model, logical_cpu_count, gpu_name, gpu_count, gpu_memory_mib, and driver_version. GPU fields come from nvidia-smi when that probe runs and are null when it is absent or fails. The run does not require a GPU. Hostname, GPU UUIDs, and the probe timestamp are on the receipt and are not identity.

### `HARDWARE_PROBE_NVIDIA_SMI`

> nvidia-smi

### `INTERIM_CADENCE_CLOCK`

> client_monotonic/inter_interim_arrival_interval

### `INTERIM_CADENCE_DEFINITION`

> Distribution of inter-interim arrival intervals against the configured cadence. Unavailable until outcomes carry interim arrival timestamps.

### `INTERIM_CADENCE_UNAVAILABLE_REASON`

> interim_arrival_timestamps_not_in_outcome

### `LAG_EXCESS_DEFINITION`

> Emission lag minus the cadence's theoretical floor: what the implementation costs above what the configuration costs. Unavailable when emission lag is unavailable. Raw emission lag is not compared across cadences.

### `LAG_EXCESS_UNAVAILABLE_REASON`

> emission_lag_unavailable

### `LANE_UNAVAILABLE_DEFINITION`

> A session whose adapter set terminal_status to lane_unavailable: an entitled lane this process could not open. The session stays in the population. It is not a crash of the run.

### `LATENCY_METRIC_RESPONSE`

> response_latency

### `LATENCY_METRIC_TTFS`

> ttfs

### `LOOKAHEAD_IDENTITY_CLIENT_DECLARED`

> client_declared

### `LOOKAHEAD_IDENTITY_DEFINITION`

> An operating point is identified by lookahead frame count. The millisecond label is carried beside it and is not identity, because vendors disagree on that convention by one frame. Policy client_declared uses the frame count the workload requested. Policy server_reported_when_present uses a server-reported frame count when an outcome carries one; none do yet, so the client count is used and server_reported_lookahead_frames is null. The policy name is part of software identity, so the two policies are not pooled.

### `LOOKAHEAD_IDENTITY_SERVER_REPORTED`

> server_reported_when_present

### `MODE_OFFLINE`

> offline

### `MODE_STREAMING`

> streaming

### `NATIVE_LATENCY_CLOCK`

> client_monotonic/first_audio_append_to_nonempty_transcription_delta

### `NORMALIZATION_CORPUS_REFERENCE`

> corpus_reference

### `NORMALIZATION_ENGINE_VS_OTHER_SERVER`

> engine_vs_other_server

### `NORMALIZATION_ENGINE_VS_OWN_ORACLE`

> engine_vs_own_oracle

### `OFFLINE_RTFX_DEFINITION`

> Successfully transcribed audio seconds divided by the wall-clock span of an unpaced offline cell. A throughput ratio. Not interchangeable with streaming RTFX. A streaming cell with pacing disabled is not this measure.

### `ORACLE_NORMALIZATION_DEFINITION`

> engine_vs_own_oracle compares raw text after whitespace tokenization only. No case-folding, no punctuation stripping, no tag stripping, and no number normalization.

### `OTHER_SERVER_NORMALIZATION_DEFINITION`

> engine_vs_other_server applies full normalization and is judged against the pass policy's declared accuracy tolerance. It is a different named policy from corpus_reference and must not be relabeled as that policy.

### `OUTCOME_ACCOUNTING_DEFINITION`

> Counts over every measured session. attempted is the population. admitted is configured. completed is a terminal transcript without timeout or protocol error, including an empty transcript. productive is completed with a non-empty transcript. timed-out and errored are mutually exclusive, and a timeout wins when both signals are present. no-first-partial overlaps those buckets. lane_unavailable counts sessions the adapter labeled as an entitled lane this process could not open. Warmup sessions are excluded from attempted and recorded as warmup_excluded.

### `PACING_FIDELITY_CLOCK`

> client_monotonic/intended_vs_actual_chunk_send

### `PACING_FIDELITY_DEFINITION`

> Deviation between intended and actual chunk send times, from the client send trace. A validity check on every other latency in the cell, not a performance number.

### `PACING_FIDELITY_UNAVAILABLE_REASON`

> client_send_trace_not_in_outcome

### `PASS_CONJUNCTION_DEFINITION`

> A cell passes only when every clause passes on that cell: admission fraction, first-partial presence, the latency SLO at the declared percentile of the named latency metric, the named sustainability policy, and accuracy within tolerance under the named normalization policy. An unevaluable clause fails the conjunction. The report names the clause. Cross-cadence transcript agreement is not a clause.

### `PERCENTILE_LINEAR_DEFINITION`

> numpy_linear, Hyndman-Fan type 7, the NumPy default percentile: rank = (n - 1) * (q / 100), linearly interpolated between the closest order statistics. q is in [0, 100]. Empty input is absent, not zero. This is the only percentile implementation in the package; nearest_rank is the named alternative.

### `PERCENTILE_METHOD_NEAREST_RANK`

> nearest_rank

### `PERCENTILE_METHOD_NUMPY_LINEAR`

> numpy_linear

### `PERCENTILE_NEAREST_RANK_DEFINITION`

> nearest_rank: 1-based rank = max(1, ceil(q / 100 * n)), clamped to n. Empty input is absent, not zero. Named alternative to numpy_linear. Not the default.

### `PER_UTTERANCE_WER_DEFINITION`

> Distribution of per-utterance word error rates under the named policy. The micro-average is a separate number. A mean of rates hides a catastrophic utterance, so both are reported. Utterances without a defined rate stay in n_total.

### `POPULATION_DEFINITION`

> n_with_measure counts outcomes, or samples where the metric says so, that possess the measure. n_total counts the measured population, including failures. Percentiles are conditional on n_with_measure. An empty sample is an absent percentile, not zero. Neither count is omitted.

### `PREFIX_VIOLATION_DEFINITION`

> Count of recorded interim hypotheses that are not a raw string prefix of the next interim, plus recorded interim hypotheses that are not a raw string prefix of the final. No normalization. An interim count without interim text is not this measure.

### `PRODUCTIVE_DEFINITION`

> A measured session that was configured, received a non-empty final transcript, and was not timed out and not errored.

### `RESPONSE_LATENCY_CLOCK`

> client_monotonic/latest_audio_append_to_nonempty_transcription_delta

### `RESPONSE_LATENCY_DEFINITION`

> Seconds from the most recent audio transmitted to each non-empty interim, on one client monotonic clock. One sample per response, not per stream. Not TTFS. Client and server clocks are never subtracted.

### `REVISION_RATE_DEFINITION`

> Whitespace tokens emitted in an interim that fall outside the common token prefix of the following interim, or of the final for the last interim, divided by whitespace tokens emitted. Not a word error rate. Cross-cadence transcript agreement is not this metric and is not a pass condition.

### `RTFX_VALIDITY_NOT_APPLICABLE`

> not_this_mode

### `RTFX_VALIDITY_PACING_DISABLED`

> pacing_disabled

### `RTFX_VALIDITY_UNQUALIFIED`

> pacing_fidelity_unavailable

### `SCHEMA_VERSION`

> asrbench-receipt/1

### `SDI_DEFINITION`

> Substitutions, deletions, and insertions counted separately. The backtrace is deterministic: on a cost tie it prefers a substitution, then a deletion, then an insertion.

### `STEADY_INTERVAL_DEFINITION`

> Sustainability over a declared steady interval of the cell. Unevaluable until outcomes carry that interval. Fail-closed: the clause does not pass.

### `STEADY_INTERVAL_UNAVAILABLE_REASON`

> steady_interval_not_in_outcome

### `STREAMING_RTFX_CLOCK`

> client_monotonic/cell_wall_span

### `STREAMING_RTFX_DEFINITION`

> Successfully transcribed audio seconds divided by the wall-clock span of the measured cell, warmup excluded from the span. Under real-time pacing this tracks stream count: it is a concurrency measure, not a speedup and not an inference-speed number. The numerator is productive audio only. Without a client send trace the ratio is unqualified, because pacing fidelity is what says the client actually paced.

### `SUSTAINABILITY_COMPLETION_FRACTION`

> completion_fraction

### `SUSTAINABILITY_STEADY_INTERVAL`

> steady_interval

### `SUSTAINED_STREAMS_DEFINITION`

> Largest offered concurrency at which the full pass conjunction still holds. One cell reports its offered lanes and whether its conjunction passed. It does not invent a sweep maximum. A sweep projection may take the maximum passing lanes across receipts that share an operating point aside from lane count.

### `SUSTAINED_STREAMS_UNAVAILABLE_REASON`

> single_cell_cannot_identify_the_largest_passing_concurrency

### `TTFS_CLOCK`

> client_monotonic/first_audio_append_to_nonempty_transcription_delta

### `TTFS_DEFINITION`

> Seconds from the first audio sample transmitted to the first non-empty partial, on one client monotonic clock. Connection, configuration, and admission wait are not included. A session with no non-empty partial stays in n_total and is absent from n_with_measure. This is not a vendor response latency and not vLLM TTFT.

### `VLLM_FORMAT_NOTE`

> Shaped like a vLLM serving-benchmark summary so a serving engineer can read the counts. ttfs_s is this project's time to first non-empty partial. response_latency_s is per response. Neither field is vLLM TTFT, and neither is a vendor-reported response latency.

### `WER_DEFINITION`

> Word error rate (substitutions + deletions + insertions) / reference words, micro-averaged, under the named normalization policy. A session with a reference and no transcript is an empty hypothesis and stays in the denominator. A session with no reference stays in n_total and is absent from n_with_measure. Undefined when the reference-word denominator is zero and any edit remains.

## Mock preset receipt shape

Types are taken from one run of `presets/mock-streaming.json`
with injected hardware and a scripted cell span.

- `schema_version`: string
- `run_id`: string
- `started_at`: string
- `finished_at`: string
- `definitions`: object
  - `AUDIO_DENOMINATOR_DEFINITION`: string
  - `CER_DEFINITION`: string
  - `COMPARISON_KEY_DEFINITION`: string
  - `COMPLETION_FRACTION_DEFINITION`: string
  - `CORPUS_NORMALIZATION_DEFINITION`: string
  - `DEFAULT_LOOKAHEAD_IDENTITY_POLICY`: string
  - `DEFAULT_NORMALIZATION_POLICY`: string
  - `DEFAULT_PERCENTILE_METHOD`: string
  - `DEFAULT_SUSTAINABILITY_POLICY`: string
  - `DIGEST_COVERS_PCM16`: string
  - `EMISSION_LAG_CLOCK`: string
  - `EMISSION_LAG_DEFINITION`: string
  - `EMISSION_LAG_UNAVAILABLE_REASON`: string
  - `FINALIZATION_CLOCK`: string
  - `FINALIZATION_DEFINITION`: string
  - `FRONTIER_DEFINITION`: string
  - `GOODPUT_DEFINITION`: string
  - `HARDWARE_IDENTITY_DEFINITION`: string
  - `HARDWARE_PROBE_NVIDIA_SMI`: string
  - `INTERIM_CADENCE_CLOCK`: string
  - `INTERIM_CADENCE_DEFINITION`: string
  - `INTERIM_CADENCE_UNAVAILABLE_REASON`: string
  - `LAG_EXCESS_DEFINITION`: string
  - `LAG_EXCESS_UNAVAILABLE_REASON`: string
  - `LANE_UNAVAILABLE_DEFINITION`: string
  - `LATENCY_METRIC_RESPONSE`: string
  - `LATENCY_METRIC_TTFS`: string
  - `LOOKAHEAD_IDENTITY_CLIENT_DECLARED`: string
  - `LOOKAHEAD_IDENTITY_DEFINITION`: string
  - `LOOKAHEAD_IDENTITY_SERVER_REPORTED`: string
  - `MODE_OFFLINE`: string
  - `MODE_STREAMING`: string
  - `NATIVE_LATENCY_CLOCK`: string
  - `NORMALIZATION_CORPUS_REFERENCE`: string
  - `NORMALIZATION_ENGINE_VS_OTHER_SERVER`: string
  - `NORMALIZATION_ENGINE_VS_OWN_ORACLE`: string
  - `OFFLINE_RTFX_DEFINITION`: string
  - `ORACLE_NORMALIZATION_DEFINITION`: string
  - `OTHER_SERVER_NORMALIZATION_DEFINITION`: string
  - `OUTCOME_ACCOUNTING_DEFINITION`: string
  - `PACING_FIDELITY_CLOCK`: string
  - `PACING_FIDELITY_DEFINITION`: string
  - `PACING_FIDELITY_UNAVAILABLE_REASON`: string
  - `PASS_CONJUNCTION_DEFINITION`: string
  - `PERCENTILE_LINEAR_DEFINITION`: string
  - `PERCENTILE_METHOD_NEAREST_RANK`: string
  - `PERCENTILE_METHOD_NUMPY_LINEAR`: string
  - `PERCENTILE_NEAREST_RANK_DEFINITION`: string
  - `PER_UTTERANCE_WER_DEFINITION`: string
  - `POPULATION_DEFINITION`: string
  - `PREFIX_VIOLATION_DEFINITION`: string
  - `PRODUCTIVE_DEFINITION`: string
  - `RESPONSE_LATENCY_CLOCK`: string
  - `RESPONSE_LATENCY_DEFINITION`: string
  - `REVISION_RATE_DEFINITION`: string
  - `RTFX_VALIDITY_NOT_APPLICABLE`: string
  - `RTFX_VALIDITY_PACING_DISABLED`: string
  - `RTFX_VALIDITY_UNQUALIFIED`: string
  - `SCHEMA_VERSION`: string
  - `SDI_DEFINITION`: string
  - `STEADY_INTERVAL_DEFINITION`: string
  - `STEADY_INTERVAL_UNAVAILABLE_REASON`: string
  - `STREAMING_RTFX_CLOCK`: string
  - `STREAMING_RTFX_DEFINITION`: string
  - `SUSTAINABILITY_COMPLETION_FRACTION`: string
  - `SUSTAINABILITY_STEADY_INTERVAL`: string
  - `SUSTAINED_STREAMS_DEFINITION`: string
  - `SUSTAINED_STREAMS_UNAVAILABLE_REASON`: string
  - `TTFS_CLOCK`: string
  - `TTFS_DEFINITION`: string
  - `VLLM_FORMAT_NOTE`: string
  - `WER_DEFINITION`: string
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
    - `transcript_policy`: string
    - `population`: list[object]
      - item: object
        - `lanes`: int
        - `cadence_ms`: int
        - `lookahead_frames`: int
        - `lookahead_ms_label`: float
  - `target`: object
    - `endpoint_url`: string
    - `transport`: string
    - `model`: string
    - `auth_env`: string
    - `tls_verify`: bool
    - `transport_options`: object
      - `interim_every_chunks`: int
      - `transcript`: string
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
- `secrets`: list[object]
  - item: object
    - `name`: string
    - `set`: bool
- `fingerprints`: object
  - `workload`: string
  - `observation`: string
- `identity`: object
  - `target`: object
    - `auth_env`: string
    - `endpoint_url`: string
    - `model`: string
    - `tls_verify`: bool
    - `transport`: string
    - `transport_options`: object
      - `interim_every_chunks`: int
      - `transcript`: string
  - `software`: object
    - `adapter`: string
    - `git_commit`: string
    - `lookahead_identity_policy`: string
    - `normalization_policy`: string
    - `package`: string
    - `package_version`: string
    - `percentile_method`: string
    - `python_version`: string
  - `hardware`: object
    - `architecture`: string
    - `cpu_model`: string
    - `driver_version`: null
    - `gpu_count`: null
    - `gpu_memory_mib`: null
    - `gpu_name`: null
    - `logical_cpu_count`: int
    - `system`: string
- `probe`: object
  - `hardware`: object
    - `collected_at`: string
    - `gpu_uuids`: list
    - `hostname`: string
    - `probe_name`: string
    - `probe_status`: string
  - `software`: object
    - `git_available`: bool
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
        - `mode`: string
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
        - `final_latency_clock`: string
    - `warmup_outcomes`: list[object]
      - item: object
        - `label`: string
        - `mode`: string
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
        - `final_latency_clock`: string
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
      - `response_latency_s`: object
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
      - `finalization_s`: object
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
    - `ttfs_p99_s`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `response_latency_p50_s`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `response_latency_p99_s`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `finalization_p50_s`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `micro_wer`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `streaming_rtfx`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `offline_rtfx`: object
      - `values`: list[null]
      - `mean`: null
      - `minimum`: null
      - `maximum`: null
      - `n_trials`: int
      - `n_trials_with_measure`: int
    - `goodput_per_second`: object
      - `values`: list[float]
      - `mean`: float
      - `minimum`: float
      - `maximum`: float
      - `n_trials`: int
      - `n_trials_with_measure`: int
  - `pooled`: object
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
    - `response_latency_s`: object
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
    - `finalization_s`: object
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
