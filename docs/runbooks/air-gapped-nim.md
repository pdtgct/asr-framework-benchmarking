# Run asrbench against an NVIDIA ASR NIM on an air-gapped network

This runbook takes asrbench to a site with no internet access, points it at any
NVIDIA ASR NIM that already serves Riva gRPC there, and lines the results up with
NVIDIA's published [ASR NIM Performance](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html)
tables. It does not pick a model, a locale, a GPU, or a profile. You choose those
at the site. The steps work the same for any model the NIM reports.

Verified against asrbench `main` at `0e93681d0e882dd15557b2172d6b03cd3de95174`
(receipt schema `asrbench-receipt/5`, measurement semantics `5`). How each step
was checked is listed in [Verification notes](#11-verification-notes). Sections
since updated for receipt schema `asrbench-receipt/6` and measurement semantics
`6` say so, and [Gaps](#10-gaps) names the change that closed each gap.

Scope:

- Covered: Riva `StreamingRecognize` over gRPC, the protocol behind NVIDIA's
  streaming tables (Parakeet, Conformer, Nemotron ASR Streaming, and other Riva
  ASR NIMs), and unary `Recognize`, the protocol behind the offline throughput
  tables (`riva_asr_client`; section 6.1).
- Not covered today: the Nemotron Transcribe tables (HTTP only, concurrent
  requests). Listed in [Gaps](#10-gaps).

Contents:

1. [What NVIDIA measures](#1-what-nvidia-measures)
2. [Package on a connected machine](#2-package-on-a-connected-machine)
3. [Install on the air-gapped host](#3-install-on-the-air-gapped-host)
4. [Point at the NIM and record its identity](#4-point-at-the-nim-and-record-its-identity)
5. [Configure a run](#5-configure-a-run)
6. [Mirror NVIDIA's methodology](#6-mirror-nvidias-methodology)
7. [Metric mapping](#7-metric-mapping)
8. [Read and export receipts](#8-read-and-export-receipts)
9. [Side-by-side comparison template](#9-side-by-side-comparison-template)
10. [Gaps](#10-gaps)
11. [Verification notes](#11-verification-notes)

## 1. What NVIDIA measures

From the performance page and the client source it names
([nvidia-riva/cpp-clients](https://github.com/nvidia-riva/cpp-clients),
`riva/clients/asr/streaming_recognize_client.cc`, read at commit
`517f303c9e8490de72043a891fd5fc47d70461e7`).

**Streaming tables.** NVIDIA runs `riva_streaming_asr_client` with:

```
riva_streaming_asr_client \
   --chunk_duration_ms=<chunk_duration> \
   --simulate_realtime=true \
   --automatic_punctuation=true \
   --num_parallel_requests=<num_streams> \
   --word_time_offsets=false \
   --print_transcripts=false \
   --interim_results=false \
   --num_iterations=<3*num_streams> \
   --audio_file=1272-135031-0000.wav \
   --output_filename=/tmp/output.json
```

- Audio: one file, LibriSpeech dev-clean `1272-135031-0000.wav` (10.885 s, 16 kHz
  mono). Each stream sends it three times (`num_iterations = 3 * num_streams`, with
  `num_streams` requests kept in flight).
- Chunking: the client sends `chunk_duration_ms` of audio per request, paced in
  real time against an absolute schedule from stream start. Client and server use
  the same chunk duration. Each table states it (for example 160 ms for
  "Streaming, low-latency" and 960 ms for "Streaming, high-throughput" on
  Parakeet CTC; 320 ms and 1600 ms on Parakeet 1.1B RNNT).
- Latency: for each stream, `latency_i = recv_time[i] - send_time[i]`, the receive
  time of response `i` minus the send time of audio chunk `i`, on the client's
  steady clock. It is only computed when the response count equals the chunk
  count, or the chunk count plus one (the trailing extra response is ignored).
  Otherwise the client prints no latency statistics at all.
- The client reports three lists: `intermediate latency` (responses with
  `is_final == false`), `final latency` (`is_final == true`), and `latency` (all
  responses). The tables report `latency`.
- Statistics: samples from every stream and every iteration in one run are pooled.
  Percentiles use `sorted[floor(q * n / 100)]`. "avg" is the arithmetic mean.
- Throughput: `RTFX = total audio sent (s) / run time (s)`, where run time spans
  from the start of the first stream to the end of the last. Under real-time
  pacing this tracks the stream count (about 1.0 at one stream).
- Trials: each table value is the average of three runs, rounded to the last
  significant digit given the standard deviation.
- Server side: the tables name the GPU, the model, the mode (`str` low latency,
  `str-thr` high throughput), the chunk size, the language model (`n-gram` where
  shown), and for newer models the server batch size and whether speaker
  diarization was on. Hardware specs for DGX A100, H100, and L40S are at the
  bottom of the page. Driver versions are not stated.

**Offline tables.** `riva_asr_client` with `--num_parallel_requests=32`,
`--num_iterations=96`, and `1272-135031-0000x5.wav` (the same clip concatenated five
times). Throughput is RTFX: total audio processed over run time, printed only
when every request succeeded. The client also prints per-request latency
percentiles. Section 6.1 reproduces this run.

**Accuracy table.** WER (or CER, or cpWER) on full public test sets per model and
locale, for "best latency", "best throughput", and offline modes. asrbench scores
one clip per run, so this table is not reproducible today.

**Nemotron Transcribe tables.** HTTP only, 30 s recordings, RTFx at 16 concurrent
requests (`low` preset) or the maximum (`high` preset), median of three runs. Not a
Riva gRPC workload.

## 2. Package on a connected machine

Do this on any internet-connected Linux machine. The output is one tarball you
carry across.

### 2.1 Decide the target interpreter and platform

You need to know the air-gapped host's Python minor version and CPU architecture.
On that host, run:

```bash
python3 -c 'import sys, platform; print(sys.version_info[:2], platform.machine())'
ldd --version | head -1
```

asrbench needs Python 3.10 or newer. The commands below assume Python 3.12 on
`x86_64` with glibc 2.17 or newer. Change `PYVER` and the `--platform` tags to
match. For `aarch64`, use `manylinux2014_aarch64` and `manylinux_2_17_aarch64`.

### 2.2 Fetch the source

```bash
export ASRBENCH_REF=main     # or a commit SHA you want to pin
export PYVER=3.12
mkdir -p asrbench-airgap/{wheelhouse,src,clips}
git clone https://github.com/pdtgct/asr-framework-benchmarking.git asrbench
git -C asrbench switch -c airgap "$ASRBENCH_REF"
git -C asrbench rev-parse HEAD > asrbench-airgap/src/COMMIT
git -C asrbench bundle create "$PWD/asrbench-airgap/src/asrbench.gitbundle" airgap
```

The local branch `airgap` pins the commit you chose, so the bundle carries exactly
that commit.

The git bundle carries the full history, the presets, and the scripts. The wheel
does not include `presets/` or `scripts/`, and an install from a plain wheel
records `git_commit` as `unknown` on every receipt. Installing from a clone of the
bundle (step 3) records the real commit.

### 2.3 Build the wheelhouse

Use a throwaway virtual environment with a recent pip.

```bash
python3 -m venv pack && . pack/bin/activate
python -m pip install --upgrade pip
python -m pip wheel --no-deps -w asrbench-airgap/wheelhouse ./asrbench
python -m pip download --dest asrbench-airgap/wheelhouse \
  --find-links asrbench-airgap/wheelhouse \
  --only-binary=:all: --implementation cp --python-version "$PYVER" \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux_2_28_x86_64 \
  "asrbench[riva]" hatchling editables pip pytest
```

What this pulls:

- `asrbench[riva]` resolves the `riva` extra from `pyproject.toml`:
  `nvidia-riva-client` and `grpcio`, plus their dependencies (`protobuf`,
  `grpcio-tools`, `websockets`, `setuptools`, `typing-extensions`). The Riva
  protobuf stubs ship inside `nvidia-riva-client`, so no proto compilation is
  needed on site.
- `hatchling` and `editables` let pip build the editable install from source with
  no index.
- `pytest` is optional. It lets you run the default test tier on site as a
  self-check.
- Add `"asrbench[riva,websocket]"` instead if you also want the `nim-realtime`
  WebSocket transport (adds `aiohttp`).

### 2.4 Prepare audio

asrbench reads 16 kHz, mono, 16-bit PCM WAV files. Convert on the connected side,
where `ffmpeg` is easy to get.

The clip NVIDIA uses is pinned in `datasets.lock`. It comes from
[LibriSpeech dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz)
(about 340 MB):

```bash
curl -LO https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz LibriSpeech/dev-clean/1272/135031/1272-135031-0000.flac
mkdir -p asrbench/clips
cp LibriSpeech/dev-clean/1272/135031/1272-135031-0000.flac asrbench/clips/
cd asrbench
python scripts/prepare_librispeech_clip.py
cd ..
cp asrbench/clips/1272-135031-0000.{flac,wav} asrbench-airgap/clips/
```

The script checks the FLAC against the lock digest and prints the WAV digest. It
should print `sha256:1fb7b660d79a51208a05678efb176d79b8fb59019e8e5cd014ad96b065bd8556`.
Add `--repeat 5` to also write `clips/1272-135031-0000x5.wav`, the five-times
concatenation NVIDIA's offline table uses, and print its digest. Copy it too
(`cp asrbench/clips/1272-135031-0000x5.wav asrbench-airgap/clips/`) and record
the printed digest so the site can check the file.

For your own audio, convert each file the same way and keep its reference
transcript beside it:

```bash
ffmpeg -i input.any -ac 1 -ar 16000 -sample_fmt s16 -c:a pcm_s16le \
  asrbench-airgap/clips/my-clip.wav
```

One asrbench run scores one clip against one reference transcript. Pick clips at
least a few encoder contexts long. See [`docs/corpus.md`](../corpus.md) for why
short clips cannot support a streaming claim.

### 2.5 Checksum and pack

Optionally save the two helper scripts from sections 4.2 and 8.4 into
`asrbench-airgap/tools/` now, so nobody has to type them on site. Then:

```bash
(cd asrbench-airgap && find . -type f ! -name SHA256SUMS -print0 | sort -z \
  | xargs -0 sha256sum > SHA256SUMS)
tar -czf asrbench-airgap.tgz asrbench-airgap
sha256sum asrbench-airgap.tgz
```

Record the tarball digest in your transfer log. With Python 3.12 wheels and the
one LibriSpeech clip, the tarball is about 15 MB.

## 3. Install on the air-gapped host

```bash
tar --no-same-owner -xzf asrbench-airgap.tgz
cd asrbench-airgap && sha256sum -c --quiet SHA256SUMS && echo CHECKSUMS_OK && cd ..

git clone --branch airgap asrbench-airgap/src/asrbench.gitbundle asrbench
git -C asrbench rev-parse HEAD          # must match asrbench-airgap/src/COMMIT

python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links asrbench-airgap/wheelhouse --upgrade pip
.venv/bin/python -m pip install --no-index --find-links asrbench-airgap/wheelhouse -e "./asrbench[riva]"
.venv/bin/python -c "import grpc, riva.client.proto.riva_asr_pb2; print('riva ok', grpc.__version__)"
```

Keep your site files outside the checkout. Any untracked file inside `asrbench/`
makes every receipt record `git_dirty: true`.

```bash
mkdir -p work/{clips,presets,tools,receipts,identity}
cp asrbench-airgap/clips/*.wav work/clips/
cp asrbench-airgap/tools/*.py work/tools/ 2>/dev/null
export PATH="$PWD/.venv/bin:$PATH"
```

Self-check with no server and no network:

```bash
cd asrbench
python -m pip install --no-index --find-links ../asrbench-airgap/wheelhouse pytest
python -m pytest -q --deselect tests/test_docs_quickstart.py::test_readme_quickstart_block_runs
asrbench run --preset presets/mock-streaming.json --out /tmp/mock-receipt.json
cd ..
```

The deselected test shells out to `uv run`, which tries to resolve packages over
the network and fails offline. Do not use `uv run` on the air-gapped host. Call
`asrbench` from the virtual environment directly.

**Runtime network use.** asrbench makes no network calls except to the target
endpoint. There are no Hugging Face Hub lookups and no revision resolution:
`checkpoint_id` and `checkpoint_revision` are strings you supply and are recorded
as given. At startup it runs two local subprocesses, `git` (to record the commit
and dirty bit) and `nvidia-smi` (a client GPU probe). Both fail soft: a missing
binary records `unknown` or `absent` and the run continues.

## 4. Point at the NIM and record its identity

Set these once per site. Ports are the NIM defaults: gRPC on 50051, HTTP on 9000.

```bash
export NIM_HOST=nim.example.internal
export NIM_GRPC="$NIM_HOST:50051"
export NIM_HTTP="http://$NIM_HOST:9000"
```

### 4.1 Health, version, and profile

```bash
curl -s "$NIM_HTTP/v1/health/ready"
curl -s "$NIM_HTTP/v1/metadata" | tee work/identity/nim-metadata.json
```

`/v1/metadata` returns `version` (the NIM version), `selectedModelProfileId`, and
`modelInfo[].modelUrl` (for example `ngc://nim/nvidia/<model>:<tag>`).

On the NIM host, also save the image, the profile selector, the profile list, and
the GPU:

```bash
docker inspect --format '{{.Config.Image}} {{.Image}}' <nim-container> | tee work/identity/nim-image.txt
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' <nim-container> \
  | grep -E '^NIM_(TAGS_SELECTOR|MODEL_PROFILE)' | tee work/identity/nim-env.txt
docker run --rm --gpus all --entrypoint nim_list_model_profiles <nim-image> \
  | tee work/identity/nim-profiles.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee work/identity/nim-gpu.txt
```

Skip any of these your site does not allow. They are evidence for the comparison,
not inputs asrbench needs to run.

### 4.2 List the models the NIM serves

Riva's `GetRivaSpeechRecognitionConfig` returns every ASR model the server loaded,
with its parameters. Save this as `work/tools/nim_asr_config.py`:

```python
"""Print every ASR model a Riva or NIM server reports, as JSON."""

import argparse
import json

import grpc
from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc

p = argparse.ArgumentParser()
p.add_argument("--server", default="localhost:50051")
p.add_argument("--use-ssl", action="store_true")
p.add_argument(
    "--ssl-root-cert", default=None, help="PEM CA bundle for a private CA"
)
p.add_argument("--model", default="", help="limit to one model_name")
a = p.parse_args()
if a.use_ssl:
    roots = open(a.ssl_root_cert, "rb").read() if a.ssl_root_cert else None
    channel = grpc.secure_channel(
        a.server, grpc.ssl_channel_credentials(root_certificates=roots)
    )
else:
    channel = grpc.insecure_channel(a.server)
stub = rasr_grpc.RivaSpeechRecognitionStub(channel)
resp = stub.GetRivaSpeechRecognitionConfig(
    rasr.RivaSpeechRecognitionConfigRequest(model_name=a.model), timeout=30
)
out = [
    {"model_name": m.model_name, "parameters": dict(m.parameters)}
    for m in resp.model_config
]
print(json.dumps(out, indent=2, sort_keys=True))
```

Run it:

```bash
python work/tools/nim_asr_config.py --server "$NIM_GRPC" | tee work/identity/nim-asr-config.json
```

Pick the streaming model you want to measure. Use its `model_name` verbatim as
`--model`. asrbench copies it into `RecognitionConfig.model`, and an unknown name
fails on the server. Use the model's language code (for example `en-US`,
`es-US`, `zh-CN`) as `--language`. If the parameters list a chunk size, that is the
server chunk duration to use for `--chunk-ms` (section 6). asrbench also runs this
same call at the start of every session and stores the result on each outcome as
`protocol_extras.server_identity`, so the receipt keeps the server's own answer.

### 4.3 Map NIM identity onto asrbench flags

| asrbench flag | Where it comes from | Role in comparison |
|---|---|---|
| `--endpoint-url` | `$NIM_GRPC` | warn-only |
| `--model` | `model_name` from 4.2 | warn-only routing name |
| `--language` | the model's language code | hard (workload fingerprint) |
| `--checkpoint-id` | `modelUrl` without `ngc://` and without `:<tag>` | hard |
| `--checkpoint-revision` | the `<tag>` from `modelUrl` | hard |
| `--checkpoint-source` | `ngc` (needs a revision), or `unverified` if you cannot read the tag | hard |
| `--nim-metadata` | `identity/nim-metadata.json`, or `$NIM_HTTP` to read it live | sets the next four rows; explicit flags win, and it wins over the preset and `ASRBENCH_*` variables. The live read goes through `urllib`, which honours `http_proxy`/`https_proxy` and checks HTTPS against the system CAs (or `SSL_CERT_FILE`/`SSL_CERT_DIR`); set `no_proxy` for the NIM host, or use the saved file |
| `--engine-name` | `nim` | comparison axis, recorded |
| `--engine-version` | `version` from `/v1/metadata` | recorded |
| `--nim-profile-id` | `selectedModelProfileId` from `/v1/metadata` | recorded (`identity.recorded.nim`) |
| `--nim-model-url` | every `modelInfo[].modelUrl`, comma-joined | recorded (`identity.recorded.nim`) |
| `--nim-tags-selector` | `NIM_TAGS_SELECTOR` from `nim-env.txt` | recorded (`identity.recorded.nim`) |
| `--server-version` | image reference from `nim-image.txt` | recorded |
| `--server-gpu-name`, `--server-gpu-count`, `--server-gpu-memory-mib` | `nim-gpu.txt` on the NIM host | hard hardware identity |
| `--server-driver-version` | `nim-gpu.txt` | warn-only |
| `--language-model` | `n-gram` or `none`, from the profile or table you are matching | hard (decode identity) |
| `--decoder` | for example `greedy`, if known | hard (decode identity) |
| `--batch-size` | server batch size, if the profile states one | warn-only |
| `--precision` | `FP16`, `FP8`, and so on, if known | recorded |
| `--client-location` | `loopback`, `same-region`, or `wan` | warn-only |

Set the server GPU fields. Without them asrbench falls back to `nvidia-smi` on the
client machine, which is usually not the GPU serving the NIM.

The NIM profile ID, model URL, and `NIM_TAGS_SELECTOR` are recorded on the receipt
under `identity.recorded.nim` (receipt schema `asrbench-receipt/6`). They sit beside
the comparison key like the engine version: a mismatch shows in `asrbench compare`
and does not refuse. `asrbench nvidia-table` prints the engine version, profile ID,
and tags selector on each row. Still keep the `work/identity/` files next to the
receipts; the image digest, profile list, and GPU output are evidence the receipt
does not hold.

### 4.4 TLS and authentication

The `riva-grpc` transport takes two settings that must agree:

| NIM endpoint | `target.tls_verify` | `transport_options.use_ssl` |
|---|---|---|
| Plaintext gRPC (the NIM default) | `false` | `false` |
| TLS with a publicly trusted or site CA | `true` | `true` |

TLS with verification off is refused, because grpcio always verifies the peer.

For a private site CA, mutual TLS, or a certificate that names a different host,
set these `riva-grpc` transport options in the preset. All need `use_ssl: true`:

| Transport option | Meaning |
|---|---|
| `ssl_root_cert` | PEM CA bundle that replaces the default trust roots |
| `ssl_client_cert`, `ssl_client_key` | PEM client certificate and key for mutual TLS; set both or neither |
| `ssl_target_name_override` | Name the server certificate is checked against, when it is not the host in `--endpoint-url` |

```json
"transport_options": {
  "use_ssl": true,
  "ssl_root_cert": "/etc/pki/site-ca.pem",
  "ssl_client_cert": "/etc/pki/asrbench-client.pem",
  "ssl_client_key": "/etc/pki/asrbench-client.key",
  "ssl_target_name_override": "nim.site.internal"
}
```

The receipt records the paths, not the file contents. The gRPC core environment
variable `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` still works for the CA, process-wide.

On-prem NIMs usually take no token. If a gateway in front of the NIM wants a
bearer token, put the token in an environment variable and name the variable, not
the value, with `--auth-env MY_TOKEN_VAR`. A token is refused on a plaintext
channel unless `transport_options.allow_insecure_auth` is `true`.

## 5. Configure a run

Configuration resolves in this order: built-in defaults, then the preset file, then
`ASRBENCH_*` environment variables, then command-line flags. The resolved config
is written into the receipt, and `asrbench run --from-receipt` replays it.

### 5.1 A site preset

Save this as `work/presets/nim-site-streaming.json`. It mirrors NVIDIA's streaming
client settings. Everything site-specific is left to flags, so one preset serves
any model.

```json
{
  "workload": {
    "clip": {
      "id": "librispeech-1272-135031-0000",
      "generator": "file",
      "duration_s": 10.885,
      "sample_rate": 16000,
      "path": "clips/1272-135031-0000.wav",
      "licence": "CC-BY-4.0",
      "attribution": "LibriSpeech ASR corpus (Panayotov et al., ICASSP 2015)",
      "note": "NVIDIA streaming perf clip. Air-gapped site run."
    },
    "reference_transcript": "BECAUSE YOU WERE SLEEPING INSTEAD OF CONQUERING THE LOVELY ROSE PRINCESS HAS BECOME A FIDDLE WITHOUT A BOW WHILE POOR SHAGGY SITS THERE A COOING DOVE",
    "mode": "streaming",
    "pacing": true,
    "chunk_ms": 160,
    "cadence_ms": 160,
    "lookahead_frames": null,
    "lookahead_ms_label": null,
    "enable_word_time_offsets": false,
    "lanes": 1,
    "iterations": 3,
    "warmup": 0,
    "trials": 3,
    "timeout_s": 30,
    "seed": 0,
    "language": "en-US",
    "punctuation": true,
    "diarization": false,
    "interim_results": false,
    "transcript_policy": "final-and-interims"
  },
  "target": {
    "endpoint_url": "localhost:50051",
    "transport": "riva-grpc",
    "model": "set-with-model-flag",
    "engine_name": "nim",
    "checkpoint_id": "set-with-checkpoint-id-flag",
    "checkpoint_source": "unverified",
    "tls_verify": false,
    "transport_options": {
      "use_ssl": false,
      "max_alternatives": 1,
      "enable_word_time_offsets": false,
      "custom_configuration": {}
    }
  },
  "run": {
    "percentile_method": "floor_index",
    "emit_tsv": true,
    "pass_policy": {
      "admission_fraction_min": 1.0,
      "first_partial_presence_min": 0.0,
      "latency_metric": "finalization",
      "latency_percentile": 99,
      "latency_slo_s": 1.0,
      "sustainability_policy": "completion_fraction",
      "sustainability_min": 1.0,
      "accuracy_tolerance": 1.0,
      "normalization_policy": "corpus_reference"
    }
  }
}
```

Notes on this preset:

- Run it from `work/`. The clip path is relative to the working directory.
- For TLS, copy the file and set `tls_verify` and `use_ssl` both to `true`. There
  is no command-line flag for `use_ssl`.
- `first_partial_presence_min` is `0.0` because NVIDIA's setting
  `interim_results: false` means a healthy server may send no non-empty partial,
  so TTFS has no samples. With interims off, judge latency from
  `riva_client_compat_s` and `finalization_s`.
- `accuracy_tolerance` is `1.0` so the pass conjunction does not fail a
  performance cell on WER. Tighten it if you want WER gating.
- The pass conjunction is asrbench's own verdict. NVIDIA's tables have no
  pass/fail, so it does not affect the comparison.

### 5.2 Knobs

All of these are `asrbench run` flags. `asrbench run --help` lists every one.

| Flag | Preset field | What it does | NVIDIA equivalent |
|---|---|---|---|
| `--chunk-ms` | `workload.chunk_ms` | Audio per gRPC request, in ms. Hard identity. | `--chunk_duration_ms` |
| `--pacing` / `--no-pacing` | `workload.pacing` | Send one chunk per `chunk_ms` of wall time. | `--simulate_realtime` |
| `--lanes` | `workload.lanes` | Closed-loop parallel streams. Each lane runs its iterations back to back. | `--num_parallel_requests` |
| `--iterations` | `workload.iterations` | Measured sessions per lane. | `--num_iterations` divided by streams |
| `--warmup` | `workload.warmup` | Unmeasured sessions per lane before the measured ones. | none (NVIDIA has no warmup) |
| `--trials` | `workload.trials` | Independent repeats. Raw trials stay on the receipt. | three runs, averaged |
| `--timeout-s` | `workload.timeout_s` | Per-session budget. A timeout stays in the denominator. | none |
| `--punctuation` / `--no-punctuation` | `workload.punctuation` | `enable_automatic_punctuation`. | `--automatic_punctuation` |
| `--interim-results` / `--no-interim-results` | `workload.interim_results` | Ask for interim hypotheses. | `--interim_results` |
| `--enable-word-time-offsets` / `--no-enable-word-time-offsets` | `workload.enable_word_time_offsets` | Request word times. Set the matching `transport_options.enable_word_time_offsets` too, or the adapter does not send it. | `--word_time_offsets` |
| `--diarization` / `--no-diarization` | `workload.diarization` | Speaker diarization on or off. | table column "Speaker Diarization" |
| `--language` | `workload.language` | `RecognitionConfig.language_code`. | `--language_code` |
| `--model` | `target.model` | `RecognitionConfig.model`. | `--model_name` |
| `--clip-path` | `workload.clip.path` | WAV to stream. Not part of the fingerprint; the PCM digest is. | `--audio_file` |
| `--cadence-ms`, `--lookahead-frames`, `--lookahead-ms-label`, `--att-context-right` | `workload.*` | Operating-point identity. `riva-grpc` does not send these. Set `--cadence-ms` equal to `--chunk-ms` so receipts group by chunk. | server profile |
| `--percentile-method` | `run.percentile_method` | `numpy_linear` (default), `nearest_rank`, or `floor_index`. | `floor_index` matches the Riva client |
| `--normalization-policy` | `run.pass_policy.normalization_policy` | `corpus_reference` (full normalization) or `engine_vs_own_oracle` (raw). | not stated |
| `--out` | none | Receipt path. With `emit_tsv`, a `.tsv` lands beside it. | `--output_filename` |

Server-side Riva options that have no workload flag can go in
`target.transport_options.custom_configuration` as string pairs. They are sent in
`RecognitionConfig.custom_configuration` and recorded as warn-only.

A clip with no `reference_transcript` (set it to `null`) still records every
latency and throughput number, but WER is absent and the pass conjunction reads
as failed.

### 5.3 Streaming and offline

`--mode offline --no-pacing` on `riva-grpc` sends the whole clip in one unary
`Recognize` per session (`RIVA_OFFLINE_RECOGNIZE_DEFINITION`), as
`riva_asr_client` does. Section 6.1 has the full offline run. `--no-pacing` on a
*streaming* run still goes through `StreamingRecognize` as fast as possible; that
is not NVIDIA's offline measurement and is not comparable to the offline tables.

### 5.4 One run

```bash
cd work
IDENT=(
  --endpoint-url "$NIM_GRPC"
  --model "<model_name from 4.2>"
  --language en-US
  --checkpoint-id "nim/nvidia/<model>"
  --checkpoint-revision "<tag>"
  --checkpoint-source ngc
  --nim-metadata identity/nim-metadata.json
  --nim-tags-selector "<NIM_TAGS_SELECTOR from nim-env.txt>"
  --server-version "<image reference>"
  --server-gpu-name "<GPU name>"
  --server-gpu-count 1
  --server-gpu-memory-mib "<MiB>"
  --language-model n-gram
  --client-location same-region
)
mkdir -p receipts/smoke
asrbench run --preset presets/nim-site-streaming.json "${IDENT[@]}" \
  --lanes 1 --iterations 1 --trials 1 \
  --out receipts/smoke/smoke.json
```

The console summary prints at the end. Check these before a sweep:

- `outcomes attempted=1 admitted=1 completed=1 productive=1`
- `riva_client_compat ... availability=measured`. If it reads `invalid`, the
  server sent neither one response per chunk nor one per chunk plus one
  trailing response (see section 7).
- `pacing_fidelity ... validity=paced`. `pacing_lagged` means the client fell
  more than one chunk behind real time; fix the client host before a sweep.
- `wer ... micro=` is near the model's known accuracy for that clip.

Each cell takes about `trials * iterations * (clip length + finalization)`, about
100 s for 3 trials of 3 iterations on the 10.885 s clip, whatever the stream count.

## 6. Mirror NVIDIA's methodology

| NVIDIA | asrbench |
|---|---|
| `--chunk_duration_ms=<C>`, same as server chunk | `--chunk-ms C --cadence-ms C` |
| `--simulate_realtime=true` | `pacing: true` (preset) |
| `--automatic_punctuation=true` | `punctuation: true` (preset) |
| `--word_time_offsets=false` | `enable_word_time_offsets: false` (preset and transport option) |
| `--interim_results=false` | `interim_results: false` (preset) |
| `--num_parallel_requests=N` | `--lanes N` |
| `--num_iterations=3*N` | `--iterations 3` (per lane) |
| no warmup | `--warmup 0` |
| average of three runs | `--trials 3`, then `asrbench nvidia-table` (section 8.4) |
| `1272-135031-0000.wav` | the clip from section 2.4 |
| `sorted[floor(q*n/100)]` percentiles | `--percentile-method floor_index` |
| one row per stream count | `asrbench sweep`: one receipt per stream count |

Choose `C` from the table you are matching. On the NIM side, the mode in
`NIM_TAGS_SELECTOR` must match: `mode=str` for "Streaming, low-latency",
`mode=str-thr` for "Streaming, high-throughput". Match the GPU, the model, the
language model, and the diarization setting the table states.

Run the stream-count sweep from `work/`, with `IDENT` set as in section 5.4. Use
NVIDIA's stream counts for the table you match. `asrbench sweep` runs one cell per
count, in order, each resolved exactly as `asrbench run --lanes N` would be, and
writes `c<chunk>-n<N>.json` (plus its `.tsv` and `.console.txt`) for each count and
`c<chunk>-nvidia-layout.tsv` for the whole sweep:

```bash
CHUNK=160
RUN=$(date +%Y%m%d)-c$CHUNK
mkdir -p "receipts/$RUN/identity"
cp identity/* "receipts/$RUN/identity/" 2>/dev/null
asrbench sweep --preset presets/nim-site-streaming.json "${IDENT[@]}" \
  --chunk-ms "$CHUNK" --cadence-ms "$CHUNK" \
  --streams 1,8,16,32,64 --iterations 3 --warmup 0 --trials 3 \
  --out-dir "receipts/$RUN"
```

Without `--streams`, the sweep reads `workload.performance_streams` from the
preset. `--lanes` and `--out` are refused; the sweep sets both per count.

A failed cell still exits `0` and the sweep continues. The failure is a result and
stays on the receipt.
Exit code `2` means a configuration error. Every count is resolved before the
first cell runs, and a repeated count is refused.

Existing receipts with the same names in `--out-dir` are overwritten, so a
same-day rerun with the `RUN` above replaces the earlier one; add a suffix to
`RUN` to keep both. An offline preset's default prefix is `offline`.

Things that differ from NVIDIA even with these settings:

- Since semantics `6`, `riva-grpc` paces against an absolute schedule from
  stream start, as NVIDIA's client does (`RIVA_PACING_SCHEDULE_DEFINITION`), so
  delays no longer accumulate. `pacing_fidelity` on each trial reports how late
  the client actually sent each chunk; it must read `paced` before a row is
  compared.
- By default asrbench opens one gRPC channel per session. The Riva client shares
  one channel across streams. Set `transport_options.share_channel: true` to do
  the same: one channel per run, reused by every lane. Each outcome records
  `protocol_extras.channel` (`shared` or `per_session`). Connection setup is
  outside the latency clocks but inside the trial wall span, so use the shared
  channel for rows at high stream counts. The shared channel connects when the
  first session asks for it, so with `--warmup 0` the connect falls inside trial
  0's span only. A shared channel is one HTTP/2 connection: a gateway's
  `MAX_CONCURRENT_STREAMS` limit (often 100 or 128 on Envoy or nginx) queues
  lanes on the client, which shows up as barrier timeouts, not server latency.
  NVIDIA's client behaves the same. A misconfigured mutual-TLS setup can read
  `lane_unavailable` or `errored`, depending on when the server rejects.
- At high stream counts (hundreds and up) the Python client may become the
  bottleneck. Watch the client host's CPU and treat a rising `e2e_rtfx` shortfall
  as a client limit until shown otherwise.

### 6.1 Offline tables

NVIDIA's offline row is `riva_asr_client --num_parallel_requests=32
--num_iterations=96` over `1272-135031-0000x5.wav`. In asrbench, 32 lanes of 3
iterations each keep 32 requests in flight for 96 requests.

Derive an offline preset from the site preset, so the receipt names the
concatenated clip and its reference:

```bash
python - <<'PY'
import json
p = json.load(open("presets/nim-site-streaming.json"))
w = p["workload"]
w["clip"].update(
    id="librispeech-1272-135031-0000x5",
    duration_s=round(w["clip"]["duration_s"] * 5, 3),
    path="clips/1272-135031-0000x5.wav",
    note="NVIDIA offline perf clip (5x). Air-gapped site run.",
)
w["clip"].pop("digest", None)
w["reference_transcript"] = " ".join([w["reference_transcript"]] * 5)
w.update(mode="offline", pacing=False, timeout_s=120)
json.dump(p, open("presets/nim-site-offline.json", "w"), indent=2)
PY
asrbench run --preset presets/nim-site-offline.json "${IDENT[@]}" \
  --model "<offline model_name from 4.2>" \
  --lanes 32 --iterations 3 --warmup 0 --trials 3 \
  --out "receipts/$RUN/offline-n32.json"
asrbench nvidia-table "receipts/$RUN/offline-n32.json"
```

Offline and streaming are usually separate `model_name`s on a NIM. The offline
row reads `compat_status=all_requests_completed` when every request succeeded,
the only case in which NVIDIA's client prints throughput. Its latency columns are
per-request latency, from `finalization_s`.

## 7. Metric mapping

asrbench's definitions are emitted constants in `asrbench.definitions`. Every
receipt carries them under `definitions`, and
[`docs/receipt-schema.md`](../receipt-schema.md) quotes them. The table below names
the constant and quotes its clock string. Read the definition on the receipt for
the full rule.

| NVIDIA column | NVIDIA formula | asrbench field | asrbench clock | Match |
|---|---|---|---|---|
| Latency avg, p50, p90, p95, p99 (ms) | per response: `recv[i] - send[i]`, pooled over streams and iterations, floor-index percentiles, then averaged over three runs | `trials[t].reduction.riva_client_compat_s` (`RIVA_CLIENT_COMPAT_DEFINITION`), averaged over trials and converted to ms by `asrbench nvidia-table` (section 8.4) | `client_monotonic/response_i_minus_chunk_i` | **Matches** in definition when `percentile_method` is `floor_index`. |
| Latency when responses = chunks + 1 | trailing response ignored, rest kept | same field; `n_trailing_extra_ignored` counts those sessions | same | **Matches** since semantics `6`. Any other count is still `invalid` (`response_count_not_chunks_or_chunks_plus_one`), where NVIDIA's client prints no statistics. |
| Intermediate and final latency | split by `is_final` | `riva_client_compat_intermediate_s`, `riva_client_compat_final_s` (`RIVA_CLIENT_COMPAT_SPLIT_DEFINITION`) | same | **Matches** since semantics `6`. NVIDIA's tables publish the pooled list. |
| Throughput (RTFX), streaming | total audio sent / run wall time | per-trial `aggregate_rtfx.value` (`AGGREGATE_RTFX_DEFINITION`); also `streaming_rtfx` (`STREAMING_RTFX_DEFINITION`) | `client_monotonic/benchmark_wall_span`; `client_monotonic/cell_wall_span` | **Close.** `aggregate_rtfx` counts offered audio, as NVIDIA does. `streaming_rtfx` counts productive sessions only. Both track the stream count under pacing. The span also covers channel setup (see section 6). |
| Throughput (RTFX), offline | total audio / run time, unpaced, unary `Recognize` | per-trial `aggregate_rtfx.value`, shown by `asrbench nvidia-table` for an offline receipt; also `offline_rtfx` (`OFFLINE_RTFX_DEFINITION`) | `client_monotonic/benchmark_wall_span` | **Close.** NVIDIA sums the server's `audio_processed`; asrbench uses the clip duration and keeps the server value as `server_audio_processed_s` on each outcome. The asrbench span also covers each session's channel open and config fetch and the lane barrier, which `riva_asr_client` does once before its timer, so asrbench reads low. |
| Offline request latency | `Finish` minus request start, per request | `finalization_s` in offline mode (`RIVA_OFFLINE_RECOGNIZE_DEFINITION`) | `client_monotonic/last_audio_sample_sent_to_final_result_received` | **Matches** in definition with `floor_index`. Printed by the client, not an NVIDIA offline table column. |
| WER (accuracy table) | full test set, per model and mode | `wer.micro_wer` (`WER_DEFINITION`) under a named normalization policy | none | **Differs.** asrbench scores one clip per receipt. Comparable only as a sanity check on that clip. |
| none | none | `ttfs_s` (`TTFS_DEFINITION`) | `client_monotonic/first_audio_append_to_nonempty_transcription_delta` | asrbench only. Not a vendor latency. Empty when interims are off. |
| none | none | `response_latency_s` (`RESPONSE_LATENCY_DEFINITION`) | `client_monotonic/latest_audio_append_to_nonempty_transcription_delta` | asrbench only. Counts non-empty interims, measured from the latest chunk sent, so it is not the Riva per-chunk pairing. |
| none | none | `finalization_s` (`FINALIZATION_DEFINITION`) | `client_monotonic/last_audio_sample_sent_to_final_result_received` | asrbench #177 headline. No NVIDIA counterpart. The `.tsv` written beside each receipt is this metric in NVIDIA's column order, not NVIDIA's latency. |
| none | none | `frontier_lag_s` (`FRONTIER_LAG_DEFINITION`), `frontier_rtfx` (`FRONTIER_RTFX_DEFINITION`) | `client_monotonic/response_receive_minus_audio_processed`, `client_monotonic/audio_processed_delta_over_receive_delta` | asrbench #177 headlines. Built on the server's `audio_processed` field. No NVIDIA counterpart. |
| none | none | `e2e_rtfx` (`E2E_RTFX_DEFINITION`) | `client_monotonic/first_audio_sent_to_final_received` | asrbench only. Per-session "kept up" check, near 1.0 under pacing. |
| none | none | `pacing_fidelity` (`PACING_FIDELITY_DEFINITION`) | `client_monotonic/intended_vs_actual_chunk_send` | asrbench only. Validity check on the client's pacing; qualifies `streaming_rtfx.validity`. |

In short:

- The Riva-compatible latency is the one asrbench number defined the same way as
  NVIDIA's table latency. It exists only when the server sends one response per
  audio chunk, optionally plus one trailing response, which is the same rule
  NVIDIA's client applies. Confirm it on the first smoke run at your site.
- Streaming throughput agrees in meaning. Check `pacing_fidelity` reads `paced`
  before comparing values.
- The #177 metrics (finalization, frontier lag, frontier RTFx) and TTFS answer
  different questions. Report them beside NVIDIA's numbers, never in NVIDIA's
  columns.
- Offline throughput is runnable (section 6.1). Test-set WER cannot be compared
  today.

## 8. Read and export receipts

### 8.1 Console

```bash
asrbench report receipts/$RUN/c160-n8.json --format console
```

### 8.2 Flat exports

```bash
mkdir -p "exports/$RUN"
asrbench report receipts/$RUN/c160-n8.json --format tsv  --out exports/$RUN/c160-n8.vendor.tsv
asrbench report receipts/$RUN/c160-n8.json --format vllm --out exports/$RUN/c160-n8.vllm.json
asrbench index receipts/$RUN --out exports/$RUN/index.jsonl
```

Keep exports out of the receipt directory. `asrbench index` reads every `*.json`
at the top of that directory and stops with "not an asrbench receipt" on the
first file that is not one.

- `--format tsv` is one wide row: identity, TTFS, response latency, finalization
  p50, WER with S/D/I, streaming RTFX, admission, and pass state.
- `--format vllm` is a vLLM-serving-benchmark-shaped summary.
- `index` writes one JSON line per receipt with its comparison key and
  fingerprints.
- The `.tsv` that `emit_tsv` writes beside each receipt is the finalization table in
  NVIDIA's column order. It is not NVIDIA's latency.

### 8.3 Across a sweep

```bash
asrbench sustained receipts/$RUN/c160-n*.json
asrbench frontier  receipts/$RUN/c160-n*.json --out exports/$RUN/frontier.json
```

`sustained` reports the largest stream count whose pass conjunction held, per
operating point. `frontier` plots accuracy against TTFS p50 per lookahead and is
mostly useful across chunk or cadence sweeps.

`asrbench compare A.json B.json` refuses (exit `2`) when the comparison keys
differ. Two cells of one sweep differ in lanes, so `compare` is for the same cell
run twice or on two engines, not for neighbouring stream counts. `--force
--reason "..."` overrides and copies the reason into the output.

### 8.4 NVIDIA-layout rows

`asrbench nvidia-table` prints one row per receipt in NVIDIA's streaming table
layout (`NVIDIA_STREAMING_ROW_DEFINITION`): stream count, latency avg/p50/p90/
p95/p99 in milliseconds (each a trial mean; `latency_source` says of what), and
throughput (the trial mean of `aggregate_rtfx`). Extra columns to the right say
whether the row may be compared and why not.

```bash
asrbench nvidia-table receipts/$RUN/c160-n*.json --out exports/$RUN/nvidia-layout.tsv
```

| Column | Meaning |
|---|---|
| `latency_source` | What filled the latency columns. `riva_compat` is `riva_client_compat_s`, NVIDIA's `recv[i] - send[i]`. When the server is not one response per chunk, `chunk_coverage/audio_processed` or `chunk_coverage/send_watermark` is `chunk_latency_s`, per-chunk coverage latency (`CHUNK_LATENCY_DEFINITION`); the watermark form is a lower bound. `n/a:final_only` means the server sent finals only, so there is no latency column. |
| `comparable` | `true` only when `latency_source` is `riva_compat` or `chunk_coverage/audio_processed` on every trial, `percentile_method` is `floor_index`, `pacing_validity` is `paced`, `interim_results` and `word_time_offsets` are `false`, and `iterations_per_lane` is at least 3. |
| `not_comparable` | Each gate above that failed, comma-separated. Empty when `comparable` is `true`. |
| `compat_status` | Distinct `riva_client_compat_s` availabilities across trials. `not_applicable` means the cell was not paced. |
| `trials_measured` | Trials whose latency source was measured, out of all trials. Must be `3/3` for a three-trial row. |
| `pacing_validity` | Distinct `pacing_fidelity.validity` across trials: `paced` (every chunk within max(5 ms, 5% of chunk) of its due time), `pacing_lagged` (within one chunk), `unpaced` (beyond), or `pacing_disabled`. Only `paced` is comparable. |
| `interim_results`, `word_time_offsets`, `iterations_per_lane` | The NVIDIA protocol knobs the row was run with. NVIDIA runs interims off, word offsets off, three iterations per stream. |
| `trailing_extra_ignored` | Sessions whose one trailing extra response was ignored, as NVIDIA's client does. Informational. |
| `intermediate_avg_ms`, `final_avg_ms` | Trial means of the `is_final` split of `riva_client_compat_s`: NVIDIA's intermediate and final latency by code. Not in NVIDIA's tables. |
| `finalization_p50_ms` | asrbench's `finalization_s` (last audio sent to final), beside `final_avg_ms` so neither is read as the other. NVIDIA's client drops the trailing final, so its final latency is not this. |
| `*_std` | Sample standard deviation over trials (`NVIDIA_TRIAL_STD_DEFINITION`). NVIDIA rounds by it. |
| `throughput_rtfx_server`, `throughput_server_gap` | Offline rows: the server's `audio_processed` sum over the span, NVIDIA's numerator, and its relative gap from the clip-duration throughput. A gap above 0.5% puts `server_audio_mismatch` in `not_comparable`. Offline latency columns are `asrbench_only`. |
| `engine_version`, `nim_profile_id`, `nim_tags_selector` | Which NIM answered, from the receipt. Match them to the profile NVIDIA's row names. |

Put a row next to NVIDIA's only when `comparable` is `true`. Anything else means
the latency columns are partial, a lower bound, or empty, and `not_comparable`
says why.

## 9. Side-by-side comparison template

Copy one block per NVIDIA table you match.

```markdown
### <Model>, <locale>, <mode>, chunk <C> ms, <GPU>

NVIDIA source: <page URL>, section <GPU / model / mode>, read on <date>.
NVIDIA settings: chunk <C> ms, language model <n-gram|none>, batch size <B or n/a>,
diarization <on|off>.

Site settings:
- NIM image: <image reference>, NIM version <version>, profile <selectedModelProfileId>
- NIM_TAGS_SELECTOR: <value>
- Model: <model_name>, checkpoint <id>:<revision>
- GPU: <name>, <MiB> MiB, driver <version>, count <n>
- Client: <host CPU>, location <loopback|same-region|wan>
- asrbench: <git commit>, schema asrbench-receipt/6, semantics 6, percentile floor_index
- Receipts: receipts/<RUN>/

| Streams | NVIDIA avg | Site avg | NVIDIA p50 | Site p50 | NVIDIA p90 | Site p90 | NVIDIA p95 | Site p95 | NVIDIA p99 | Site p99 | NVIDIA RTFX | Site RTFX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | | | |
| 8 | | | | | | | | | | | | |

Latency in ms. Site latency is riva_client_compat_s, trial mean.
Site RTFX is aggregate_rtfx, trial mean.

asrbench-only measures (not in NVIDIA's table), pooled over trials:

| Streams | finalization p50 (s) | finalization p99 (s) | frontier_lag p50 (s) | e2e_rtfx mean | WER (policy) | outcomes attempted/completed |
|---|---|---|---|---|---|---|
| 1 | | | | | | |

Differences that apply to this comparison: <pacing_fidelity, per-session channel,
client host, network path, anything else>.
```

## 10. Gaps

Things NVIDIA parity needed that asrbench could not do at `0e93681`, each filed
as an issue. Where a change has since addressed one, it is named under the gap.

1. **No offline mode on Riva gRPC.** NVIDIA's offline RTFX uses unary `Recognize`
   with 32 parallel requests over a 5x concatenated clip. `riva-grpc` refuses
   `--mode offline`.
   Issue: "riva-grpc: implement offline Recognize so offline_rtfx can match riva_asr_client"
   Addressed: `--mode offline` sends unary `Recognize` (section 6.1).
2. **No stream-count sweep in one command.** `performance_streams` in a preset is
   recorded but not executed. Each stream count is a separate run, looped in the
   shell.
   Issue: "run: execute a lanes sweep from performance_streams and emit one receipt per stream count"
   Addressed by `asrbench sweep` (section 6).
3. **Riva compat refuses the chunks + 1 case.** The Riva client accepts one extra
   trailing response and ignores it. asrbench marks the session, and so the whole
   statistic, invalid.
   Issue: "reducers: accept responses == chunks + 1 in riva_client_compat_s, as riva_streaming_asr_client does"
   Addressed by the streaming-parity change (semantics `6`).
4. **No intermediate and final latency split.** The Riva client reports both.
   asrbench reports only the pooled list.
   Issue: "reducers: split riva_client_compat_s into intermediate and final by is_final"
   Addressed by the streaming-parity change (semantics `6`).
5. **No built-in NVIDIA-layout projection of the compat latency.** The perf TSV
   uses finalization, in seconds.
   Issue: "reporters: NVIDIA streaming table row from riva_client_compat_s (ms, trial mean, aggregate RTFX)"
   Addressed by `asrbench nvidia-table` (section 8.4).
6. **Pacing drifts.** Each chunk is followed by a relative sleep, so delays add up
   over a session. The Riva client paces against an absolute schedule.
   Issue: "riva-grpc: pace chunks against an absolute schedule from first send"
   Addressed by the streaming-parity change (semantics `6`).
7. **Pacing fidelity is not computed.** Streaming RTFX validity reads
   `pacing_fidelity_unavailable` even though `chunk_send_times_s` is recorded.
   Issue: "reducers: compute pacing fidelity from chunk_send_times_s"
   Addressed by the streaming-parity change (semantics `6`).
8. **One gRPC channel per session.** The Riva client shares a channel across
   streams. Connection cost differs at high stream counts.
   Issue: "riva-grpc: option to share one channel across lanes like the Riva C++ client"
   Addressed: `transport_options.share_channel` (section 6).
9. **TLS options are minimal.** No CA bundle option (only the
   `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` environment variable), no client
   certificate for mutual TLS, and no target-name override.
   Issue: "riva-grpc: ssl_root_cert, client cert and key, and ssl_target_name_override transport options"
   Addressed: those four transport options (section 4.4).
10. **NIM identity is manual.** Version, image, profile ID, tags selector, and GPU
    are typed as flags or kept in side files. There is no receipt field for the
    profile ID or the tags selector.
    Issue: "identity: record NIM /v1/metadata (version, profile id, model URL) and NIM_TAGS_SELECTOR on the receipt"
    Addressed: `--nim-metadata` and the `nim_*` target fields, recorded under
    `identity.recorded.nim` (sections 4.3 and 5.4). Image digest and GPU are
    still typed or kept in side files.
11. **One clip per run.** NVIDIA's accuracy table uses full test sets. asrbench has
    no manifest of clips and references, so corpus WER takes many receipts and
    hand aggregation.
    Issue: "workload: accept a manifest of clips and reference transcripts for corpus WER"
12. **English-only number normalization.** `corpus_reference` normalizes English
    cardinals only. WER for other locales is not comparable to NVIDIA's normalized
    figures.
    Issue: "wer: locale-aware corpus_reference normalization for non-English models"
13. **No high-concurrency client characterization.** NVIDIA's tables reach thousands
    of streams with a C++ client. asrbench's Python asyncio client has not been
    measured at that scale.
    Issue: "runner: characterize client overhead at 256+ lanes and report client CPU per cell"
14. **No Nemotron Transcribe adapter.** Its tables use HTTP with concurrent
    requests, not Riva gRPC.
    Issue: "adapter: Nemotron Transcribe HTTP transport for concurrent-request RTFx"
15. **The README quickstart test needs the network.** It shells out to `uv run`,
    which resolves packages online.
    Issue: "tests: skip the README quickstart test when uv cannot resolve offline"
    Addressed: the test now skips with uv's reason when it cannot resolve.

## 11. Verification notes

Checked on a Linux box on 2026-11-12 against `0e93681`:

- Section 2: wheelhouse built with pip 26.3 for CPython 3.12 `manylinux` x86_64;
  git bundle created; LibriSpeech WAV prepared with
  `scripts/prepare_librispeech_clip.py`. The plain `ffmpeg` command in 2.4 gave a
  byte-identical WAV.
- Section 3: install, `pytest` (228 passed, 2 skipped, 1 deselected), and the mock
  run were done inside a network namespace with no network (`unshare -rn`), so
  `--no-index` was proven offline.
- Sections 4 to 8 (verified at `0e93681`, before receipt/6: the chunks + 1
  case now reads `measured`, and `nvidia_row.py` is superseded by
  `asrbench nvidia-table`): run inside the same offline namespace against a local fake Riva
  server that answers `GetRivaSpeechRecognitionConfig` and sends one response per
  chunk. Checked: `nim_asr_config.py` over plaintext and over TLS with a private
  CA; `asrbench run` with the site preset at 1 and 4 lanes, 3 iterations, 3
  trials; TLS through `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` (and the failure without
  it); the chunks + 1 case reading `invalid`; `--mode offline` being refused;
  `report --format tsv`, `index`, `sustained`, `frontier`, `compare` refusing
  across lane counts and accepting a repeat of the same cell; and
  `nvidia_row.py`. The commands in sections 2, 3, 4.2, 5.4, 6, and 8 were run as
  written, with only the placeholders filled and the sweep cut to 1 and 2 streams.
  The fake answers chunk `i` when chunk `i + 1` arrives, so its latency values
  (about 160 ms) mean nothing. Its throughput does: 0.992 RTFX at one stream,
  against NVIDIA's typical 0.9995, which is the pacing drift in gap 6.
- Not run here: anything against a real NIM, so `curl` to `/v1/health/ready` and
  `/v1/metadata`, the `docker inspect` and `nim_list_model_profiles` commands, and
  real NIM response counts with interims off are taken from NVIDIA's documentation
  and not verified. Confirm `riva_client_compat_s` reads `measured` on the first
  site smoke run.
