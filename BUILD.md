# Build and run Lightning Backend V6 on NVIDIA Quadro K6000

This source-only fork includes the complete backend; original Kindi need not be
installed separately. V6 was accepted on K6000 in development on 2026-10-31. The
public Docker recipe was clean-built and GPU-smoke-tested for V3 on 2026-10-30.
This publication does not claim a new clean V6 Docker build or GPU installation
run. See [VALIDATION.md](VALIDATION.md) for the separate records and
[INSTALLATION.md](INSTALLATION.md) for the detailed installation and upgrade guide.

## Requirements

- Linux x86_64, Quadro K6000 12 GB, a working compatible NVIDIA driver and NVIDIA Container Toolkit.
- Docker with BuildKit and Docker Compose v2 or later; permission to run Docker (prefix commands with sudo if required).
- The existing pinned GGUF described in [MODEL.md](MODEL.md). This procedure neither includes nor downloads model weights.
- At least 20 GB free disk for image layers and build cache. The initial build downloads a roughly 2.65 GB CUDA installer plus packages.
- Several GB of free RAM for CUDA extraction through BuildKit tmpfs. The earlier clean-build host had 32 GB RAM; the running V6 service was accepted with a separate 4 GiB container limit and no swap.

Check `nvidia-smi`. The accepted stack is CUDA 10.2, CuPy 9.6.0 and Python 3.6.9
on driver 470.256.02. The runtime lives inside Docker; preserve a working host
driver. The GPU must have room for the model and workspace. Before inference,
stop other model workloads on that GPU through their normal controls.

Run these commands from the cloned or extracted repository directory.

## 1. Prepare private settings before invoking Compose

```sh
mkdir -p .local/pack .local/state
cp lightning.conf.example .local/kindi.conf
cp .env.example .env
```

Edit `.env`: set `MODEL_DIRECTORY` to the existing absolute model directory on the
Linux Docker host and `GPU_DEVICE` to the K6000 index. The default host API port
is 18095; change `LIGHTNING_PORT` if occupied. Keep `.env` and `.local/` private.
Configure your own random `api_key` in `.local/kindi.conf` before exposing
inference beyond localhost, together with authenticated transport. Compose binds
to `127.0.0.1` by default and ships no credentials.

The example selects 65536 total prompt-plus-output tokens, `head = exact`,
`lightning_kv = fp16`, `checkpoint_budget_mb = 1024`, `prefix_max = 8` and
`prefix_min = 512`. API/CLI entry points pass KV dtype explicitly from config;
they need no duplicate `KINDI_LIGHTNING_KV` variable. Direct Python callers must
supply `kv_dtype` or the corresponding environment value themselves.

## 2. Build from public sources

```sh
docker compose build lightning
```

The first stage follows upstream's `docker/Dockerfile`, starting from public
`ubuntu:18.04` and installing CUDA 10.2 and pinned primary Python dependencies.
CUDA extraction uses RAM-backed BuildKit temporary storage. The final stage
includes the complete source and checks [SOURCE-FILES.txt](SOURCE-FILES.txt).
No private base image is required.

The historical clean V3 build used `docker compose build --no-cache lightning`.
V6 publication has not repeated that build on a fresh host. Image bytes are not
reproducible: the OS tag, APT packages and some transitive dependencies are not
digest locked. Save the image identity and `python3 -m pip freeze` when recording
a new installation validation.

Compose and the entrypoint select these twelve optimization variables:

| Variable | Value |
| --- | --- |
| `KINDI_LIGHTNING_KERNELS` | `optimized` |
| `KINDI_LIGHTNING_DENSE_PREFILL` | `tile24` |
| `KINDI_LIGHTNING_EXPERT_PREFILL` | `tile24` |
| `KINDI_LIGHTNING_Q2_INPUT` | `vec4` |
| `KINDI_LIGHTNING_Q2_DECODE` | `or` |
| `KINDI_LIGHTNING_PREFILL_CHUNK` | `256` |
| `KINDI_LIGHTNING_ATTENTION` | `split` |
| `KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB` | `512` |
| `KINDI_LIGHTNING_KV_READ` | `direct` |
| `KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY` | `gqa16_vec2_32_bank4` |
| `KINDI_LIGHTNING_SSM_STRATEGY` | `auto` |
| `KINDI_LIGHTNING_HEAD_LAYOUT` | `split` |

V6 attention requires FP16 KV, direct reads, optimized split attention and pinned
32-query-head/2-KV-head/128-dimension geometry. It rejects capacity above 65536.
Changing just KV to FP32 or Q8_0 is not a valid version of this profile. Other
strategies need compatible configuration and their own validation; see
[LIGHTNING_ADAPTERS.md](LIGHTNING_ADAPTERS.md).

## 3. Register and verify the existing GGUF

```sh
docker compose run --rm lightning convert /models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --out /workspace/pack
docker compose run --rm lightning verify
```

Use the actual file name if different, after checking [MODEL.md](MODEL.md).
`convert` creates a small manifest pointing to the existing read-only `/models`
file; it does not copy weights. Keep the same mount for conversion and serving.
`verify` uses the GPU to check model identity, three full-vocabulary outputs
against the matching FP16 reference and bit-exact snapshot restoration. It is a
short numerical check, not a 64K retrieval benchmark.

## 4. Start and check the API

```sh
docker compose up -d lightning
curl --fail http://127.0.0.1:18095/health
curl --fail -o corresponding-source.zip http://127.0.0.1:18095/source/archive
```

Adjust the port if changed. Health should report `ok:true`, context 65536,
`head:exact`, `kv:fp16` and prefill chunk 256. Model loading and first compilation
take time; inspect `docker compose logs --tail 30 lightning`. Client base URL:
`http://127.0.0.1:18095/v1`; model: `nemotron-lightning-30b-a3b-ternary`.
Send your configured Bearer API key for inference.

The source/license routes are public regardless of the inference key. Their ZIP
uses the reviewed manifest and excludes weights/private configuration. The split
Q8 head still evaluates the exact vocabulary; it is distinct from Kindi's
approximate fast-head option. The backend has no chat UI.

Compose sets 4 GiB RAM, no additional swap and restart-unless-stopped. Docker
unhealthy status alone does not restart a process that is still running.

## CPU checks and shutdown

Use an existing Python environment with the required dependencies. These checks
do not load a model or use the GPU:

```sh
python3 scripts/check_release.py
python3 -m unittest discover -s tests -p test_source_offer.py -v
# Also require NumPy; protocol tests require Jinja2:
PYTHONPATH=. python3 tests/test_protocol.py
PYTHONPATH=. python3 tests/test_sampling.py
PYTHONPATH=. python3 tests/test_gqa_dispatch_cpu.py
PYTHONPATH=. python3 tests/test_prefix_capture.py
PYTHONPATH=. python3 tests/test_vector2_cpu.py
# Stop only this Compose project; mounted models and state remain:
docker compose down
```

For a separate installation use a unique Compose project (`docker compose -p
YOUR_TEST_NAME ...` consistently), separate `.local/` state/configuration and a
different host port. After diagnosing an unrecoverable CUDA error, use
`docker compose restart lightning`. Inspect GPU acceptance scripts before use;
legacy `cuda-memcheck` is excluded for this K6000 profile.
