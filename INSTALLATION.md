# Install Lightning V6 on a Quadro K6000

This guide installs the independent [slavalevski-tech/kindi fork](https://github.com/slavalevski-tech/kindi),
based on Kindi 0.3.0 by Amir Husain, as a separate native inference server.
Original Kindi need not be installed first. Clients connect to the backend's
OpenAI-compatible API; the backend includes no chat interface.

V6 follows the development profile accepted on 31 October 2026. The Docker recipe
passed a V3 clean-install test on 30 October; no fresh V6 Docker/GPU test is claimed.
Read [VALIDATION.md](VALIDATION.md) for the exact
scope, and [CHANGES-V3-TO-V6.md](CHANGES-V3-TO-V6.md) for the changes.
Performance tables are intentionally withheld until comparable full measurements.

## 1. Installation target and resources

Use a native Linux x86_64 host with a Quadro K6000 12 GB, a compatible working
NVIDIA driver and Docker GPU support. A Windows computer can use SSH and connect
as an API client. Docker Desktop/WSL GPU support is not the tested K6000 path;
K6000 is Kepler, outside NVIDIA's documented Pascal-and-newer WSL GPU support.

| Item | Requirement or accepted value |
| --- | --- |
| GPU | Quadro K6000, 12 GB, compute capability 3.5 |
| Host driver | Existing compatible driver; 470.256.02 was used |
| Containers | Docker Engine, Compose v2 or later, BuildKit/buildx, NVIDIA Container Toolkit |
| Command-line tools | Bash, Git, curl, OpenSSL, coreutils; OpenSSH on a remote client |
| Build RAM | Several GB free for RAM-backed CUDA extraction; earlier clean-build host had 32 GB |
| Service memory limit | 4 GiB RAM, no additional container swap |
| Docker storage | At least 20 GB free for image layers/build cache |
| Model | Existing 9,472,656,320-byte GGUF outside the checkout |
| V6 GPU memory plan | About 10.26 GiB including reserves, for the declared profile |
| Image runtime | CUDA 10.2.89, CuPy 9.6.0, Python 3.6.9 |
| Context | 65,536 total tokens, including input formatting and output |

The Ubuntu 18.04 image is separate from the host OS. Preserve working host
drivers/shared installations; CUDA 10.2 is installed inside Docker.

Use Bash on the Linux Docker host; apply Docker `sudo` consistently if needed.
Proceed after each step succeeds.

## 2. Check existing prerequisites before installing anything

```bash
command -v bash git curl openssl sha256sum stat
uname -m
cat /etc/os-release
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
docker version
docker compose version
docker buildx version
docker info --format '{{json .Runtimes}}'
docker info --format '{{.DockerRootDir}}'
free -h
df -h
```

Confirm x86_64, a visible K6000, Docker/Compose/buildx and free RAM/storage on
Docker's actual filesystem. Reuse compatible installations; official prerequisite
references are in section 15.

Check the selected GPU's current processes before verification or serving. A
second Compose project does not provide a second GPU or additional VRAM. Build
without loading a model while another service is running, but account for build
RAM/disk use. Before loading V6, stop only workloads you own and intend to replace.
For an existing V3 installation, follow the separate upgrade/rollback procedure
in section 13 rather than changing its directory or configuration.

## 3. Clone V6 into a new directory and record its revision

First check whether a suitable clone already exists. For a new installation:

```bash
git clone https://github.com/slavalevski-tech/kindi.git kindi-lightning-v6
cd kindi-lightning-v6
git rev-parse HEAD
git show HEAD:kindi/__init__.py
```

The public `main` branch should identify `0.3.0-lightning-v6`. Record the actual
commit printed by Git with the installation log. The internal source identity in VALIDATION.md is provenance, not a public
checkout target. Do not select the old V3 revision for this install.

Leave other Kindi installations in their own directories. The new clone has
separate private configuration, pack manifest, state, image tag and Compose
project while reusing the existing GGUF.

## 4. Identify and reuse the correct model

This profile is for **boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF**,
not Nemotron Bolt E4B or another Kindi checkpoint.

| Property | Expected identity |
| --- | --- |
| GGUF revision | `570610701b08e16ffcfaa899a0bdec94f0cda819` |
| Filename | `Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf` |
| Size | `9472656320` bytes |
| SHA-256 | `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3` |

Set the path to your existing model. The following location is only an example:

```bash
MODEL_FILE='/srv/models/nemotron/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf'
test -f "$MODEL_FILE"
stat -c '%s' "$MODEL_FILE"
sha256sum "$MODEL_FILE"
```

Check unfamiliar/newly downloaded files; reuse an already verified unchanged
GGUF without repeated hashing. Obtain a genuinely missing file from the [pinned repository](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/570610701b08e16ffcfaa899a0bdec94f0cda819),
store it outside the checkout and verify it. Neither Docker build nor conversion
downloads the model. See [MODEL.md](MODEL.md) for provenance and the separate license.

## 5. Prepare private configuration before Compose

The following setup preserves existing private copies if repeated:

```bash
umask 077
mkdir -p .local/pack .local/state
cp -n lightning.conf.example .local/kindi.conf
cp -n .env.example .env
chmod 600 .env .local/kindi.conf
```

Edit `.env` with your own Linux host directory and GPU index:

```dotenv
MODEL_DIRECTORY='/srv/models/nemotron'
GPU_DEVICE=0
LIGHTNING_PORT=18095
```

`MODEL_DIRECTORY` is a directory, not the GGUF filename. It is mounted read-only
at `/models`. `GPU_DEVICE` is the host index from `nvidia-smi`; the GPU may appear
as index 0 inside the container. `LIGHTNING_PORT` is the host API port, while the
container port stays 8093. Choose an unused host port, such as 18195, for an
isolated installation alongside a stopped or existing configured service.

Generate a key for `.local/kindi.conf`:

```bash
openssl rand -hex 32
```

Use the generated value without quotes on the `api_key` line below. Do not keep
the literal placeholder. Retain these V6 settings:

```ini
[kindi]
pack = /workspace/pack
state_dir = /workspace/state
host = 0.0.0.0
port = 8093
model_name = nemotron-lightning-30b-a3b-ternary
ctx = 65536
head = exact
lightning_kv = fp16
checkpoint_budget_mb = 1024
prefix_max = 8
prefix_min = 512
max_tokens = 1024
temperature = 0.7
top_p = 0.8
top_k = 20
api_key = YOUR_GENERATED_VALUE
```

Keep `.env` and `.local/` private. Context, exact-head and KV settings belong in
this config: the launcher clears inherited `KINDI_*` values except the
`KINDI_LIGHTNING_*` optimization settings. `host = 0.0.0.0` listens inside the
container; Compose publishes only on the host's loopback interface by default.

The twelve V6 optimization settings are supplied by Compose/entrypoint; see
[BUILD.md](BUILD.md). Vector2 attention requires FP16/direct/optimized/split,
Lightning geometry 32/2/128 and capacity ≤65536. Changing only KV to FP32/Q8_0
while leaving vector2 enabled is invalid.

Check configuration before building:

```bash
docker compose -p kindi-lightning-v6 config --quiet
```

Use this project name consistently. Projects isolate containers, not VRAM.
Parallel experiments also need distinct local image tags; V6 uses
`kindi-lightning:v6-publication`.

## 6. Build the backend from public sources

```bash
docker compose -p kindi-lightning-v6 build lightning
```

The first build downloads a roughly 2.65 GB CUDA installer and packages. BuildKit
uses temporary RAM; weights are not copied. Reuse compatible layers. Historical
V3 used `--no-cache`; no fresh V6 clean-build claim is made here.

OS/APT/transitive packages are not completely digest locked; record the built
image and package identities with your installation.

Check GPU visibility without loading a model:

```bash
docker compose -p kindi-lightning-v6 run --rm --entrypoint nvidia-smi lightning -L
```

## 7. Convert the manifest and run numerical verification

Ensure enough VRAM is free and no other model instance uses this K6000. Replace
the filename if your verified file has a different name:

```bash
docker compose -p kindi-lightning-v6 run --rm lightning convert /models/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --out /workspace/pack && \
  docker compose -p kindi-lightning-v6 run --rm lightning verify
```

Conversion writes `.local/pack/kindi_pack.json`, referring to `/models/...`
inside the container. It does not write another model copy. Keep the same mount
for verification and serving. If a pack exists, inspect its manifest first;
use a separate private pack directory when changing models.

`verify` loads the model, checks its identity and three full-vocabulary outputs
against the matching FP16 reference, then checks exact snapshot restoration.
Initial load/compilation can take time. Success includes:

```json
{"passed": true, "snapshot_bit_exact": true}
```

The output includes errors/hash/profile. Resolve failures before serving. This
short check is not a 64K retrieval test. For later verification, stop this API:

```bash
docker compose -p kindi-lightning-v6 stop lightning && \
  docker compose -p kindi-lightning-v6 run --rm lightning verify && \
  docker compose -p kindi-lightning-v6 up -d lightning
```

A failed verify leaves the service stopped. Never run a second model-loading
CLI process beside the service on one K6000.

## 8. Start the API and wait for readiness

```bash
docker compose -p kindi-lightning-v6 up -d lightning
docker compose -p kindi-lightning-v6 ps
docker compose -p kindi-lightning-v6 logs --tail 50 lightning
BASE_URL='http://127.0.0.1:18095'
curl --fail --show-error "$BASE_URL/health"
```

Use your selected port. Inspect logs if still loading; the 180-second healthcheck
grace period may be shorter than first initialization. Ready health includes:

```json
{
  "ok": true,
  "backend": "kindi_lightning",
  "model": "nemotron-lightning-30b-a3b-ternary",
  "ctx": 65536,
  "head": "exact",
  "kv": "fp16",
  "busy": false,
  "phase": "idle",
  "prefill_chunk": 256
}
```

Health also reports memory/cache state and `prefill`/`decode` request phase.
Public health does not establish inference authentication.

## 9. Test authentication and ordinary/streamed requests

Read the configured key without echoing it:

```bash
read -r -s -p 'Paste your configured API key: ' API_KEY
printf '\n'
curl --fail --show-error "$BASE_URL/v1/models" -H "Authorization: Bearer $API_KEY"
```

Send a short deterministic request:

```bash
curl --fail --show-error --max-time 1800 "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "nemotron-lightning-30b-a3b-ternary",
  "messages": [{"role": "user", "content": "Reply with only the number that equals six times seven."}],
  "temperature": 0,
  "max_tokens": 32,
  "stream": false
}
JSON
```

The answer is in `choices[0].message.content`. Historical engineering checks
included 42; this is a connectivity check rather than a broad quality test.

For streaming:

```bash
curl --no-buffer --fail --show-error --max-time 1800 "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "nemotron-lightning-30b-a3b-ternary",
  "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
  "temperature": 0,
  "max_tokens": 64,
  "stream": true
}
JSON
```

SSE uses `data:` events and ends with `data: [DONE]`. Requests are serialized;
avoid parallel calls while checking installation. Verify rejection without a key:

```bash
curl --include --max-time 10 "$BASE_URL/v1/models"
```

With `api_key` configured, this should return 401. If it returns 200, check the
actual config mount/key and restart this service.

## 10. Connect a client or SSH tunnel

| Client field | Value |
| --- | --- |
| Provider | OpenAI-compatible chat completions |
| Host-local base URL | `http://127.0.0.1:18095/v1` |
| Model | `nemotron-lightning-30b-a3b-ternary` |
| API key | Your configured private key |
| Context | 65536 total, including formatting and generated output |
| First output budget | 128 or 256 tokens |
| Initial request timeout | 30 minutes; allow more for long prompts |

For a remote desktop client, leave the service on loopback and open a tunnel on
the client computer, replacing the placeholders:

```bash
ssh -N -L 18195:127.0.0.1:18095 YOUR_USER@YOUR_SERVER
```

Keep SSH running and use `http://127.0.0.1:18195/v1` with the same model/key.
OpenSSH in Windows PowerShell also works. The first port is client-local;
the last is server-local. Containers have separate networking.

## 11. Check corresponding source and licenses

```bash
curl --fail --show-error "$BASE_URL/source"
curl --fail --show-error --output .local/corresponding-source.zip "$BASE_URL/source/archive"
curl --fail --show-error "$BASE_URL/license"
```

The public ZIP contains running-image source/notices, excluding weights, `.env`,
private config and state. Preserve these routes through a proxy and rebuild after
source edits so the offer matches execution.

## 12. Lifecycle, CPU checks and installation records

Run only the lifecycle command you need:

```bash
docker compose -p kindi-lightning-v6 stop lightning
docker compose -p kindi-lightning-v6 up -d lightning
docker compose -p kindi-lightning-v6 restart lightning
```

`stop` exits and releases service VRAM. `up -d` starts it; `restart` reloads
bind-mounted config. Compose restarts a crashed/exited service unless stopped;
healthcheck failure alone does not restart a live failed process. For port/GPU/
mount changes in `.env`, recreate this project:

```bash
docker compose -p kindi-lightning-v6 up -d --force-recreate lightning
```

Source changes require a rebuild, verification and API checks. CPU checks in
[BUILD.md](BUILD.md) need no model or GPU.

Record the installation while it is running:

```bash
git rev-parse HEAD
LIGHTNING_CONTAINER_ID="$(docker compose -p kindi-lightning-v6 ps -q lightning)"
docker inspect "$LIGHTNING_CONTAINER_ID" --format '{{.Image}}'
docker compose -p kindi-lightning-v6 exec -T lightning python3 -m pip freeze > .local/python-packages.txt
docker compose -p kindi-lightning-v6 logs --no-color lightning > .local/startup.log
```

Keep logs private if your own logging adds requests, host details or credentials.
`docker compose -p kindi-lightning-v6 down` removes this project's containers and
network while preserving external GGUF files and bind-mounted `.local/` files.
Preserve shared models, tools and working images when cleaning up experiments.

## 13. Isolated V3 upgrade and rollback

Preserve the V3 checkout, private files, container and image. In its directory,
record the running revision/image using the actual existing project name:

```bash
OLD_PROJECT='kindi-lightning-v3'
git rev-parse HEAD
OLD_CONTAINER_ID="$(docker compose -p "$OLD_PROJECT" ps -q lightning)"
test -n "$OLD_CONTAINER_ID"
OLD_IMAGE_ID="$(docker inspect "$OLD_CONTAINER_ID" --format '{{.Image}}')"
docker image tag "$OLD_IMAGE_ID" kindi-lightning:v3-local-rollback
```

Keep a private backup of V3 config/key and leave its clone unchanged. Prepare
V6 separately using sections 3–6, the same read-only weights and another host
port, such as 18195. Record that port in the new `.env` and client URLs.

When ready to test V6, stop only the original V3 service from its own directory:

```bash
docker compose -p "$OLD_PROJECT" stop lightning
```

Change into the V6 directory. Run its manifest conversion, numerical verify and
API/auth/source checks. Start only one model instance on the K6000. The old V3
pack/state/config must not be mounted into the new service. V6 snapshots use a
new genuine fingerprint; do not import old numerical-profile state by overriding
that check. Fresh V6 conversion reuses weights without making a second copy.

After V6 passes, update the client connection. Keep the preserved V3 container
stopped for rollback; the V6 build uses a separate image tag.

If V6 verification or API checks fail, stop V6 from its directory:

```bash
docker compose -p kindi-lightning-v6 stop lightning
```

Return to the original V3 directory and start the preserved container without
rebuilding or recreating it:

```bash
docker compose -p "$OLD_PROJECT" start lightning
docker compose -p "$OLD_PROJECT" ps
```

Check the old health/API and restore the previous client connection settings.
This preserves the original container configuration. If that container was
removed independently, use your recorded source/config and saved rollback image
in an explicit local Compose override; do not substitute a new image silently.

## 14. Troubleshooting

- Missing model/pack: check `.env`, the read-only `/models` mount, filename and
  pinned hash. Inspect an existing pack before replacing it.
- GPU/NVML/no-kernel errors: check host `nvidia-smi`, the selected GPU, Container
  Toolkit and the supplied CUDA 10.2 image. Do not replace working host CUDA/drivers.
- GPU OOM: stop only competing workloads you control. Stop the API before GPU
  CLI operations; separate project names do not isolate VRAM.
- Build failure: inspect the actual download error, Docker filesystem, free RAM
  and BuildKit support. Keep the supplied tmpfs extraction step.
- Connection refused/port conflict: inspect startup logs and selected host port;
  recreate the project after changing mappings.
- HTTP 401: check the actual key/config mount. Use the exact Bearer value without
  literal placeholder or quotes.
- Wrong profile: require context 65536, FP16 KV, exact head and the complete V6
  optimization settings. The vector2 guard rejects incompatible combinations.
- HTTP 400: use supported text parameters and fit prompt/formatting/output within
  65536 tokens. A failed engine requires diagnosis and restart.
- Source edit absent: rebuild the image; restart alone does not replace code.

Speed tables await a full comparable campaign. Earlier 64-forward timing is not
full 256-token measurement. See [VALIDATION.md](VALIDATION.md) and
[BENCHMARKS-LIGHTNING.md](BENCHMARKS-LIGHTNING.md) for distinct acceptance scopes.

## 15. Prerequisites, attribution and primary references

Reuse existing Docker/Compose/Container Toolkit installations. A missing or
broken driver needs a host-specific plan; the newest driver is not automatically
a K6000 replacement. Host Docker runtime configuration and daemon restarts may
affect other containers, so they are not routine backend-install steps.

- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Compose plugin](https://docs.docker.com/compose/install/linux/)
- [BuildKit](https://docs.docker.com/build/buildkit/)
- [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [K6000 compute capability](https://developer.nvidia.com/cuda/gpus/legacy)
- [Quadro Kepler driver support](https://nvidia.custhelp.com/app/answers/detail/a_id/5210)
- [NVIDIA WSL support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Original Kindi is by **Amir Husain**. This independent fork is maintained by
**slavalevski-tech**, with Lightning changes developed using ChatGPT/Codex.
Source is AGPL-3.0-only; model materials use OpenMDW 1.1; CUDA and dependencies
retain their licenses. No CLA signature is required to install/use the fork.
See [LICENSE](LICENSE), [NOTICE](NOTICE), [LICENSING.md](LICENSING.md),
[MODEL.md](MODEL.md) and [VALIDATION.md](VALIDATION.md).
