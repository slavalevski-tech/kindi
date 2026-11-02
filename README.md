# Lightning Backend V10 for NVIDIA Quadro K6000

An independent **Kindi 0.3.0 custom inference backend for Quadro K6000
(12 GB, Kepler sm_35)**. V10 names the backend implementation, not the model.
This source release contains backend code and build/launch tools. Obtain the
model separately; no weights, CUDA distribution or container image are included.

Original [Kindi](https://github.com/ahtx/kindi): **Amir Husain**, copyright 2026.
Fork-specific Lightning development is maintained by **slavalevski-tech**.
This is an independent fork, without a claim of upstream endorsement or merge.
See [NOTICE](NOTICE) and [MODIFICATIONS.md](MODIFICATIONS.md).

The supported model is
**boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF**: NVIDIA base model,
Nota AI Lightning adaptation, boro64 ternary/GGUF distribution. Names,
authorship and model licensing remain separate; see [MODEL.md](MODEL.md).

## V10 profile and changes

- CUDA 10.2, CuPy 9.6.0 and native kernels for Quadro K6000.
- 65,536 **total** tokens, including formatting, prompt and answer.
- FP16 KV with direct reads; FP32 Mamba state; exact full-vocabulary Q8 head.
- Exact `split_lane4` output head, inherited from V7.
- Q2 prefill with `or_aos_lb4` and four-halfword attention staging with `load4`.
- Chunk 256, tile24 dense/expert prefill, one GPU inference worker.

V7–V10 change the output-head layout, Q2 prefill and attention memory staging.
V10's newest change uses a vector AoS shared-input layout. The tested arithmetic,
model weights and quantization are retained. See [CHANGES-V6-TO-V10.md](CHANGES-V6-TO-V10.md).
This release does not claim support for another GPU or contexts above 65,536.

Use [INSTALLATION.md](INSTALLATION.md) for the detailed guide and [BUILD.md](BUILD.md)
for concise build/configuration commands. Existing installations should use a
separate checkout, image tag, private config, pack and state directory.
Compose/entrypoint select the complete V10 profile; changing just the context or
model of an earlier installation does not select V10.

## Validation status

The development V10 passed finite numerical, actual-image and API/service gates.
On 2026-11-02 this public recipe also passed a **clean build without cache and
real K6000 installation/API acceptance**, including tool calls, streaming,
cache isolation, cancellation, restart and restoration of the original service.
All 153 files offered by the tested API matched the frozen input source.
Documentation/receipts were refreshed afterwards; runtime files remain identical.
See [VALIDATION.md](VALIDATION.md) and [the validation receipt](VALIDATION-LIGHTNING.json)
for the exact tested identities and limits.

## Saved speed measurements on our overclocked K6000

**Kindi 0.3.0 + custom Lightning Backend V10 · Quadro K6000 12 GB · CUDA 10.2**

Ternary-Nemotron-3.5-Lightning-30B-A3B (boro64). FP16 direct KV, FP32 Mamba,
chunk 256, capacity 65,536 total tokens. **256 output tokens; one observation per row.**

| Input tokens | Read / prefill (tok/s) | Generation (tok/s) |
| ---: | ---: | ---: |
| 1,000 | 117.86 | 36.49 |
| 4,000 | 114.19 | 35.28 |
| 8,000 | 108.99 | 33.96 |
| 16,000 | 99.81 | 31.84 |
| 64,000 | 65.75 | 22.74 |

Our particular board reports **993 MHz graphics / 3350 MHz memory**, with a
**250 W power limit**. These are a later read-only snapshot of the reported maxima,
not proof of constant clocks in each run. Stock-clock performance is not measured.
64K was measured in a separate session. See [benchmark details](BENCHMARKS-LIGHTNING.md)
for timing, warmup, hardware and single-observation limits. These are development
image results; no model-quality, universal correctness or statistical claim is made.

## AI-assisted development and DevOps

ChatGPT/Codex assisted implementation, review, testing, documentation and DevOps.
The maintainer supplied goals, hardware, personal resources and experimental
direction. Previously disclosed tools include GPT-6 Astra and GPT-6.1 Sol;
an exact model-by-model record for every later V10 edit is not available.
The same AI tools assisted the provenance review; no independent human legal
or code audit is claimed. Known references and review limits are in [PROVENANCE.md](PROVENANCE.md).

## License and corresponding source

Program code: **AGPL-3.0-only**, without warranty. Preserve original copyright,
license and modification notices; see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).
Dependencies, CUDA and the separately obtained model retain their own licenses.

The shipped Lightning API offers matching source at `/` and `/source`, a complete
download at `/source/archive`, and the license at `/license`, without an inference
API key. Keep these routes and source links accessible through reverse proxies.
Rebuild after source changes so the archive matches the running build.

The maintainer accepted upstream **CLA 1.1**, explicitly covering PR #16.
This concerns the maintainer's contributions and does not relicense third-party
work or give the public an exception to the fork's AGPL. Original commercial
licensing information is retained in COMMERCIAL-LICENSE.md.

## Versions and history

- Current release: [Lightning Backend V10](https://github.com/slavalevski-tech/kindi/releases/tag/lightning-k6000-v10).
- Previous source snapshots: [V6](https://github.com/slavalevski-tech/kindi/tree/lightning-k6000-v6) and [V3](https://github.com/slavalevski-tech/kindi/tree/lightning-k6000-v3).
- [Changes from V6 to V10](CHANGES-V6-TO-V10.md); Git history and original authorship are preserved.

Original upstream documentation remains in `docs/` and UPSTREAM-README.md.
Its historical model and benchmark claims do not describe this K6000 backend.
