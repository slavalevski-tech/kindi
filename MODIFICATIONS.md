# V6 modified-version notice — 2026-11-01

Release scope: Lightning Backend V6 is the Kindi inference backend optimized and
validated for NVIDIA Quadro K6000. V6 names the backend implementation; the external
Nemotron Lightning model is separately distributed and retains its own authorship.

Kindi 0.3.0 + Lightning backend V6 is maintained by slavalevski-tech under
AGPL-3.0-only. Original Kindi: Copyright (c) 2026 Amir Husain. The full upstream
license/commercial-license text and original notices remain preserved.

Accepted development source: `3f03ba3fad655088c5d7819f30f18a3da109d17d`.
Public V3 base: `51c134248e9ff48ff1fa051adefbf228134359cb`.
The public Git history continues the original upstream/fork history; private
snapshot histories, host wrappers, credentials and development evidence are not imported.

Changes since published V3: eight accepted core files (kv.py, lightning.cu,
lightning.py, lightning_batch.cu, lightning_batch.py, lightning_kv.cu, moe.py,
prefix_cache.py), the matching cache lifetime/admission changes in
lightning_server.py, new CPU/quantization tests, and publication documentation,
source-offer labels, release identity, explicit launch/config defaults and source manifest.
Numerical CUDA/adapter code matches the accepted source after line-ending normalization.
The server retains the public AGPL source offer. New tests use explicit model
arguments instead of internal host paths; the host-specific RTX launcher is excluded.
See VALIDATION.md and PROVENANCE.md for validation and licensing-review boundaries.

ChatGPT/Codex (GPT-6 Astra and GPT-6.1 Sol) generated implementation and assisted
development, testing, DevOps and the provenance review. The maintainer supplied
goals, hardware/resources and experimental direction. No independent human code
or legal audit is claimed. The model weights are unchanged and not distributed.

SOURCE-FILES.txt lists the complete V6 public payload. The following original
publication record documents V3 and its earlier modifications; its profile and
file inventory are historical and are superseded by the V6 manifest/defaults above.

---

# Modified version notice — 2026-10-30

This is Kindi 0.3.0 + Lightning backend V3, maintained by slavalevski-tech,
under **AGPL-3.0-only**. Original Kindi copyright (c) 2026 Amir Husain.
This fork is not endorsed by the original author or the model publishers.

## Provenance

- Original repository: https://github.com/ahtx/kindi
- Base commit: `6ac903e5cb00a6d55b6cb60f6ad7445cafe93051` (Kindi 0.3.0).
- Accepted development V3: `3e6f680eb6a886d0fded3cbba392c79b49baaacb`.
- Earlier development V2: `d31f3acc8e46fa410e06601acef09191beec7669`.
- The public preparation starts from the real upstream commit graph. Internal
  snapshot-based development history, server wrappers and private reports are
  not imported. The development IDs above identify provenance, not ancestors
  available in this repository.

## Changes to upstream

Native Lightning model adapter and CLI dispatch; packed Q2_0/Q8_0/F32 loading;
Mamba2/MoE/attention kernels; tile16 dense/expert prefill and split attention;
FP32/FP16/Q8 KV storage; tokenizer/tools, OpenAI-compatible API, prefix snapshots,
cancellation and failure handling; numerical fixtures and acceptance scripts.

Publication preparation adds restored Git archive revision substitution, explicit V3 launch defaults, self-contained Docker
build instructions with RAM-backed CUDA extraction, release identity, source-only packaging and ASGI source/license
routes. No numerical CUDA kernel logic or model weight is modified by this preparation.
Original upstream line endings are retained to keep the change diff focused.

`LICENSE` and `COMMERCIAL-LICENSE.md` are preserved verbatim. The original NOTICE
is retained verbatim within NOTICE and separately in docs/UPSTREAM-NOTICE.txt.
Original source copyright/SPDX headers are retained. New source-offer/packaging
code is also AGPL-3.0-only. The original README is preserved as UPSTREAM-README.md.

## File inventory relative to upstream

The following files are changed or added by this modified version. Changes are
dated 2026-10-30; the upstream changelog is retained separately within CHANGELOG.md.
SOURCE-FILES.txt is the complete public payload, including inherited source.

- `.dockerignore`
- `.env.example`
- `.gitattributes`
- `.gitignore`
- `BENCHMARKS-LIGHTNING.md`
- `BUILD.md`
- `CHANGELOG.md`
- `Dockerfile.lightning`
- `LICENSING.md`
- `LIGHTNING.md`
- `LIGHTNING_ADAPTERS.md`
- `MODEL.md`
- `MODIFICATIONS.md`
- `NOTICE`
- `README.md`
- `SOURCE-FILES.txt`
- `UPSTREAM-README.md`
- `VALIDATION.md`
- `compose.yaml`
- `docs/UPSTREAM-NOTICE.txt`
- `examples/prompt_cache_bench.py`
- `kindi.conf.example`
- `kindi/COMMIT`
- `kindi/__init__.py`
- `kindi/__main__.py`
- `kindi/config.py`
- `kindi/convert.py`
- `kindi/gemma4_cache.py`
- `kindi/gemma4_tp.py`
- `kindi/kv.py`
- `kindi/lightning.cu`
- `kindi/lightning.py`
- `kindi/lightning_batch.cu`
- `kindi/lightning_batch.py`
- `kindi/lightning_cli.py`
- `kindi/lightning_kv.cu`
- `kindi/lightning_kv_reference.json`
- `kindi/lightning_reference-fp16.npz`
- `kindi/lightning_reference-q8_0.npz`
- `kindi/lightning_reference.npz`
- `kindi/lightning_server.py`
- `kindi/lightning_tokenizer.py`
- `kindi/moe.py`
- `kindi/packed.py`
- `kindi/prefix_cache.py`
- `kindi/server.py`
- `kindi/source_offer.py`
- `licenses/Nemotron-OpenMDW-1.1.txt`
- `lightning.conf.example`
- `release_entry.py`
- `scripts/check_release.py`
- `tests/accept_api.py`
- `tests/accept_batch.py`
- `tests/accept_capacity.py`
- `tests/accept_context.py`
- `tests/accept_lightning.py`
- `tests/accept_long.py`
- `tests/bench_attention.py`
- `tests/bench_full.py`
- `tests/bench_linear.py`
- `tests/bench_prefill_chunks.py`
- `tests/bench_prefill_linear.py`
- `tests/bench_prefill_variants.py`
- `tests/calibrate_kv.py`
- `tests/diagnose_prefill.py`
- `tests/profile_lightning.py`
- `tests/stress_api.py`
- `tests/test_failure.py`
- `tests/test_generic_moe.py`
- `tests/test_kv.py`
- `tests/test_lightning_batch.py`
- `tests/test_lightning_ops.py`
- `tests/test_loader.py`
- `tests/test_protocol.py`
- `tests/test_sampling.py`
- `tests/test_source_offer.py`
