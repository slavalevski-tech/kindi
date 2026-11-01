# Lightning Backend V6 for NVIDIA Quadro K6000

**Lightning Backend V6 is the fork-specific Kindi 0.3.0 inference backend for
NVIDIA Quadro K6000 (12 GB, Kepler sm_35).** It provides native CUDA/CuPy execution
for the separately distributed Nemotron Lightning model. This release contains
backend source and build/launch tools; model weights and a chat frontend are not included.
**V6 is the backend implementation version.** The external model retains its own
name, authors and license.

An independent fork of [Kindi](https://github.com/ahtx/kindi), originally written
by **Amir Husain**. Fork-specific Lightning work is maintained by
**slavalevski-tech**, with modifications dated 2026-10-30 through 2026-11-01.
See [MODIFICATIONS.md](MODIFICATIONS.md) for the change history and attribution.

This backend runs **boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF** using
native CUDA/CuPy kernels. NVIDIA created the base model, Nota AI produced the
ternary checkpoint, and boro64 converted it to GGUF. **Only backend source is
distributed; no model weights are included or automatically downloaded.**
The external Nemotron Lightning model is distinct from Nemotron Bolt E4B.

## AI-assisted development and DevOps

ChatGPT/Codex generated the fork-specific implementation, including CUDA kernels,
and carried out build, testing and deployment workflows. The AI models used were
**GPT-6 Astra and GPT-6.1 Sol**. **slavalevski-tech** set the goals, provided
hardware and personally funded tools, and guided the experiments in spare time.
The provenance/license review was also AI-assisted, not an independent human
code or legal audit. Known sources and review limits are in [PROVENANCE.md](PROVENANCE.md).
Original Kindi remains credited to Amir Husain.

## Accepted V6 profile

- Linux x86_64, Quadro K6000 12 GB, Kepler sm_35.
- CUDA 10.2, CuPy 9.6.0, Python 3.6.9; accepted driver 470.256.02.
- **65,536 tokens total: prompt plus generated output.**
- FP16 KV storage with direct reads; FP32 arithmetic and recurrent state.
- tile24 dense/expert prefill, chunk 256, vec4/OR Q2 decoding, SSM auto.
- GQA16 vector2 stage32 attention with per-function four-byte bank requests.
- Exact full-vocabulary split Q8 output head; no approximate fast head.
- 512 MiB attention workspace; host prefix cache bounded to 1024 MiB/eight entries.
- One inference worker. The Compose example uses 4 GiB host RAM without swap.

Use the [step-by-step installation guide](INSTALLATION.md),
[BUILD.md](BUILD.md) for the concise procedure and [MODEL.md](MODEL.md) for the exact
existing GGUF identity. The Compose file and entrypoint set the complete V6
profile explicitly; changing only the context of a V3 setup does not select V6.
No RTX GPU, MLX, llama.cpp or PyTorch is required for production inference.
The retained upstream installer is not the Lightning installation procedure.

## Validation

The accepted development V6 passed full-model 64K retrieval and numerical checks,
plus authenticated HTTP/SSE, cache, cancellation, restart and rollback checks on
K6000 on 2026-10-31. See [VALIDATION.md](VALIDATION.md) for the exact scope and the
separate checks of this publication copy. The 64K retrieval result is a declared
fixture, not a guarantee of universal model quality. 128K is not an accepted V6 profile.

See [changes from V3 to V6](CHANGES-V3-TO-V6.md) for the detailed explanation of
context, memory, kernels, cache and upgrade behavior. Performance tables and
percentage-gain claims are deferred until a matched full benchmark is available;
[benchmark methodology](BENCHMARKS-LIGHTNING.md) states the planned protocol.

## License and corresponding source

Program code: **AGPL-3.0-only**, without warranty; see [LICENSE](LICENSE),
[NOTICE](NOTICE) and [LICENSING.md](LICENSING.md). Preserve original attribution,
license notices and modification notices. Model and dependency licenses are separate.
This is an independent fork, not an official upstream release or endorsement.

The Lightning API prominently offers the shipped corresponding source at `/`
and `/source`, downloads it at `/source/archive`, and serves `/license`.
Source/license routes require no inference API key. Preserve them through any
reverse proxy, and rebuild/restart after source changes. No registry image is published.
The original author's commercial offer is retained in COMMERCIAL-LICENSE.md;
the public license of this fork remains AGPL-3.0-only.

The previous published V3 remains in Git history at commit
`51c134248e9ff48ff1fa051adefbf228134359cb`.
Original upstream documentation is preserved in `docs/` and UPSTREAM-README.md;
its other models and benchmark claims do not describe Lightning V6.
