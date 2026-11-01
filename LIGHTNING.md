# Lightning Backend V6 for NVIDIA Quadro K6000

This extension runs the pinned Nota Nemotron 3.5 Lightning 30B-A3B ternary
checkpoint with native CUDA kernels on Kepler sm_35. Model computation stays on
the GPU. Tokenization, request-local sampling and bounded prefix caching use CPU.
The accepted graph has 23 Mamba-2, 23 MoE and 6 attention blocks. Experts use ReLU
squared, sigmoid routing and a shared expert. Attention uses NoPE; misleading
GGUF metadata does not select Gemma or Qwen RoPE.

## Runtime and model

The accepted K6000 stack is CUDA 10.2, CuPy 9.6 and Python 3.6.9 on driver 470.256.02.
[BUILD.md](BUILD.md) gives the source recipe. Its base stage installs the runtime;
the extension stage adds no packages or weights. RTX 4080 was used in earlier
reference work and is not required for K6000 use.

| Artifact | Identity |
| --- | --- |
| Original Kindi | `6ac903e5cb00a6d55b6cb60f6ad7445cafe93051`, version 0.3.0 |
| Accepted V6 development source | `3f03ba3fad655088c5d7819f30f18a3da109d17d` |
| GGUF repository | `boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF` |
| GGUF revision | `570610701b08e16ffcfaa899a0bdec94f0cda819` |
| GGUF size | 9,472,656,320 bytes |
| GGUF SHA-256 | `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3` |
| Nota revision | `85889ce1c6c89bc6c999e28cf01c2dce0d0c4ef4` |

`kindi convert EXISTING.gguf --out PACK` writes a manifest pointing to existing
weights; it does not duplicate them. `verify` checks three complete vocabulary
outputs against the matching storage reference and bit-exact restoration.
`chat`, `bench`, `info` and `serve` dispatch to the native adapter when the pack
manifest names backend `kindi_lightning`.

## Published V6 profile

| Setting | Value |
| --- | --- |
| Total context capacity | 65,536 tokens, prompt plus generated output |
| Packed GPU weights before head conversion | 9,464,761,344 bytes |
| Recurrent/convolution state | 49,930,240 bytes, FP32 |
| KV storage/read policy | FP16, direct; 384 MiB at 64K |
| Attention | Optimized split, `gqa16_vec2_32_bank4`, 512 MiB workspace |
| General GPU reserve | 512 MiB |
| Output head | Exact full vocabulary, split Q8 layout |
| Prefill | tile24 dense/experts, vec4 Q2 input, chunk 256 |
| Q2 decode / SSM strategy | `or` / `auto` |
| GPU concurrency | One serialized worker |
| Host prefix cache | 1024 MiB budget, at most 8 entries |
| Container RAM/swap | 4 GiB RAM, no additional swap |

The split Q8 head replaces the packed GPU head after conversion. Steady extra
allocation is 21 MiB; startup temporarily allocates 378 MiB before retiring the
original 357 MiB head. This changes GPU layout, not disk weights or vocabulary.
The accepted memory plan is about 10.26 GiB including reserves for this profile.

Compose and the entrypoint explicitly select the twelve settings in
[BUILD.md](BUILD.md). Direct library use retains conservative defaults unless
configured. The vector2 strategy requires FP16 KV, direct reads, optimized split
attention, geometry 32/2/128 and capacity at most 65536. Incompatible profiles are
rejected before GPU allocation where possible. Changing just KV dtype does not
preserve this V6 configuration.

Mamba recurrence/state remain FP32. FP16 KV is storage, not FP16 model arithmetic
or universal equivalence to FP32 storage. Vector2 attention pairs FP32 shared V
loads, retaining scalar global FP16 reads, 128-key leaves and existing FP32
reduction/merge order. It requests four-byte shared-bank configuration on tile
and merge functions individually. Failed requests reject the profile.

## API and state

Endpoints include `/health`, `/v1/models`, `/v1/chat/completions` and
`/v1/completions`. Public source/legal routes are described in
[LICENSING.md](LICENSING.md). Requests support UTF-8 streaming, exact stops,
temperature/top-p/top-k, seed, usage, thinking and the function-call template.
`tool_choice` accepts `auto` and `none`; unsupported options are rejected before
streaming. Multimodal input and continuous batching are outside this adapter.

Every request creates its own RNG. Prefix reuse does not inherit sampling state.
Snapshots contain recurrent state, convolution history, used KV, consumed IDs
and valid logits. Compatibility includes model SHA, adapter/tokenizer and kernel
source, numerical profile, CUDA runtime and compute capability. V6 does not
override a fingerprint to import another source/profile snapshot. Snapshots
stay in host RAM and disappear on restart.

The cache checks prospective snapshot size before copying device data and
rejects oversized entries without making that copy. The server releases its
restored-entry reference before later LRU eviction. This avoids retaining an
evicted checkpoint or copying one that cannot fit the cache budget.

Cancellation is checked at bounded synchronized chunk boundaries. An
unrecoverable CUDA error rejects further work and makes health return 503 until
restart. Compose restarts an exited process; unhealthy status alone does not
restart a process that remains running.

## Validation and limits

[VALIDATION.md](VALIDATION.md) separates V6 development evidence from publication
packaging and historical V3 installation. V6 acceptance includes a genuine
65280-token retrieval input at capacity 65536, three declared facts correct in
both roles, 354 state-array digest comparisons, 8 raw-array pairs and API/cache/
cancellation/restart checks. This is bounded evidence, not universal quality.

Speed tables and numerical performance comparisons are withheld pending a full
comparable campaign. The historical short V6 protocol used 64 forwards at
1000/4000 inputs; it is not a measured 256-token run or warm V6 8K/16K test. Cold 64K
observations are diagnostics. See [BENCHMARKS-LIGHTNING.md](BENCHMARKS-LIGHTNING.md).

Historical V1/V2/V3 work covered operators, tokenizer/templates and sequential/
chunk parity. V3 acceptance was 32768 total tokens with FP32 KV and tile16/chunk 128.
V6's separate 64K acceptance applies only to its declared profile. Earlier failed
V5 evidence is preserved; its 64K outlier cause remains unresolved. The BF16 teacher
is untested. There is no V6 128K or million-token claim, new UI, speculative
decoding, multi-GPU support or integration of future routed16/persistent/chunk 512
prototypes. CUDA 11.4 was researched, not installed or benchmarked for this release.

The implementation does not change host clocks, power limits, driver or BIOS.
Legacy `cuda-memcheck` is excluded following a historical `nvidia_uvm` oops whose
original cause was not conclusively established.

## Licensing and origins

Original Kindi is by Amir Husain. The extension retains AGPL-3.0-only LICENSE
and original notices. Changes and AI assistance are disclosed in
[MODIFICATIONS.md](MODIFICATIONS.md), [NOTICE](NOTICE) and [README.md](README.md).
Weights are separately governed by OpenMDW 1.1, whose notice is retained in
[licenses/Nemotron-OpenMDW-1.1.txt](licenses/Nemotron-OpenMDW-1.1.txt).

Architecture/layout were cross-checked against the pinned Nota checkpoint,
MLX-LM Nemotron/SSM and llama.cpp Nemotron. Transformers and Mamba were also
cross-check references, as disclosed in NOTICE. Production inference does not
import those implementations or PyTorch.

- [Nota checkpoint](https://huggingface.co/nota-ai/Nemotron-3.5-Lightning-30B-A3B-Nota-Ternary-MLX-1014/tree/85889ce1c6c89bc6c999e28cf01c2dce0d0c4ef4)
- [GGUF checkpoint/license](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF/tree/570610701b08e16ffcfaa899a0bdec94f0cda819)
- [CUDA 10.2 Kepler tuning guide](https://docs.nvidia.com/cuda/archive/10.2/kepler-tuning-guide/index.html)
- MLX-LM Nemotron revision `28e9ccd9cb52d1d4a6f674b7914bae2de49d9aad`; SSM revision `1e511c1f72a93f47e3c6d73254f1b5da2ae92d2d`.
- llama.cpp Nemotron revision `68d20b884c3878217de79eade0a249efde96ab38`.

The private development workspace retains experiment evidence/source inventories.
Public files provide backend source and summaries, not private experiment folders
or infrastructure configuration.
