# Provenance, AI assistance and third-party licenses

## V10 review — 2026-11-02

The preparation starts from published V6 commit
`a6604b7c99d8ef703b26faba7eef76fa19e1b0b2`. All 58 development runtime
files were verified against the stable V10 SHA-256 manifest and read-only
container source. Nine numerical/dispatch files are imported. Three additional
paths retain public release identity, CLI notices and source-offer middleware;
other upstream source is retained after accounting for line endings.

The assembled CUDA source for `or_aos_lb4` + `load4` has SHA-256
`f79d9a537bf8c8b6992ae974afa5a0105c4aa1b27a3774308fcef132f6fb4390`,
matching the accepted development assembly. See
`licenses/Lightning-V10-source-provenance.json` for the separate source identities.

The lane4 head and Q2/attention staging derive from recorded native implementations
in this AGPL fork. V6-to-V10 changed ranges were compared with 32 pinned external
source files; all saved reference hashes were rechecked. Exact-token,
meaningful-line and identifier-normalized comparisons found no matches at the
declared thresholds. A synthetic known match was detected as a control.
This is not proof of every AI training source or absence of transformed copying.

Static import comparison found no new absolute import roots. The subsequent
clean public-image installation recorded Python 3.6.9 and 25 package metadata
records: the 24 previously inventoried component names, plus the distro copy of
pip 9.0.1 alongside active pip 21.3.1. The recorded primary runtime versions match
the table below. This is not an exhaustive Ubuntu/bundled dependency audit;
transitive/OS resolutions can vary in future builds.

NVIDIA's CUDA 10.2 Kepler tuning and best-practices guides were conceptual/API
references. This update vendors no SDK headers, samples, CUDA binaries or external
inference implementation. Existing third-party and model disclosures are retained.

ChatGPT/Codex assisted development, testing, DevOps and this review. Previously
disclosed model names include GPT-6 Astra and GPT-6.1 Sol; exact per-edit model
records for every later V10 change are not available. The maintainer's existing
CLA 1.1 acceptance covers later contributions under that version and explicitly
names earlier PR #16. It does not transfer upstream or third-party ownership.

No concrete source-license barrier to this source-only update or the maintainer's
own contribution permissions was identified within this scope. This does not
certify patents, copyright status in every jurisdiction, or binary/CUDA
redistribution. OpenAI's output terms do not override third-party licenses.
The V6 review below is retained as historical evidence; it is not a fresh V10
image installation test or an independent human legal audit.

## Historical V6 review

Reviewed 1 November 2026 for the source-only Lightning V6 publication.
Original Kindi 0.3.0: Amir Husain, upstream revision
`6ac903e5cb00a6d55b6cb60f6ad7445cafe93051`.
Accepted internal V6 source: `3f03ba3fad655088c5d7819f30f18a3da109d17d`.
Published V3 baseline: `51c134248e9ff48ff1fa051adefbf228134359cb`.

## Authorship and scope

Original Kindi remains credited to Amir Husain. The maintainer's contribution is
the fork-specific Lightning additions and changes, not ownership of upstream
Kindi, dependencies, model weights or reference projects. The program remains
AGPL-3.0-only; original notices and commercial-licensing information are retained.
See [LICENSE](LICENSE), [NOTICE](NOTICE), [LICENSING.md](LICENSING.md) and
[MODIFICATIONS.md](MODIFICATIONS.md).

The maintainer, **slavalevski-tech**, set the goals, supplied personal hardware and
funds, and directed the experiments in spare time. ChatGPT/Codex generated the
fork-specific implementation, including CUDA kernels, and carried out build,
testing and deployment workflows. The disclosed AI models were **GPT-6 Astra**
and **GPT-6.1 Sol**. The provenance review was also performed with AI and automated
tools; it was not an independent human code or legal audit.

[OpenAI's Terms of Use](https://openai.com/policies/row-terms-of-use/) assign
OpenAI's rights, if any, in output to the user to the extent permitted by law.
This does not supply rights in third-party material or establish originality of
every generated fragment. [Service Terms, section 4](https://openai.com/policies/service-terms/)
expressly preserve possible third-party licenses for generated code, including Codex output.

The maintainer [accepted Kindi CLA 1.1 for PR #16](https://github.com/ahtx/kindi/pull/16#issuecomment-6551899747),
explicitly naming the earlier submission. This does not make upstream or
third-party material the maintainer's work, remove its license conditions, or
promise that the author will merge the contribution.

## Development and comparison references

The following 11 files were included in the earlier source comparison and
rechecked for V6. Links identify fixed revisions. Some pins were selected for
the audit after development had cited a mutable branch; these pins are comparison
snapshots, not proof of the exact historical revision consulted.

| Reference source | License | Role |
| --- | --- | --- |
| [mlx_nemotron_h.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/28e9ccd9cb52d1d4a6f674b7914bae2de49d9aad/mlx_lm/models/nemotron_h.py) | MIT | Model/routing/SSM cross-check |
| [mlx_ssm.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/1e511c1f72a93f47e3c6d73254f1b5da2ae92d2d/mlx_lm/models/ssm.py) | MIT | Model/routing/SSM cross-check |
| [llama_nemotron_h.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/68d20b884c3878217de79eade0a249efde96ab38/src/models/nemotron-h.cpp) | MIT | Model/routing/SSM cross-check |
| [transformers_modeling_nemotron_h.py](https://raw.githubusercontent.com/huggingface/transformers/a672eb5f474b8b30173e9f96eab66adf292da635/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | Apache-2.0 | Model/routing/SSM cross-check |
| [mamba2.py](https://raw.githubusercontent.com/state-spaces/mamba/6b72c122713bb769cc82c6b8e6d019c53d27d6a1/mamba_ssm/modules/mamba2.py) | Apache-2.0 | Model/routing/SSM cross-check |
| [ssd_minimal.py](https://raw.githubusercontent.com/state-spaces/mamba/a70e3d85e2db4bb5ded2e01aa4e6a1cbda355b32/mamba_ssm/modules/ssd_minimal.py) | Apache-2.0 | Model/routing/SSM cross-check |
| [llama_conversion_nemotron.py](https://raw.githubusercontent.com/ggml-org/llama.cpp/68d20b884c3878217de79eade0a249efde96ab38/conversion/nemotron.py) | MIT | Model/routing/SSM cross-check |
| [ggml-common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/68d20b884c3878217de79eade0a249efde96ab38/ggml/src/ggml-common.h) | MIT | Packed-format or dispatch cross-check |
| [ggml-quants.c](https://raw.githubusercontent.com/ggml-org/llama.cpp/68d20b884c3878217de79eade0a249efde96ab38/ggml/src/ggml-quants.c) | MIT | Packed-format or dispatch cross-check |
| [vecdotq.cuh](https://raw.githubusercontent.com/ggml-org/llama.cpp/68d20b884c3878217de79eade0a249efde96ab38/ggml/src/ggml-cuda/vecdotq.cuh) | MIT | Packed-format or dispatch cross-check |
| [mlx_switch_layers.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/28e9ccd9cb52d1d4a6f674b7914bae2de49d9aad/mlx_lm/models/switch_layers.py) | MIT | Packed-format or dispatch cross-check |

Additional optimization references are disclosed separately from implementation
reuse. The development record describes the following as concept references,
not imported inference backends or vendored kernels. The selected revisions below
are supplementary audit snapshots, not a reconstruction of every historical read.

| Project and selected revision | License | Relationship to this work |
| --- | --- | --- |
| [NVIDIA FasterTransformer, `df4a753`](https://github.com/NVIDIA/FasterTransformer/tree/df4a7534860137e060e18d2ebf019906120ea204) | Apache-2.0 | Thread groups per key and reductions, used as background for native subwarp/GQA attention. |
| [vLLM, `3a90cc6`](https://github.com/vllm-project/vllm/tree/3a90cc6fd0504b52bf47d58e8d70cff4b03f2c19) | Apache-2.0 | Paged-attention thread grouping/layout background. Classic [v0.6.6, `f49777b`](https://github.com/vllm-project/vllm/tree/f49777ba62b4926d0f8c100ab06edb03c5c10098) was an additional comparison target. |
| [FlashAttention, `4616f51`](https://github.com/Dao-AILab/flash-attention/tree/4616f5116d33e3ba3cbb71d582caffcfcb49649c) | BSD-3-Clause | Tiled/split-KV attention and memory-traffic background; no FlashAttention runtime is used. |
| [FLUTE, `9eb83a1`](https://github.com/HanGuo97/flute/tree/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761) | Apache-2.0 | Low-bit layout/LUT background for unintegrated experiments. Its kernels are not included. |
| [LUT-GEMM, `647d49f`](https://github.com/naver-aics/lut-gemm/tree/647d49fcf0b054b6dafee04b5ce1c8c2adb047e5) | Apache-2.0; repository NOTICE also reviewed | Activation-dependent lookup background for rejected/unintegrated LUT experiments. Its kernels are not included. |

V6's vector2 attention derives from this fork's frozen GQA16 kernel. Q2 OR decode
and the split Q8 output head likewise reuse the fork's recorded prototypes.
The split head is not an imported FLUTE or LUT-GEMM implementation.

NVIDIA's [Kepler tuning guide](https://docs.nvidia.com/cuda/archive/10.2/kepler-tuning-guide/index.html)
and [CUDA 10.2 driver API](https://docs.nvidia.com/cuda/archive/10.2/cuda-driver-api/group__CUDA__EXEC.html)
were consulted for shared-memory behavior and `cuFuncSetSharedMemConfig`.
The per-function request uses Python's standard-library `ctypes` and the externally
installed driver. No NVIDIA SDK headers or sample implementations are vendored.

The development bank-driver diagnostic proposals cited CuPy 13.6.0 API sources
(`raw.pyx`, `function.pyx`, `function.pxd`) for the existing RTX tooling. The audit
checked [revision `25e552d`](https://github.com/cupy/cupy/tree/25e552d5d679dcdc6f7cc3d81310a9b265463137)
and its [MIT license](https://github.com/cupy/cupy/blob/25e552d5d679dcdc6f7cc3d81310a9b265463137/LICENSE).
This was a development reference, not a change to the K6000 production runtime.

For the installed K6000 version, this audit additionally checked CuPy 9.6.0
[revision `3a1d844`](https://github.com/cupy/cupy/tree/3a1d84421cdfaf3ee14a5631add6b1761cec8575)
([MIT license](https://github.com/cupy/cupy/blob/3a1d84421cdfaf3ee14a5631add6b1761cec8575/LICENSE))
for public function handles and compiler flags:
[`raw.pyx`](https://github.com/cupy/cupy/blob/3a1d84421cdfaf3ee14a5631add6b1761cec8575/cupy/_core/raw.pyx),
[`function.pyx`](https://github.com/cupy/cupy/blob/3a1d84421cdfaf3ee14a5631add6b1761cec8575/cupy/cuda/function.pyx),
[`function.pxd`](https://github.com/cupy/cupy/blob/3a1d84421cdfaf3ee14a5631add6b1761cec8575/cupy/cuda/function.pxd),
[`compiler.py`](https://github.com/cupy/cupy/blob/3a1d84421cdfaf3ee14a5631add6b1761cec8575/cupy/cuda/compiler.py).
It remains an external dependency; its implementation is not substituted into this fork.

If copied or adapted third-party expression is later identified, its original
license, copyright, attribution and change-notice requirements still apply.
Consulting an algorithm or API does not itself establish that an implementation
was copied; absence of a textual match does not establish the opposite either.

## Separately installed Python components

This inventory describes the saved clean V3 build used for the earlier license
review: eight direct runtime packages, fourteen transitive packages and two build
tools. The V6 production Python imports add no new external package and the public
CUDA 10.2 build recipe retains the direct versions. This is not a fresh inspection
of the V6 image filesystem or a guarantee of future transitive resolutions.

| Component | Reviewed version | License summary | Role |
| --- | --- | --- | --- |
| numpy | 1.19.5 | BSD-3-Clause (NumPy); BSD-3-Clause (OpenBLAS/LAPACK); GPL-3.0-or-later WITH GCC-exception-3.1 (bundled GCC runtimes) | Direct runtime |
| fastrlock | 0.8.3 | MIT | Direct runtime |
| cupy-cuda102 | 9.6.0 | MIT (CuPy); BSD-3-Clause (NumPy/SciPy/CUB/Jitify portions) | Direct runtime |
| fastapi | 0.79.1 | MIT | Direct runtime |
| uvicorn | 0.16.0 | BSD-3-Clause | Direct runtime |
| pydantic | 1.9.2 | MIT | Direct runtime |
| Jinja2 | 3.0.3 | BSD-3-Clause | Direct runtime |
| MarkupSafe | 2.0.1 | BSD-3-Clause | Direct runtime |
| starlette | 0.19.1 | BSD-3-Clause | Transitive runtime |
| click | 8.0.4 | BSD-3-Clause | Transitive runtime |
| h11 | 0.13.0 | MIT | Transitive runtime |
| asgiref | 3.4.1 | BSD-3-Clause | Transitive runtime |
| typing-extensions | 4.1.1 | PSF-2.0 and historical Python license notices | Transitive runtime |
| dataclasses | 0.8 | Apache-2.0 | Transitive runtime |
| contextlib2 | 21.6.0 | PSF-2.0 (implementation) AND Apache-2.0 (type hints) | Transitive runtime |
| anyio | 3.6.2 | MIT | Transitive runtime |
| importlib-metadata | 4.8.3 | Apache-2.0 | Transitive runtime |
| sniffio | 1.2.0 | MIT OR Apache-2.0 | Transitive runtime |
| idna | 3.10 | BSD-3-Clause | Transitive runtime |
| contextvars | 2.4 | Apache-2.0 | Transitive runtime |
| zipp | 3.6.0 | MIT | Transitive runtime |
| immutables | 0.19 | Apache-2.0 AND MIT (pythoncapi_compat.h) | Transitive runtime |
| pip | 21.3.1 | MIT (top-level; vendored dependency details not exhaustively inventoried) | Build tool |
| setuptools | 59.6.0 | MIT (top-level; vendored dependency details not exhaustively inventoried) | Build tool |

The review read 32 LICENSE/NOTICE files from exact package artifacts; their saved
SHA-256 values were rechecked for this review. Twenty-one small artifacts had been
fully hash-verified; NumPy, CuPy and Pydantic license entries were read using HTTP
byte ranges, without recomputing their complete large-wheel hashes. Build-tool
vendored dependencies and the full Ubuntu package closure were not exhaustively
inventoried. Direct pins do not lock all transitive, apt or base-image revisions.

Dependency packages are obtained separately, not published as source inside this
repository. Preserve their actual license files when distributing them. Production
inference does not require PyTorch, Transformers, MLX, Mamba, vLLM or FlashAttention.
An internal RTX diagnostic used an already installed PyTorch package to locate
CUDA DLLs; that machine-specific launcher is outside the K6000 publication.

## Model, CUDA and distribution boundaries

No model weights, CUDA SDK distribution, dependency wheels, compiled kernels or
container images are published by this source release. The Lightning weights are
obtained separately under OpenMDW 1.1; see [MODEL.md](MODEL.md) and
[`licenses/Nemotron-OpenMDW-1.1.txt`](licenses/Nemotron-OpenMDW-1.1.txt).
The inherited small numerical validation fixtures are outputs, not model parameters.

CUDA 10.2 is proprietary and separately obtained under the
[NVIDIA CUDA 10.2 EULA](https://docs.nvidia.com/cuda/archive/10.2/eula/index.html).
Using its documented interfaces does not grant ownership of the SDK. The source
license does not relicense the toolkit or authorize unrestricted redistribution
of an image containing it. A binary/image release would need its own review of
CUDA redistribution, dependency notices and corresponding-source obligations.

## Review result and limits

Within the reviewed source and documented references, no concrete instance of
unlicensed copying or an incompatible license on the fork-specific additions was
identified. The expanded comparison included 32 selected external source files,
using exact normalized-token, meaningful-line and identifier-normalized checks,
alongside source-lineage and manual mechanism review.

The review cannot recover every source behind AI output, exclude all transformed
or unknown-source copying, clear every patent, or determine copyright protection
in every jurisdiction. It is a bounded technical and license-evidence review,
not a legal certification or guarantee against claims. Third-party rights and
license conditions remain in force.
