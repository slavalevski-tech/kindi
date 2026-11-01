# Lightning benchmark methodology

Performance tables, projected rates and V3-to-V6 percentage gains are intentionally
withheld from this V6 publication. A complete benchmark will be run and published
separately. No throughput gain should be inferred from the optimization names or
the correctness checks.

## Planned comparison

- Use the same K6000, pinned GGUF, rendered input token IDs, sampling policy and
  output length. Record source/image revisions, driver, CUDA/CuPy/Python versions,
  device clocks and temperature, context capacity, KV dtype and every profile flag.
- Measure input lengths 1,000, 4,000, 8,000 and 16,000 tokens, including chat template.
- Use **256 generation steps** for both versions, with the same documented EOS policy.
- Warm each fresh process consistently, then reset before every full prefill.
  Disable prefix reuse. Use repeated interleaved runs and report medians and spread.
- Synchronize GPU boundaries. State whether tokenization, loading, compilation,
  sampling, scalar transfers, HTTP, logging and validation copies are timed.
- Check token/logit/state results under a declared numerical comparison. FP16 KV
  and FP32 KV are different profiles; document this rather than claiming byte parity.
- Separate operator timing, engine timing, time-to-first-token and HTTP/SSE throughput.
  Save commands and evidence so another person can reproduce the test.

## Existing evidence and its limits

Historical V3 generation measurements used 256 steps at capacity 32,768 with FP32
KV. The saved short V6 engineering comparison used 64 steps at capacity 65,536
with FP16 KV and compared V5 with V6. Those are different protocols. They do not
establish a matched V3/V6 speed comparison, 256-step V6 rates, or an 8K/16K V6 series.
The new release does not republish their speed tables or extrapolate their values.

Correctness and API acceptance are documented in [VALIDATION.md](VALIDATION.md).
Use [INSTALLATION.md](INSTALLATION.md) for setup. Do not benchmark a GPU that is
serving another workload without coordinating that workload first.
