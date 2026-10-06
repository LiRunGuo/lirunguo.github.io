---
title: "vLLM — Attention Kernels & KV Cache Quantization"
excerpt: "Contributor to vLLM: merged a fix that lets the int4 per-token-head KV cache run on head sizes that are not powers of two, where a power-of-two-only Hadamard rotation aborted engine initialization instead of quantizing."
collection: portfolio
category: contribution
order: 2
permalink: /portfolio/vllm/
---

I contribute to [vLLM](https://github.com/vllm-project/vllm), focusing on the numerics of the KV-cache quantization paths and on the models those paths quietly refuse to serve.

### Merged contribution

**[PR #56198 — Support `int4_per_token_head` on non-power-of-two head sizes](https://github.com/vllm-project/vllm/pull/56198)** · Merged October 5, 2026 (+43 / −6 across 2 files). Fixes [#56197](https://github.com/vllm-project/vllm/issues/56197).

`--kv-cache-dtype int4_per_token_head` is accepted for every head size, but the write path rotates K/V with a Walsh–Hadamard transform that asserted a power-of-two dimension:

```text
reshape_and_cache_int4 -> single_rht -> fast_hadamard_transform
  assert d & (d - 1) == 0, f"Requires power-of-2 dim, got {d}"
```

So a model whose head size is not a power of two — Phi-3-mini-4k-instruct, which vLLM supports as `Phi3ForCausalLM`, has `head_size = 96` — selected `TRITON_ATTN`, logged that the mode "reduces the GPU memory footprint and boosts the performance", and then aborted during engine initialization. The assertion fires inside the engine-core process, so the only thing the user sees is an opaque wrapper:

```text
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

Walsh–Hadamard transforms exist for any composite length, so the rotation can be performed rather than refused. For `d = b * n`, with `b` the largest power-of-two divisor of `d`, applying `H_b` block-diagonally inside each of the `n` blocks is still an orthogonal transform (`H @ H.T = b * I`). That keeps the cache layout untouched — no padding of the last dimension, no extra cache bytes — and `b == d` for power-of-two head sizes, so those models take exactly the code path they took before. The change touches two places: `single_rht` becomes block-diagonal while `fast_hadamard_transform` keeps its power-of-two contract for every other caller, and `unified_attention_int4` scales its inner-product and output compensations by `b` instead of `head_size`, since `sqrt(b)` is the norm the two `1 / head_size` factors were cancelling.

End to end on an H200, one command against `main` and against the PR, with the dtype as the only difference:

| | main | this PR |
|:--|:--|:--|
| Phi-3-mini + `int4_per_token_head` | `AssertionError: Requires power-of-2 dim, got 96`, engine init aborts | starts and generates |

Controls on this PR — `auto`, `int8_per_token_head`, `fp8_per_token_head`, and Qwen2.5-1.5B (`head_size = 128`) with `int4_per_token_head` — all pass, i.e. the power-of-two path is unaffected. The transform was also checked against an independent fp64 block-diagonal reference (relative error 3.56e-3 for the unchanged 128 path, 3.99e-3 for 96, 3.52e-3 for 192), which is the bf16 round-trip rather than the block decomposition. On `lm_eval` gsm8k 5-shot / 500 samples, int4 lands 0.03 below bf16 while `int8_per_token_head` — which never uses the transform — lands at the same level, so that gap is the cost of per-token-head quantisation, not of this change. Serving measurements are reported honestly as a trade: the INT4 decode path is slower at Phi-3's 1k-token, concurrency-32 operating point, and no throughput win is claimed — it buys a roughly 4× smaller KV cache.

[View my vLLM pull requests](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+author%3ALiRunGuo)
