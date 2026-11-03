---
title: "FlashInfer — Attention Planning Metadata"
excerpt: "Contributor to FlashInfer: merged a fix that makes strided indptr, indices, and last_page_len metadata contiguous before attention planning and kernel use, preventing illegal memory accesses and wrong outputs across ragged, paged, and block-sparse backends."
collection: portfolio
category: contribution
order: 15
permalink: /portfolio/flashinfer/
---

I contribute to [FlashInfer](https://github.com/flashinfer-ai/flashinfer), focusing on the contract between the Python attention wrappers and the C++ planners and kernels underneath them.

### Merged contribution

**[PR #5385 — Accept non-contiguous index tensors in `plan()`](https://github.com/flashinfer-ai/flashinfer/pull/5385)** · Merged October 27, 2026 (+452 / −2 across 4 files). Fixes [#5260](https://github.com/flashinfer-ai/flashinfer/issues/5260).

`plan()` passed user-supplied index metadata (`indptr`, `indices`, and `last_page_len`) to the C++ planner and kernels without checking whether the tensors were contiguous. A stride-2 view such as `indptr_full[::2]` was therefore read with the wrong layout. The failure depended on device:

- With CUDA metadata, `.to(self.device)` was a no-op, so the kernel buffers stayed strided, while `.to("cpu")` produced a contiguous host copy for the planner. The plan and kernel could then disagree.
- With CPU metadata, `.to("cpu")` was a no-op, so the strided host tensor went directly into the C++ planner.

On an H200 (SM90), identical stride-2 metadata produced illegal memory accesses, planner check failures, or wrong output with maximum absolute errors of 2.6--4.6 across ragged and paged prefill, paged decode, and block-sparse attention. The fix makes the metadata contiguous at the Python boundary before planning or kernel use. Already-contiguous tensors return the same object, so the common path does not allocate, and CUDA-graph paths continue copying into preallocated buffers.

The regression suite adds `tests/attention/test_plan_noncontiguous_metadata.py` with 32 cases covering ragged and paged prefill, paged decode with and without tensor cores, `fast_decode_plan`, and block-sparse attention, with metadata on CUDA and CPU. Before the fix, 24 of the 29 cases that existed then failed. With the fix, all 32 pass, and a sampled 1,498-case regression run across the affected kernels reports 1,140 passed, 358 skipped, and zero failures.

[View my FlashInfer pull requests](https://github.com/flashinfer-ai/flashinfer/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged)
