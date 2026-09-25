---
title: "PyTorch — CUDA Kernels & cdist"
excerpt: "Contributor to PyTorch: merged a fix for an int32 overflow in the cdist CUDA backward kernel, where indices wrapped negative past 2^31 and slipped past the bounds check into an illegal memory access."
collection: portfolio
category: contribution
order: 1
permalink: /portfolio/pytorch/
---

I contribute to [PyTorch](https://github.com/pytorch/pytorch), focusing on CUDA kernel correctness and on the boundary cases where 32-bit index arithmetic stops being safe.

### Merged contribution

**[PR #198452 — Fix int32 overflow in the cdist CUDA backward kernel](https://github.com/pytorch/pytorch/pull/198452)** · Merged September 25, 2026 (+21 / −4 across 2 files). Fixes [#128791](https://github.com/pytorch/pytorch/issues/128791).

`torch.cdist(x1, x2, p=1)` backward raised `CUDA error: an illegal memory access was encountered` on an H200 once either of two quantities reached 2³¹:

```text
(b, r, m) = (2, 8191, 32)   OK
(b, r, m) = (2, 8192, 32)   illegal memory access   # r1 * r2 * m = 2^31
(b, r, m) = (31, 8192, 1)   OK
(b, r, m) = (32, 8192, 1)   illegal memory access   # dist.numel() = 2^31
```

The forward half of this issue had already been fixed by [#188006](https://github.com/pytorch/pytorch/pull/188006), which capped the cdist/pdist forward grid and grid-strides over the outputs. The backward kernel still failed at the same scale, because `cdist_backward_kernel_cuda_impl` computed its indices in `int` while the forward kernel already used `int64_t` for the same quantities:

- `y = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y`. The launcher rounds the grid up, so once `dist.numel()` reaches 2³¹ the extra threads have `y >= 2^31`. That wraps negative, passes the `y >= count` guard, and reads `grad[y]` / `dist[y]` out of bounds.
- `l_size = r_size * m` is the per-batch stride of the `(batch, r2, r1, m)` scratch buffer. When `r1 * r2 * m` reaches 2³¹ it wraps, so `buffer + l * l_size` points *before* the buffer for every batch `l >= 1`.

The fix makes `y`, `l`, `k` and `l_size` `int64_t`, casting at the first multiply. They are computed once per thread before the inner loop, so the loop keeps its existing pointer-increment form. I considered templating the kernel on `index_t` and dispatching on `canUse32BitIndexMath` instead, but rejected it: that doubles the number of kernel instantiations to save a few 64-bit multiplies per thread, and it would leave the backward kernel inconsistent with the forward one, which uses `int64_t` unconditionally. The launch configuration itself was already fine — `grid_y`/`grid_z` are split to stay under 65535 — so it only overflows when `dist.numel()` exceeds roughly 6.9e10.

New test `test_cdist_backward_large_index` covers both overflows (`largeTensorTest('32GB')`; measured peaks 16.1 GiB and 24.3 GiB). It compares the last batch against that same batch computed on its own, which stays within int32; integer-valued `grad` with p=1 makes the reductions exact, so the comparison is bitwise rather than order-dependent. On 1× H200 with CUDA 12.8: the targeted test fails on the nightly and passes with the fix, 29 pass on the `pdist or cdist` subset, 80 pass across `test_ops.py`, and `compute-sanitizer --tool memcheck` reports 0 errors on both reproducers. Backward timing is unchanged (differences within run-to-run noise across eight shape/p combinations).

[View my PyTorch pull requests](https://github.com/pytorch/pytorch/pulls?q=is%3Apr+author%3ALiRunGuo)
