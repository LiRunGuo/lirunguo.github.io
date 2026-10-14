---
title: "Triton — GPU Compiler Correctness"
excerpt: "Contributor to Triton: two merged compiler fixes — loop fusion no longer trusts an assumption that does not dominate the loop, and fp64 atomic_xchg on AMD buffer atomics compiles instead of aborting the process."
collection: portfolio
category: contribution
order: 11
permalink: /portfolio/triton/
---

I contribute to [Triton](https://github.com/triton-lang/triton), with a focus on GPU compiler correctness and reproducible regression tests.

### Merged contributions

**[PR #11521 — FuseNestedLoops: only trust llvm.assume that dominates the loop](https://github.com/triton-lang/triton/pull/11521)** · Merged September 8, 2026.

The nested-loop fusion pass could use an assumption from an unrelated branch to conclude that an inner loop must execute. This could remove the guard for a zero-iteration loop and produce incorrect results or unintended memory writes.

I added a dominance check so the pass only uses assumptions that hold before the loop executes, preserving the zero-trip guard when required. The contribution includes an MLIR regression test and verification of the reproducer on an NVIDIA H200 GPU.

**[PR #11904 — [AMD] Swap floating-point values as integers in buffer atomic xchg](https://github.com/triton-lang/triton/pull/11904)** · Merged October 13, 2026 (+94 / −1 across 4 files).

`tl.atomic_xchg` on an fp64 tensor aborted the process on gfx942, gfx950 and gfx1250:

```text
LLVM ERROR: Cannot select: t90: f64,ch = AMDGPUISD::BUFFER_ATOMIC_SWAP<(volatile dereferenceable load store (s64) ...
```

`ConvertToBufferOps` turns the atomic into a buffer atomic exchange, which lowered to `llvm.amdgcn.raw.ptr.buffer.atomic.swap` on the value type, and the AMDGPU backend has no f64 buffer swap. A swap only moves bits, so `BufferEmitter::emitAtomicRMW` now emits the exchange on the same-width integer type and bitcasts around it. Writing the end-to-end test also showed that the Triton interpreter rejected floating-point `atomic_xchg` although the GPU path accepts it, so `float` and `double` were added there too.

Validated with MLIR lit tests (the f64 case fails before and passes after; all 124 AMD lit tests pass), a new `test_core.py` test passing on H200 and in the interpreter, and a compile-only matrix of `atomic_xchg`/`atomic_add`/`atomic_max` over four dtypes and two memory orderings on gfx942/gfx950/gfx1250: 6 fp64 exchange variants aborted before, and all 72 compile after.

[View my Triton pull requests](https://github.com/triton-lang/triton/pulls?q=is%3Apr+author%3ALiRunGuo)
