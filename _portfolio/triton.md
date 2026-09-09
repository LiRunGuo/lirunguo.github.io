---
title: "Triton — GPU Compiler Correctness"
excerpt: "Contributor to Triton: merged a loop-fusion correctness fix that prevents invalid assumptions from removing zero-trip guards and causing unintended memory writes."
collection: portfolio
category: contribution
order: 2
permalink: /portfolio/triton/
---

I contribute to [Triton](https://github.com/triton-lang/triton), with a focus on GPU compiler correctness and reproducible regression tests.

### Selected merged contribution

**[PR #11521 — FuseNestedLoops: only trust llvm.assume that dominates the loop](https://github.com/triton-lang/triton/pull/11521)** · Merged September 8, 2026.

The nested-loop fusion pass could use an assumption from an unrelated branch to conclude that an inner loop must execute. This could remove the guard for a zero-iteration loop and produce incorrect results or unintended memory writes.

I added a dominance check so the pass only uses assumptions that hold before the loop executes, preserving the zero-trip guard when required. The contribution includes an MLIR regression test and verification of the reproducer on an NVIDIA H200 GPU.

[View my Triton pull requests](https://github.com/triton-lang/triton/pulls?q=is%3Apr+author%3ALiRunGuo)
