---
title: "Apache TVM — ONNX Frontend & Symbolic Shapes"
excerpt: "Contributor to Apache TVM: merged a Relax/ONNX frontend fix that lets Hardmax lower when the reduced axis has a symbolic extent, instead of failing on a static-only attribute."
collection: portfolio
category: contribution
order: 8
permalink: /portfolio/tvm/
---

I contribute to [Apache TVM](https://github.com/apache/tvm), focusing on the ONNX frontend for Relax and on what happens when a graph carries symbolic shape information rather than fixed extents.

### Merged contribution

**[PR #20416 — support Hardmax over a symbolic axis extent (Fix, Relax, ONNX)](https://github.com/apache/tvm/pull/20416)** · Merged September 25, 2026 (+21 / −0 across 2 files).

The ONNX importer lowers `Hardmax` to `argmax` followed by `one_hot`, which works until the reduced axis has a symbolic extent. `relax.op.one_hot` takes its depth as a *static integer attribute*, so passing a `tir.Var` fails outright:

```text
TypeError: Mismatched type on argument #3 when calling relax.op.one_hot
  ... Expected int but got ir.Var
```

There is a second path into the same failure, and it is the more common one in practice. For opset 12 and below the importer first flattens the input to 2-D, so the reduced extent becomes the *product* of the trailing dimensions. Any one of those being symbolic turns the extent into a `prim.Mul` expression rather than an `int`, which breaks the same way — for example `Hardmax(axis=1)` on an `[N, 3, W]` input.

The fix keeps the `one_hot` lowering whenever the extent is static, so the IR emitted for existing models is unchanged. When the extent is symbolic it takes a different route: take the `keepdims` argmax index, compare it against an `arange` over the axis broadcast along it, and cast the resulting boolean mask to the input dtype. The result is the same one-hot mask, built from ops whose attributes do not have to be static.

One detail matters for spec compliance: `argmax` returns the *first* maximum, so ties continue to resolve to the lowest index, as the ONNX spec requires. The tests exercise that directly — they generate inputs from a small integer range so that rows genuinely tie.

Coverage covers opsets 11 and 13 with a symbolic reduced axis, a flattened extent that is a product involving a symbolic dimension, a static axis on an otherwise symbolic input, and tied inputs, all compared against onnxruntime. Six of the seven cases fail without the change; the seventh is a guard that the static-extent path is untouched.

[View my TVM pull requests](https://github.com/apache/tvm/pulls?q=is%3Apr+author%3ALiRunGuo)
