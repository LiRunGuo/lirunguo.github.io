---
title: "JAX — Batched Linear Algebra"
excerpt: "Contributor to JAX: merged a fix so jax.scipy.linalg.lu accepts the batched inputs its documentation promises, and stops mixing real and complex operands for permute_l."
collection: portfolio
category: contribution
order: 6
permalink: /portfolio/jax/
---

I contribute to [JAX](https://github.com/jax-ml/jax), focusing on the numerical-library surface where the documented contract and the implementation disagree.

### Merged contribution

**[PR #40931 — Support batched inputs in `jax.scipy.linalg.lu`](https://github.com/jax-ml/jax/pull/40931)** · Merged September 30, 2026 (+19 / −4 across 3 files).

`jax.scipy.linalg.lu` documents inputs of shape `(..., M, N)` — as does `scipy.linalg.lu` — but any input with more than two dimensions failed:

```python
import numpy as np, jax.scipy.linalg as jsl
jsl.lu(np.ones((2, 3, 3)))
# ValueError: too many values to unpack (expected 2)
```

`lax.linalg.lu` already handled batch dimensions; the post-processing in `_lu` assumed a 2-D input (`m, n = np.shape(a)`, then indexed `P`, `L` and `U` along the leading axes). The existing `testLuGrad` works around this by `vmap`-ing its batched shape, which is why the gap survived.

The fix wraps `_lu` with `jnp.vectorize` under the signature `(m,n)->(m,m),(m,k),(k,n)` (or `(m,n)->(m,k),(k,n)` for `permute_l=True`), as `lu_solve` already does, leaving `_lu` itself two-dimensional. The same pass fixes a second latent bug: the permutation matrix is now built in the input dtype and only its real part is returned, so `P @ L` for `permute_l=True` no longer mixes real and complex operands — previously complex inputs with `permute_l=True` raised `TypePromotionError` under `jax_numpy_dtype_promotion='strict'`. The returned `P` is still real, as in SciPy, and 2-D behavior is unchanged.

`testLu` now covers shapes `(0, 3, 3)`, `(2, 4, 5)` and `(3, 2, 6, 6)`, checks output shapes, and exercises `permute_l=True` — those cases fail before the change (10 failures) and pass after. `tests/linalg_test.py`, `tests/lax_scipy_test.py` and `tests/lax_numpy_test.py` pass with and without `JAX_ENABLE_X64`, the `testLu*` family passes with `JAX_NUM_GENERATED_CASES=100` in x32 and x64, and `tests/export_back_compat_test.py` passes.

[View my JAX pull requests](https://github.com/jax-ml/jax/pulls?q=is%3Apr+author%3ALiRunGuo)
