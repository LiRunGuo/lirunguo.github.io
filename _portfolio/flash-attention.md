---
title: "FlashAttention — CuTe Kernels & Block Sparsity"
excerpt: "Contributor to FlashAttention: merged an SM90 tile-selection fix that takes block-sparse attention on Hopper from 8 of 40 head-dim/block-size combinations working to 32."
collection: portfolio
category: contribution
order: 7
permalink: /portfolio/flash-attention/
---

I contribute to [FlashAttention](https://github.com/Dao-AILab/flash-attention), focusing on the CuTe kernels and on whether the tile heuristics agree with the block-sparsity configurations callers can actually pass in.

### Merged contribution

**[PR #2903 — fit the forward tile to the block-sparse block size (CuTe, SM90)](https://github.com/Dao-AILab/flash-attention/pull/2903)** · Merged September 24, 2026 (+102 / −1 across 3 files).

`flash_attn_func` / `flash_attn_varlen_func` — and flex attention's FLASH backend, which calls them — take no tile sizes. On SM90 the forward tile comes from `_tile_size_fwd_sm90`, which adjusts `tile_m` for a sparse Q block only when `head_dim <= 96`, and never looks at the sparse KV block at all. SM90 has no KV subtiling (`kv_subtile_factor` must be 1), and a sparse Q block smaller than `tile_m` cannot be subtiled either, so any `BlockSparseTensorsTorch.block_size` that differed from the heuristic tile raised a `ValueError` during the forward pass — even though the kernel itself runs that tile perfectly well through `_flash_attn_fwd(..., tile_mn=...)`.

The gap was wide and had been silent for a while:

- head_dim 192: flex attention's default `BLOCK_SIZE=128` failed outright, because the heuristic wanted `tile_n=112`.
- head_dim 256: every KV block size except 80 failed; and the one size that did work, (128, 80), was then rejected by the backward pass's `tile_n=64` — so SM90 block sparsity at head_dim 256 through autograd was not usable at all.
- head_dim ≤ 128: (64, 128), (128, 64), (64, 64), (128, 112) and others all failed.

It also accounted for all 120 SM90 failures of `test_mask_mod.py::test_parameterized_masks` in the autograd variants, which had been failing on Hopper since that variant was added in #2485. The FA4 CI runs only on B200, which skips non-128×128 tiles, so nothing had caught them.

The fix adds `_fit_sm90_fwd_tile_to_block_sparsity`, a post-pass that runs after the SM90 heuristic when the caller supplies a `block_size`. If the sparse Q block is not a multiple of `tile_m` but is a multiple of 64, it uses `tile_m=64`; larger sparse Q blocks keep using `q_subtile_factor`. If the sparse KV block is smaller than `tile_n` and a multiple of 16, `tile_n` follows the block. The tile only ever shrinks, so it never adds shared-memory or register pressure, and the rule fires only in configurations that `normalize_block_sparse_config` would already have rejected — meaning anything that works today keeps the same tiles, kernels, and performance. The PR also corrects the SM90 KV-subtile error message, which printed `tile_n` where it meant the sparse KV block size.

Measured on H200 (bf16, B=2 H=4 S=1024, block-aligned causal BlockMask against the fp32 flex reference), across 40 combinations of head_dim (64/96/128/192/256) × block size:

| | main | this PR |
|:--|--:|--:|
| forward | 8 / 40 | 32 / 40 |
| forward + backward | 6 / 40 | 11 / 40 |

Forward max absolute error against the fp32 reference is ≤ 0.0039 for every newly accepted case. The remaining backward rejections come from the SM90 backward's own `tile_n` (128 for head_dim ≤ 128, 96 for 192, 64 for 256) — a separate heuristic, left to a follow-up.

Validation included a new `test_sm90_block_sparse_fwd_tile_fits_block_size` (7 cases, failing 7/7 on main and passing 7/7 here), a full run of `tests/cute/test_mask_mod.py` on H200 going from 1256 passed / 120 failed to **1383 passed / 0 failed**, and `test_block_sparsity.py` at 4883 passed / 0 failed.

[View my FlashAttention pull requests](https://github.com/Dao-AILab/flash-attention/pulls?q=is%3Apr+author%3ALiRunGuo)
