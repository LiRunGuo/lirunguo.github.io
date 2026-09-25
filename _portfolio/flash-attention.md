---
title: "FlashAttention — CuTe Kernels & Block Sparsity"
excerpt: "Contributor to FlashAttention: merged an SM90 tile-selection fix that takes block-sparse attention on Hopper from 8 of 40 head-dim/block-size combinations working to 32, plus a causal-forward fix that stops re-applying an all-true mask on unmasked KV blocks."
collection: portfolio
category: contribution
order: 5
permalink: /portfolio/flash-attention/
---

I contribute to [FlashAttention](https://github.com/Dao-AILab/flash-attention), focusing on the CuTe kernels and on whether the tile heuristics agree with the block-sparsity configurations callers can actually pass in.

### Merged contributions

**[PR #2904 — skip the causal/local mask on unmasked KV blocks in forward (CuTe, SM90)](https://github.com/Dao-AILab/flash-attention/pull/2904)** · Merged September 24, 2026 (+4 / −2 across 1 file).

On H200, the FA4 SM90 causal forward trailed the C++ FA3 forward by 8–20% at every sequence length, while the non-causal forward was at parity. Fitting kernel time against `seqlen_k` showed the per-KV-block cost of the *non-causal* kernels was identical between the two (FA3 1.626 µs vs FA4 1.620 µs per block per SM), so the causal gap had to come from the causal kernel itself.

The cause was in the SM90 mainloop: it splits KV iterations into blocks that need masking (the diagonal, or the window edges for local attention) and blocks that do not, but the second loop — the one that by construction needs no masking — still passed `mask_fn`, bound with `mask_causal=self.is_causal` / `mask_local=self.is_local`. Every unmasked block therefore re-applied the causal mask element-wise, even though the mask is all-true there. The SASS made it visible: the hdim-128 causal kernel's unmasked inner loop had 580 instructions with 66 `FSEL` and 8 `R2P`, against 457 instructions / 2 `FSEL` / 0 `R2P` in the non-causal kernel. `flash_fwd_sm100.py` already handled this loop correctly (it only passes `mask_fn` when `mask_mod` is set); the fix applies the same rule on SM90, dropping the unmasked inner loop to 477 instructions, 2 `FSEL`, 0 `R2P`, with the masked loop unchanged.

Measured throughput on H200 (bf16, 32k total tokens, CUDA-event timing), in TFLOPS:

| forward | FA3 | FA4 main | FA4 this PR |
|:--|--:|--:|--:|
| hdim128 causal 1k | 509 | 414 | 431 |
| hdim128 causal 4k | 691 | 607 | 643 |
| hdim128 causal 8k | 679 | 637 | 667 |
| hdim128 causal 16k | 682 | 613 | **660** |
| hdim64 causal 16k | 514 | 463 | **531** |

Non-causal is untouched, as it runs the same kernel. Correctness is strong for a loop-level change: forward `out` and `lse` are bitwise identical to main across nine configurations (causal with `sq == sk`, `sq < sk`, `sq > sk`, GQA, head dims 64/128/192/256, local windows (256,0) and (300,100), and non-causal).

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

Validation included a new `test_sm90_block_sparse_fwd_tile_fits_block_size` (7 cases, failing 7/7 on main and passing 7/7 here), a full run of `tests/cute/test_mask_mod.py` on H200 going from 1256 passed / 120 failed to **1383 passed / 0 failed**, and `test_block_sparsity.py` at 4883 passed / 0 failed. The two PRs are an interesting pair: #2903 fixes what the SM90 forward *accepts*, while #2904 fixes what it *computes* on the blocks it was already running.

[View my FlashAttention pull requests](https://github.com/Dao-AILab/flash-attention/pulls?q=is%3Apr+author%3ALiRunGuo)
