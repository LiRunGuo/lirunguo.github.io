---
title: "LMCache — KV Cache Management"
excerpt: "Contributor to LMCache: merged two fixes — an allocator that now rejects invalid sizes instead of silently emptying the cache, and a lint gate that enforces lazy %-format logging."
collection: portfolio
category: contribution
order: 9
permalink: /portfolio/lmcache/
---

I contribute to [LMCache](https://github.com/LMCache/LMCache), a KV cache layer for LLM serving, focusing on the memory allocator and on keeping the logging path cheap.

### Merged contributions

**[PR #5117 — reject non-positive size in `AddressManager`](https://github.com/LMCache/LMCache/pull/5117)** · Merged September 23, 2026 (+186 / −0 across 6 files). Closes [#5115](https://github.com/LMCache/LMCache/issues/5115).

`AddressManager.allocate()` and `batched_allocate()` document that `size` must be greater than zero, but neither checked it, and a zero-byte request did not fail cleanly. `allocate(0)` carved out a zero-length block; freeing it inserted a zero-sized entry whose neighbours could never be coalesced again, so `get_free_size()` kept reporting the whole heap as free while the next full-heap allocation failed with "no memory is available". `batched_allocate(0, n)` raised `ZeroDivisionError` inside `block.size // aligned_size`, and because `TensorMemoryAllocator` only translates `RuntimeError`, that exception escaped all the way up through `LMCacheEngine.store()` / `store_layer()`.

The review process changed the fix for the better. The first iteration raised `RuntimeError` — the exception the allocation stack uses for "out of memory" — and a reviewer showed that this was the wrong signal: the stack does not merely fail, it *reacts*. With a busy loop enabled the allocator never returned, repeating "Local cpu memory is under pressure" every 0.1 s; without one, the observed behaviour was worse still, with ten cached chunks all evicted (`hot_cache: 10 -> 0`) by a single invalid request. Since a zero-byte request is a caller bug, the final version raises `ValueError` instead, which matches the convention already documented in this allocator API for unsupported arguments.

The second half of the change removes the reason the invalid size could arrive at all: `store()` and `store_layer()` now skip degenerate ranges (`end <= start`), which is what a leading, trailing, or doubled separator produced in the first place. Tests cover both the rejection and the preserved behaviour that `batched_allocate(size, 0)` with a *valid* size still returns an empty list.

**[PR #5125 — enforce lazy `%`-format logging (ruff G004)](https://github.com/LMCache/LMCache/pull/5125)** · Merged September 17, 2026 (+82 / −2 across 1 file). Refs [#5118](https://github.com/LMCache/LMCache/issues/5118).

`logging` renders a record lazily as `msg % args`, so an f-string pays the formatting cost even when the level is disabled. Nothing prevented a new `logger.info(f"...")` from landing in a file that had already been migrated, which is why the repository kept generating one-file migration pull requests — 15 or more had been merged since March 2026, with a further five still open at the time of this change.

This pull request enables ruff's `G004` and adds temporary `per-file-ignores` for the files that have not been migrated yet, so the rule starts holding the line immediately without waiting for the backlog to clear. The ignore list is written per file rather than per directory, because ruff matches `per-file-ignores` patterns with `*` crossing directory separators, which would have silently exempted far more than intended.

[View my LMCache pull requests](https://github.com/LMCache/LMCache/pulls?q=is%3Apr+author%3ALiRunGuo)
