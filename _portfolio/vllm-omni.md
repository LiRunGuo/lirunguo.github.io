---
title: "vLLM-Omni — Omni-Modal Inference"
excerpt: "Contributor to vLLM-Omni: block-sparse attention in the Qwen2.5-Omni Token2Wav DiT (a 4,000-frame sample went from 10.8 s to 2.87 s), plus a fix that turns a malformed prompt from an engine-killing crash into a rejected request."
collection: portfolio
category: contribution
order: 15
permalink: /portfolio/vllm-omni/
---

I contribute to [vLLM-Omni](https://github.com/vllm-project/vllm-omni), the framework for efficient inference with omni-modality models, focusing on the latency and input-processing correctness of multi-modal models.

### Merged contributions

**[PR #7975 — [Perf][Model] Use block-sparse attention in Qwen2.5-Omni Token2Wav DiT](https://github.com/vllm-project/vllm-omni/pull/7975)** · Merged October 8, 2026 (+133 / −26 across 2 files).

After the thinker and talker, Qwen2.5-Omni's Token2Wav stage is the slowest part of a request: about 10–20 s on an H200 for 30–40 s of audio, almost all of it in a 22-layer fp32 DiT. The DiT only attends within 24-frame mel blocks (19 layers to their own block, 3 to one neighbour as well), but every layer built a dense `(batch, heads, n, n)` mask and ran dense SDPA over all `n` keys. On a 4,000-frame sample, about 35% of the CUDA time went to mask operations and about 30% to the dense attention kernel, where each query kept only 24–48 of the 4,000 keys.

The new `block_sparse_attention()` reshapes queries, keys and values into blocks and attends to the own block plus the shifted neighbours, masking only the padded tail of a partial last block. The math is the same as the masked dense path, but the work is linear in sequence length instead of quadratic.

| Measurement (H200, Qwen2.5-Omni-3B) | Before | After |
| --- | --- | --- |
| Standalone DiT, 4,000 mel frames | 10.8 s | **2.87 s** |
| Peak extra memory, same sample | 2.74 GiB | **0.78 GiB** |
| Token2Wav, 26.8 s of audio (cleanest request) | 8.61 s | **2.48 s** |
| Token2Wav, 40.3 s of audio (cleanest request) | 20.71 s | **3.23 s** |

Outputs are unchanged: identical thinker text and talker token counts, WAV correlation of at least 0.99982 against main, and identical Whisper-large-v3 transcriptions. Nine new CPU unit tests compare the block-sparse path with dense masked SDPA for sequences shorter than a block, an exact multiple of it, and a partial last block.


**[PR #7959 — reject a stray `<|AUDIO|>` placeholder with `use_audio_in_video` in Qwen2.5-Omni](https://github.com/vllm-project/vllm-omni/pull/7959)** · Merged September 22, 2026 (+59 / −2 across 3 files).

Running the Qwen2.5-Omni offline example with `--query-type use_audio_in_video` killed the thinker engine outright:

```text
File ".../qwen2_5_omni/qwen2_5_omni.py", line 489, in get_mrope_input_positions
    audio_seqlen = audio_seqlens[audio_idx]
IndexError: list index out of range
...
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

The root cause is an interaction between two prompt shapes. The example prompt carried `<|vision_bos|><|VIDEO|><|vision_eos|>` *plus* a separate `<|audio_bos|><|AUDIO|><|audio_eos|>`. With `use_audio_in_video=True`, `_maybe_apply_prompt_updates` drops the standalone audio prompt updates and instead derives the audio placeholder from the video placeholder — because the audio is meant to be interleaved into the video span. That left the standalone `<|AUDIO|>` token unexpanded with no audio item behind it. It passed input processing, reached the engine, and then `get_mrope_input_positions` indexed past the end of `audio_feature_lengths`.

What makes this worth fixing beyond the single example is the failure mode: one malformed request took down the whole stage rather than failing on its own. The fix adds a check in `Qwen2_5OmniThinkerMultiModalProcessor` that rejects any `<|AUDIO|>` token lying outside a video placeholder with a clear `ValueError` at input-processing time, and removes the stray placeholder from the offline example so the documented prompt shape matches what the model actually expects.

Validated with unit tests on CPU (including a test that interleaved audio inside a video is still accepted, and one that standalone audio outside a video is rejected), an end-to-end run on 2× H200 with `Qwen2.5-Omni-3B`, and a malformed-prompt check through the offline `Omni.generate` path.

[View my vLLM-Omni pull requests](https://github.com/vllm-project/vllm-omni/pulls?q=is%3Apr+author%3ALiRunGuo)
