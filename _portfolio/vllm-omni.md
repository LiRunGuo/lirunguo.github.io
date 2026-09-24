---
title: "vLLM-Omni — Omni-Modal Inference"
excerpt: "Contributor to vLLM-Omni: merged a bug fix that turns a malformed Qwen2.5-Omni prompt from an engine-killing crash into a rejected request."
collection: portfolio
category: contribution
order: 6
permalink: /portfolio/vllm-omni/
---

I contribute to [vLLM-Omni](https://github.com/vllm-project/vllm-omni), the framework for efficient inference with omni-modality models, focusing on input-processing correctness for multi-modal models.

### Merged contribution

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
