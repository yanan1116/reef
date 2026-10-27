# TransformerEngine 2.16.1: allow FlashAttention-2 at head_dim 256 on sm89

`utils.py` is `transformer_engine/pytorch/attention/dot_product_attention/utils.py` from `reef:sao-31f75d74` (sha256 `025db641f10b6a21e480840ccf059c979af2deac80b59fcad4aef7a73edbe038`) with one line changed: the device list of the FlashAttention-2 head-dim rule gains `(8, 9)`.

```
- and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
+ and device_compute_capability not in ((8, 0), (8, 9), (9, 0), (10, 0), (12, 0))
```

Why: TE drops FlashAttention-2 for head_dim > 192 on sm89 (RTX 6000 Ada). Fused attention has no head_dim-256 kernel there either, so Qwen3.5 (head_dim 256) falls to the Unfused backend, whose S x S scores bound the trainable sample length (~12k on one 48 GB GPU). The rule is stale for the installed flash_attn 2.8.3: on .16 (sm89) it runs head_dim 256 forward and backward (probed 2026-10-08, 8192 tokens, no dropout).

Used only when `STACK_TE_FA2_SM89_HDIM256=1` is set for `scripts/start_stack.sh`, which then checks the image's original file against the sha256 above and mounts this copy read-only over it. Unset, the stack is unchanged.
