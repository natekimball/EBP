---
name: project-ebp
description: "EBP (Energy-Based Pre-Training) codebase — research idea, structure, and cleanup history"
metadata: 
  node_type: memory
  type: project
  originSessionId: d922b6a3-938a-419a-b6d4-19ac51951aa1
---

EBP applies the EBFT feature-matching objective to LLM pre-training. For each (context, reference) pair it: (1) samples rollouts, (2) extracts hidden-state features at 25%/50%/75% depth, (3) computes alignment+diversity rewards, (4) updates via REINFORCE+RLOO. Mixed with optional CE term (γ).

Two model variants: EMAEBPModel (separate EMA feature network) and OnlineEBPModel (live model, single forward pass). Both inherit from BaseEBPModel which holds shared `generate_rollouts()` and `forward()`.

**Why:** Novel pre-training approach; comparing EBP vs CE-only baseline on Dolma/Qwen3-0.6B.

**How to apply:** When modifying training, be careful about the two backward paths (regular vs memory_constrained). The `_train_kernel` function may be torch.compiled via `global _train_kernel` in `train()`.

## Key design decisions
- `--memory_constrained` (default True): backprops CE early before rollout generation to reduce peak VRAM
- `--whitening` (default True): whitens features per EBFT Eq. 9 before computing rewards
- Grad accumulation via `--grad_accum_steps`; LR scheduler is cosine with `--warmup_steps` linear warmup
- `policy_nll` metric = per-token NLL of rollouts (proxy for policy confidence, not true entropy)

## Cleanup done (2026-06-21)
- Extracted BaseEBPModel; eliminated duplicate generate_rollouts/forward between EMA and Online variants
- Fixed pin_memory-before-assignment bug in val_dataloader creation
- Replaced `globals()["_train_kernel"] = ...` hack with `global _train_kernel`
- Removed dead commented-out dynamo/inductor config code
- Extracted `_reset_cuda_peak`/`_record_cuda_peak` as module-level helpers
- Fixed `--memory_constrained` default (was False, now True to match README)
- Fixed `--whitening` default (was undocumented as True, README said false)
- Added `loss_scale` parameter to all step functions for grad accumulation
- Added cosine LR scheduler with optional linear warmup
- Added `--grad_accum_steps` for gradient accumulation
- Renamed `entropy` metric to `policy_nll` (was confusingly named)
