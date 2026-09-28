---
name: ebp-results-2026-09
description: "Measured EBP vs CPT results and reward-signal diagnostics (Sept 2026) — EBP loses; anisotropy ruled out as the cause"
metadata:
  type: project
---

Head-to-head run 2026-09-22/24 on a rented RTX 3090. Qwen3-0.6B-Base, open-web-math
(streamed; Nemotron-CC-Math is gated and the home upstream ~170 kB/s made transfer
impractical), context 512, gen 16, 8 rollouts, batch 4, 3000 steps/arm, seed 42,
gamma=1.0, whitening OFF, loss_agg=token, chunk 16. Artifacts in `runpod_results/`.

**EBP lost, consistently.** CPT val_ce 2.0378 -> 2.0229 (-0.0149 nats); EBP 2.0391 ->
2.0260 (-0.0130). CPT better at 12/12 checkpoints, sign-test p=0.0005, mean gap
+0.0026 +/- 0.0006. EBP captured 87% of CPT's gain for ~10x the compute
(4.1 s/step vs 0.41 s/step), so compute-matched it is not close.

**The reward works, weakly — it is not broken.** Offline diagnostics on the CPT
checkpoint:
- Degradation ladder is monotone: mean alignment temp0.7 1.678 > 1.0 1.562 >
  1.5 1.282 > 3.0 1.231 > random tokens 0.942. Coarse quality IS encoded.
- Within-batch ranking vs token overlap: Spearman rho=+0.279 +/- 0.083 over 27
  contexts (8 rollouts), 77.8% positive, p~0.002. Real but weak signal.
  NOTE: an earlier n=16 estimate gave +0.168 with a 50/50 sign split and was
  wrongly called "chance" — it was underpowered. Do not repeat that error.

**Anisotropy is NOT the bottleneck — this is the load-bearing negative result.**
Features are strongly anisotropic: effective rank 106 of a possible 1279 in 3072
dims (per-layer ~90-99), mean cosine 0.409 between two DIFFERENT real continuations
(isotropic would be ~0 +/- 0.018). But decorrelating does not help ranking: every
transform lands within noise of identity (+0.279) — shrinkage whitening +0.309,
drop-top-25-PCs +0.296, PCA-whiten top-64 +0.174 (worse). Whitener estimated on
1280 held-out real continuations, so it is rank-valid, unlike the per-context
4-sample Sigma in the code.

**Why:** This retires two planned directions. (1) Running/global-covariance
whitening: tested offline, changes nothing. (2) The anisotropy rationale for
SIGReg: unsupported. SIGReg *trains* isotropy rather than imposing it post hoc so
it is not formally refuted, but the proposed mechanism is dead — do not spend a
training run on that reasoning alone.

**How to apply:** Frame EBP's problem as cost-effectiveness (weak real signal at
10x compute), not correctness. Before any further EBP training run, demand a
mechanism that raises rho well above 0.28 or cuts the 10x. Reward-variance is
dominated by the single-reference alignment term (its variance is (n-1)x the
diversity term's, verified for n=2..32), so ADDING ROLLOUTS CANNOT FIX THE NOISE
THAT MATTERS. See [[ebp-infra-gotchas]] and [[project-ebp]].
