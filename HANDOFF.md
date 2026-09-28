# EBP handoff — 2026-09-28

Written because the GB10 box is being given up on 2026-09-29. Everything needed to
continue is in this repo; nothing required is left only on that machine.

## Bottom line: EBP does not beat CE-only continued pretraining

Three completed runs, Qwen3-0.6B-Base on `open-web-math` (streamed), context 512,
gen 16, 8 rollouts, batch 4, seed 42, `gamma=1.0`, whitening OFF, `loss_agg=token`.

| run | steps | val_ce start → end | delta | vs CPT |
|---|---|---|---|---|
| `cpt` (CE only) | 3000 | 2.0378 → 2.0229 | −0.0149 | baseline |
| `ebp_g1` (online) | 3000 | 2.0391 → 2.0260 | −0.0130 | lost 12/12, p=0.0005 |
| `ebp_ema` (EMA target) | 2000 | 2.0386 → 2.0284 | −0.0103 | lost 0/8, gap grows to +0.0051 |

CPT also costs ~10x less per step (0.41 s vs 4.1 s), so compute-matched it is not close.
Adding the EMA feature network made things **worse**, not better.

## The diagnosis, and three repair hypotheses that are now DEAD

The reward is not broken — it is too weak for policy gradient to exploit.

**Reward carries real signal.** Degradation ladder is monotone (mean alignment:
temp0.7 1.678 > 1.0 1.562 > 1.5 1.282 > 3.0 1.231 > random tokens 0.942). Within-batch
Spearman vs token overlap = **+0.279 ± 0.083** over 27 contexts, 77.8% positive, p≈0.002.
(An earlier n=16 estimate of +0.168 with a 50/50 sign split was underpowered and was
wrongly called "chance" — do not repeat that.)

**But REINFORCE never climbs its own reward.** first-third → last-third `mean_reward`:
- online: +0.0089 ± 0.0114 → FLAT
- EMA:    +0.0163 ± 0.0141 → FLAT

With rho≈0.28 the advantage is ~92% noise: large gradient norm, no consistent direction.

Ruled out, each with a measurement:
1. **Gradient imbalance / gamma.** REINFORCE gradient measured 25–33x a gamma=0.1 CE
   gradient. Fixed by gamma=1.0. Did not help. Also: `rl_weight` is redundant with
   `gamma` under AdamW — only the ratio matters (verified: identical updates to 2e-8).
2. **Anisotropy / whitening / the SIGReg rationale.** Features ARE strongly anisotropic
   (effective rank **106 of 1279** in 3072 dims; mean cosine 0.409 between two different
   real continuations vs ~0.018 expected isotropic). But decorrelating does **not** help
   ranking — every transform lands within noise of identity (+0.279): shrinkage whiten
   +0.309, drop-top-25-PCs +0.296, PCA-whiten top-64 +0.174 (worse). So the running-
   covariance whitening plan is retired, and the *anisotropy* argument for SIGReg is
   unsupported. SIGReg trains isotropy rather than imposing it post hoc so it is not
   formally refuted — but do not spend a run on that reasoning alone.
3. **Self-referential (moving) reward target.** Tested with `--model_type ema`. Reward
   still flat, and val_ce worse than the online variant.

**Structural fact that constrains any fix:** reward variance is dominated by the
single-reference alignment term — its variance is exactly (n−1)x the diversity term's
(verified n=2..32). So **adding rollouts cannot reduce the noise that matters**; it only
smooths the already-quiet side. A fix must raise rho well above 0.28 or cut the 10x cost.

## Bugs fixed in this campaign (see also results/memory/ebp_infra_gotchas.md)

- **Validation was measuring training data.** `--val_split test` with `--dataset_split train`
  makes them unequal so `val_skip_docs` stays 0; the saved dataset is a plain `Dataset`
  so `split="test"` is silently ignored and both loaders start at row 0. **Every val
  number before 2026-09-22 is invalid** (incl. CPT 1.1039 / EBP 1.2179 on Nemotron).
  Fixed: use `--val_split train`, which triggers the existing disjoint carve-out.
- **Whitening was mathematically broken.** Sigma was estimated from n=4 rollouts in 3072
  dims and jittered, so null-space directions were amplified by 1/sqrt(eps) and the
  `L > eps` mask was numerical noise. Alignment came out ~1e-3 instead of ~2. Rewritten
  via the (n,n) Gram matrix with a proper pseudo-inverse cutoff: exact, and 6090ms → 5.7ms.
- `TORCHDYNAMO_DISABLE=1` is **required**: transformers >=5 compiles inside `generate()`
  and varying rollout shapes recompile forever, yielding ZERO training steps while the
  GPU looks busy. Verify with `grep -c symbolic_shapes <log>` == 0.
- Run python with **`-u`**. Buffered stdout means `^Step` lines never appear and any
  progress-gated watchdog kills a healthy run.
- `--dataset_config` defaults to `v1_7` (Dolma leftover); pass `default` for open-web-math.
- Also fixed: post-EOS rollout tokens got REINFORCE credit; last-token pooling mis-indexed
  padded batches; GPUPrefetcher had a CUDA stream race (missing `record_stream`); EMA
  weights were not checkpointed; warmup started at lr=0.

## What is where

- `results/metrics/*.pkl` — the three runs' full metric histories (the actual science).
- `results/logs/` — training logs. `results/diagnostics/` — every diagnostic script plus
  its output, and `rankdata.pt` (1280 corpus + 32x8 rollout features; ~80 min CPU to regenerate).
- `results/memory/` — copies of the session memories, which otherwise live only in
  `~/.claude/` on the GB10 and would be lost.
- **NOT preserved (gitignored, ~22GB, still on the GB10):** `runpod_results/*/final`,
  `step_*` checkpoints + optimizer state. Re-derivable for ~$1 and ~3.5h on a rented 3090.
  `data/` (398GB tokenized Nemotron) and `data/nemotron_subset` (3.2GB) are also local-only;
  Nemotron-CC-Math is gated on HF and needs a token.
- Nothing is in wandb: all runs used `--wandb_mode disabled`.

## Renting GPUs (lessons that cost real money)

- **Gate every host before shipping data or launching.** ~3 of 4 community pods reach
  RUNNING with SSH up and still fail `torch._C._cuda_init()`. nvidia-smi passing, torch
  importing, and `device_count()==1` are all NOT evidence. `results/diagnostics/gate.py`
  asserts a real bf16 forward+backward. It is host-level and intermittent, not SKU-specific.
- **Gate network too.** One host passed CUDA and pulled at 9 kB/s, so pip never finished.
  Use `curl -sL` — without `-L` you measure HuggingFace's 302 redirect body (~1006 bytes)
  and reject healthy hosts. This cost 4 good pods.
- **Auto-terminate must be a detached process at ppid 1**, never a harness object (a
  Monitor/cron dies with the agent session and the pod bills on). `results/diagnostics/watchdog.sh`
  gates on step progress + GPU utilisation, mirrors results, then terminates and **verifies
  against a pod list** — never trusting the delete response. Check `ppid` after launching;
  `setsid` alone does not always reparent.
- Home uplink was ~170 kB/s, so streaming a corpus from HF (17–33 MB/s on-pod) beats
  uploading one. `--public-ip` gives community pods real port-22 SSH (rsync works);
  without it the ssh.runpod.io proxy needs `-tt`, ignores trailing remote commands, and
  eats piped stdin before bash is ready.
- Beware `pkill -f <pattern>` matching your own command line — it killed my launch chain
  three separate times. Use a script file instead.
- Cheapest verified working SKUs: RTX 3090 $0.22/hr (24GB) and A40 $0.49/hr (48GB).
  The EMA variant **needs 48GB**: it holds a second model copy and cannot use the chunked
  rollout path (`compute_rollout_features` exists only on `OnlineEBPModel`), so it runs the
  unchunked 32-seq backward — 24.04GB measured, 39GB observed in practice.
- VRAM at `--rollout_chunk_size`: 32 → 24.04GB, 16 → 13.23GB, 8 → 7.82GB, 4 → 5.12GB.
  A 0.6B model does not need a big card; it needs a sane chunk size.

## Open decisions for the user

1. **Discard the 22GB of checkpoints, or upload before the box goes?** They are
   re-derivable (~$1, ~3.5h). At ~3 MB/s shared uplink a full upload is ~2h and would
   hog the link. My recommendation: keep `runpod_results/cpt/final` only (1.2GB, ~7min)
   if anything, discard the rest.
2. **Is EBP worth more runs?** Three repair hypotheses are dead and the structural noise
   argument says more rollouts cannot help. I would not fund another run without a
   mechanism that moves rho well above 0.28 or cuts the 10x compute.
3. The `data/` corpus (398GB) is local-only and gated on HF. Re-acquire or re-tokenize
   if needed; `open-web-math` is an ungated substitute that streams fine.
