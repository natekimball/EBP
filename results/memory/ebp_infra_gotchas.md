---
name: ebp-infra-gotchas
description: "EBP training gotchas that silently produce wrong or zero results (val contamination, dynamo hang, stdout buffering)"
metadata:
  type: project
---

Three failures that produce confident-looking but wrong output.

**1. Validation was measuring training data.** `--val_split test` with
`--dataset_split train` makes the two unequal, so `val_skip_docs` stays 0 and
train never skips; the saved dataset is a plain `Dataset` (not a `DatasetDict`)
so `split="test"` is silently ignored and BOTH loaders start at row 0. Every
val number before 2026-09-22 (CPT 1.1039, EBP 1.2179 on Nemotron) was measured on
training data. Fix: `--val_split train`, which triggers the existing disjoint
carve-out path (`skip_documents` for train, `max_documents` for val).

**2. transformers >=5 compiles inside `generate()`.** Even with
`--no-compile_model`, rollout generation triggers dynamo; varying rollout shapes
cause endless recompiles (counters climbing to [12/55]) and the run produces ZERO
training steps while the GPU looks busy. Fix: `TORCHDYNAMO_DISABLE=1`. Verify with
`grep -c symbolic_shapes <log>` — it must be 0.

**3. Run python with `-u`.** Stdout block-buffers when redirected to a file, so
`^Step` lines do not appear until the process exits. Any watchdog gating on log
progress will conclude a healthy run is dead and kill it. This nearly produced a
false "EBP crashed again" report.

Also: `--dataset_config` defaults to `v1_7` (a Dolma leftover) and breaks any
other corpus — pass `default` for open-web-math. `nvidia/Nemotron-CC-Math-v1` is
gated and needs an HF token.

**Why:** Each of these fails silently or looks like a different problem.

**How to apply:** Before trusting any EBP training number, confirm the val split
is disjoint, `symbolic_shapes` count is 0, and step lines appear live.
See [[ebp-results-2026-09]].
