"""Does the feature-matching reward actually rank continuations by quality?

Two tests, both on real contexts from the local tokenized math corpus:

 A) Degradation ladder. Sample at increasing temperature (and finally uniform
    random tokens). If phi(c:y_hat).phi(c:y) encodes continuation quality,
    alignment must fall monotonically as samples get worse.

 B) Within-batch rank correlation. Among n rollouts at a single temperature --
    exactly the comparison RLOO exploits -- does alignment rank them the same
    way an independent quality proxy (token overlap with the reference) does?
    Spearman ~0 here means the reward carries no usable within-batch signal,
    regardless of how much variance it has.
"""
import sys
sys.path.insert(0, "/home/natekimball/Projects/EBP")
import torch, statistics as st, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from ebp.model import _features_from_hidden_states, _get_transformer_layers

CKPT = "runpod_results/cpt/final"
DEV = "cpu"
CTX, GEN, N = 512, int(sys.argv[1]) if len(sys.argv) > 1 else 16, 8
NCTX = int(sys.argv[2]) if len(sys.argv) > 2 else 16

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).to(DEV).eval()
nl = len(_get_transformer_layers(model))
FEAT = [max(0, min(round(f*nl)-1, nl-1)) for f in (0.25, 0.50, 0.75)]

ds = load_from_disk("data/nemotron_subset")

def feats(ids, cs, pool="last"):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True, return_dict=True, use_cache=False)
    return _features_from_hidden_states(out.hidden_states, FEAT, torch.ones_like(ids),
                                        cs, detach=True, pool_type=pool)

def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for p, i in enumerate(s): r[i] = p
        return r
    a, b = rk(x), rk(y); n = len(x)
    if n < 3: return float('nan')
    m = n*(n*n-1)
    return 1 - 6*sum((a[i]-b[i])**2 for i in range(n))/m

lad = {t: [] for t in (0.7, 1.0, 1.5, 3.0, "random")}
rhos, rewards_spread = [], []
t0 = time.time()
row = 0
for ci in range(NCTX):
    while True:
        toks = ds[row]["tokens"]; row += 1
        if len(toks) >= CTX+GEN: break
    ids = torch.tensor(toks[:CTX+GEN], device=DEV).unsqueeze(0)
    ctx, ref_full = ids[:, :CTX], ids
    phi_y = feats(ref_full, CTX)[0]

    # --- A: degradation ladder ---
    for temp in (0.7, 1.0, 1.5, 3.0, "random"):
        if temp == "random":
            gen = torch.randint(1000, model.config.vocab_size-1000, (1, GEN), device=DEV)
            cand = torch.cat([ctx, gen], 1)
        else:
            with torch.no_grad():
                cand = model.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx),
                                      max_new_tokens=GEN, do_sample=True, temperature=temp,
                                      pad_token_id=model.config.eos_token_id)
        lad[temp].append(float(feats(cand, CTX)[0] @ phi_y))

    # --- B: within-batch rank correlation at temperature 1.0 ---
    with torch.no_grad():
        roll = model.generate(input_ids=ctx.repeat(N,1), attention_mask=torch.ones(N,CTX,dtype=torch.long),
                              max_new_tokens=GEN, do_sample=True, temperature=1.0,
                              pad_token_id=model.config.eos_token_id)
    F = feats(roll, CTX)
    align = (F @ phi_y).tolist()
    tgt = ids[0, CTX:]
    overlap = [float((roll[j, CTX:] == tgt).float().mean()) for j in range(N)]
    rhos.append(spearman(align, overlap))
    rewards_spread.append(st.pstdev(align))
    print(f"  ctx {ci+1}/{NCTX}  rho={rhos[-1]:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== A: degradation ladder (GEN={GEN}, {NCTX} contexts) ===")
print("  sampling        mean alignment  (higher should = better)")
for t in (0.7, 1.0, 1.5, 3.0, "random"):
    v = lad[t]; print(f"  temp={str(t):8s}    {st.mean(v):+.4f}  +/- {st.pstdev(v):.4f}")
print(f"\n=== B: within-batch rank corr vs token overlap (n={N}) ===")
good = [r for r in rhos if r == r]
print(f"  mean Spearman rho = {st.mean(good):+.4f}   sd {st.pstdev(good):.4f}   n={len(good)}")
print(f"  fraction of contexts with rho > 0: {sum(1 for r in good if r>0)/len(good):.1%}  (chance = 50%)")
print(f"  mean within-batch alignment spread = {st.mean(rewards_spread):.4f}")
