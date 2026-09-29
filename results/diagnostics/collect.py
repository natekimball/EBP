"""Collect features once; sweep decorrelation transforms offline afterwards.

Saves (a) a corpus feature sample for estimating the whitener -- from REAL
continuations, independent of the rollouts being ranked, so there is no leakage;
(b) per-context rollout features + reference feature + a quality proxy.
"""
import sys
sys.path.insert(0, "/home/natekimball/Projects/EBP")
import torch, time
from transformers import AutoModelForCausalLM
from datasets import load_from_disk
from ebp.model import _features_from_hidden_states, _get_transformer_layers

CKPT, DEV, CTX, GEN, N = "runpod_results/cpt/final", "cpu", 512, 16, 8
M_BATCH, NCTX = 160, 32          # 160*8=1280 corpus samples; 32 ranking contexts
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).to(DEV).eval()
nl = len(_get_transformer_layers(model))
FEAT = [max(0, min(round(f*nl)-1, nl-1)) for f in (0.25, 0.50, 0.75)]
ds = load_from_disk("data/nemotron_subset")
row = 0
def nxt():
    global row
    while True:
        t = ds[row]["tokens"]; row += 1
        if len(t) >= CTX+GEN: return t[:CTX+GEN]
def feats(ids):
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                  output_hidden_states=True, return_dict=True, use_cache=False)
    return _features_from_hidden_states(o.hidden_states, FEAT, torch.ones_like(ids),
                                        CTX, detach=True, pool_type="last")

t0 = time.time()
# (a) corpus sample for the whitener
corpus = []
for b in range(M_BATCH):
    ids = torch.tensor([nxt() for _ in range(8)], device=DEV)
    corpus.append(feats(ids))
    if (b+1) % 20 == 0: print(f"  corpus {(b+1)*8}/{M_BATCH*8}  ({time.time()-t0:.0f}s)", flush=True)
corpus = torch.cat(corpus)

# (b) ranking data
R, P, O = [], [], []
for c in range(NCTX):
    toks = nxt()
    ids = torch.tensor(toks, device=DEV).unsqueeze(0)
    ctx = ids[:, :CTX]
    P.append(feats(ids)[0])                                  # reference feature
    with torch.no_grad():
        roll = model.generate(input_ids=ctx.repeat(N,1),
                              attention_mask=torch.ones(N,CTX,dtype=torch.long),
                              max_new_tokens=GEN, do_sample=True, temperature=1.0,
                              pad_token_id=model.config.eos_token_id)
    R.append(feats(roll))                                    # (N, D) rollout features
    tgt = ids[0, CTX:]
    O.append(torch.tensor([float((roll[j, CTX:] == tgt).float().mean()) for j in range(N)]))
    print(f"  ctx {c+1}/{NCTX}  ({time.time()-t0:.0f}s)", flush=True)

torch.save({"corpus": corpus, "rollouts": torch.stack(R), "ref": torch.stack(P),
            "overlap": torch.stack(O)}, ".runpod/rankdata.pt")
print(f"\nSAVED corpus={tuple(corpus.shape)} rollouts={tuple(torch.stack(R).shape)}")
