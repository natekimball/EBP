"""How anisotropic are the EBP features?

If the feature distribution occupies a low-dimensional subspace, similar
continuations map to near-identical embeddings and fine-grained ranking dies,
while coarse distinctions survive. That is the failure signature observed in
the ranking test, and it is exactly what SIGReg (isotropy) targets.

Effective rank = participation ratio (sum L)^2 / sum L^2 of the covariance
spectrum. Isotropic in D dims -> D. Collapsed to one direction -> 1.
"""
import sys
sys.path.insert(0, "/home/natekimball/Projects/EBP")
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from ebp.model import _features_from_hidden_states, _get_transformer_layers

CKPT, DEV, CTX, GEN, NB, BS = "runpod_results/cpt/final", "cpu", 512, 16, 128, 8
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).to(DEV).eval()
nl = len(_get_transformer_layers(model))
FEAT = [max(0, min(round(f*nl)-1, nl-1)) for f in (0.25, 0.50, 0.75)]
ds = load_from_disk("data/nemotron_subset")

rows, row, t0 = [], 0, time.time()
while len(rows) < NB:
    batch = []
    while len(batch) < BS:
        t = ds[row]["tokens"]; row += 1
        if len(t) >= CTX+GEN: batch.append(t[:CTX+GEN])
    ids = torch.tensor(batch, device=DEV)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                  output_hidden_states=True, return_dict=True, use_cache=False)
    f = _features_from_hidden_states(o.hidden_states, FEAT, torch.ones_like(ids),
                                     CTX, detach=True, pool_type="last")
    rows.append(f)
    print(f"  {len(rows)*BS}/{NB*BS}  ({time.time()-t0:.0f}s)", flush=True)
F = torch.cat(rows).double()
print(f"\nfeatures: {tuple(F.shape)}")

def spec(X, name):
    Xc = X - X.mean(0, keepdim=True)
    C = (Xc.T @ Xc) / (Xc.shape[0]-1)
    L = torch.linalg.eigvalsh(C).clamp(min=0).flip(0)
    er = (L.sum()**2 / (L**2).sum()).item()
    tot = L.sum()
    print(f"\n{name}: dim={X.shape[1]}  n={X.shape[0]}")
    print(f"  effective rank (participation ratio) = {er:.2f}   (isotropic would be min(n-1,dim)={min(X.shape[0]-1,X.shape[1])})")
    print(f"  top-1 eigenvalue explains {100*L[0]/tot:.1f}% of variance")
    print(f"  top-5  {100*L[:5].sum()/tot:.1f}%   top-10 {100*L[:10].sum()/tot:.1f}%   top-50 {100*L[:50].sum()/tot:.1f}%")
    return er

spec(F, "concatenated (all 3 layers)")
d = F.shape[1]//3
for i in range(3):
    spec(F[:, i*d:(i+1)*d], f"layer block {i} (transformer layer {FEAT[i]})")
# how close are two random real continuations in this space?
Fn = torch.nn.functional.normalize(F, dim=1)
G = Fn @ Fn.T
off = G[~torch.eye(len(G), dtype=bool)]
print(f"\nmean cosine between two DIFFERENT real continuations = {off.mean():.4f} (sd {off.std():.4f})")
print("-> near 1.0 means the space cannot separate distinct real texts, let alone similar rollouts.")
