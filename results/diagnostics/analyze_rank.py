"""Does removing feature anisotropy recover within-batch ranking signal?

Baseline reward ranks rollouts at chance (Spearman ~ +0.17, 50% sign split)
while feature effective rank is 106/1023. If anisotropy is the cause, a
decorrelating transform should lift the correlation. The whitener is estimated
on REAL continuations, never on the rollouts being ranked.
"""
import sys
sys.path.insert(0, "/home/natekimball/Projects/EBP")
import torch, statistics as st

d = torch.load(".runpod/rankdata.pt")
C, R, P, O = d["corpus"].double(), d["rollouts"].double(), d["ref"].double(), d["overlap"].double()
NC, N, D = R.shape
print(f"corpus {tuple(C.shape)}  rollouts {tuple(R.shape)}  contexts={NC} n={N} D={D}")

mu = C.mean(0, keepdim=True)
Cc = C - mu
cov = (Cc.T @ Cc) / (Cc.shape[0] - 1)
lam, V = torch.linalg.eigh(cov)
lam, V = lam.flip(0).clamp(min=0), V.flip(1)          # descending
er = (lam.sum()**2 / (lam**2).sum()).item()
print(f"corpus effective rank = {er:.1f} / {min(C.shape[0]-1, D)}\n")

def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for p, i in enumerate(s): r[i] = p
        return r
    a, b = rk(x), rk(y); n = len(x)
    return 1 - 6*sum((a[i]-b[i])**2 for i in range(n))/(n*(n*n-1))

def score(fn, label):
    rhos = []
    for c in range(NC):
        rr, pp = fn(R[c]), fn(P[c].unsqueeze(0))[0]
        rr = rr / rr.norm(dim=1, keepdim=True).clamp(min=1e-9)
        pp = pp / pp.norm().clamp(min=1e-9)
        al = (rr @ pp).tolist()
        ov = O[c].tolist()
        if len(set(ov)) < 2 or len(set(al)) < 2: continue
        rhos.append(spearman(al, ov))
    m, sd, n = st.mean(rhos), st.pstdev(rhos), len(rhos)
    se = sd/(n**0.5)
    pos = sum(1 for r in rhos if r > 0)/n
    star = "  <<<" if m - 2*se > 0 else ""
    print(f"{label:34s} rho={m:+.4f} +/-{se:.4f}  pos={pos:5.1%}  n={n}{star}")
    return m

print("transform                           mean Spearman rho vs token overlap")
print("-"*76)
score(lambda X: X, "identity (current reward)")
score(lambda X: X - mu, "centered")
for k in (16, 32, 64, 128, 256):
    Vk, lk = V[:, :k], lam[:k].clamp(min=1e-12)
    score(lambda X, Vk=Vk, lk=lk: (X - mu) @ Vk / lk.sqrt(), f"PCA-whiten top-{k}")
for k in (16, 32, 64, 128, 256):
    Vk = V[:, :k]
    score(lambda X, Vk=Vk: (X - mu) @ Vk, f"PCA project top-{k} (no whiten)")
for j in (1, 5, 10, 25):
    Vj = V[:, j:]
    score(lambda X, Vj=Vj: (X - mu) @ Vj, f"drop top-{j} PCs")
tr = lam.sum()/D
for f in (1e-3, 1e-2, 1e-1):
    W = V @ torch.diag(1.0/(lam + f*tr).sqrt()) @ V.T
    score(lambda X, W=W: (X - mu) @ W, f"shrinkage whiten lam={f:g}*tr/D")
print("\n'<<<' marks transforms whose rho is >2 SE above zero.")
