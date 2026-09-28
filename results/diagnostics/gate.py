import os, sys, traceback
out = []
out.append("torch_import_start")
import torch
out.append(f"torch={torch.__version__} cuda={torch.version.cuda}")
out.append(f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')!r} NVD={os.environ.get('NVIDIA_VISIBLE_DEVICES')!r}")
out.append(f"device_count={torch.cuda.device_count()}")
try:
    # Assert truthiness, do not merely catch: one observed failure mode returns
    # is_available()==False without raising, which a try/except-only gate passes
    # silently. Both surfaces share the same root cause.
    assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
    assert torch.cuda.device_count() > 0, "device_count()==0"
    torch.zeros(1).cuda()
    out.append("alloc=OK")
    import torch.nn as nn
    m = nn.Linear(512, 512).cuda().to(torch.bfloat16)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    loss = m(x).float().pow(2).mean()
    loss.backward()
    g = m.weight.grad.float().abs().sum().item()
    free, total = torch.cuda.mem_get_info()
    out.append(f"GATE_PASS loss={loss.item():.4f} grad={g:.2f} vram={total/1e9:.1f}GB name={torch.cuda.get_device_name(0)}")
except Exception:
    out.append("GATE_FAIL")
    out.append(traceback.format_exc()[-600:])
open("/tmp/gate_result.txt", "w").write("\n".join(out))
