# bench_ab.py  (fixed dtype mismatch)
import time, torch, torch.nn as nn
from gemlite.helper import A8W8_INT8_dynamic

M = N = K = 2**14
ITERS = 200
DEV = "cuda:0"

class TinyFP16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(K, N, bias=False)
        # Convert weights/bias to fp16 to match input dtype
        self.fc = self.fc.to(dtype=torch.float16)
    def forward(self, x):
        return self.fc(x)

class TinyINT8(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Linear(K, N, bias=False)
        # Patch to INT8 path (A8W8 dynamic activ quant)
        self.fc = A8W8_INT8_dynamic(device=DEV).from_linear(base)
    def forward(self, x):
        return self.fc(x)

def time_model(model, x, iters=ITERS, warmup=20):
    # Warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False  # avoid TF32 noise

    # Common input (fp16), sized to hit TC-friendly tiles
    x = torch.randn(M, K, device=DEV, dtype=torch.float16)

    # Baseline FP16
    fp16 = TinyFP16().to(DEV)
    t_fp16 = time_model(fp16, x)
    print(f"FP16: {t_fp16*1e3/ITERS:.3f} ms/iter")

    # INT8 (A8W8 dynamic)
    int8 = TinyINT8().to(DEV)
    t_int8 = time_model(int8, x)
    print(f"INT8: {t_int8*1e3/ITERS:.3f} ms/iter")

    speedup = (t_fp16 / t_int8) if t_int8 > 0 else float("inf")
    print(f"Speedup INT8 vs FP16: {speedup:.2f}×")
    print(f"Layers: FP16={type(fp16.fc)}  INT8={type(int8.fc)}  CC={torch.cuda.get_device_capability(0)}")

if __name__ == "__main__":
    main()
