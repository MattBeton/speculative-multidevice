# verify_int8_matmul.py
# Usage:
#   python verify_int8_matmul.py --in 4096 --out 4096 --batch 32 --steps 10 --log cublaslt.log
#
# What it does:
#   - Sets cuBLASLt logging env vars
#   - Builds a tiny nn.Module with a single Linear
#   - Patches the Linear using your A8W8 INT8 processor
#   - Runs a few forwards to force cuBLASLt to choose a kernel
#   - Parses the log to report whether INT8/IMMA kernels were used

import os, re, argparse, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Enable cuBLASLt logging to a file we can parse afterwards
# (Ref: cuBLASLt logging vars documented by NVIDIA)
os.environ["CUBLASLT_LOG_LEVEL"] = "2"         # 'Trace' -> kernel-launch params
os.environ["CUBLASLT_LOG_MASK"]  = "30"        # Error|Trace|Hints|Info (bitmask 1|2|4|8 = 15; 30 adds API Trace)
os.environ["CUBLASLT_LOG_FILE"]  = "cublaslt.log"  # can be overridden by --log

import torch
import torch.nn as nn

# ---- your processor comes from gemlite ----
try:
    from gemlite.helper import A8W8_INT8_dynamic
except Exception as e:
    print("ERROR: gemlite.helper not importable; ensure gemlite is installed and on PYTHONPATH.")
    raise

# Minimal copy of your patcher (kept simple, Linear-only)
def patch_model(model, device, processor, skip_modules=()):
    model = model.to(device, non_blocking=True)
    # mark names
    for name, module in model.named_modules():
        setattr(module, "name", name)

    def convert(layer):
        if any(s in layer.name for s in skip_modules):
            return layer
        if isinstance(layer, nn.Linear):
            return processor(device=device).from_linear(layer)
        return layer

    def recurse(m):
        for name, child in list(m.named_children()):
            new_child = convert(child) if isinstance(child, nn.Linear) else recurse(child) or child
            setattr(m, name, new_child)
        return m

    recurse(model)
    torch.cuda.empty_cache()
    return model

class Tiny(nn.Module):
    def __init__(self, din, dout, bias=False):
        super().__init__()
        self.fc = nn.Linear(din, dout, bias=bias)
    def forward(self, x):
        return self.fc(x)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="din", type=int, default=4096)
    ap.add_argument("--out", dest="dout", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--log", type=str, default=os.environ.get("CUBLASLT_LOG_FILE", "cublaslt.log"))
    return ap.parse_args()

def main():
    args = parse_args()
    os.environ["CUBLASLT_LOG_FILE"] = args.log
    # Clean previous log so we only parse this run
    try:
        if os.path.exists(args.log):
            os.remove(args.log)
    except OSError:
        pass

    assert torch.cuda.is_available(), "Need a CUDA device"
    dev = "cuda:0"
    cc = torch.cuda.get_device_capability(0)
    print(f"Device: {torch.cuda.get_device_name(0)}  CC={cc}  torch={torch.__version__}")

    torch.backends.cuda.matmul.allow_tf32 = False  # remove TF32 ambiguity
    torch.set_float32_matmul_precision("high")

    model = Tiny(args.din, args.dout, bias=False)
    model = patch_model(model, dev, processor=A8W8_INT8_dynamic, skip_modules=("lm_head",))
    print(f"Patched layer type -> {type(model.fc)}")

    x = torch.randn(args.batch, args.din, device=dev, dtype=torch.float16)

    # warmup
    for _ in range(args.warmup):
        y = model(x)
    torch.cuda.synchronize()

    # run a few times to get a stable cuBLASLt choice and get it into the log
    t0 = time.perf_counter()
    for _ in range(args.steps):
        y = model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"Ran {args.steps} steps in {dt*1000:.1f} ms")

    # Parse cuBLASLt log
    if not os.path.exists(args.log):
        print(f"cuBLASLt log not found at {args.log}. Did cuBLASLt run?")
        sys.exit(1)

    log = open(args.log, "r", errors="ignore").read()

    # Heuristics: look for datatype fields & IMMA mention
    # Common tokens in cuBLASLt logs:
    #  - CUDA_R_8I / CUDA_R_16F / CUDA_R_32F etc. (Atype/Btype/Ctype)
    #  - computeType=CUBLAS_COMPUTE_32I (for int8 accum)
    #  - algo/kernels mentioning IMMA (integer MMA tensor cores)
    atypes = re.findall(r"(?:Atype|AType)\s*=?\s*(CUDA_R_[0-9A-Z_]+)", log)
    btypes = re.findall(r"(?:Btype|BType)\s*=?\s*(CUDA_R_[0-9A-Z_]+)", log)
    ctypes = re.findall(r"(?:Dtype|DType|Ctype|CType)\s*=?\s*(CUDA_R_[0-9A-Z_]+)", log)
    compt  = re.findall(r"(?:computeType|ComputeType)\s*=?\s*([A-Z_0-9]+)", log)
    imma   = re.findall(r"IMMA", log)
    order  = re.findall(r"(CUBLASLT_ORDER_[A-Z0-9_]+)", log)

    # Count evidence
    int8_A = sum(t == "CUDA_R_8I" for t in atypes)
    int8_B = sum(t == "CUDA_R_8I" for t in btypes)
    int8_C = sum(t == "CUDA_R_8I" for t in ctypes)
    compute_32i = sum("CUBLAS_COMPUTE_32I" in t for t in compt)
    imma_hits = len(imma)

    print("\n==== cuBLASLt Evidence ====")
    print(f"Log file           : {args.log}")
    print(f"A types (sample)   : {atypes[:5]}")
    print(f"B types (sample)   : {btypes[:5]}")
    print(f"C/D types (sample) : {ctypes[:5]}")
    print(f"computeType sample : {compt[:5]}")
    if order:
        # Useful to see COL32_2R_4R4 etc. for INT8 fast paths
        print(f"Layouts seen       : {sorted(set(order))[:6]}")

    used_int8_inputs = (int8_A + int8_B) > 0
    used_int8_math   = compute_32i > 0 or imma_hits > 0

    print("\n==== Verdict ====")
    if used_int8_inputs and used_int8_math:
        print("✅ INT8 GEMM paths are in use (int8 inputs + INT32 accumulation / IMMA).")
    elif used_int8_inputs:
        print("⚠️  INT8 inputs detected, but no clear INT32/IMMA compute evidence in log.")
    else:
        print("❌ No INT8 inputs detected in cuBLASLt matmuls for this run.")

    # Quick hints if not hitting INT8 TC kernels
    if not used_int8_inputs or not used_int8_math:
        print("\nHints:")
        print(" • Ensure your patched Linear actually dispatches to cuBLASLt INT8 kernels (not dequant -> FP16 GEMM).")
        print(" • Use sizes that are multiples of 32 (in/out/features), e.g., 4096, 8192, to hit COL32 int8 kernels.")
        print(" • Keep inputs on GPU, dtype fp16/bf16 is fine if your kernel is fpA x int8B;")
        print("   for pure int8 x int8 you’ll need activation quantization too.")
        print(" • Try increasing batch to saturate TC pipelines.")
        print(" • Double-check your processor=A8W8_INT8_dynamic truly emits an int8 GEMM call, not a Q->deQ path.")

if __name__ == "__main__":
    main()

