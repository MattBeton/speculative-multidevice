# int8_nvtx.py
import torch, torch.nn as nn, time
from gemlite.helper import A8W8_INT8_dynamic

M = 4096  # batch or rows
K = 4096  # in_features
N = 4096  # out_features

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(K, N, bias=False)
    def forward(self, x):
        return self.fc(x)

def patch_linear_to_a8w8(model, device="cuda:0"):
    for name, mod in model.named_modules():
        setattr(mod, "name", name)
    def walk(m):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                setattr(m, name, A8W8_INT8_dynamic(device=device).from_linear(child))
            else:
                walk(child)
    return walk(model) or model

def main():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda:0"
    model = Tiny().to(dev)
    model = patch_linear_to_a8w8(model, dev)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)  # activations will be quantized dynamically

    # warm-up
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    # mark the region we want ncu to focus on
    torch.cuda.nvtx.range_push("GEMLITE_VERIFY")
    for _ in range(50):
        _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(f"layer: {type(model.fc)}  device_cap={torch.cuda.get_device_capability(0)}")

if __name__ == "__main__":
    main()

