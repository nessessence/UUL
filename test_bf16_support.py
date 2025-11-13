import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device")

print("Device:", torch.cuda.get_device_name())
print("Compute capability:", torch.cuda.get_device_capability())
probe = getattr(torch.cuda, "is_bf16_supported", None)
print("torch.cuda.is_bf16_supported():", probe() if callable(probe) else "N/A")

# 1) Direct BF16 matmul (cuBLAS)
try:
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    print("BF16 matmul: OK (out dtype:", c.dtype, ")")
except Exception as e:
    print("BF16 matmul FAILED:", e)

# 2) AMP autocast with BF16
try:
    from torch.amp import autocast  # PyTorch 2.x
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.randn(16, 64, 64, 64, device="cuda")  # fp32 input
        conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        y = conv(x)
    print("AMP autocast(bfloat16): OK (conv out dtype:", y.dtype, ")")
except Exception as e:
    print("AMP autocast(bfloat16) FAILED:", e)
