import torch

print("--- PyTorch GPU Check ---")
available = torch.cuda.is_available()
print(f"GPU Available: {available}")

if available:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("PyTorch is currently using the CPU.")
    print("If you have an NVIDIA GPU, you may need to install the CUDA-enabled version of PyTorch.")