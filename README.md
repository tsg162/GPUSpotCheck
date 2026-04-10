# GPUSpotCheck

A ~5 minute PyTorch benchmark to decide whether to keep or cancel a freshly
rented GPU instance before investing real work in it. Measures fp16/bf16/fp32
matmul TFLOPS, HBM bandwidth, PCIe h2d/d2h, VRAM integrity, and sustained
throughput (thermal throttle detection), then prints a verdict you can paste
straight into GPUHunter's "GPU spot check" form.

## Quick start

On the rented box:

```bash
git clone https://github.com/tsg162/GPUSpotCheck.git
cd GPUSpotCheck
python gpu_spot_check.py
```

One-liner (clone + run):

```bash
git clone https://github.com/tsg162/GPUSpotCheck.git && cd GPUSpotCheck && python gpu_spot_check.py
```

No-clone one-shot (fetch the script and run it directly):

```bash
curl -fsSL https://raw.githubusercontent.com/tsg162/GPUSpotCheck/main/gpu_spot_check.py | python -
```

## Common options

```bash
# label the run with the host / instance id
python gpu_spot_check.py --host vast-12345

# shorten the sustained burn-in (default 90s)
python gpu_spot_check.py --sustain-seconds 60

# smaller matmul for low-VRAM cards (default 8192)
python gpu_spot_check.py --matmul-n 4096
```

## Requirements

- NVIDIA GPU with a working CUDA driver (`nvidia-smi` should list it)
- Python 3.8+
- `torch` with CUDA support (most rented GPU images ship with this already)
