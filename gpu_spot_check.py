#!/usr/bin/env python3
"""
GPU spot check — a ~5 minute torch benchmark to decide whether to keep
or cancel a freshly rented GPU instance before investing real work in it.

Usage (on the rented box):
    pip install torch            # if the image doesn't already have it
    python gpu_spot_check.py
    python gpu_spot_check.py --host vast-12345        # add a label
    python gpu_spot_check.py --sustain-seconds 60     # shorten the burn-in

What it measures (budget ~4-5 min on modern NVIDIA cards):
  1. FP16 / BF16 / FP32 matmul TFLOPS        (peak compute)
  2. Device memory bandwidth (d2d copy)      (HBM / GDDR health)
  3. PCIe H2D / D2H bandwidth                (catches x4-in-an-x16-slot)
  4. VRAM integrity                          (fill + checksum, ~60% of free)
  5. Sustained fp16 matmul                   (thermal throttle detection)

Output is a structured text block you paste into GPUHunter's "GPU spot
check" form. The parser in GPUHunter/lib/parser.js is the authoritative
consumer of this format, so if you change the table layout update both.
"""

import argparse
import datetime as dt
import platform
import subprocess
import sys
import time


def _fail(msg):
    print(f"gpu_spot_check: {msg}", file=sys.stderr)
    sys.exit(1)


try:
    import torch
except ImportError:
    _fail("torch is not installed. Try: pip install torch")


# Realistic achievable numbers (roughly 70-80% of marketing peak) for the
# GPUs we actively rent. The script uses these for an on-the-box verdict;
# GPUHunter stores its own copy in gpu_types and is the source of truth
# for the dashboard. Update both when you add a new card.
EXPECTED = {
    "RTX 3090":      {"fp16": 110,  "mem_bw":  820},
    "RTX 4090":      {"fp16": 300,  "mem_bw":  950},
    "RTX 5090":      {"fp16": 450,  "mem_bw": 1700},
    "H100":          {"fp16": 700,  "mem_bw": 3000},
    "H100 SXM":      {"fp16": 800,  "mem_bw": 3300},
    "H200":          {"fp16": 800,  "mem_bw": 4500},
    "A100 80GB":     {"fp16": 250,  "mem_bw": 2000},
    "A100":          {"fp16": 250,  "mem_bw": 1500},
    "A6000":         {"fp16": 140,  "mem_bw":  720},
    "L40":           {"fp16": 180,  "mem_bw":  860},
}


def _match_expected(gpu_name):
    norm = gpu_name.upper().replace(" ", "")
    # Longest key first so "A100 80GB" wins over "A100".
    for key in sorted(EXPECTED, key=len, reverse=True):
        if key.upper().replace(" ", "") in norm:
            return key, EXPECTED[key]
    return None, None


def _nvidia_smi_query(field):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return None


def _device_info():
    props = torch.cuda.get_device_properties(0)
    pcie_gen = _nvidia_smi_query("pcie.link.gen.current")
    pcie_width = _nvidia_smi_query("pcie.link.width.current")
    pcie_link = f"Gen{pcie_gen} x{pcie_width}" if pcie_gen and pcie_width else None
    return {
        "name": props.name,
        "vram_gb": props.total_memory / (1024 ** 3),
        "cuda": torch.version.cuda or "?",
        "driver": _nvidia_smi_query("driver_version") or "?",
        "pcie_link": pcie_link,
        "compute_cap": f"{props.major}.{props.minor}",
    }


def _time_matmul(dtype, n, iters, warmup=5):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tflops = (2 * (n ** 3) * iters) / elapsed / 1e12
    return tflops


def _time_d2d_bandwidth(size_gb=1.0, iters=20):
    n = int(size_gb * (1024 ** 3) / 4)
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty(n, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gbytes = n * 4 * iters / (1024 ** 3)
    return gbytes / elapsed


def _time_pcie(direction, size_gb=1.0, iters=10):
    n = int(size_gb * (1024 ** 3) / 4)
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float32, device="cuda")
    if direction == "h2d":
        src, dst = host, dev
    else:
        dev.random_()
        src, dst = dev, host
    for _ in range(2):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    gbytes = n * 4 * iters / (1024 ** 3)
    return gbytes / elapsed


def _vram_integrity():
    """Fill ~60% of free VRAM with a constant and checksum it. Catches
    gross corruption; not a substitute for a full memtest."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    size_bytes = int(free * 0.6)
    n = size_bytes // 4
    try:
        x = torch.full((n,), 42.0, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        actual = float(x.sum().item())
        expected = 42.0 * n
        rel_err = abs(actual - expected) / expected
        gb = size_bytes / (1024 ** 3)
        del x
        torch.cuda.empty_cache()
        return gb, rel_err
    except RuntimeError:
        torch.cuda.empty_cache()
        return 0.0, float("inf")


def _sustained(n, seconds, dtype=torch.float16):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    chunks = []
    iters_per_chunk = 20
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < seconds:
        t0 = time.perf_counter()
        for _ in range(iters_per_chunk):
            _ = a @ b
        torch.cuda.synchronize()
        dt_ = time.perf_counter() - t0
        chunks.append((2 * (n ** 3) * iters_per_chunk) / dt_ / 1e12)
    return chunks


def _status(value, expected, warn=0.90, fail=0.75):
    if expected is None or value is None:
        return "info"
    if value >= expected * warn:
        return "pass"
    if value >= expected * fail:
        return "warn"
    return "fail"


def main():
    p = argparse.ArgumentParser(description="GPU spot check for GPUHunter")
    p.add_argument("--host", default=None, help="host label (e.g. vast-12345)")
    p.add_argument("--sustain-seconds", type=int, default=90,
                   help="duration of the sustained matmul test (default 90s)")
    p.add_argument("--matmul-n", type=int, default=8192,
                   help="matrix dimension for matmul tests (default 8192)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False — no GPU detected")

    run_start = time.perf_counter()
    info = _device_info()
    exp_key, expected = _match_expected(info["name"])
    exp_fp16 = expected["fp16"] if expected else None
    exp_mem = expected["mem_bw"] if expected else None
    n = args.matmul_n

    def step(i, label):
        print(f"[{i}/7] {label}...", file=sys.stderr, flush=True)

    step(1, "fp16 matmul")
    fp16 = _time_matmul(torch.float16, n, iters=30)

    step(2, "bf16 matmul")
    try:
        bf16 = _time_matmul(torch.bfloat16, n, iters=30)
    except (RuntimeError, TypeError):
        bf16 = None

    step(3, "fp32 matmul")
    fp32 = _time_matmul(torch.float32, n, iters=15)

    step(4, "device memory bandwidth")
    d2d = _time_d2d_bandwidth(size_gb=1.0, iters=20)

    step(5, "PCIe h2d / d2h")
    h2d = _time_pcie("h2d", size_gb=1.0, iters=10)
    d2h = _time_pcie("d2h", size_gb=1.0, iters=10)

    step(6, "VRAM integrity")
    vram_gb, vram_err = _vram_integrity()

    step(7, f"sustained fp16 matmul ({args.sustain_seconds}s)")
    sustain = _sustained(n, args.sustain_seconds)
    s_avg = sum(sustain) / len(sustain) if sustain else 0.0
    s_min = min(sustain) if sustain else 0.0
    s_max = max(sustain) if sustain else 0.0
    s_drop = ((s_max - s_min) / s_max * 100.0) if s_max > 0 else 0.0

    run_duration = time.perf_counter() - run_start

    rows = [
        ("fp16_matmul_tflops", fp16, "TFLOPS",
         _status(fp16, exp_fp16),
         f"{n}x{n}, 30 iters"),
        ("bf16_matmul_tflops",
         bf16, "TFLOPS",
         _status(bf16, exp_fp16) if bf16 is not None else "skip",
         f"{n}x{n}, 30 iters" if bf16 is not None else "unsupported on this GPU"),
        ("fp32_matmul_tflops", fp32, "TFLOPS", "info",
         f"{n}x{n}, 15 iters"),
        ("d2d_bandwidth", d2d, "GB/s",
         _status(d2d, exp_mem),
         "device-to-device copy, 1 GB"),
        ("h2d_bandwidth", h2d, "GB/s", "info",
         "pinned host -> device, 1 GB"),
        ("d2h_bandwidth", d2h, "GB/s", "info",
         "device -> pinned host, 1 GB"),
        ("vram_alloc_gb", vram_gb, "GB",
         "pass" if vram_err < 1e-4 else "fail",
         f"rel_err={vram_err:.2e}"),
        ("sustain_avg_tflops", s_avg, "TFLOPS",
         _status(s_avg, exp_fp16),
         f"{args.sustain_seconds}s fp16 matmul"),
        ("sustain_min_tflops", s_min, "TFLOPS",
         _status(s_min, exp_fp16 * 0.85 if exp_fp16 else None),
         "thermal floor"),
        ("sustain_drop_pct", s_drop, "%",
         "pass" if s_drop < 5 else ("warn" if s_drop < 10 else "fail"),
         "peak -> min drop"),
    ]

    statuses = [r[3] for r in rows if r[3] in ("pass", "warn", "fail")]
    if "fail" in statuses:
        verdict, reason = "BAD", "one or more tests failed"
    elif "warn" in statuses:
        verdict, reason = "MARGINAL", "some tests below expected"
    elif exp_key:
        verdict, reason = "GOOD", f"within expected range for {exp_key}"
    else:
        verdict, reason = "GOOD", "no baseline for this GPU — inspect values manually"

    # ---- render -----------------------------------------------------------
    header = f"GPU SPOT CHECK — {info['name']}  [cuda {info['cuda']}, driver {info['driver']}]"
    bar = "=" * 80
    print(bar)
    print("  " + header)
    print(bar)
    print()

    meta = [
        ("Host",               args.host or platform.node()),
        ("VRAM total (GB)",    f"{info['vram_gb']:.2f}"),
        ("PCIe link",          info["pcie_link"] or "unknown"),
        ("Compute capability", info["compute_cap"]),
        ("Run duration (s)",   f"{run_duration:.1f}"),
        ("Timestamp",          dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    ]
    mw = max(len(k) for k, _ in meta)
    for k, v in meta:
        print(f" {k.ljust(mw)} | {v}")
    print()

    cm, cv, cu, cs = 24, 10, 8, 8
    print(
        f"  {'Metric'.ljust(cm)}|"
        f" {'Value'.rjust(cv)} |"
        f" {'Unit'.ljust(cu)}|"
        f" {'Status'.ljust(cs)}| Notes"
    )
    print(
        "  " + "-" * cm + "+" +
        "-" * (cv + 2) + "+" +
        "-" * (cu + 1) + "+" +
        "-" * (cs + 1) + "+" + "-" * 30
    )
    for metric, value, unit, status, notes in rows:
        val_str = "N/A" if value is None else f"{value:.2f}"
        print(
            f"  {metric.ljust(cm)}|"
            f" {val_str.rjust(cv)} |"
            f" {unit.ljust(cu)}|"
            f" {status.ljust(cs)}| {notes}"
        )
    print()
    print(f" Verdict: {verdict}  ({reason})")
    print()


if __name__ == "__main__":
    main()
