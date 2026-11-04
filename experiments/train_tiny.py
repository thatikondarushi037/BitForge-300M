"""Offline synthetic teacher regression; never constructs a large model."""

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn

from bitforge import BitLinear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_training.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be between 1 and 10000")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; explicitly use --device cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = nn.Sequential(BitLinear(32, 64), nn.ReLU(), BitLinear(64, 8)).to(device)
    # Generate on CPU so the dataset is independent of the selected device.
    x = torch.randn(128, 32).to(device)
    target = (x.cpu() @ torch.randn(32, 8) / 32**0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    initial = nn.functional.mse_loss(model(x), target).item()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), target)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        optimizer.step()
    final = nn.functional.mse_loss(model(x), target).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = dict(seed=args.seed, steps=args.steps, device=str(device),
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  parameters=sum(p.numel() for p in model.parameters()),
                  initial_loss=initial, final_loss=final,
                  elapsed_seconds=time.perf_counter() - start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
