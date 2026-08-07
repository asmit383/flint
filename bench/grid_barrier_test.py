#!/usr/bin/env python3
"""Is a hand-rolled resident-grid barrier competitive with cooperative-groups grid.sync()?
Builds three variants and times one barrier in isolation. If hand-rolled ≈ grid.sync, the CDNA/AMD
path (no cooperative groups) is a small step.

    python bench/grid_barrier_test.py
"""
import os, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "kernels/grid_barrier_bench.cu")

def build(name, flags):
    return load(name=name, sources=[SRC], extra_cuda_cflags=["-O3"] + flags, verbose=False)

if __name__ == "__main__":
    assert torch.cuda.is_available()
    ITERS = 100000
    variants = [
        ("grid.sync()",   build("flint_bar_cg",  [])),
        ("hand-rolled C++", build("flint_bar_hr", ["-DHANDROLL"])),
        ("hand-rolled PTX", build("flint_bar_px", ["-DHANDROLL", "-DPTX"])),
    ]
    base = variants[0][1].barrier_ns(ITERS)
    for label, mod in variants:
        t = mod.barrier_ns(ITERS)
        print(f"  {label:16s} {t:7.1f} ns / barrier   ({t/base:.2f}× grid.sync)")
    print("\nread: if the hand-rolled paths are within ~1.5× of grid.sync, hand-rolling the barrier on")
    print("      CDNA is cheap — the barrier under cooperative groups really is atomics + a fence.")
