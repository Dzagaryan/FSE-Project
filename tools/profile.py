"""Large-input profiling workload for python3 -m cProfile."""

import torch

from fse_ops import bifurcation_step


def main():
    x = torch.linspace(0.0, 1.0, 1_000_000, dtype=torch.float32)
    r = torch.full_like(x, 3.5)
    for _ in range(10):
        bifurcation_step(x, r)


if __name__ == "__main__":
    main()
