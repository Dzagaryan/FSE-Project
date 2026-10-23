# fse_ops: one logistic-map step

`fse_ops.bifurcation_step(x, r)` applies one elementwise logistic-map update to
CPU tensors. At each position it multiplies the growth parameter `r` by the
state `x` and by one minus that state. The exact one-line PyTorch specification is:

```python
expected = r * x * (1.0 - x)
```

The function accepts two dense, strided, CPU `torch.float32` tensors with the
same shape. Every `x` value must be finite and in `[0, 1]`; every `r` value
must be finite and in `[0, 4]`. Neither tensor may require gradients. It
returns a new tensor with the same shape and dtype. The C++ extension checks
the contract with `TORCH_CHECK` and raises `RuntimeError` for invalid inputs,
including integer tensors. Empty and noncontiguous tensors are supported.

## Build and test in the course container

Start the course lab from its directory containing `compose.yaml`:

```bash
docker compose up -d
docker compose exec lab bash
cd /course/team-repo  # use the actual mounted checkout path
```

Build the wheel and run the full installed-wheel proof:

```bash
make wheel
make test
```

Or run the one-command smoke proof:

```bash
./tools/smoke.sh
```

The script builds a Linux wheel, installs CPU PyTorch and the wheel into a
separate fresh virtual environment, imports from `/tmp`, checks one exact
result, and runs pytest on a copy of the tests from `/tmp`. It prints `OK`
only after all checks pass. The first run downloads PyTorch and can take
several minutes.

For a manual wheel install, use `make wheel`, create a new venv, install
PyTorch from `https://download.pytorch.org/whl/cpu`, and then install
`dist/*.whl` with `pip`.
`make install` installs `dist/*.whl` into the current Python environment;
create and activate a fresh venv first if you want isolation.

## Docker image

The Dockerfile builds on the public `ubuntu:24.04` image. Its builder stage
installs Python, pip, and a C++ compiler, builds the wheel, and the final
stage installs CPU PyTorch and then the wheel. From this repository run:

```bash
docker build -t fse-ops:0.1 .
docker run --rm fse-ops:0.1
```

The image builds its own wheel and runs pytest against its installed copy.



## Example

```python
import torch
from fse_ops import bifurcation_step

x = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32)
r = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
print(bifurcation_step(x, r))  # tensor([0.0000, 0.3750, 0.7500, 0.0000])
```

See `NOTE.md` for a bug found during development, a rejected input, and the
large-input profiling command.
