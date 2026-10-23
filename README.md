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

> The bounds `x ∈ [0, 1]` and `r ∈ [0, 4]` are artificial requirements API restrictions. The PyTorch expression can be evaluated outside them, but the extension rejects out-of-range inputs with `RuntimeError`.

## Build and test

Build and test the Linux wheel inside the course container. Keep the checkout
in the course directory's `FSE-Project` subdirectory; the course directory
contains `compose.yaml` and is mounted at `/course` inside the container.

**On your laptop**, starting from the `FSE-Project` repository root, enter the
parent course directory and open a shell in the lab container:

```bash
cd .. &&
docker compose up -d &&
docker compose exec lab bash
```

**Inside the lab container**, run the complete build/install/test proof:

```bash
cd /course/FSE-Project
./tools/smoke.sh
```

The script creates fresh build and test environments, builds and installs a Linux wheel, and runs tests from `/tmp`. Success: `7 passed` and `OK`.

### Manual alternative

To run the Makefile targets manually instead of the smoke script, execute the
following **inside the lab container**. This creates the virtual environment
inside Linux and removes earlier build outputs, including any macOS wheels:

```bash
cd /course/FSE-Project &&
python3 -m venv --clear /tmp/fse-check &&
source /tmp/fse-check/bin/activate &&
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu &&
python3 -m pip install pytest &&
make clean
```

With the environment activated, build, install, and test the wheel:

```bash
make wheel && make install && make test
```

`make install` installs `dist/*.whl` into the current Python environment;
`make test` runs the repository's tests in that environment. Use the smoke
script above for the full proof with separate build and test environments
and tests copied outside the source tree.

## Docker image

The two-stage `ubuntu:24.04` image builds the wheel and runs pytest against its installed copy.

Build and run the image **on your laptop**, from the `FSE-Project` repository
directory:

```bash
docker build -t fse-ops:0.1 . &&
docker run --rm fse-ops:0.1
```

The container runs pytest against the installed wheel. The `&&` ensures that
tests run only after a successful image build, so a failed build cannot lead
to testing an older image with the same tag. The image builds its own wheel and
runs pytest against its installed copy.

## Example

After a successful smoke run, open Python in its test environment **from your
laptop terminal**, in the `FSE-Project` directory:

```bash
docker compose exec lab /tmp/fse-smoke-try/bin/python
```

If you used the manual alternative instead, replace
`/tmp/fse-smoke-try/bin/python` with `/tmp/fse-check/bin/python` in this
command. Both variants open Python inside the lab container. Enter the
following in that Python session:

```python
import torch
from fse_ops import bifurcation_step

x = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32)
r = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
print(bifurcation_step(x, r))  # tensor([0.0000, 0.3750, 0.7500, 0.0000])
```

See `NOTE.md` for the bug report, unsupported case, and profiling results.
