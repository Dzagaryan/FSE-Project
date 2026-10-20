# fse_ops: Single iteration of the logistic map

A PyTorch C++ extension for one elementwise update over an arbitrary grid
of normalized population states and growth parameters. Each grid point has
its own state and strictly positive parameter.

**Version:** `0.1.0`.

## Operation and interpretation

For a two-dimensional grid,

$$
X^{(n+1)}_{ij}=R_{ij}\,X^{(n)}_{ij}\left(1-X^{(n)}_{ij}\right),
\qquad 0\leq X^{(0)}_{ij}\leq 1,
\qquad R_{ij}>0.
$$

The same elementwise rule applies to tensors of any common shape. The
indices identify independent grid points; both initial populations and
parameters may be chosen independently at every point.

$X^{(n)}_{ij}$ is the population divided by a fixed reference population scale (normalized population).

$R_{ij}$ is the reproduction parameter.

The factor $1-X^{(n)}_{ij}$
models density feedback from limited resources and competition.

In words, multiply each current state by its corresponding growth parameter
and by one minus that state. One call computes one update for the
entire grid.

## Python API and contract

**Function:** `fse_ops.bifurcation_step`.

```text
bifurcation_step(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor
```

| Property | Contract |
|---|---|
| Inputs | Dense CPU `torch.float32` tensors with layout `torch.strided` and the same arbitrary shape |
| Input population | Every value of `x` is finite and must belong to `[0, 1]`, including both endpoints; this input is the initial state for the current step |
| Growth parameter | Every value of `r` is finite and must belong to `[0, 4]` |
| Output | A new CPU `torch.float32` tensor with the same shape as the inputs |
| Input modification | Input tensors remain unaffected |

### Input validation

The function validates both inputs inside the C++ extension using
`TORCH_CHECK`. It raises `RuntimeError` if:

- Any value of `x` lies outside `[0, 1]` or is non-finite.
- Any value of `r` lies outside `[0, 4]` or is non-finite.
- Either tensor has an unsupported dtype, device, or layout.
- The input shapes differ.
- Any of `x`, `r` requires gradient calculation.

If any element is invalid, the entire call is rejected. The same input
checks apply on every call; output values follow the formula directly.

The one-line pure-PyTorch specification is

```python
expected = r * x * (1.0 - x)
```
