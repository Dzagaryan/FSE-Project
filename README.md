# fse_ops

Custom PyTorch operation, CPU-only, C++ extension.

## Function

**Python name:** `fse_ops.clipped_square`

**Formula in words:** clamp every element of the input tensor to `[-3, 3]`,
then square the result elementwise.

**Pure-PyTorch specification line:**

```python
x.clamp(-3, 3).square()
```

**Accepted dtype:** `torch.float32` on CPU. Other dtypes (e.g. integer
tensors) raise `RuntimeError` from the C++ `TORCH_CHECK`.

## Build / Install / Test

```bash
python3 -m pip wheel . --no-deps -w dist
python3 -m pip install dist/*.whl
python3 -m pytest -q tests
```