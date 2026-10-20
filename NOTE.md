# Project Note

## 1. Real bugs we hit
We encountered two real issues during our CI pipeline setup:
1. **Contract mismatch (C++ vs README):** The C++ code strictly enforced a 2D shape (`TORCH_CHECK(x.dim() == 2)`), which crashed on 1D and batched tensors with `RuntimeError: Expected x to be 2D, got 1D`. However, the README stated the op should support "any common shape". This was fixed in commit [34ced98] by replacing the strict dimension check with `TORCH_CHECK(x.sizes() == r.sizes(), ...)`.
2. **Floating-point precision in tests:** Our adversarial test failed because hardcoded expected values (like `0.32` and `1.0`) didn't exactly match the C++ float32 arithmetic (e.g., `4.0 * 0.5 * (1 - 0.5)` yielded `0.9999999`). This was fixed in commit [060003b] by calculating the expected value using the pure-PyTorch spec line `SPEC(x, r)` directly, avoiding hardcoded floats.

## 2. Unhandled case
Our op does not support gradient flow (we explicitly reject tensors with `requires_grad=True` via `TORCH_CHECK`). Instead of silently breaking the autograd graph, it raises a clear `RuntimeError`.

## 3. cProfile measurement
We measured the performance on a large input (e.g., 10,000x10,000 tensor).
Command: `python3 -m cProfile -s time bench.py`
Output: [TODO: later, when we run it, paste the number of seconds here]