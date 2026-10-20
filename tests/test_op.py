"""Tests for fse_ops.bifurcation_step.

Spec line (from README.md):  r * x * (1.0 - x)

Five kinds required by the project brief:
  1. output shape
  2. output dtype
  3. exact numeric input vs the spec line (torch.equal, no tolerance)
  4. adversarial input: values the formula treats differently
  5. error path: TORCH_CHECK rejects bad dtype or invalid values -> RuntimeError from C++
"""

import pytest
import torch

from fse_ops import bifurcation_step

# Spec line from README.md — the single source of truth.
SPEC = lambda x, r: r * x * (1.0 - x)


def test_shape_preserved():
    """Kind 1: output shape equals input shape."""
    x = torch.tensor([[0.2, 0.5], [0.8, 0.1]], dtype=torch.float32)
    r = torch.tensor([[2.0, 3.0], [3.5, 4.0]], dtype=torch.float32)
    y = bifurcation_step(x, r)
    assert y.shape == x.shape


def test_dtype_is_float32():
    """Kind 2: output dtype is float32."""
    x = torch.tensor([0.2, 0.5], dtype=torch.float32)
    r = torch.tensor([2.0, 3.0], dtype=torch.float32)
    y = bifurcation_step(x, r)
    assert y.dtype == torch.float32


def test_exact_numeric_matches_spec():
    """Kind 3: exact values match the spec line, value for value."""
    x = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float32)
    r = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    expected = SPEC(x, r)
    y = bifurcation_step(x, r)
    assert torch.equal(y, expected)


def test_adversarial_input_not_noop_not_constant():
    """Kind 4: values the formula treats differently.
    
    For x=[0.2, 0.5] and r=[2.0, 4.0]:
    expected = [2*0.2*(1-0.2), 4*0.5*(1-0.5)] = [0.32, 1.0]
    Not the input back, not one value copied to every position.
    """
    x = torch.tensor([0.2, 0.5], dtype=torch.float32)
    r = torch.tensor([2.0, 4.0], dtype=torch.float32)
    y = bifurcation_step(x, r)
    
    # Fixed Bag
    # expected = torch.tensor([0.32, 1.0], dtype=torch.float32)
    expected = SPEC(x, r)
    assert torch.equal(y, expected)
    assert not torch.equal(y, x)                # not a no-op
    assert y.unique().numel() > 1               # not one element reused


def test_integer_tensor_raises_runtime_error():
    """Kind 5: TORCH_CHECK rejects integer dtype -> RuntimeError from C++."""
    x = torch.tensor([1, 2, 3], dtype=torch.int64)
    r = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        bifurcation_step(x, r)

def test_invalid_x_range_raises_runtime_error():
    """Kind 5 (second test for error path): x out of [0, 1] -> RuntimeError."""
    x = torch.tensor([-0.1, 0.5, 1.1], dtype=torch.float32)
    r = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    with pytest.raises(RuntimeError):
        bifurcation_step(x, r)