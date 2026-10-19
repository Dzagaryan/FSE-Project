"""Tests for fse_ops.clipped_square.

Spec line (from README.md):  x.clamp(-3, 3).square()

Five kinds required by the project brief (one test per kind, 5..10 total):
  1. output shape
  2. output dtype
  3. exact numeric input vs the spec line (torch.equal, no tolerance)
  4. adversarial input: values the formula treats differently, so the
     correct output differs from the input and from element to element
  5. error path: TORCH_CHECK rejects a dtype -> RuntimeError from C++
"""

import pytest
import torch

from fse_ops import clipped_square


# Spec line from README.md — the single source of truth.
SPEC = lambda x: x.clamp(-3, 3).square()  # noqa: E731


def test_shape_preserved():
    """Kind 1: output shape equals input shape."""
    x = torch.tensor([[-5.0, 0.5, 2.0], [1.0, -1.0, 0.0]], dtype=torch.float32)
    y = clipped_square(x)
    assert y.shape == x.shape


def test_dtype_is_float32():
    """Kind 2: output dtype is float32."""
    x = torch.tensor([-5.0, 0.5, 2.0], dtype=torch.float32)
    y = clipped_square(x)
    assert y.dtype == torch.float32


def test_exact_numeric_matches_spec():
    """Kind 3: exact values match the spec line, value for value."""
    x = torch.tensor([-5.0, 0.5, 2.0], dtype=torch.float32)
    y = clipped_square(x)
    assert torch.equal(y, SPEC(x))


def test_adversarial_input_not_noop_not_constant():
    """Kind 4: values the formula treats differently.

    For [-5.0, 0.5, 2.0] the correct output is [9.0, 0.25, 4.0]:
      - not the input back (that would be a no-op),
      - not one value copied to every position (that would be one
        element computed and reused).
    """
    x = torch.tensor([-5.0, 0.5, 2.0], dtype=torch.float32)
    y = clipped_square(x)
    assert torch.equal(y, torch.tensor([9.0, 0.25, 4.0], dtype=torch.float32))
    assert not torch.equal(y, x)                # not a no-op
    assert y.unique().numel() > 1               # not one element reused


def test_integer_tensor_raises_runtime_error():
    """Kind 5: TORCH_CHECK rejects integer dtype -> RuntimeError from C++."""
    x = torch.tensor([1, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        clipped_square(x)