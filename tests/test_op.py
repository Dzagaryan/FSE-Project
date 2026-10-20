"""Tests for the installed fse_ops wheel."""

import pytest
import torch

from fse_ops import bifurcation_step


def spec(x, r):
    return r * x * (1.0 - x)


def test_output_shape():
    x = torch.tensor([[0.1, 0.2], [0.5, 1.0]])
    r = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert bifurcation_step(x, r).shape == x.shape


def test_output_dtype():
    x = torch.tensor([0.2, 0.5], dtype=torch.float32)
    r = torch.tensor([2.0, 3.0], dtype=torch.float32)
    assert bifurcation_step(x, r).dtype == torch.float32


def test_exact_values():
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    r = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert torch.equal(bifurcation_step(x, r), spec(x, r))


def test_adversarial_input_not_noop_not_constant():
    """Kind 4: values the formula treats differently.
    
    For x=[0.2, 0.5] and r=[2.0, 4.0]:
    expected = [2*0.2*(1-0.2), 4*0.5*(1-0.5)] = [0.32, 1.0]
    Not the input back, not one value copied to every position.
    """
    x = torch.tensor([0.2, 0.5], dtype=torch.float32)
    r = torch.tensor([2.0, 4.0], dtype=torch.float32)
    y = bifurcation_step(x, r)
    
    expected = torch.tensor([0.32, 1.0], dtype=torch.float32)
    assert torch.equal(y, expected)
    assert not torch.equal(y, x)                # not a no-op
    assert y.unique().numel() > 1               # not one element reused


def test_integer_tensor_raises_in_extension():
    with pytest.raises(RuntimeError, match="x must have dtype float32"):
        bifurcation_step(torch.tensor([1, 2, 3]), torch.ones(3))


def test_out_of_range_raises():
    with pytest.raises(RuntimeError, match=r"x must be in \[0, 1\]"):
        bifurcation_step(torch.tensor([-0.1, 0.5]), torch.ones(2))
