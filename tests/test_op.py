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


def test_batch_and_noncontiguous_values():
    x = torch.tensor([[0.1, 0.25, 0.5, 0.9]] * 4).T
    r = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4).T
    actual = bifurcation_step(x, r)
    assert torch.equal(actual, spec(x, r))
    assert not torch.equal(actual, x)
    assert actual.unique().numel() > 1


def test_integer_tensor_raises_in_extension():
    with pytest.raises(RuntimeError, match="x must have dtype float32"):
        bifurcation_step(torch.tensor([1, 2, 3]), torch.ones(3))


def test_out_of_range_raises():
    with pytest.raises(RuntimeError, match=r"x must be in \[0, 1\]"):
        bifurcation_step(torch.tensor([-0.1, 0.5]), torch.ones(2))
