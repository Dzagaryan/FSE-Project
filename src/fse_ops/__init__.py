# Load PyTorch shared libraries before loading the compiled extension.
import torch  # noqa: F401

from ._C import bifurcation_step

__all__ = ["bifurcation_step"]
