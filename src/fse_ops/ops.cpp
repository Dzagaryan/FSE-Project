#include <torch/extension.h>

torch::Tensor bifurcation_step(torch::Tensor x, torch::Tensor r)
{
    // device check
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(r.device().is_cpu(), "r must be on CPU");
    // dtype check
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Expected dtype of x is float32, got ", x.scalar_type());
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "Expected dtype of r is float32, got ", r.scalar_type());
    // layout check
    TORCH_CHECK(x.layout() == torch::kStrided, "Expected x to have strided layout, got ", x.layout());
    TORCH_CHECK(r.layout() == torch::kStrided, "Expected r to have strided layout, got ", r.layout());
    // gradient check
    TORCH_CHECK(x.requires_grad() == false, "x does not support gradient flow");
    TORCH_CHECK(r.requires_grad() == false, "r does not support gradient flow");
    // dimensions check
    // TORCH_CHECK(x.dim() == 2, "Expected x to be 2D, got ", x.dim(), "D");
    // TORCH_CHECK(r.dim() == 2, "Expected r to be 2D, got ", r.dim(), "D");
    TORCH_CHECK(x.sizes() == r.sizes(), "Shapes of x and r must match");
    // sizes check
    TORCH_CHECK(x.size(0) == r.size(0), "Sizes of x and r are expected to be equal in dimension 0");
    TORCH_CHECK(x.size(1) == r.size(1), "Sizes of x and r are expected to be equal in dimension 1");
    // inf, nan check
    TORCH_CHECK(torch::isfinite(x).all().item<bool>(), "x must contain only finite values");
    TORCH_CHECK(torch::isfinite(r).all().item<bool>(), "r must contain only finite values");
    // value limits check
    TORCH_CHECK(((x >= 0) & (x <= 1)).all().item<bool>(), "Expected all elements of x to be in [0, 1]");
    TORCH_CHECK(((r >= 0) & (r <= 4)).all().item<bool>(), "Expected all elements of r to be in [0, 4]");

    // ATen-based formula implementation
    return r * x * (1 - x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bifurcation_step", &bifurcation_step, "Single bifurcation step.");
}