#include <torch/extension.h>

torch::Tensor bifurcation_step(torch::Tensor x, torch::Tensor r)
{
    // device check
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(r.device().is_cpu(), "r must be on CPU");
    // dtype check
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Expected dtype of x is float32, got ", x.scalar_type());
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "Expected dtype of r is float32, got ", r.scalar_type());
    // dimensions check
    TORCH_CHECK(x.dim() == 2, "Expected x to be 2D, got ", x.dim(), "D");
    TORCH_CHECK(r.dim() == 2, "Expected r to be 2D, got ", r.dim(), "D");
    // sizes check
    TORCH_CHECK(x.size(0) == r.size(0), "Sizes of x and r are expected to be equal in dimension 0");
    TORCH_CHECK(x.size(1) == r.size(1), "Sizes of x and r are expected to be equal in dimension 1");

    // ATen-based formula implementation
    return r * x * (1 - x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bifurcation_step", &bifurcation_step, "Clamp a CPU tensor to [0, 1]");
}