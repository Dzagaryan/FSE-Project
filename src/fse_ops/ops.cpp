#include <torch/extension.h>

torch::Tensor bifurcation_step(torch::Tensor x, torch::Tensor r)
{
    // device check
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(r.device().is_cpu(), "r must be on CPU");
    // ATen-based formula implementation
    return r * x * (1 - x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bifurcation_step", &bifurcation_step, "Clamp a CPU tensor to [0, 1]");
}