#include <torch/extension.h>

torch::Tensor bifurcation_step(const torch::Tensor& x, const torch::Tensor& r) {
    TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
    TORCH_CHECK(r.device().is_cpu(), "r must be on CPU");
    TORCH_CHECK(x.layout() == torch::kStrided, "x must have strided layout");
    TORCH_CHECK(r.layout() == torch::kStrided, "r must have strided layout");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must have dtype float32");
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "r must have dtype float32");
    TORCH_CHECK(!x.requires_grad(), "x must not require gradients");
    TORCH_CHECK(!r.requires_grad(), "r must not require gradients");
    TORCH_CHECK(x.sizes() == r.sizes(), "x and r must have the same shape");
    TORCH_CHECK(torch::isfinite(x).all().item<bool>(), "x must be finite");
    TORCH_CHECK(torch::isfinite(r).all().item<bool>(), "r must be finite");
    TORCH_CHECK(((x >= 0) & (x <= 1)).all().item<bool>(), "x must be in [0, 1]");
    TORCH_CHECK(((r >= 0) & (r <= 4)).all().item<bool>(), "r must be in [0, 4]");

    // Earlier version used a manual per-element loop; see git history (commit 2743746^).
    return r * x * (1.0f - x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bifurcation_step", &bifurcation_step, "One CPU logistic-map step");
}
