# Project note

## Bug found during development

The original C++ checks assumed two-dimensional inputs. A one-dimensional
tensor raised `RuntimeError` instead of producing a result, even though the
README described arbitrary shapes. Commit `bac5055` removed the dimension
check; commit `34ced98` removed the remaining `size(1)` check. The current
tests include one-dimensional input.

Another recorded fix is commit `5ed79fa`: the adversarial test replaced a
hardcoded float32 expected value with the pure-PyTorch specification, so exact
comparison follows the specified operation.

## Unsupported case

Inputs with `requires_grad=True` are rejected by `TORCH_CHECK` in C++ and
raise `RuntimeError`; this extension does not implement autograd.

## Large-input profile

Run this from the repository root in any environment where the wheel is installed (for example the /tmp/fse-smoke-try venv that ./tools/smoke.sh creates):



```bash
python3 -m cProfile -s cumulative tools/profile.py
```

The workload calls the installed extension ten times on 1,000,000-element
float32 tensors. In the course container, `cProfile` reported 10 calls to
`fse_ops._C.bifurcation_step`, with 0.110 seconds cumulative time and
0.011 seconds per call:

```text
10    0.110    0.011    0.110    0.011 {built-in method fse_ops._C.bifurcation_step}
```
