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

After `./tools/smoke.sh`, run this from the repository root:

```bash
/tmp/fse-smoke-try/bin/python -m cProfile -s cumulative tools/profile.py
```

The workload calls the installed extension ten times on 1,000,000-element
float32 tensors. The measured cumulative time still needs to be recorded here
from a successful run in the course container.
