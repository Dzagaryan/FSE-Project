#!/bin/bash
set -e

rm -rf dist && mkdir -p dist
python3 -m venv --clear /tmp/fse-smoke-build &&
/tmp/fse-smoke-build/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
/tmp/fse-smoke-build/bin/pip install setuptools wheel &&
/tmp/fse-smoke-build/bin/pip wheel . --no-deps --no-build-isolation -w dist &&
ls dist/*.whl | grep -E 'linux(_x86_64|_aarch64)?\.whl' &&
test "$(ls dist/*.whl | wc -l)" -eq 1 &&
python3 -m venv --clear /tmp/fse-smoke-try &&
/tmp/fse-smoke-try/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
/tmp/fse-smoke-try/bin/pip install pytest &&
/tmp/fse-smoke-try/bin/pip install --no-deps dist/*.whl &&
rm -rf /tmp/fse-smoke-tests && mkdir -p /tmp/fse-smoke-tests &&
cp -r tests/. /tmp/fse-smoke-tests/ &&
(cd /tmp && /tmp/fse-smoke-try/bin/python -c 'import torch; from fse_ops import bifurcation_step; x=torch.tensor([0.0, 0.25, 0.5, 1.0]); r=torch.tensor([1.0, 2.0, 3.0, 4.0]); assert torch.equal(bifurcation_step(x, r), r*x*(1.0-x))') &&
(cd /tmp && /tmp/fse-smoke-try/bin/python -m pytest -q /tmp/fse-smoke-tests) &&
echo OK