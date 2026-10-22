.PHONY: wheel install test clean

wheel:
	python3 -m venv /tmp/fse-build
	/tmp/fse-build/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	/tmp/fse-build/bin/pip install setuptools wheel
	/tmp/fse-build/bin/pip wheel . --no-deps --no-build-isolation -w dist

install: wheel
	python3 -m venv /tmp/fse-install
	/tmp/fse-install/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	/tmp/fse-install/bin/pip install --no-deps dist/*.whl

test:
	./tools/smoke.sh

clean:
	rm -rf build dist src/fse_ops.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
