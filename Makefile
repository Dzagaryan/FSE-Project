.PHONY: wheel install test clean smoke

wheel:
	python3 -m pip wheel . --no-deps -w dist

install:
	python3 -m pip install dist/*.whl

test:
	python3 -m pytest -q tests

smoke:
	./tools/smoke.sh

clean:
	rm -rf build dist src/fse_ops.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +