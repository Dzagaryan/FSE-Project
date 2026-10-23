FROM ubuntu:24.04 AS build
WORKDIR /src

# Python, pip, and a C++ compiler for CppExtension
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --break-system-packages setuptools wheel

COPY . .
RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels

FROM ubuntu:24.04

# Python and pip for running tests; the runtime stage does not need a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --break-system-packages pytest
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages --no-deps /wheels/*.whl
COPY tests /opt/tests
WORKDIR /tmp
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]