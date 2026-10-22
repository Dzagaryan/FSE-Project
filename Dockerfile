FROM fse-09-packaging AS build
WORKDIR /src
RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --break-system-packages setuptools wheel
COPY . .
RUN python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels

FROM fse-09-packaging
RUN python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
COPY --from=build /wheels /wheels
RUN python3 -m pip install --break-system-packages --no-deps /wheels/*.whl
COPY tests /opt/tests
WORKDIR /tmp
CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
