from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fse_ops",
    version="0.1.0",
    package_dir={"": "src"},
    packages=["fse_ops"],
    ext_modules=[
        CppExtension("fse_ops._C", ["src/fse_ops/ops.cpp"])
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
