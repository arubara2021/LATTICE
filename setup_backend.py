from setuptools import setup, Extension
import numpy as np

module = Extension(
    'lattice_backend',
    sources=['lattice/lattice_backend.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-ffast-math', '-march=native', '-fopenmp'],
    extra_link_args=['-lm', '-fopenmp'],
)

setup(
    name='lattice_backend',
    version='1.0',
    description='High-performance C backend for Formula2ONNX',
    ext_modules=[module],
)
