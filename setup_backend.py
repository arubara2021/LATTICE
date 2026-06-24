from setuptools import setup, Extension
import numpy as np
import os

if os.name == 'nt':
    # Windows (MSVC)
    extra_compile_args = ['/O2', '/fp:fast', '/openmp:llvm']
    extra_link_args = ['/openmp:llvm']
else:
    # Linux/Mac (GCC/Clang)
    extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp']
    extra_link_args = ['-lm', '-fopenmp']

module = Extension(
    'lattice_backend',
    sources=['lattice/lattice_backend.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='lattice_backend',
    version='1.0',
    description='High-performance C backend for Formula2ONNX',
    ext_modules=[module],
)
