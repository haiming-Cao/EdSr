from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

ext_modules = [Extension(
    "core_c", 
    sources=["core_c.pyx"],
    extra_compile_args = ["-O3", "-ffast-math","-march=native", "-fopenmp" ],
    extra_link_args = ["-fopenmp"]
)]

setup(
    ext_modules = cythonize(ext_modules, annotate = True ),
    include_dirs=[np.get_include()],
)