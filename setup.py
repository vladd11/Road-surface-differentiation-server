from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Colorify',
    ext_modules=cythonize("color.pyx"), include_dirs=[numpy.get_include()],
    zip_safe=False,
)
