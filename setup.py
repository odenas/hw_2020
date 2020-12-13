import glob
from setuptools import *
import numpy


setup(
    name='ghw',
    description='Python utils for hollywood naming data',
    author='Olgert Denas',
    author_email='gertidenas@gmail.com',
    version=0.1,
    packages=['ghw'],
    include_dirs=[numpy.get_include()],
    install_requires=['numpy > 1.5', 'numba', 'pandas']
)
