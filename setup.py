import glob
from setuptools import *
from Cython.Build import cythonize


setup(
        name='ghw',
        description='Python utils for hollywood naming data',
        author='Olgert Denas',
        author_email='gertidenas@gmail.com',
        version=0.1,
        packages=['ghw'],
        #package_dir={'ghw': 'src'},
        #scripts=glob.glob('scripts/*py'),
        ext_modules=cythonize("ghw/cosine_cy.pyx"),
        install_requires=['numpy > 1.5']
)
