from distutils.core import setup
from setuptools import find_packages

setup(
    name='rlframework',
    version='0.2.1dev',
    packages=find_packages(),
    license='GNU Lesser General Public License v3.0',
    long_description=open('README.md').read(),
)
