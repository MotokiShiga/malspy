# -*- coding: utf-8 -*-

# Learn more: https://github.com/MotokiShiga/malspy/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='malspy',
    version='0.4.0',
    description='Python package for MAchine Learning based Spectral imaging data analysis',
    long_description=readme,
    author='Motoki Shiga',
    author_email='shiga_m@gifu-u.ac.jp',
    url='https://github.com/MotokiShiga/malspy',
    install_requires=['numpy','scipy','pandas','matplotlib'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples', 'mytest'))
)
