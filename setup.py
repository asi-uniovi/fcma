#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "pint",
    "pulp",
    "rich",
    "cloudmodel @ git+https://github.com/asi-uniovi/cloudmodel.git#egg=cloudmodel",
]

test_requirements = ['pytest>=3', ]

setup(
    author="Jose Maria Lopez Lopez",
    author_email='chechu@uniovi.es',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="Fast Container to Machine Allocator",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fcma',
    name='fcma',
    packages=find_packages(include=['fcma', 'fcma.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/uochechu/fcma',
    version='0.1.0',
    zip_safe=False,
)
