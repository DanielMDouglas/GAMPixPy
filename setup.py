#!/usr/bin/env python

import setuptools

VER = "0.1.6"

reqs = ['numpy',
        'h5py',
        'tqdm',
        # 'torch', # skip this for now.  Users must fend for themselves!
        # 'ROOT', # skip this for now.  Users must fend for themselves!
        'torchist',
        'particle',
        'pyyaml',
        ]
links = ['https://download.pytorch.org/whl/cpu']

setuptools.setup(
    name="gampixpy",
    version=VER,
    author="SLAC National Accelerator Laboratory",
    author_email="dougl215@slac.stanford.edu",
    description="Gampix Readout Simulation",
    url="https://github.com/DanielMDouglas/GAMPixPy",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    package_data={"gampixpy": ["*_config/*.yaml"]},
    dependency_links=links,
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
