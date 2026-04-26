from setuptools import find_packages, setup


HDF5_DEPS = [
    "numpy>=1.23",
    "h5py>=3.8",
]

TEST_DEPS = [
    "pytest>=7",
    *HDF5_DEPS,
]

setup(
    name="orca_parser",
    version="1.0.0",
    description="Modular parser for ORCA quantum chemistry output files",
    author="Henrique Castro",
    packages=find_packages(),
    python_requires=">=3.10",
    extras_require={
        "hdf5": HDF5_DEPS,
        "rmsd": ["numpy>=1.23"],
        "test": TEST_DEPS,
        "dev": TEST_DEPS,
    },
    entry_points={
        "console_scripts": [
            "orca_parser=orca_parser.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
