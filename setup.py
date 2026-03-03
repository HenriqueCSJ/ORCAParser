from setuptools import setup, find_packages

setup(
    name="orca_parser",
    version="1.0.0",
    description="Modular parser for ORCA quantum chemistry output files",
    author="Henrique Castro",
    packages=find_packages(),
    python_requires=">=3.10",
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
