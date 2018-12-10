from setuptools import setup, find_packages

setup(
    name="bgbb",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "lifetimes",
        "numba",
        "scipy",
        "pandas",
        "numpy",
        "pyspark",
    ],
    license=["Apache 2.0", "MIT"],
    extras_require={"test": ["pytest"]},
    classifiers=["Programming Language :: Python :: 3.6"],
)
