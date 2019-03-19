from setuptools import setup, find_packages

setup(
    name="bgbb",
    version="0.1.4a1",
    packages=find_packages(),
    install_requires=[
        "lifetimes==0.9",
        "numba==0.42.0",
        "scipy",
        "pandas",
        "numpy",
        "pyarrow",
        "pyspark",
    ],
    license=["Apache 2.0", "MIT"],
    extras_require={"test": ["pytest"]},
    classifiers=["Programming Language :: Python :: 3.5"],
)
