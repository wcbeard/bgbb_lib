from setuptools import setup, find_packages

setup(
    name = "bgbb",
    version = "0.1.1",
    packages = find_packages(),
    install_requires=[
        "lifetimes",
        "numba",
        "scipy",
        "pandas",
        "numpy",
        "pyspark",
    ],
    extras_require={
        "test": ["pytest"]
    }
)
