from setuptools import setup, find_packages

setup(
    name="simopt-tools",
    version="0.1.0",
    author="T. Henry Lian",
    author_email="tlian@fudan.edu.cn",
    description="Private Tools",
    packages=find_packages(exclude=["_tests"]),
    install_requires=[
        "numpy"
    ]
)
