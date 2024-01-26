from setuptools import setup, find_packages
import os

with open("requirements.txt", mode="r") as f:
    requirement_list = f.read().splitlines()
    
setup(
    name="bort",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirement_list,
    python_requires=">=3.9",
)