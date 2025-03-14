# Project management code

from setuptools import setup, find_packages

# read the requirements
with open("requirements.txt") as rf:
    requirements = rf.read().splitlines()

setup(
    name="Hotel_reservation",
    version="0.1",
    author="Karthik Pai",
    packages=find_packages(),
    install_requires=requirements,
)
