from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Planet Pipeline",
    version="0.0.1",
    description="A planet finding pipeline for TESS Data",
    author="Erik Gillis",
    author_email="gillie1@mcmaster.ca",
    packages=find_packages(),
)