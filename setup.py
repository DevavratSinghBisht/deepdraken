import os
import re
from io import open

from setuptools import find_packages, setup


def parse_requirements(filename):
    """
    Parse a requirements pip file returning the list of required packages. It exclude commented lines and --find-links directives.
    
    :param filename: pip requirements requirements
    :return: list of required package with versions constraints
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()
    parsed_requirements = [line.strip()
                           for line in parsed_requirements
                           if not ((line.strip()[0] == "#") or line.strip().startswith('--find-links') or ("git+https" in line))]
    
    return parsed_requirements


def get_dependency_links(filename):
    """
    Parse a requirements pip file looking for the --find-links directive.
    
    :param filename:  pip requirements requirements
    :return: list of find-links's url
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()
    dependency_links = list()
    for line in parsed_requirements:
        line = line.strip()
        if line.startswith('--find-links'):
            dependency_links.append(line.split('=')[1])
    return dependency_links


dependency_links = get_dependency_links('requirements.txt')
parsed_requirements = parse_requirements('requirements.txt')


def versionfromfile(*filepath):
    infile = os.path.join(*filepath)
    with open(infile) as fp:
        version_match = re.search(
                r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string in {}.".format(infile))


here = os.path.abspath(os.path.dirname(__file__))
_version: str = versionfromfile(here, "deepdraken", "_version.py")

setup(
    name="deepdraken",
    version=_version,
    author="Devavrat Singh Bisht",
    author_email="devavratsinghbisht@gmail.com",
    description="Package for CV tasks",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Computer-Vision Transfer-Learning",
    license="Apache License 2.0",
    url="https://github.com/DevavratSinghBisht/deepdraken",
    download_url=f"https://github.com/DevavratSinghBisht/deepdraken", # TODO see this
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]), # TODO see this
    dependency_links=dependency_links,
    install_requires=parsed_requirements,
    python_requires=">=3.6.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)