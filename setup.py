import setuptools
from sertit_utils.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sertit_utils',
    version=__version__,  # Semantic Versioning (see https://semver.org/)
    author="Rémi BRAUN",
    author_email="remi.braun@unistra.fr",
    description="SERTIT python library for Sentinel Downloading purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.sertit.unistra.fr/SERTIT/satdownload",
    packages=setuptools.find_packages(),
    install_requires=[
        'lxml',
        'html5lib',
        'beautifulsoup4',
        'geopandas',
        'sentinelsat',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ],
    package_data={'sertit_utils': ['*.ui']},
    include_package_data=True
)
