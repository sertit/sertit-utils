import setuptools
from sertit import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sertit',
    version=__version__,  # Semantic Versioning (see https://semver.org/)
    author="Rémi BRAUN",
    author_email="remi.braun@unistra.fr",
    description="SERTIT python library for generic tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.sertit.unistra.fr/SERTIT/sertit-utils",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm',
        "numpy",
        "lxml"
    ],
    extras_require={
        'full': ["geopandas",
                 "rasterio",
                 "colorlog"],
        'rasters_rio': ["geopandas",
                        "rasterio"],
        'rasters': ["geopandas",
                    "xarray",
                    "rioxarray"],
        'vectors': ["geopandas"]
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ]
)
