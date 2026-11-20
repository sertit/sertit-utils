"""
Curated module import rules, shared by the import tests.

For a given set of missing dependencies, which sertit modules must
fail to import, and with which message. Every module not listed for
a scenario must import cleanly.

test_optional_imports.py checks these rules against simulated gaps;
minimal/test_minimal_imports.py checks them against a real base-only
install. Both tests assert these strings verbatim, so if a module's
import-failure message changes in sertit/, update it here as well.

Cascade note: geoutils itself imports rasterio and rioxarray at
module level, so when one of those is missing, `import geoutils`
fails too, and the modules that hard-require geoutils report the
geoutils message (as they do when geoutils is genuinely missing).
"""

# Messages, shared between scenarios.
_RASTERIO = {
    "rasters": "Please install 'rasterio' to use the 'rasters' package.",
    "rasters_rio": "Please install 'rasterio' to use the 'rasters_rio' package.",
    # imports rasters at module level, so fails with rasters' message
    "ci": "Please install 'rasterio' to use the 'rasters' package.",
}

# Modules that must fail to import when rasterio is missing.
NO_RASTERIO = {
    **_RASTERIO,
    # geoutils imports rasterio at module level, so its absence
    # breaks the geoutils import of these modules too:
    "vectors": "Please install 'geoutils' to use the 'vectors' package.",
    "geometry": "Please install 'geoutils' to use the 'vectors' package.",
}

# Modules that must fail to import when click is missing: arcpy
# directly, plus rasterio's modules transitively (rasterio itself
# depends on click).
NO_CLICK = {
    "arcpy": "Please install 'click' to use the 'arcpy' package.",
    **NO_RASTERIO,
}

# Modules that must fail to import when rioxarray is missing:
# geoutils imports rioxarray at module level, so every module that
# hard-requires geoutils (directly or via vectors/rasters) fails
# with the geoutils message.
NO_RIOXARRAY = {
    "vectors": "Please install 'geoutils' to use the 'vectors' package.",
    # imports vectors at module level, so fails with vectors' message
    "geometry": "Please install 'geoutils' to use the 'vectors' package.",
    "rasters": "Please install 'geoutils' to use the 'rasters' package.",
    # imports vectors at module level, so fails with vectors' message
    "rasters_rio": "Please install 'geoutils' to use the 'vectors' package.",
    # imports rasters at module level, so fails with rasters' message
    "ci": "Please install 'geoutils' to use the 'rasters' package.",
}

# Modules that must fail to import when geopandas is missing.
NO_GEOPANDAS = {
    "vectors": "Please install 'geopandas' to use the 'vectors' package.",
    "geometry": "Please install 'geopandas' to use the 'vectors' package.",
    "rasters": "Please install 'geopandas' to use the 'rasters' package.",
    "rasters_rio": "Please install 'geopandas' to use the 'rasters_rio' package.",
    "ci": "Please install 'geopandas' to use the 'rasters' package.",
}

# Modules that must fail to import when geoutils is missing.
NO_GEOUTILS = {
    "vectors": "Please install 'geoutils' to use the 'vectors' package.",
    # imports vectors at module level, so fails with vectors' message
    "geometry": "Please install 'geoutils' to use the 'vectors' package.",
    "rasters": "Please install 'geoutils' to use the 'rasters' package.",
    # imports vectors at module level, so fails with vectors' message
    "rasters_rio": "Please install 'geoutils' to use the 'vectors' package.",
    # imports rasters at module level, so fails with rasters' message
    "ci": "Please install 'geoutils' to use the 'rasters' package.",
}

# Modules that must fail to import in a base-only install (no optional
# extras at all). In that state the geopandas blocks fail before the
# geoutils/rasterio ones (import order), so only arcpy fails with its
# own (click) message.
MINIMAL = {
    "arcpy": NO_CLICK["arcpy"],
    **NO_GEOPANDAS,
}
