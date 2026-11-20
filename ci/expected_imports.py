"""
Curated module import rules, shared by the import tests.

For a given set of missing dependencies, which sertit modules must
fail to import, and with which message. Every module not listed for
a scenario must import cleanly.

test_optional_imports.py checks these rules against simulated gaps;
minimal/test_minimal_imports.py checks them against a real base-only
install. Both tests assert these strings verbatim, so if a module's
import-failure message changes in sertit/, update it here as well.
"""

# Modules that must fail to import when rasterio is missing.
NO_RASTERIO = {
    "rasters": "Please install 'rasterio' to use the 'rasters' package.",
    "rasters_rio": "Please install 'rasterio' to use the 'rasters_rio' package.",
    # imports rasters at module level, so fails with rasters' message
    "ci": "Please install 'rasterio' to use the 'rasters' package.",
}

# Modules that must fail to import when click is missing: arcpy
# directly, plus rasterio's modules transitively (rasterio itself
# depends on click).
NO_CLICK = {
    "arcpy": "Please install 'click' to use the 'arcpy' package.",
    **NO_RASTERIO,
}

# Modules that must fail to import in a base-only install (no optional
# extras at all). Only rasterio's and click's consumers fail at import
# level; the other optional dependencies degrade to guarded functions
# instead (see test_guarded_function_rules).
MINIMAL = {**NO_RASTERIO, **NO_CLICK}
