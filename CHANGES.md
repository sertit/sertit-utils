# Release History

## 1.3.2 (2021-04-27)
- Allowing `gpd.GeoDataFrames` in `crop`/`mask`
- **API break**: `rasters.write(array, path, meta)` becomes `rasters.write(array, meta, path)` !

## 1.3.1 (2021-04-27)
- Do not lose attributes when using `rasters.set_nodata`
- Discard locks when reading `xarrays` (in `rasters.read`)

## 1.3.0 (2021-04-26)
- Going Open Source
