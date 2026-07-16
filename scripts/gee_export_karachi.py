"""
Pull a Sentinel-2 surface-reflectance composite for the Karachi informal
settlement AOI, 2017 - matching the same band selection / resolution / CRS
approach used for Mumbai.

Run in Colab, in a cell AFTER your existing ee.Authenticate()/ee.Initialize().
"""

import ee

# Bounding box derived from EO4SD_KARACHI_INFORMAL_2017.shp
# (297267, 2742976) to (321421, 2756469) in UTM 42N -> converted to lon/lat
AOI = ee.Geometry.Rectangle([66.99464, 24.78813, 67.23174, 24.91294])
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate("2017-01-01", "2017-04-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

count = collection.size().getInfo()
print(f"Karachi 2017: {count} scenes found")

if count == 0:
    print("No usable scenes in this window - try widening the date range "
          "(e.g. 2017-01-01 to 2017-06-30) before giving up.")
else:
    composite = collection.median().select(BANDS)

    task = ee.batch.Export.image.toDrive(
        image=composite.clip(AOI),
        description="karachi_sentinel2_2017",
        folder="mumbai_slum_project",  # same Drive folder, fine to mix cities
        fileNamePrefix="karachi_sentinel2_2017",
        region=AOI,
        scale=10,
        crs="EPSG:32642",  # UTM 42N - correct zone for Karachi's longitude
        maxPixels=1e10,
    )
    task.start()
    print("Export started - check Tasks tab at code.earthengine.google.com "
          "or your Drive folder once it finishes.")
