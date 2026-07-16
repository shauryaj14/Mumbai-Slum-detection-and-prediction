"""
Pull a Sentinel-2 surface-reflectance composite for Mumbai, early 2016,
matching the band selection / resolution / CRS used in the existing
training stack (mumbai_stack.tif).

Run this locally (needs your own Earth Engine account):
    earthengine authenticate   # one-time
    python gee_export_sentinel2_2016.py
"""

import ee
import geemap

ee.Initialize()

# Same AOI as the KML mask / existing Mumbai.tif (lon min/max, lat min/max)
AOI = ee.Geometry.Rectangle([72.78134, 18.90838, 72.97509, 19.26522])

# Same 10 bands used in the existing stack: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# NOTE: ESA's globally-reprocessed Level-2A archive does reach back to 2015-2016
# for most regions, but coverage/quality this early can be patchier than later
# years (fewer cloud-free scenes, some regions only got L2A reprocessing later).
# If this collection comes back empty for your date range, switch to
# "COPERNICUS/S2_HARMONIZED" (Level-1C, top-of-atmosphere) instead - the
# archive there is more complete from 2015 onward, but note that TOA
# reflectance is NOT the same scale/processing as your existing Mumbai stack,
# so it would need separate normalization if you go that route.
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate("2016-01-01", "2016-04-30")   # early 2016, dry season = less cloud
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

print("Number of scenes found:", collection.size().getInfo())

# Median composite to reduce cloud/noise impact, then select + resample
# 20m bands (B5,B6,B7,B8A,B11,B12) up to 10m to match everything else.
composite = collection.median().select(BANDS)
composite = composite.resample("bilinear").reproject(crs="EPSG:32643", scale=10)

# Export to Google Drive as a GeoTIFF (large area, so Drive export is more
# reliable than a direct download URL).
task = ee.batch.Export.image.toDrive(
    image=composite.clip(AOI),
    description="mumbai_sentinel2_2016",
    folder="mumbai_slum_project",
    fileNamePrefix="mumbai_sentinel2_2016",
    region=AOI,
    scale=10,
    crs="EPSG:32643",
    maxPixels=1e10,
)
task.start()
print("Export started - check the 'Tasks' tab at https://code.earthengine.google.com/ "
      "or your Google Drive 'mumbai_slum_project' folder once it finishes (can take a few minutes).")
