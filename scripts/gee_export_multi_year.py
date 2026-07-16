"""
Run in Google Colab (same setup as your 2016 pull - ee.Authenticate() etc.
already done in your notebook). Loops over multiple years, exporting one
Sentinel-2 median composite per year to Drive.

Paste this in a Colab cell AFTER your existing ee.Initialize(...) cell.
"""

import ee

AOI = ee.Geometry.Rectangle([72.78134, 18.90838, 72.97509, 19.26522])
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Adjust this list to whichever years you actually want
YEARS = [2016, 2017, 2018, 2020, 2021]  # 2019 already have

for year in YEARS:
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate(f"{year}-01-01", f"{year}-04-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    count = collection.size().getInfo()
    print(f"{year}: {count} scenes found")

    if count == 0:
        print(f"  Skipping {year} - no usable scenes in this date range.")
        continue

    # No manual .resample()/.reproject() - let the export step handle it
    composite = collection.median().select(BANDS)

    task = ee.batch.Export.image.toDrive(
        image=composite.clip(AOI),
        description=f"mumbai_sentinel2_{year}",
        folder="mumbai_slum_project",
        fileNamePrefix=f"mumbai_sentinel2_{year}",
        region=AOI,
        scale=10,
        crs="EPSG:32643",
        maxPixels=1e10,
    )
    task.start()
    print(f"  Export started for {year}")

print("\nAll exports queued. Check Tasks tab at code.earthengine.google.com or your Drive folder.")
