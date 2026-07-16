"""
Convert the KML slum polygons into a raster mask (0/1) matching the exact
pixel grid of a reference image - typically the Sentinel-2 image you just
pulled from GEE, so the two are born already aligned (no offset-fixing
needed, unlike the original Mumbai.tif / Mumbai_ground_truth.tif pair).

Usage:
    python rasterize_kml_mask.py slums_2015.kml mumbai_sentinel2_2016.tif mumbai_ground_truth_2016.tif
"""

import sys

import fiona
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

# Some GDAL builds need the KML driver explicitly enabled for read+write
fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"


def rasterize_kml(kml_path, reference_tif_path, out_path):
    gdf = gpd.read_file(kml_path)
    print(f"Loaded {len(gdf)} polygons from {kml_path}")
    print(f"Original CRS: {gdf.crs}")

    with rasterio.open(reference_tif_path) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)
        target_crs = ref.crs
        print(f"Reference grid: {out_shape[1]} x {out_shape[0]}, CRS: {target_crs}")

    # KML is lon/lat (EPSG:4326) - reproject to match the reference image's CRS
    gdf = gdf.to_crs(target_crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    slum_fraction = (mask > 0).mean()
    print(f"Rasterized mask - slum pixel fraction: {slum_fraction:.4f}")

    with rasterio.open(reference_tif_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"Wrote mask to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rasterize_kml_mask.py <slums.kml> <reference_image.tif> <out_mask.tif>")
        sys.exit(1)
    rasterize_kml(sys.argv[1], sys.argv[2], sys.argv[3])
