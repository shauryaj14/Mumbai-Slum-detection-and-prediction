"""
Build a single combined GeoTIFF: 10 Sentinel-2 bands + 1 ground-truth mask
band, aligned to the same pixel grid, with proper GeoTIFF georeferencing
tags copied from the original image so it stays readable in QGIS/rasterio/
GDAL for the tiling step later.

Usage:
    python build_combined_stack.py Mumbai.tif Mumbai_ground_truth.tif \
        data/processed/mumbai_stack.tif
"""

import sys
import numpy as np
import tifffile

from align_and_overlay import read_geotiff, align_by_offset  # reuse earlier logic


def get_geotiff_tags(path):
    """Pull the raw GeoTIFF georeferencing tag VALUES (materialized while the
    file is still open) to copy onto the output file."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {}
        for code in (34735, 34736, 34737):  # GeoKeyDirectory, GeoDoubleParams, GeoASCIIParams
            t = page.tags.get(code)
            if t is not None:
                tags[code] = t.value  # force read now, not a lazy reference
        return tags


def build_stack(img_path, mask_path, out_path):
    img, img_t = read_geotiff(img_path)
    mask, mask_t = read_geotiff(mask_path)

    img_c, mask_c = align_by_offset(img, img_t, mask, mask_t)

    # Combine: 10 satellite bands (float32) + 1 mask band (cast to float32,
    # values 0.0/1.0) so the whole stack is a single consistent dtype.
    combined = np.concatenate(
        [img_c.astype(np.float32), mask_c.astype(np.float32)], axis=0
    )
    n_bands, h, w = combined.shape
    print(f"Combined stack shape: {n_bands} bands, {h} x {w} "
          f"(bands 1-10 = satellite, band {n_bands} = mask)")

    # New origin = the aligned image's cropped origin. Since align_by_offset
    # crops from the top-left of whichever raster starts further along,
    # recompute the true-world origin of pixel (0,0) of the cropped array.
    xres, yres = img_t["xres"], img_t["yres"]
    dx = round((mask_t["ulx"] - img_t["ulx"]) / xres)
    dy = round((img_t["uly"] - mask_t["uly"]) / yres)
    crop_row0 = max(0, dy)
    crop_col0 = max(0, dx)
    new_ulx = img_t["ulx"] + crop_col0 * xres
    new_uly = img_t["uly"] - crop_row0 * yres

    geo_tags = get_geotiff_tags(img_path)  # copy CRS (EPSG:32643) from original

    extratags = [
        (33550, "d", 3, (xres, yres, 0.0), False),           # ModelPixelScaleTag
        (33922, "d", 6, (0.0, 0.0, 0.0, new_ulx, new_uly, 0.0), False),  # ModelTiepointTag
    ]
    dtype_map = {34735: "H", 34736: "d", 34737: "s"}
    for code, value in geo_tags.items():
        count = len(value) + 1 if isinstance(value, str) else len(value)
        extratags.append((code, dtype_map[code], count, value, False))

    import os, shutil
    local_tmp = "/home/claude/_tmp_combined_stack.tif"
    tifffile.imwrite(
        local_tmp,
        combined,
        photometric="minisblack",
        planarconfig="separate",
        extratags=extratags,
    )
    shutil.copy(local_tmp, out_path)
    os.remove(local_tmp)
    print(f"Wrote combined GeoTIFF to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_combined_stack.py <image.tif> <mask.tif> <out.tif>")
        sys.exit(1)
    build_stack(sys.argv[1], sys.argv[2], sys.argv[3])
