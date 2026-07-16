"""
The 2019 real mask comes from a slightly smaller/different AOI than the
2016-2021 GEE pulls (3927x1992 vs 3976x2088). This pads it onto an
identical canvas/origin as the other years, using the known offset, so
that grid_cells.py produces grid_row/grid_col values that line up exactly
across every year - required before build_expansion_table.py can safely
merge any transition involving 2019.

Usage:
    python pad_2019_mask.py real_mask_2019.tif predicted_mask_2018.tif real_mask_2019_aligned.tif
    (second argument is any file already on the "big" 2016-2021 grid, used
    only as a template for shape/transform/CRS)
"""

import sys

import numpy as np
import rasterio


def pad_mask(mask_2019_path, template_path, out_path):
    with rasterio.open(mask_2019_path) as src:
        mask_2019 = src.read(1)
        src_transform = src.transform
        src_ulx, src_uly = src_transform.c, src_transform.f

    with rasterio.open(template_path) as tmpl:
        template_profile = tmpl.profile.copy()
        target_h, target_w = tmpl.height, tmpl.width
        target_transform = tmpl.transform
        tgt_ulx, tgt_uly = target_transform.c, target_transform.f
        xres = target_transform.a
        yres = -target_transform.e

    col_offset = round((src_ulx - tgt_ulx) / xres)
    row_offset = round((tgt_uly - src_uly) / yres)
    print(f"Placing 2019 mask at row_offset={row_offset}, col_offset={col_offset} "
          f"within a {target_h}x{target_w} canvas")

    padded = np.zeros((target_h, target_w), dtype=np.uint8)
    h, w = mask_2019.shape
    padded[row_offset:row_offset + h, col_offset:col_offset + w] = mask_2019

    out_profile = template_profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(padded[np.newaxis, ...])

    print(f"Wrote aligned 2019 mask to {out_path}, shape {padded.shape}")
    print(f"Slum fraction (should roughly match original): {(padded > 0).mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python pad_2019_mask.py <real_mask_2019.tif> <template.tif> <out_aligned.tif>")
        sys.exit(1)
    pad_mask(sys.argv[1], sys.argv[2], sys.argv[3])
