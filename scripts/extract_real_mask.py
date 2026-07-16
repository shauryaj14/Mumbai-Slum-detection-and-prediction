"""
For years where you have a REAL ground-truth mask already baked into an
11-band stack (2016, 2019) - extract just the mask band as its own
single-band GeoTIFF, matching the format predict_mask.py produces for
other years. This lets grid_cells.py treat real and predicted masks
identically.

Usage:
    python extract_real_mask.py mumbai_stack.tif real_mask_2019.tif
    python extract_real_mask.py mumbai_stack_2016.tif real_mask_2016.tif
"""

import sys

import numpy as np
import rasterio


def extract_mask(stack_path, out_path, mask_band=11):
    with rasterio.open(stack_path) as src:
        mask = (src.read(mask_band) > 0).astype(np.uint8)
        profile = src.profile.copy()

    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, ...])

    print(f"Extracted mask band {mask_band} from {stack_path}")
    print(f"Slum fraction: {mask.mean():.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_real_mask.py <stack.tif> <out_real_mask.tif>")
        sys.exit(1)
    extract_mask(sys.argv[1], sys.argv[2])
