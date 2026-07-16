"""
Run this locally (you already have rasterio + geopandas installed).

Does three things:
  1. Rasterizes the KML slum polygons onto the Sentinel-2 image's exact grid
  2. Combines image + mask into one 11-band GeoTIFF stack (same format as
     your existing mumbai_stack.tif)
  3. Saves a true-color, percentile-stretched PNG preview with the mask
     overlaid in red, so you can eyeball it without opening GIS software

Usage:
    python build_2016_dataset.py mumbai_sentinel2_2016.tif slums_2015.kml mumbai_stack_2016.tif mumbai_2016_preview.png
"""

import sys

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt

fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"


def percentile_stretch(band, low=2, high=98):
    band = band.astype(np.float64)
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    lo, hi = np.nanpercentile(valid, [low, high])
    if hi <= lo:
        return np.zeros_like(band)
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    out = np.nan_to_num(out, nan=0.0)  # NaN gaps render as black, not blank
    return out


def main(image_path, kml_path, out_stack_path, out_png_path):
    # --- 1. Read the Sentinel-2 image ---
    with rasterio.open(image_path) as src:
        image = src.read()  # shape: (bands, rows, cols)
        profile = src.profile.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        target_crs = src.crs

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"CRS: {target_crs}")
    for i in range(image.shape[0]):
        b = image[i]
        valid = b[~np.isnan(b)]
        p5, p50, p95 = np.percentile(valid, [5, 50, 95])
        print(f"  band {i}: p5={p5:.1f} p50={p50:.1f} p95={p95:.1f} "
              f"(nan={np.isnan(b).sum()}/{b.size})")

    # --- 2. Rasterize the KML mask onto this exact grid ---
    gdf = gpd.read_file(kml_path)
    print(f"Loaded {len(gdf)} polygons from {kml_path}")
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

    # --- 3. Combine into one 11-band stack (10 satellite + 1 mask), float32 ---
    combined = np.concatenate(
        [image.astype(np.float32), mask[np.newaxis, ...].astype(np.float32)], axis=0
    )

    stack_profile = profile.copy()
    stack_profile.update(count=combined.shape[0], dtype="float32")
    with rasterio.open(out_stack_path, "w", **stack_profile) as dst:
        dst.write(combined)
    print(f"Wrote combined stack to {out_stack_path}")

    # --- 4. True-color PNG preview with mask overlay ---
    # Band order assumed: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (0-indexed)
    # True color = Red:B4(idx2), Green:B3(idx1), Blue:B2(idx0)
    r = percentile_stretch(image[2])
    g = percentile_stretch(image[1])
    b = percentile_stretch(image[0])
    rgb = np.dstack([r, g, b])

    overlay = rgb.copy()
    m = mask > 0
    alpha = 0.45
    overlay[..., 0] = np.where(m, overlay[..., 0] * (1 - alpha) + 1.0 * alpha, overlay[..., 0])
    overlay[..., 1] = np.where(m, overlay[..., 1] * (1 - alpha), overlay[..., 1])
    overlay[..., 2] = np.where(m, overlay[..., 2] * (1 - alpha), overlay[..., 2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 2016 (true color)")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Rasterized KML mask (2015)")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    print(f"Wrote preview PNG to {out_png_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python build_2016_dataset.py <image.tif> <slums.kml> <out_stack.tif> <out_preview.png>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
