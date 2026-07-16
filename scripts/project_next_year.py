"""
Projects a 2022 slum mask from the 2021 grid, using the trained Random
Forest's expansion predictions. This is NOT a segmentation of real 2022
satellite imagery (none exists yet) - it's a forward projection: take
2021's actual state, ask the Random Forest which cells are likely to
newly cross the slum threshold, and paint those predicted-expansion
cells onto the map alongside cells that were already slum.

Output:
  - projected_mask_2022.tif : full-resolution raster (same grid as the
    2021 mask), each 160m block painted 0/1 based on the projected state
  - projected_2022_preview.png : true-color 2021 image with existing
    slum in one color and NEWLY predicted expansion in a second color

Usage:
    python project_next_year.py grid_2021.csv predicted_mask_2021.tif \
        mumbai_sentinel2_2021.tif random_forest_expansion.pkl \
        projected_mask_2022.tif projected_2022_preview.png
"""

import sys

import joblib
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


def percentile_stretch(band, low=2, high=98):
    band = band.astype(np.float64)
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    lo, hi = np.nanpercentile(valid, [low, high])
    if hi <= lo:
        return np.zeros_like(band)
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(out, nan=0.0)


def main(grid_csv, mask_tif, image_tif, model_pkl, out_tif, out_png):
    grid = pd.read_csv(grid_csv)

    saved = joblib.load(model_pkl)
    rf = saved["model"]
    threshold = saved["threshold"]
    print(f"Loaded Random Forest, using threshold {threshold}")

    # Build the exact feature set the model was trained on
    feature_cols = ["slum_fraction_before", "neighbor_slum_fraction", "dist_to_nearest_slum_m"]
    X = grid.rename(columns={"slum_fraction": "slum_fraction_before"})[feature_cols]

    probs = rf.predict_proba(X)[:, 1]
    grid["expansion_prob"] = probs
    grid["predicted_expand"] = (probs > threshold).astype(int)

    # A cell is slum in the 2022 projection if it already was, OR the
    # model predicts it newly crosses the threshold. This assumes slums
    # don't spontaneously disappear - a reasonable assumption over a
    # single year, and consistent with how "expanded" was defined during
    # training.
    grid["is_slum_2022_projected"] = ((grid["is_slum"] == 1) | (grid["predicted_expand"] == 1)).astype(int)

    n_new = ((grid["is_slum"] == 0) & (grid["predicted_expand"] == 1)).sum()
    print(f"Cells already slum in 2021: {grid['is_slum'].sum()}")
    print(f"NEW cells predicted to expand by 2022: {n_new}")
    print(f"Total projected slum cells in 2022: {grid['is_slum_2022_projected'].sum()}")

    # --- Paint the grid-level projection back onto the full pixel raster ---
    with rasterio.open(mask_tif) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()
        transform = src.transform

    n_rows = grid["grid_row"].max() + 1
    n_cols = grid["grid_col"].max() + 1
    cell_px_h = h // n_rows
    cell_px_w = w // n_cols

    projected_raster = np.zeros((h, w), dtype=np.uint8)
    new_expansion_raster = np.zeros((h, w), dtype=np.uint8)

    for _, row in grid.iterrows():
        r, c = int(row["grid_row"]), int(row["grid_col"])
        r0, r1 = r * cell_px_h, (r + 1) * cell_px_h
        c0, c1 = c * cell_px_w, (c + 1) * cell_px_w
        if row["is_slum_2022_projected"] == 1:
            projected_raster[r0:r1, c0:c1] = 1
        if row["is_slum"] == 0 and row["predicted_expand"] == 1:
            new_expansion_raster[r0:r1, c0:c1] = 1

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(projected_raster[np.newaxis, ...])
    print(f"Wrote projected raster to {out_tif}")

    # --- Build the preview PNG ---
    with rasterio.open(image_tif) as src:
        image = src.read()
    image = np.nan_to_num(image, nan=0.0)

    r = percentile_stretch(image[2])
    g = percentile_stretch(image[1])
    b = percentile_stretch(image[0])
    rgb = np.dstack([r, g, b])

    overlay = rgb.copy()
    existing = (projected_raster == 1) & (new_expansion_raster == 0)
    new_exp = new_expansion_raster == 1

    alpha = 0.5
    # Existing slum (2021, carried forward) in red
    overlay[..., 0] = np.where(existing, overlay[..., 0] * (1 - alpha) + 1.0 * alpha, overlay[..., 0])
    overlay[..., 1] = np.where(existing, overlay[..., 1] * (1 - alpha), overlay[..., 1])
    overlay[..., 2] = np.where(existing, overlay[..., 2] * (1 - alpha), overlay[..., 2])
    # NEW predicted expansion in bright yellow
    overlay[..., 0] = np.where(new_exp, overlay[..., 0] * (1 - alpha) + 1.0 * alpha, overlay[..., 0])
    overlay[..., 1] = np.where(new_exp, overlay[..., 1] * (1 - alpha) + 1.0 * alpha, overlay[..., 1])
    overlay[..., 2] = np.where(new_exp, overlay[..., 2] * (1 - alpha), overlay[..., 2])

    fig, ax = plt.subplots(figsize=(10, 18))
    ax.imshow(overlay)
    ax.set_title(f"Projected 2022 - existing slum (red) + predicted new expansion (yellow), threshold={threshold}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote preview to {out_png}")


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python project_next_year.py <grid_2021.csv> <predicted_mask_2021.tif> "
              "<mumbai_sentinel2_2021.tif> <random_forest_expansion.pkl> <out_mask.tif> <out_preview.png>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
