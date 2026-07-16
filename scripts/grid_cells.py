"""
Turn a full-scene mask GeoTIFF (predicted by SegFormer, or a real ground
truth mask) into a table of 160m grid cells, each with:
  - its row/col position in the grid
  - its center coordinate (in the raster's CRS, and reprojected to lat/lon)
  - the fraction of its pixels that are slum
  - a binary is_slum flag using the >50% rule

160m was chosen to divide evenly into your 256-pixel training tiles at 10m
resolution (2560m per tile / 160m = 16 cells per tile side), though this
script operates directly on the full scene rather than the individual
256x256 tile files - the grid cell boundaries land in the same place either
way, this is just simpler and avoids double bookkeeping.

Usage:
    python grid_cells.py mumbai_predicted_mask_2018.tif grid_2018.csv --cell-size-m 160
"""

import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.spatial import cKDTree


def build_grid(mask_path, out_csv, cell_size_m=160, threshold=0.5):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask = (mask > 0).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        xres = transform.a
        yres = -transform.e  # transform.e is negative (north-up raster)

    cell_px = int(round(cell_size_m / xres))
    h, w = mask.shape
    n_rows = h // cell_px
    n_cols = w // cell_px
    print(f"Scene {h}x{w} px -> {n_rows} x {n_cols} grid cells of {cell_size_m}m "
          f"({cell_px}x{cell_px} px each)")

    rows_out = []
    is_slum_grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    slum_fraction_grid = np.zeros((n_rows, n_cols), dtype=np.float64)
    centers = np.zeros((n_rows, n_cols, 2))

    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = r * cell_px, (r + 1) * cell_px
            c0, c1 = c * cell_px, (c + 1) * cell_px
            slum_fraction = float(mask[r0:r1, c0:c1].mean())
            is_slum = int(slum_fraction > threshold)

            slum_fraction_grid[r, c] = slum_fraction
            is_slum_grid[r, c] = is_slum

            center_row_px = (r0 + r1) / 2
            center_col_px = (c0 + c1) / 2
            center_x, center_y = transform * (center_col_px, center_row_px)
            centers[r, c] = (center_x, center_y)

    # Neighborhood context: fraction of the 8 surrounding cells that are
    # already slum, and distance to the nearest existing slum cell
    # citywide - both computed directly from the grid we just built, no
    # external data needed. These tend to be far more predictive of
    # expansion than any single distance-to-infrastructure feature, since
    # slums overwhelmingly grow by spreading from their own edges.
    slum_cell_coords = centers[is_slum_grid == 1]
    if len(slum_cell_coords) > 0:
        slum_tree = cKDTree(slum_cell_coords)
    else:
        slum_tree = None

    for r in range(n_rows):
        for c in range(n_cols):
            r_lo, r_hi = max(0, r - 1), min(n_rows, r + 2)
            c_lo, c_hi = max(0, c - 1), min(n_cols, c + 2)
            neighborhood = is_slum_grid[r_lo:r_hi, c_lo:c_hi]
            n_neighbors = neighborhood.size - 1  # exclude self
            neighbor_slum_count = neighborhood.sum() - is_slum_grid[r, c]
            neighbor_slum_fraction = neighbor_slum_count / n_neighbors if n_neighbors > 0 else 0.0

            if slum_tree is not None:
                dist_to_nearest_slum, _ = slum_tree.query(centers[r, c])
                # 0 if this cell IS slum itself (nearest slum cell = itself)
                if is_slum_grid[r, c] == 1:
                    dist_to_nearest_slum = 0.0
            else:
                dist_to_nearest_slum = np.nan

            rows_out.append({
                "grid_row": r,
                "grid_col": c,
                "center_x": centers[r, c, 0],
                "center_y": centers[r, c, 1],
                "slum_fraction": round(float(slum_fraction_grid[r, c]), 5),
                "is_slum": int(is_slum_grid[r, c]),
                "neighbor_slum_fraction": round(float(neighbor_slum_fraction), 5),
                "dist_to_nearest_slum_m": round(float(dist_to_nearest_slum), 2),
            })

    df = pd.DataFrame(rows_out)

    # Also compute lat/lon for each cell center, needed for OSM distance lookups
    lons, lats = warp_transform(crs, "EPSG:4326", df["center_x"].tolist(), df["center_y"].tolist())
    df["center_lon"] = lons
    df["center_lat"] = lats

    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} grid cells to {out_csv}")
    print(f"Cells flagged is_slum=1: {df['is_slum'].sum()} / {len(df)} "
          f"({df['is_slum'].mean()*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_path")
    parser.add_argument("out_csv")
    parser.add_argument("--cell-size-m", type=float, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    build_grid(args.mask_path, args.out_csv, args.cell_size_m, args.threshold)
