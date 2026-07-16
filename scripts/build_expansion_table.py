"""
Joins a "before" year's grid and an "after" year's grid (matched by
grid_row/grid_col, which is the same for every year since they all share
the same raster grid) to produce an expansion label per cell, then attaches
the static OSM distance features (computed once, from any single year's
grid_with_features.csv - the geometry doesn't change year to year).

Usage:
    python build_expansion_table.py grid_2016.csv grid_2019.csv grid_2019_with_features.csv training_table_2016_2019.csv
"""

import sys

import pandas as pd


def main(before_csv, after_csv, features_csv, out_csv):
    before = pd.read_csv(before_csv)[
        ["grid_row", "grid_col", "slum_fraction", "is_slum", "neighbor_slum_fraction", "dist_to_nearest_slum_m"]
    ]
    before = before.rename(columns={"slum_fraction": "slum_fraction_before", "is_slum": "was_slum"})

    after = pd.read_csv(after_csv)[["grid_row", "grid_col", "slum_fraction", "is_slum"]]
    after = after.rename(columns={"slum_fraction": "slum_fraction_after", "is_slum": "is_slum_after"})

    features = pd.read_csv(features_csv)
    feature_cols = [c for c in features.columns if c.startswith("dist_to_")]
    features = features[["grid_row", "grid_col"] + feature_cols]

    merged = before.merge(after, on=["grid_row", "grid_col"])
    merged = merged.merge(features, on=["grid_row", "grid_col"])

    merged["expanded"] = ((merged["is_slum_after"] == 1) & (merged["was_slum"] == 0)).astype(int)
    merged["unexpected_shrink"] = ((merged["was_slum"] == 1) & (merged["is_slum_after"] == 0)).astype(int)

    merged.to_csv(out_csv, index=False)

    print(f"Wrote {len(merged)} cells to {out_csv}")
    print(f"Cells that expanded: {merged['expanded'].sum()}")
    print(f"Cells with unexpected shrink (flag for review): {merged['unexpected_shrink'].sum()}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python build_expansion_table.py <before_grid.csv> <after_grid.csv> <features.csv> <out_table.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
