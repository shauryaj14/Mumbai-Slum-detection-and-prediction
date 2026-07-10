"""
Tile the combined 11-band GeoTIFF (10 satellite bands + 1 mask band) into
256x256 chunks for SegFormer training.

For each tile this:
  - Writes out a standalone georeferenced GeoTIFF (so it can be reopened,
    inspected in QGIS, or re-stitched later using its own transform).
  - Records the tile's origin (ulx, uly), its row/col position in the
    full scene, and its slum-pixel fraction (band 11 mean).
  - Skips tiles that fall outside the valid data area at the right/bottom
    edge (where the scene doesn't divide evenly by 256).

Output: a folder of tiles/tile_r{row}_c{col}.tif plus a tile_index.csv
you can use for the geographic train/val/test split and imbalance-aware
sampling.

Usage:
    python tile_stack.py mumbai_stack.tif tiles_out/ --tile-size 256
"""

import argparse
import csv
import os

import numpy as np
import tifffile

from align_and_overlay import read_geotiff  # reuse the manual GeoTIFF reader


def get_geotiff_crs_tags(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {}
        for code in (34735, 34736, 34737):
            t = page.tags.get(code)
            if t is not None:
                tags[code] = t.value
        return tags


def write_tile(out_path, tile_data, ulx, uly, xres, yres, crs_tags):
    extratags = [
        (33550, "d", 3, (xres, yres, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, ulx, uly, 0.0), False),
    ]
    dtype_map = {34735: "H", 34736: "d", 34737: "s"}
    for code, value in crs_tags.items():
        count = len(value) + 1 if isinstance(value, str) else len(value)
        extratags.append((code, dtype_map[code], count, value, False))

    tifffile.imwrite(
        out_path,
        tile_data,
        photometric="minisblack",
        planarconfig="separate",
        extratags=extratags,
    )


def tile_stack(stack_path, out_dir, tile_size=256, min_valid_fraction=0.9,
               nodata_check_band=0):
    data, transform = read_geotiff(stack_path)
    crs_tags = get_geotiff_crs_tags(stack_path)

    n_bands, h, w = data.shape
    xres, yres = transform["xres"], transform["yres"]
    base_ulx, base_uly = transform["ulx"], transform["uly"]

    os.makedirs(out_dir, exist_ok=True)
    index_rows = []

    n_rows = h // tile_size
    n_cols = w // tile_size
    print(f"Scene {h}x{w} -> {n_rows} x {n_cols} full {tile_size}x{tile_size} tiles "
          f"(dropping partial edge tiles: {h % tile_size} leftover rows, "
          f"{w % tile_size} leftover cols)")

    kept, skipped = 0, 0
    for r in range(n_rows):
        for c in range(n_cols):
            row0 = r * tile_size
            col0 = c * tile_size
            tile = data[:, row0:row0 + tile_size, col0:col0 + tile_size]

            # Skip tiles that are mostly nodata/black (common at scene edges,
            # coastline, or gaps) using band 1 as a proxy.
            valid_fraction = (tile[nodata_check_band] > 0).mean()
            if valid_fraction < min_valid_fraction:
                skipped += 1
                continue

            slum_fraction = float(tile[-1].mean())  # band 11 = mask
            tile_ulx = base_ulx + col0 * xres
            tile_uly = base_uly - row0 * yres

            fname = f"tile_r{r:03d}_c{c:03d}.tif"
            out_path = os.path.join(out_dir, fname)
            write_tile(out_path, tile, tile_ulx, tile_uly, xres, yres, crs_tags)

            index_rows.append({
                "filename": fname,
                "row": r,
                "col": c,
                "ulx": tile_ulx,
                "uly": tile_uly,
                "slum_fraction": round(slum_fraction, 6),
                "valid_fraction": round(float(valid_fraction), 4),
            })
            kept += 1

    index_path = os.path.join(out_dir, "tile_index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Kept {kept} tiles, skipped {skipped} (mostly-nodata) tiles.")
    print(f"Wrote tile index to {index_path}")

    fractions = np.array([row["slum_fraction"] for row in index_rows])
    print(f"Slum-fraction stats across tiles: "
          f"mean={fractions.mean():.4f}, "
          f"tiles with >0 slum pixels={(fractions > 0).sum()}/{len(fractions)}, "
          f"tiles with >50% slum={(fractions > 0.5).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stack_path")
    parser.add_argument("out_dir")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--min-valid-fraction", type=float, default=0.9,
                         help="Skip tiles with less than this fraction of non-zero band-1 pixels")
    args = parser.parse_args()

    tile_stack(args.stack_path, args.out_dir, args.tile_size, args.min_valid_fraction)
