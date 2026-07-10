"""
Align and overlay a Sentinel-2 satellite image with a (mis-registered) mask,
using only tifffile + numpy (no GDAL/rasterio available offline).

Assumes:
  - Both rasters share the same pixel size (10m) and same array shape.
  - The mismatch is a pure translation (different origin/tiepoint), not a
    rotation or scale difference. This is the normal case for two tiles
    clipped from the same larger scene with different bounding boxes.

Usage:
    python align_and_overlay.py Mumbai.tif Mumbai_ground_truth.tif
"""

import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Minimal GeoTIFF reader: pulls the array + affine transform manually
#    from the ModelPixelScaleTag (33550) and ModelTiepointTag (33922).
# ----------------------------------------------------------------------
def read_geotiff(path):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        data = page.asarray()

        # Normalize to (bands, rows, cols)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3 and data.shape[2] < data.shape[0]:
            # looks like (rows, cols, bands) -> move bands first
            data = np.moveaxis(data, -1, 0)

        tags = page.tags
        pixel_scale = tags.get(33550)
        tiepoint = tags.get(33922)

        transform = None
        if pixel_scale is not None and tiepoint is not None:
            sx, sy, _sz = pixel_scale.value
            i, j, _k, x, y, _z = tiepoint.value[:6]
            # Pixel (i, j) maps to model coords (x, y). Build origin of
            # pixel (0,0), i.e. the upper-left corner:
            ulx = x - i * sx
            uly = y + j * sy
            transform = dict(ulx=ulx, uly=uly, xres=sx, yres=sy)

        return data, transform


def describe(name, data, transform):
    print(f"\n--- {name} ---")
    print(f"shape (bands, rows, cols): {data.shape}, dtype: {data.dtype}")
    if transform:
        print(f"origin (ulx, uly): ({transform['ulx']:.2f}, {transform['uly']:.2f})")
        print(f"pixel size (xres, yres): ({transform['xres']}, {transform['yres']})")
    else:
        print("WARNING: no GeoTIFF georeferencing tags found in this file.")
    for b in range(data.shape[0]):
        band = data[b].astype(np.float64)
        finite = band[np.isfinite(band)]
        print(f"  band {b+1}: min={finite.min():.1f} max={finite.max():.1f} "
              f"p2={np.percentile(finite,2):.1f} p98={np.percentile(finite,98):.1f}")


# ----------------------------------------------------------------------
# 2. Compute pixel offset between two rasters from their geotransforms
#    and shift/crop to the overlapping region.
# ----------------------------------------------------------------------
def align_by_offset(img, img_t, mask, mask_t):
    if img_t is None or mask_t is None:
        raise ValueError(
            "Missing georeferencing on one of the files — can't compute "
            "offset automatically. Send me the tag dump above and I'll "
            "adjust the reader."
        )

    xres = img_t["xres"]
    yres = img_t["yres"]

    # How many pixels is the mask's origin shifted from the image's origin?
    dx = round((mask_t["ulx"] - img_t["ulx"]) / xres)
    dy = round((img_t["uly"] - mask_t["uly"]) / yres)

    print(f"\nComputed offset (mask relative to image): dx={dx} px, dy={dy} px")

    # Shift the mask onto the image's grid, then crop both to the
    # common overlapping window.
    h, w = img.shape[1], img.shape[2]

    # Build slices for image and mask that correspond to the same
    # ground area.
    img_row0, img_row1 = max(0, dy), min(h, h + dy)
    img_col0, img_col1 = max(0, dx), min(w, w + dx)
    msk_row0, msk_row1 = max(0, -dy), min(h, h - dy)
    msk_col0, msk_col1 = max(0, -dx), min(w, w - dx)

    img_crop = img[:, img_row0:img_row1, img_col0:img_col1]
    mask_crop = mask[:, msk_row0:msk_row1, msk_col0:msk_col1]

    print(f"Aligned overlap size: {img_crop.shape[1]} x {img_crop.shape[2]} "
          f"(cropped from {h} x {w})")

    return img_crop, mask_crop


# ----------------------------------------------------------------------
# 3. Percentile stretch (fixes all-white / washed-out composites caused
#    by outlier bright pixels in raw min-max stretch).
# ----------------------------------------------------------------------
def percentile_stretch(band, low=2, high=98):
    band = band.astype(np.float64)
    lo, hi = np.percentile(band, [low, high])
    if hi <= lo:
        return np.zeros_like(band)
    out = (band - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def make_rgb(data, r_idx, g_idx, b_idx, low=2, high=98):
    """
    data: (bands, rows, cols) array, 0-indexed band selection.
    Set r_idx/g_idx/b_idx to match your actual band order once you've
    inspected the per-band stats printed by describe().
    """
    r = percentile_stretch(data[r_idx], low, high)
    g = percentile_stretch(data[g_idx], low, high)
    b = percentile_stretch(data[b_idx], low, high)
    return np.dstack([r, g, b])


# ----------------------------------------------------------------------
# 4. Overlay mask on top of the RGB composite
# ----------------------------------------------------------------------
def overlay_mask(rgb, mask_band, color=(1, 0, 0), alpha=0.45, threshold=0):
    """
    mask_band: 2D array, nonzero (or > threshold) = slum pixel.
    """
    out = rgb.copy()
    m = mask_band > threshold
    for c in range(3):
        out[..., c] = np.where(m, out[..., c] * (1 - alpha) + color[c] * alpha, out[..., c])
    return out


def main(img_path, mask_path,
         r_idx=3, g_idx=2, b_idx=1,   # GUESS: true-color-ish default, adjust after inspecting stats
         out_path="/mnt/user-data/outputs/mumbai_overlay.png"):

    img, img_t = read_geotiff(img_path)
    mask, mask_t = read_geotiff(mask_path)

    describe("Mumbai.tif (satellite image)", img, img_t)
    describe("Mumbai_ground_truth.tif (mask)", mask, mask_t)

    img_c, mask_c = align_by_offset(img, img_t, mask, mask_t)

    rgb = make_rgb(img_c, r_idx, g_idx, b_idx)
    overlay = overlay_mask(rgb, mask_c[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite (stretched composite)")
    axes[1].imshow(mask_c[0], cmap="gray")
    axes[1].set_title("Ground truth mask (aligned)")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved overlay figure to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python align_and_overlay.py <image.tif> <mask.tif>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
