"""
Run the trained SegFormer model on a full Sentinel-2 scene (any size) and
write out a predicted slum mask as a GeoTIFF, matching the input's exact
georeferencing.

Since the model was trained on 256x256 tiles, this slides a non-overlapping
256x256 window across the full scene, predicts each patch, and stitches the
results back together. Edge patches smaller than 256x256 are zero-padded
before prediction and cropped back afterward.

Usage:
    python predict_mask.py mumbai_sentinel2_2016_v3.tif ../models/segformer_mumbai_2019.pt predicted_mask_2016.tif
    python predict_mask.py ../data/raw/Mumbai.tif ../models/segformer_mumbai_2019.pt predicted_mask_2019.tif
"""

import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    old_conv = model.segformer.stages[0].patch_embeddings.proj
    new_conv = torch.nn.Conv2d(
        in_channels=10,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    model.segformer.stages[0].patch_embeddings.proj = new_conv
    return model


def predict_scene(image_path, weights_path, out_path, tile_size=256, device_str="cpu"):
    device = torch.device(device_str)

    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model weights from {weights_path}")

    with rasterio.open(image_path) as src:
        image = src.read()  # (10, H, W)
        profile = src.profile.copy()

    image = np.nan_to_num(image, nan=0.0)
    image = np.clip(image.astype(np.float32) / 10000.0, 0, 1)

    n_bands, h, w = image.shape
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    n_rows = (h + tile_size - 1) // tile_size
    n_cols = (w + tile_size - 1) // tile_size
    print(f"Scene {h}x{w} -> {n_rows} x {n_cols} tiles to predict")

    with torch.no_grad():
        for r in range(n_rows):
            for c in range(n_cols):
                row0, col0 = r * tile_size, c * tile_size
                row1, col1 = min(row0 + tile_size, h), min(col0 + tile_size, w)

                patch = np.zeros((n_bands, tile_size, tile_size), dtype=np.float32)
                patch[:, : row1 - row0, : col1 - col0] = image[:, row0:row1, col0:col1]

                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1, 10, 256, 256)
                outputs = model(pixel_values=patch_tensor)
                logits = F.interpolate(outputs.logits, size=(tile_size, tile_size),
                                        mode="bilinear", align_corners=False)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (256, 256)

                pred_mask[row0:row1, col0:col1] = pred[: row1 - row0, : col1 - col0]

    slum_fraction = (pred_mask > 0).mean()
    print(f"Predicted slum pixel fraction: {slum_fraction:.4f}")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(pred_mask[np.newaxis, ...])
    print(f"Wrote predicted mask to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python predict_mask.py <image.tif> <model_weights.pt> <out_predicted_mask.tif>")
        sys.exit(1)
    predict_scene(sys.argv[1], sys.argv[2], sys.argv[3])
