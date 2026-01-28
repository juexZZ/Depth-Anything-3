# Semantic segment embedding visualization.
# 1. Load the semantic segment embedding (H, W, D).
# 2. Assign a consistent color to identical embeddings for a segment mask view.

import argparse
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_IMAGE_PATH = "/local_data/jz4725/metacam/data_v3/8thfloor_add/images/left_1769336983165570000.png"
DEFAULT_FEATURE_PATH = "/local_data/jz4725/metacam/data_v3/8thfloor_add/semantic_emb/left_1769336983165570000.npz"


def _load_feature_array(feature_path: str) -> np.ndarray:
    with np.load(feature_path, allow_pickle=True) as data:
        feat = data["embedding"]
    feat = np.asarray(feat)
    while feat.ndim > 3 and 1 in feat.shape:
        feat = np.squeeze(feat, axis=tuple(i for i, s in enumerate(feat.shape) if s == 1))
    if feat.ndim != 3:
        raise ValueError(f"Expected feature shape (H, W, D), got {feat.shape}")
    return feat


def _colorize_by_embedding(
    feat: np.ndarray,
    quantize_decimals: int = 5,
    min_count: int = 1,
    outlier_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    h, w, d = feat.shape
    flat = feat.reshape(-1, d).astype(np.float32)
    valid = np.isfinite(flat).all(axis=1)
    if not np.any(valid):
        raise ValueError("No finite embeddings found in feature array.")
    quantized = np.round(flat, decimals=quantize_decimals)
    uniq, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)

    # Deterministic color per unique embedding using a hash-based HSV mapping.
    # Use a fixed-seed random projection to handle high-dimensional embeddings.
    rng = np.random.default_rng(0)
    weight = rng.standard_normal(uniq.shape[1]).astype(np.float32)
    hashed = np.abs(uniq) @ weight
    hue = (hashed * 0.61803398875) % 1.0  # golden ratio for spread
    sat = np.full_like(hue, 0.75, dtype=np.float32)
    val = np.full_like(hue, 0.95, dtype=np.float32)

    i = np.floor(hue * 6.0).astype(np.int32)
    f = (hue * 6.0) - i
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    i_mod = i % 6
    rgb = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    idx = i_mod == 0
    rgb[idx] = np.stack([val[idx], t[idx], p[idx]], axis=1)
    idx = i_mod == 1
    rgb[idx] = np.stack([q[idx], val[idx], p[idx]], axis=1)
    idx = i_mod == 2
    rgb[idx] = np.stack([p[idx], val[idx], t[idx]], axis=1)
    idx = i_mod == 3
    rgb[idx] = np.stack([p[idx], q[idx], val[idx]], axis=1)
    idx = i_mod == 4
    rgb[idx] = np.stack([t[idx], p[idx], val[idx]], axis=1)
    idx = i_mod == 5
    rgb[idx] = np.stack([val[idx], p[idx], q[idx]], axis=1)

    if min_count > 1:
        low_count = counts < min_count
        rgb[low_count] = np.array(outlier_color, dtype=np.float32)

    out = np.zeros((flat.shape[0], 3), dtype=np.float32)
    out[valid] = rgb[inverse[valid]]
    return out.reshape(h, w, 3)


def visualize_semantic_embeddings(
    image_path: str,
    feature_path: str,
    output_path: str,
    quantize_decimals: int = 5,
    min_count: int = 1,
    show: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(image_path).convert("RGB"))
    feat = _load_feature_array(feature_path)
    mask_rgb = _colorize_by_embedding(
        feat,
        quantize_decimals=quantize_decimals,
        min_count=min_count,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(mask_rgb, interpolation="nearest")
    axes[1].set_title("Mask")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return image, mask_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic segment embeddings.")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--feature_path", type=str, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output_path", type=str, default="semantic_embedding_viz.png")
    parser.add_argument("--quantize_decimals", type=int, default=5, help="Decimals for embedding quantization.")
    parser.add_argument("--min_count", type=int, default=1, help="Min pixels per segment to keep color.")
    parser.add_argument("--show", action="store_true", help="Display visualization window.")
    args = parser.parse_args()

    visualize_semantic_embeddings(
        image_path=args.image_path,
        feature_path=args.feature_path,
        output_path=args.output_path,
        quantize_decimals=args.quantize_decimals,
        min_count=args.min_count,
        show=args.show,
    )
