import argparse
import gc
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def collect_images(image_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def to_4x4(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = extrinsic
        return out
    raise ValueError(f"Unexpected extrinsic shape {extrinsic.shape}")


def intrinsic_to_4x4(intrinsic):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = intrinsic[:3, :3]
    return out


def write_matrix(path, mat):
    with open(path, "w") as f:
        for row in mat:
            f.write(" ".join(f"{v:.10f}" for v in row) + "\n")


def plan_chunks(n_frames, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    if n_frames <= chunk_size:
        return [(0, n_frames)]

    stride = chunk_size - overlap
    ranges = []
    start = 0
    while True:
        end = min(start + chunk_size, n_frames)
        ranges.append((start, end))
        if end >= n_frames:
            break
        start += stride
    return ranges


def scale_c2w(c2w, s):
    """Scale only the translation part of c2w by s. Rotations are scale-invariant."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[..., :3, 3] = float(s) * out[..., :3, 3]
    return out


def estimate_chunk_scale_ratio(
    d_prev,
    d_curr,
    conf_prev=None,
    conf_curr=None,
    min_depth=0.05,
    max_depth=100.0,
    conf_quantile=0.5,
):
    """
    Robust per-chunk depth scale ratio.

    Given matching overlap-frame depths from two chunks (each in its own local scale),
    return r such that r * d_curr ≈ d_prev, computed as the median pixel-wise ratio
    over all valid overlap pixels.

    Returns: (ratio, n_valid_pixels)
    """
    d_prev = np.asarray(d_prev, dtype=np.float64).reshape(-1)
    d_curr = np.asarray(d_curr, dtype=np.float64).reshape(-1)

    valid = (
        np.isfinite(d_prev)
        & np.isfinite(d_curr)
        & (d_prev > min_depth)
        & (d_curr > min_depth)
        & (d_prev < max_depth)
        & (d_curr < max_depth)
    )

    if conf_prev is not None and conf_curr is not None:
        c_min = np.minimum(
            np.asarray(conf_prev, dtype=np.float64).reshape(-1),
            np.asarray(conf_curr, dtype=np.float64).reshape(-1),
        )
        if valid.any():
            thresh = float(np.quantile(c_min[valid], conf_quantile))
            valid &= c_min >= thresh

    n_valid = int(valid.sum())
    if n_valid < 100:
        return 1.0, n_valid

    ratio = d_prev[valid] / d_curr[valid]
    return float(np.median(ratio)), n_valid


def estimate_se3_alignment(c2w_src, c2w_tgt, axis_scale=0.1):
    """
    Solve for rigid T (4x4) such that T @ c2w_src[i] ≈ c2w_tgt[i].
    Augments each pose into 4 anchor points (camera center + 3 axis tips)
    so both rotation and translation contribute to the Procrustes solve.
    """
    c2w_src = np.asarray(c2w_src, dtype=np.float64)
    c2w_tgt = np.asarray(c2w_tgt, dtype=np.float64)
    if c2w_src.shape != c2w_tgt.shape or c2w_src.ndim != 3 or c2w_src.shape[-2:] != (4, 4):
        raise ValueError(f"Bad pose array shapes: {c2w_src.shape}, {c2w_tgt.shape}")

    n = c2w_src.shape[0]
    R_s = c2w_src[:, :3, :3]
    t_s = c2w_src[:, :3, 3]
    R_t = c2w_tgt[:, :3, :3]
    t_t = c2w_tgt[:, :3, 3]

    eye = np.eye(3) * float(axis_scale)
    src_pts = np.empty((n * 4, 3), dtype=np.float64)
    tgt_pts = np.empty((n * 4, 3), dtype=np.float64)
    src_pts[0::4] = t_s
    tgt_pts[0::4] = t_t
    for j in range(3):
        src_pts[(j + 1)::4] = t_s + R_s @ eye[:, j]
        tgt_pts[(j + 1)::4] = t_t + R_t @ eye[:, j]

    src_mean = src_pts.mean(0)
    tgt_mean = tgt_pts.mean(0)
    src_c = src_pts - src_mean
    tgt_c = tgt_pts - tgt_mean

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    t = tgt_mean - R @ src_mean

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def chunked_inference(model, image_paths, chunk_size, overlap, log=print):
    n = len(image_paths)
    ranges = plan_chunks(n, chunk_size, overlap)

    chunks = []
    for k, (s, e) in enumerate(ranges):
        log(f"[chunk {k+1}/{len(ranges)}] frames [{s}, {e})  size={e-s}")
        with torch.no_grad():
            pred = model.inference(image_paths[s:e])
            # breakpoint()
        if pred.intrinsics is None or pred.extrinsics is None:
            raise RuntimeError("Model did not return intrinsics/extrinsics.")
        if not getattr(pred, "is_metric", 0):
            log("  warning: chunk prediction is_metric=False")

        n_c = pred.depth.shape[0]
        c2w_local = np.empty((n_c, 4, 4), dtype=np.float64)
        for i in range(n_c):
            c2w_local[i] = np.linalg.inv(to_4x4(pred.extrinsics[i]))

        chunks.append({
            "start": s,
            "end": e,
            "depth": np.asarray(pred.depth),
            "conf": np.asarray(pred.conf) if pred.conf is not None else None,
            "intrinsics": np.asarray(pred.intrinsics),
            "processed_images": np.asarray(pred.processed_images),
            "c2w_local": c2w_local,
        })

        del pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return chunks


def stitch_chunks(
    chunks,
    n_total,
    axis_scale=0.1,
    min_depth=0.05,
    max_depth=100.0,
    conf_quantile=0.5,
    log=print,
):
    """
    Iteratively align chunk k to chunk k-1 using overlap frames.
      - Scale: depth-driven. r_k = median(d_{k-1}_local / d_k_local) over overlap pixels.
               Cumulative s_k = s_{k-1} * r_k, applied to chunk k's depths and to its c2w
               translations (rotations are scale-invariant).
      - Rigid offset: SE(3) Procrustes on the now-scale-correct overlap c2w poses,
                     so all chunk origins are glued into chunk-0's frame.

    First-chunk-wins for overlapping frames in the final output.
    Returns per-frame arrays in chunk-0's world frame, with metric depth.
    """
    depths = [None] * n_total
    confs = [None] * n_total
    intrinsics = [None] * n_total
    c2ws_global = [None] * n_total
    images = [None] * n_total
    sources = [-1] * n_total

    chunk_scales = []     # cumulative scale per chunk
    chunk_se3 = []        # 4x4 rigid R|t per chunk (applied AFTER translation scaling)
    chunk_ratios = []     # incremental r_k between chunk k-1 and chunk k (1.0 for k=0)

    for k, ch in enumerate(chunks):
        s, e = ch["start"], ch["end"]
        n_c = ch["depth"].shape[0]

        if k == 0:
            s_k_cum = 1.0
            r_k = 1.0
            T_k = np.eye(4, dtype=np.float64)
            log(f"[chunk 0] frames [{s},{e})  scale=1.0 (reference)")
        else:
            prev = chunks[k - 1]
            ov_lo = s
            ov_hi = min(prev["end"], e)
            n_ov = ov_hi - ov_lo
            if n_ov < 1:
                raise RuntimeError(
                    f"chunk {k} has no overlap with chunk {k-1}: "
                    f"prev=[{prev['start']},{prev['end']}), curr=[{s},{e})"
                )

            prev_idx = np.arange(ov_lo, ov_hi) - prev["start"]
            curr_idx = np.arange(ov_lo, ov_hi) - s

            d_prev_local = prev["depth"][prev_idx]
            d_curr_local = ch["depth"][curr_idx]
            c_prev = prev["conf"][prev_idx] if prev["conf"] is not None else None
            c_curr = ch["conf"][curr_idx] if ch["conf"] is not None else None

            # 1. Depth-ratio scale (median over all valid overlap pixels)
            r_k, n_valid = estimate_chunk_scale_ratio(
                d_prev_local, d_curr_local, c_prev, c_curr,
                min_depth=min_depth, max_depth=max_depth, conf_quantile=conf_quantile,
            )
            s_k_cum = chunk_scales[k - 1] * r_k

            # 2. Build prev's overlap poses in global frame (already aligned in prior iters)
            T_prev = chunk_se3[k - 1]
            s_prev = chunk_scales[k - 1]
            prev_c2w_local_overlap = prev["c2w_local"][prev_idx]
            prev_c2w_global_overlap = np.einsum(
                "ij,njk->nik", T_prev, scale_c2w(prev_c2w_local_overlap, s_prev)
            )

            # 3. Build curr's overlap poses with scaled translations
            curr_c2w_local_overlap = ch["c2w_local"][curr_idx]
            curr_c2w_scaled_overlap = scale_c2w(curr_c2w_local_overlap, s_k_cum)

            # 4. Rigid Procrustes to glue chunk k's origin into chunk k-1's global frame
            T_k = estimate_se3_alignment(
                curr_c2w_scaled_overlap, prev_c2w_global_overlap, axis_scale=axis_scale
            )

            # Diagnostics
            aligned = np.einsum("ij,njk->nik", T_k, curr_c2w_scaled_overlap)
            t_err = np.linalg.norm(
                aligned[:, :3, 3] - prev_c2w_global_overlap[:, :3, 3], axis=1
            )
            log(
                f"[chunk {k}] overlap={n_ov}  r_k={r_k:.6f}  s_k_cum={s_k_cum:.6f}  "
                f"valid_px={n_valid}  trans_err mean={t_err.mean():.4f}m max={t_err.max():.4f}m"
            )

        chunk_ratios.append(r_k)
        chunk_scales.append(s_k_cum)
        chunk_se3.append(T_k)

        # Apply (s_k_cum, T_k) to all of chunk k's frames (first-chunk-wins on overlap)
        ch_c2w_scaled = scale_c2w(ch["c2w_local"], s_k_cum)
        ch_c2w_global = np.einsum("ij,njk->nik", T_k, ch_c2w_scaled)

        for i in range(n_c):
            f = s + i
            if c2ws_global[f] is not None:
                continue
            c2ws_global[f] = ch_c2w_global[i]
            depths[f] = s_k_cum * ch["depth"][i]
            confs[f] = ch["conf"][i] if ch["conf"] is not None else None
            intrinsics[f] = ch["intrinsics"][i]
            images[f] = ch["processed_images"][i]
            sources[f] = k

    if any(c is None for c in c2ws_global):
        missing = [i for i, c in enumerate(c2ws_global) if c is None]
        raise RuntimeError(f"Missing frames after stitching: {missing[:10]}...")

    return {
        "depths": depths,
        "confs": confs,
        "intrinsics": intrinsics,
        "c2ws": c2ws_global,
        "images": images,
        "sources": sources,
        "chunk_scales": chunk_scales,
        "chunk_ratios": chunk_ratios,
        "chunk_se3": chunk_se3,
    }


def export_scene(stitched, output_dir, image_paths, chunks, depth_scale=1000.0, max_uint16=65535):
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    pose_dir = os.path.join(output_dir, "pose")
    intrinsic_dir = os.path.join(output_dir, "intrinsic")
    for d in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        os.makedirs(d, exist_ok=True)

    images = stitched["images"]
    depths = stitched["depths"]
    intrinsics = stitched["intrinsics"]
    c2ws = stitched["c2ws"]
    sources = stitched["sources"]

    n_frames = len(depths)
    for i in range(n_frames):
        Image.fromarray(images[i]).save(os.path.join(color_dir, f"{i}.jpg"), quality=95)

        depth_m = depths[i].astype(np.float32)
        depth_mm = depth_m * depth_scale
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0, max_uint16)
        depth_u16 = depth_mm.astype(np.uint16)
        Image.fromarray(depth_u16).save(os.path.join(depth_dir, f"{i}.png"))

        write_matrix(os.path.join(pose_dir, f"{i}.txt"), c2ws[i])

    K = intrinsic_to_4x4(intrinsics[0])
    write_matrix(os.path.join(intrinsic_dir, "intrinsic_color.txt"), K)
    write_matrix(os.path.join(intrinsic_dir, "intrinsic_depth.txt"), K)

    mapping = [
        {
            "index": i,
            "original": os.path.basename(image_paths[i]),
            "source_chunk": int(sources[i]),
        }
        for i in range(n_frames)
    ]
    with open(os.path.join(output_dir, "filename_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    with open(os.path.join(output_dir, "filename_mapping.txt"), "w") as f:
        f.write("# index\toriginal_filename\tsource_chunk\n")
        for entry in mapping:
            f.write(f"{entry['index']}\t{entry['original']}\t{entry['source_chunk']}\n")

    chunk_meta = {
        "num_chunks": len(chunks),
        "chunks": [
            {
                "index": k,
                "start": int(c["start"]),
                "end": int(c["end"]),
                "incremental_depth_ratio": float(stitched["chunk_ratios"][k]),
                "cumulative_scale": float(stitched["chunk_scales"][k]),
                "rigid_T_se3": stitched["chunk_se3"][k].tolist(),
            }
            for k, c in enumerate(chunks)
        ],
        "note": (
            "global_depth = cumulative_scale * local_depth.  "
            "global_c2w = rigid_T_se3 @ scale_translation(local_c2w, cumulative_scale)."
        ),
    }
    with open(os.path.join(output_dir, "chunk_alignment.json"), "w") as f:
        json.dump(chunk_meta, f, indent=2)

    return n_frames


def run(
    image_dir,
    output_root,
    model_name,
    scene_name=None,
    depth_scale=1000.0,
    chunk_size=64,
    overlap=8,
    axis_scale=0.1,
    min_depth=0.05,
    max_depth=100.0,
    conf_quantile=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # breakpoint()
    model = DepthAnything3.from_pretrained(model_name).to(device=device)
    model.eval()

    image_paths = collect_images(image_dir)
    n = len(image_paths)
    print(f"Found {n} images. chunk_size={chunk_size}, overlap={overlap}.")

    if scene_name is None:
        scene_name = os.path.basename(os.path.normpath(image_dir))
    scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    chunks = chunked_inference(model, image_paths, chunk_size, overlap)

    # Free model before stitching to reclaim VRAM during heavy numpy work.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stitched = stitch_chunks(
        chunks,
        n,
        axis_scale=axis_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        conf_quantile=conf_quantile,
    )
    n_written = export_scene(
        stitched, scene_dir, image_paths, chunks, depth_scale=depth_scale
    )
    print(f"Wrote {n_written} frames to {scene_dir}")
    print(f"  color/  depth/  pose/  intrinsic/  filename_mapping.{{json,txt}}  chunk_alignment.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Metric monocular depth with DA3, processed in overlapping chunks "
            "and iteratively SE(3)-aligned."
        )
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Folder of RGB images.")
    parser.add_argument(
        "--output_root", type=str, default="outputs", help="Parent folder for the scene."
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Scene folder name. Defaults to the input folder's basename.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help=(
            "Pretrained DA3 model id. Must have pose+intrinsic heads. "
            "DA3NESTED-GIANT-LARGE = metric depth + poses + intrinsics in one model. "
            "(da3metric-large is depth-only and will fail here.)"
        ),
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="depth_uint16 = depth_meters * scale (default 1000 → mm).",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=64, help="Frames per chunk for inference."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=8,
        help="Number of overlapping frames between consecutive chunks (used for alignment).",
    )
    parser.add_argument(
        "--axis_scale",
        type=float,
        default=0.1,
        help="Length (meters) of camera-axis tips used for pose Procrustes alignment.",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.05,
        help="Pixels with depth (in meters) below this are excluded from scale ratio.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=100.0,
        help="Pixels with depth above this are excluded from scale ratio. It does not exclude pixels with depth below min_depth, only for estimating the scale ratio.",
    )
    parser.add_argument(
        "--conf_quantile",
        type=float,
        default=0.5,
        help=(
            "When confidence maps are available, keep only pixels whose min-conf "
            "is above this quantile (0 = use all valid pixels)."
        ),
    )
    args = parser.parse_args()

    run(
        args.image_dir,
        args.output_root,
        args.model_name,
        scene_name=args.scene_name,
        depth_scale=args.depth_scale,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        axis_scale=args.axis_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        conf_quantile=args.conf_quantile,
    )
