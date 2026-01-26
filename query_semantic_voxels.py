import argparse
import os
import sys

import torch
from transformers import CLIPProcessor, CLIPModel

# Allow running from repo root.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from da3_streaming.semantic_voxels.semantic_voxel import SemanticVoxelMap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query semantic voxel map with a clip embedding vector.")
    parser.add_argument("--voxel_map_dir", type=str, required=True, help="Directory containing semantic voxel map.")
    parser.add_argument("--query_prompt", type=str, default="", help="Text query.")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k voxels.")
    parser.add_argument("--port", type=int, default=8081, help="Viser port.")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization.")
    parser.add_argument("--vis_pcd", action="store_true", help="Visualize the retrieval results in the original point cloud (ply file).")
    parser.add_argument("--pcd_path", type=str, default=None, help="Path to the pcd file to visualize the retrieval results.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images to retrieve.")
    parser.add_argument("--output_dir", type=str, default="retrieval_results", help="Directory to save results.")
    args = parser.parse_args()

    voxel_map = SemanticVoxelMap.load_from_directory(args.voxel_map_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    clip_model.to(device)
    clip_model.eval()

    inputs = clip_processor(text=args.query_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    topk_indices, topk_coords, topk_similarities = voxel_map.query_with_embedding(
        text_embedding.cpu().numpy(), top_k=args.top_k
    )

    print("Query:", args.query_prompt)
    save_dir = os.path.join(args.output_dir, "query_" + "_".join(args.query_prompt.split(" ")) + "_results")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(topk_indices)):
        voxel_index = topk_indices[i]
        voxel_coord = topk_coords[i]
        similarity = topk_similarities[i]
        resolved = voxel_map.get_latest_frame_at_voxel(voxel_index)
        print(f"Top {i+1}: voxel={voxel_index} coord={voxel_coord} sim={similarity}")
        if resolved is not None:
            frame_name, submap_id, frame_id = resolved
            print(f"  frame={frame_name} submap={submap_id} frame_id={frame_id}")
            if args.image_dir is not None:
                from PIL import Image
                import matplotlib.pyplot as plt

                image_path = os.path.join(args.image_dir, frame_name)
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    plt.imshow(img)
                    plt.axis("off")
                    save_path = os.path.join(save_dir, f"top{i+1}_retrieval_result.png")
                    plt.savefig(save_path)
                    plt.close()

    if not args.no_viz:
        if args.vis_pcd:
            if args.pcd_path is not None:
                import open3d as o3d
                import viser
                import numpy as np
                pcd = o3d.io.read_point_cloud(args.pcd_path)
                pts = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                server = viser.ViserServer(host="0.0.0.0", port=args.port)
                handle = server.scene.add_point_cloud(
                    name="pcd",
                    points=pts,
                    colors=colors,
                    point_size=0.01,
                )
                for i in range(len(topk_indices)):
                    voxel_index = topk_indices[i]
                    voxel_coord = topk_coords[i]
                    voxel_center = (voxel_coord[0] + 0.5) * voxel_map.voxel_size, (voxel_coord[1] + 0.5) * voxel_map.voxel_size, (voxel_coord[2] + 0.5) * voxel_map.voxel_size
                    server.scene.add_box(
                        name=f"query_voxel_{i}",
                        position=voxel_center,
                        dimensions=(voxel_map.voxel_size*5, voxel_map.voxel_size*5, voxel_map.voxel_size*5),
                        color=(1.0, 0.0, 0.0),
                        wireframe=False,
                        opacity=1.0,
                    )
                # keep server alive
                print(f"Viser server running on port {args.port}. Press Enter to exit...")
                try:
                    input()
                except KeyboardInterrupt:
                    pass
            else:
                print("pcd path is not provided, please provide the pcd path")
        else:
            voxel_map.visualize(
                port=args.port,
                render_mode="points",
                color_mode="query",
                max_voxels=None,
                base_color=(0.75, 0.75, 0.75),
                highlight_color=(1.0, 0.0, 0.0),
                wireframe=False,
                opacity=0.5,
                keep_alive=True,
                query_voxel_indices=topk_indices,
            )