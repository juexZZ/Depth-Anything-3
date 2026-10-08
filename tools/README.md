# Local reconstruction utilities

Run these tools from the repository root using the DA3 environment.

- `relative_pose.py`: infer the relative camera pose between two images. Pass `--image_dir`; the default `test_images/` contains local inputs excluded from Git.
- `run_metric_depth.py`: infer overlapping chunks, align their scale and poses, and export RGB-D, camera poses and intrinsics. See `--help` for model and export options.
- `view_ply.py`: display a point cloud with Viser (`--pcd_path`).
- `helper.sh`: original archive preparation recipe. It contains machine-specific source paths; review and edit before running.

The baseline geometry batch entry point remains `basic_usage.py`. The DejaView repository owns the current semantic voxel fusion and query evaluation code.
