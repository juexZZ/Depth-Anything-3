# Validated DejaView baseline runtime (2026-10-23)

The Basic installation now declares the dependencies actually imported by the
source: `addict==2.4.0` and `scikit-learn==1.7.2`. The former is the attribute
dictionary used by model outputs/registries; the latter is used by reconstruction
utilities. Neither introduces another neural model.

Torch 2.9.1, torchvision 0.24.1 and xFormers 0.0.33.post2 are pinned together.
The formerly unconstrained install selected Torch 2.14.1/cu130 while the installed
xFormers extension had been compiled for Torch 2.10/cu128. Merely passing
`pip check` did not detect this C++ ABI mismatch. The pinned combination was
verified by actually running `xformers.ops.memory_efficient_attention` on H200,
not by suppressing its warning or relying on another attention implementation.
The existing `numpy<2` constraint remains unchanged.

Use the README's updated Basic installation. `gsplat` is optional and needed
only for Gaussian rendering, not the depth/pose/point-cloud baseline. Its
optional-module warning alone is not a failed depth reconstruction.

Validation used the README default DA3NESTED-GIANT-LARGE model at HF revision
`8615eefb62f2db4f8d6ebaa59160086981672829`, with weight SHA-256
`8899faf998dedbc230261ab736fa57015280727399429122d44d4f9e7aac2ddd`.
The unmodified `basic_usage.py` generated four per-frame NPZs and a combined PLY
from four real DejaView v2 RGB images; depth/confidence/intrinsics/extrinsics were
checked for shape and finite values. Source image identities were recorded.

This is an inference smoke, not the historical 198-query semantic voxel score.
Those scores additionally require the shared semantic cache, causal full-scene
fusion, and the exact historical query/readout protocol. Do not replace them
with these smoke outputs.
