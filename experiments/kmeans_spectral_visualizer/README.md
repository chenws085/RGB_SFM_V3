# Minimal K-means vs spectral K-means visualization

This reduced demo produces one PNG for each computational step. It deliberately
uses a simple two-moons dataset so the difference between centroid-based and
graph-connectivity-based clustering is visible without the main training
pipeline.

Run from the repository root:

```bash
.venv/bin/python experiments/kmeans_spectral_visualizer/generate_step_figures.py
```

The command writes `00_overview.png`, seven numbered step images, and
`metrics.json` to `outputs/`. Parameters such as `--seed`, `--noise`, and
`--neighbors` are optional.

For two separate overview figures using the same unnormalized 3D double-helix
dataset, run:

```bash
.venv/bin/python experiments/kmeans_spectral_visualizer/generate_3d_overviews.py
```

This writes one overview for the research K-means pipeline and one for spectral
K-means to `outputs_3d/`. The research pipeline normalizes the raw input as an
internal algorithm step; spectral K-means builds its graph directly from the
raw, unnormalized 3D coordinates.

To visualize the L2 projection itself with radial 3D data and a unit-sphere
wireframe, run:

```bash
.venv/bin/python experiments/kmeans_spectral_visualizer/generate_l2_sphere_projection.py
```

To rerun both clustering pipelines with that same radial dataset, camera angle,
and sphere wireframe, run:

```bash
.venv/bin/python experiments/kmeans_spectral_visualizer/generate_radial_kmeans_overviews.py
```

## Exact methods used

- **Normalized-feature Euclidean K-means:** every three-dimensional input
  feature is L2-normalized, followed by scikit-learn `KMeans`. Centroids are not
  projected back onto the unit sphere, so this is not spherical K-means.
- **Spectral K-means:** the same normalized features form a symmetric (union)
  binary k-nearest-neighbor graph. The two smallest eigenvectors of
  `L_sym = I - D^-1/2 W D^-1/2` are row-normalized, then clustered with standard
  K-means (the Ng-Jordan-Weiss-style embedding step).
- Labels are Hungarian-aligned to the synthetic ground truth before colors and
  pointwise disagreement are compared. This changes names only, not partitions.

The visible dataset has two coordinates. A constant third feature (`z=3`) is
added before L2 normalization so normalization remains part of the method while
the two-moons geometry stays easy to inspect. The normalized-space image shows
the first two coordinates of those actual three-dimensional features.
