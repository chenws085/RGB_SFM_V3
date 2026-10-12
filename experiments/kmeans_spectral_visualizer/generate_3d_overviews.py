#!/usr/bin/env python3
"""Generate two overview figures from the same unnormalized 3D dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


COLORS = np.array(["#2878B5", "#D95319"])


def make_double_helix(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create two non-convex 3D clusters without normalizing their norms."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(seed)
    sizes = (n_samples // 2, n_samples - n_samples // 2)
    samples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, size in enumerate(sizes):
        t = np.linspace(-2.2 * np.pi, 2.2 * np.pi, size)
        phase = label * np.pi
        points = np.column_stack(
            (np.cos(t + phase), np.sin(t + phase), 0.28 * t)
        )
        points += rng.normal(scale=noise, size=points.shape)
        samples.append(points)
        labels.append(np.full(size, label, dtype=int))
    X = np.vstack(samples)
    y = np.concatenate(labels)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def align_labels(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overlap = np.array(
        [[np.sum((reference == i) & (labels == j)) for j in (0, 1)] for i in (0, 1)]
    )
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {col: row for row, col in zip(rows, cols)}
    return np.array([mapping[value] for value in labels])


def spectral_kmeans(
    X: np.ndarray, neighbors: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    graph = kneighbors_graph(
        X, n_neighbors=neighbors, mode="connectivity", include_self=False
    )
    affinity = graph.maximum(graph.T).toarray()
    n_components = int(connected_components(affinity, directed=False)[0])
    degree = affinity.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(len(X)) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, 1])
    embedding = normalize(eigenvectors)
    labels = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(embedding)
    return labels, affinity, embedding, n_components


def scatter_3d(ax, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    for label in (0, 1):
        mask = labels == label
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2], s=22,
            color=COLORS[label], edgecolor="white", linewidth=0.25,
            alpha=0.92, label=f"cluster {label}", depthshade=False,
        )
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((1, 1, 1.35))


def add_graph(ax, X: np.ndarray, affinity: np.ndarray) -> None:
    rows, cols = np.nonzero(np.triu(affinity, k=1))
    for row, col in zip(rows, cols):
        ax.plot(
            X[[row, col], 0], X[[row, col], 1], X[[row, col], 2],
            color="#555555", linewidth=0.5, alpha=0.42, zorder=1,
        )


def save_kmeans_overview(
    X: np.ndarray, y: np.ndarray, X_normalized: np.ndarray,
    labels: np.ndarray, centroids: np.ndarray, output: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(16, 5.4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    scatter_3d(ax1, X, y, "1. Raw 3D input (unnormalized)")
    scatter_3d(ax2, X_normalized, y, "2. Per-sample L2 normalization")
    scatter_3d(ax3, X_normalized, labels, "3. Euclidean K-means in normalized space")
    ax3.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="X", s=190,
        c=COLORS, edgecolor="black", linewidth=1.2, depthshade=False,
    )
    ax1.legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Research method: normalized-feature Euclidean K-means",
        fontsize=18, weight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Input is unnormalized; normalization is an internal step. Centroids are not reprojected to the unit sphere.",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_spectral_overview(
    X: np.ndarray, y: np.ndarray, affinity: np.ndarray, embedding: np.ndarray,
    labels: np.ndarray, n_components: int, output: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(20, 5.4))
    ax1 = fig.add_subplot(141, projection="3d")
    ax2 = fig.add_subplot(142, projection="3d")
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144, projection="3d")
    scatter_3d(ax1, X, y, "1. Raw 3D input (unnormalized)")
    add_graph(ax2, X, affinity)
    scatter_3d(ax2, X, y, "2. k-nearest-neighbor affinity graph")
    for label in (0, 1):
        mask = y == label
        ax3.scatter(
            embedding[mask, 0], embedding[mask, 1], s=24,
            color=COLORS[label], edgecolor="white", linewidth=0.3,
            label=f"cluster {label}", alpha=0.92,
        )
    ax3.set_title("3. Row-normalized spectral embedding", fontsize=12, weight="bold")
    ax3.set_xlabel("eigenvector 1")
    ax3.set_ylabel("eigenvector 2")
    ax3.grid(alpha=0.18)
    ax3.set_aspect("equal", adjustable="datalim")
    scatter_3d(ax4, X, labels, "4. K-means labels mapped to input space")
    ax1.legend(frameon=False, loc="upper left")
    ax2.text2D(
        0.03, 0.93, f"Connected components: {n_components}",
        transform=ax2.transAxes, fontsize=9, color="#555555",
    )
    fig.suptitle(
        "Spectral K-means: clustering by graph connectivity",
        fontsize=18, weight="bold",
    )
    fig.text(
        0.5, 0.015,
        "The raw 3D input is used directly to build the graph; no L2 input normalization is applied.",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs_3d")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.035)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_double_helix(args.n_samples, args.noise, args.seed)
    X_normalized = normalize(X)

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=args.seed).fit(X_normalized)
    kmeans_labels = align_labels(y, kmeans.labels_)
    spectral_raw, affinity, embedding, n_components = spectral_kmeans(
        X, args.neighbors, args.seed
    )
    spectral_labels = align_labels(y, spectral_raw)

    save_kmeans_overview(
        X, y, X_normalized, kmeans_labels, kmeans.cluster_centers_,
        args.output_dir / "01_research_kmeans_overview_3d.png", args.dpi,
    )
    save_spectral_overview(
        X, y, affinity, embedding, spectral_labels, n_components,
        args.output_dir / "02_spectral_kmeans_overview_3d.png", args.dpi,
    )

    metrics = {
        "dataset": "unnormalized_3d_double_helix",
        "n_samples": len(X),
        "knn_neighbors": args.neighbors,
        "graph_connected_components": n_components,
        "research_kmeans_ari": float(adjusted_rand_score(y, kmeans_labels)),
        "spectral_kmeans_ari": float(adjusted_rand_score(y, spectral_labels)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved two overview figures to {args.output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
