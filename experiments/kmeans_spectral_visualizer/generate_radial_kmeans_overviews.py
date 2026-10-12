#!/usr/bin/env python3
"""Compare both K-means pipelines on the radial 3D sphere-projection dataset."""

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

from generate_l2_sphere_projection import (
    COLORS,
    decorate,
    draw_unit_sphere,
    make_radial_clusters,
    scatter_clusters,
)


def align_labels(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    values = np.unique(reference)
    overlap = np.array(
        [[np.sum((reference == ref) & (predicted == pred)) for pred in values]
         for ref in values]
    )
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {values[col]: values[row] for row, col in zip(rows, cols)}
    return np.array([mapping[value] for value in predicted])


def build_spectral_result(
    X: np.ndarray, n_clusters: int, neighbors: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    graph = kneighbors_graph(
        X, n_neighbors=neighbors, mode="connectivity", include_self=False
    )
    affinity = graph.maximum(graph.T).toarray()
    n_components = int(connected_components(affinity, directed=False)[0])
    degree = affinity.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(len(X)) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    eigenvalues, eigenvectors = eigh(
        laplacian, subset_by_index=[0, n_clusters - 1]
    )
    embedding = normalize(eigenvectors)
    labels = KMeans(
        n_clusters=n_clusters, n_init=20, random_state=seed
    ).fit_predict(embedding)
    return labels, affinity, embedding, eigenvalues, n_components


def add_graph(ax, X: np.ndarray, affinity: np.ndarray) -> None:
    rows, cols = np.nonzero(np.triu(affinity, k=1))
    for row, col in zip(rows, cols):
        ax.plot(
            X[[row, col], 0], X[[row, col], 1], X[[row, col], 2],
            color="#555555", linewidth=0.45, alpha=0.32,
        )


def save_research_overview(
    X: np.ndarray, y: np.ndarray, X_normalized: np.ndarray,
    labels: np.ndarray, centroids: np.ndarray, output: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(17, 5.6))
    axes = [fig.add_subplot(1, 3, i, projection="3d") for i in range(1, 4)]

    scatter_clusters(axes[0], X, y)
    draw_unit_sphere(axes[0], alpha=0.12)
    decorate(axes[0], "1. Raw 3D input: varying feature norms", 3.7)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)

    draw_unit_sphere(axes[1], alpha=0.30)
    scatter_clusters(axes[1], X_normalized, y, size=28)
    decorate(axes[1], "2. L2-normalized samples on unit sphere", 1.15)

    draw_unit_sphere(axes[2], alpha=0.30)
    scatter_clusters(axes[2], X_normalized, labels, size=28)
    projected_centroids = normalize(centroids)
    for label in range(3):
        axes[2].plot(
            [centroids[label, 0], projected_centroids[label, 0]],
            [centroids[label, 1], projected_centroids[label, 1]],
            [centroids[label, 2], projected_centroids[label, 2]],
            color=COLORS[label], linestyle="--", linewidth=2.0, alpha=0.9,
        )
    axes[2].scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="X", s=180,
        color=COLORS, edgecolor="black", linewidth=1.1, depthshade=False,
        label="actual centroid",
    )
    axes[2].scatter(
        projected_centroids[:, 0], projected_centroids[:, 1],
        projected_centroids[:, 2], marker="*", s=210, facecolor="none",
        edgecolor=COLORS, linewidth=1.8, depthshade=False,
        label="normalized centroid (reference only)",
    )
    decorate(axes[2], "3. Euclidean K-means (K=3)", 1.15)
    centroid_norms = np.linalg.norm(centroids, axis=1)
    axes[2].text2D(
        0.02, 0.96,
        "actual centroid norms: " + ", ".join(f"{value:.3f}" for value in centroid_norms),
        transform=axes[2].transAxes, fontsize=8, color="#444444",
    )
    axes[2].text2D(
        0.02, 0.90, "X stays inside sphere; ★ is not used by the algorithm",
        transform=axes[2].transAxes, fontsize=8, color="#444444",
    )
    norm_inset = axes[2].inset_axes([0.62, 0.02, 0.34, 0.22])
    norm_inset.bar(range(3), centroid_norms, color=COLORS, width=0.65)
    norm_inset.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    norm_inset.set_ylim(0.90, 1.01)
    norm_inset.set_xticks(range(3), ["c1", "c2", "c3"])
    norm_inset.set_yticks([0.90, 0.95, 1.00])
    norm_inset.set_title("centroid norm (< 1)", fontsize=7)
    norm_inset.tick_params(labelsize=6)
    norm_inset.grid(axis="y", alpha=0.18)

    fig.suptitle(
        "Research method: normalized-feature Euclidean K-means",
        fontsize=18, weight="bold",
    )
    fig.text(
        0.5, 0.012,
        "Samples have norm 1, but their arithmetic-mean centroids generally have norm < 1 and are not reprojected.",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_spectral_overview(
    X: np.ndarray, y: np.ndarray, affinity: np.ndarray, embedding: np.ndarray,
    labels: np.ndarray, eigenvalues: np.ndarray, n_components: int,
    output: Path, dpi: int,
) -> None:
    fig = plt.figure(figsize=(21, 5.6))
    axes = [fig.add_subplot(1, 4, i, projection="3d") for i in range(1, 5)]

    scatter_clusters(axes[0], X, y)
    draw_unit_sphere(axes[0], alpha=0.12)
    decorate(axes[0], "1. Same raw 3D input", 3.7)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)

    add_graph(axes[1], X, affinity)
    scatter_clusters(axes[1], X, y, size=20)
    decorate(axes[1], "2. KNN affinity graph in raw space", 3.7)
    axes[1].text2D(
        0.03, 0.94, f"Connected components: {n_components}",
        transform=axes[1].transAxes, fontsize=8, color="#555555",
    )

    scatter_clusters(axes[2], embedding, y, size=28)
    decorate(axes[2], "3. Row-normalized spectral embedding", 1.15)
    axes[2].text2D(
        0.03, 0.94,
        "eigenvalues: " + ", ".join(f"{value:.3f}" for value in eigenvalues),
        transform=axes[2].transAxes, fontsize=8, color="#555555",
    )

    scatter_clusters(axes[3], X, labels)
    draw_unit_sphere(axes[3], alpha=0.12)
    decorate(axes[3], "4. Spectral K-means labels in input space", 3.7)

    fig.suptitle(
        "Spectral K-means: KNN graph, Laplacian embedding, then K-means",
        fontsize=18, weight="bold",
    )
    fig.text(
        0.5, 0.012,
        "The same unnormalized input is grouped through local graph connectivity rather than input-space centroids.",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "outputs_radial_comparison",
    )
    parser.add_argument("--samples-per-cluster", type=int, default=80)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_radial_clusters(args.samples_per_cluster, args.seed)
    X_normalized = normalize(X)

    research_model = KMeans(n_clusters=3, n_init=20, random_state=args.seed).fit(
        X_normalized
    )
    research_labels = align_labels(y, research_model.labels_)
    aligned_centroids = np.vstack(
        [X_normalized[research_labels == label].mean(axis=0) for label in range(3)]
    )
    spectral_raw, affinity, embedding, eigenvalues, n_components = (
        build_spectral_result(X, 3, args.neighbors, args.seed)
    )
    spectral_labels = align_labels(y, spectral_raw)

    save_research_overview(
        X, y, X_normalized, research_labels, aligned_centroids,
        args.output_dir / "01_research_kmeans_radial_3d.png", args.dpi,
    )
    save_spectral_overview(
        X, y, affinity, embedding, spectral_labels, eigenvalues, n_components,
        args.output_dir / "02_spectral_kmeans_radial_3d.png", args.dpi,
    )

    metrics = {
        "dataset": "unnormalized_3d_radial_clusters",
        "n_clusters": 3,
        "n_samples": len(X),
        "knn_neighbors": args.neighbors,
        "graph_connected_components": n_components,
        "research_kmeans_ari": float(adjusted_rand_score(y, research_labels)),
        "spectral_kmeans_ari": float(adjusted_rand_score(y, spectral_labels)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved radial comparison to {args.output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
