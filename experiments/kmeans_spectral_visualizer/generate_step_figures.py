#!/usr/bin/env python3
"""Generate a minimal, step-by-step K-means vs spectral K-means demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


COLORS = np.array(["#2878B5", "#D95319"])
STEP_TITLES = [
    "1. Input: two interlocking moons",
    "2. L2-normalized feature space",
    "3. Normalized-feature Euclidean K-means",
    "4. k-nearest-neighbor affinity graph",
    "5. Spectral embedding (row-normalized)",
    "6. Spectral K-means, shown in input space",
    "7. Where the two methods disagree",
]


@dataclass(frozen=True)
class DemoResult:
    raw_2d: np.ndarray
    normalized: np.ndarray
    truth: np.ndarray
    kmeans_labels: np.ndarray
    centroids: np.ndarray
    affinity: np.ndarray
    eigenvalues: np.ndarray
    embedding: np.ndarray
    spectral_labels: np.ndarray
    components: int


def align_labels(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Relabel ``labels`` to maximize agreement with ``reference``."""
    ref_values = np.unique(reference)
    pred_values = np.unique(labels)
    overlap = np.array(
        [[np.sum((reference == ref) & (labels == pred)) for pred in pred_values]
         for ref in ref_values]
    )
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {pred_values[col]: ref_values[row] for row, col in zip(rows, cols)}
    return np.array([mapping[value] for value in labels])


def run_demo(n_samples: int, noise: float, neighbors: int, seed: int) -> DemoResult:
    if not 1 <= neighbors < n_samples:
        raise ValueError("neighbors must be between 1 and n_samples - 1")

    raw_2d, truth = make_moons(
        n_samples=n_samples, noise=noise, random_state=seed
    )
    # A constant third feature keeps the visible moon geometry after L2
    # normalization. This is still a genuine per-sample 3D normalization.
    features = np.column_stack((raw_2d, np.full(n_samples, 3.0)))
    normalized = normalize(features, norm="l2")

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=seed).fit(normalized)
    kmeans_labels = align_labels(truth, kmeans.labels_)

    graph = kneighbors_graph(
        normalized, n_neighbors=neighbors, mode="connectivity", include_self=False
    )
    affinity = graph.maximum(graph.T).toarray()
    components = int(connected_components(affinity, directed=False)[0])

    degree = affinity.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    inv_sqrt_degree[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(n_samples) - (
        inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :]
    )
    eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, 1])
    embedding = normalize(eigenvectors, norm="l2")

    spectral_raw = KMeans(
        n_clusters=2, n_init=20, random_state=seed
    ).fit_predict(embedding)
    spectral_labels = align_labels(truth, spectral_raw)

    return DemoResult(
        raw_2d=raw_2d,
        normalized=normalized,
        truth=truth,
        kmeans_labels=kmeans_labels,
        centroids=kmeans.cluster_centers_,
        affinity=affinity,
        eigenvalues=eigenvalues,
        embedding=embedding,
        spectral_labels=spectral_labels,
        components=components,
    )


def _scatter(ax: Axes, points: np.ndarray, labels: np.ndarray) -> None:
    for label in (0, 1):
        mask = labels == label
        ax.scatter(
            points[mask, 0], points[mask, 1], s=28, color=COLORS[label],
            edgecolor="white", linewidth=0.35, alpha=0.92,
            label=f"cluster {label}", zorder=2,
        )


def _decorate(ax: Axes, title: str, xlabel: str = "feature 1", ylabel: str = "feature 2") -> None:
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16)
    ax.set_aspect("equal", adjustable="datalim")


def draw_steps(result: DemoResult) -> list[tuple[str, plt.Figure]]:
    figures: list[tuple[str, plt.Figure]] = []

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.raw_2d, result.truth)
    _decorate(ax, STEP_TITLES[0], "input x1", "input x2")
    ax.legend(frameon=False)
    figures.append(("01_input.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.normalized[:, :2], result.truth)
    _decorate(ax, STEP_TITLES[1], "normalized x1", "normalized x2")
    ax.text(
        0.02, 0.02, "Displayed: first 2 of 3 normalized features",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    figures.append(("02_l2_normalized.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.normalized[:, :2], result.kmeans_labels)
    ax.scatter(
        result.centroids[:, 0], result.centroids[:, 1], marker="X", s=180,
        c=COLORS, edgecolor="black", linewidth=1.2, label="centroids", zorder=4,
    )
    _decorate(ax, STEP_TITLES[2], "normalized x1", "normalized x2")
    ax.text(
        0.02, 0.02, "Centroid distance favors compact/convex partitions",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    figures.append(("03_kmeans.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    rows, cols = np.nonzero(np.triu(result.affinity, k=1))
    for row, col in zip(rows, cols):
        ax.plot(
            result.raw_2d[[row, col], 0], result.raw_2d[[row, col], 1],
            color="#9A9A9A", linewidth=0.35, alpha=0.35, zorder=1,
        )
    _scatter(ax, result.raw_2d, result.truth)
    _decorate(ax, STEP_TITLES[3], "input x1", "input x2")
    ax.text(
        0.02, 0.02, f"Connected components: {result.components}",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    figures.append(("04_affinity_graph.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.embedding, result.truth)
    _decorate(ax, STEP_TITLES[4], "eigenvector 1", "eigenvector 2")
    ax.text(
        0.02, 0.02,
        f"Smallest eigenvalues: {result.eigenvalues[0]:.4f}, {result.eigenvalues[1]:.4f}",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    figures.append(("05_spectral_embedding.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.raw_2d, result.spectral_labels)
    _decorate(ax, STEP_TITLES[5], "input x1", "input x2")
    ax.text(
        0.02, 0.02, "K-means is applied in the embedding, not this display space",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    figures.append(("06_spectral_kmeans.png", fig))

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, result.raw_2d, result.spectral_labels)
    disagreement = result.kmeans_labels != result.spectral_labels
    ax.scatter(
        result.raw_2d[disagreement, 0], result.raw_2d[disagreement, 1],
        s=85, facecolors="none", edgecolors="#111111", linewidth=1.4,
        label="different assignment", zorder=5,
    )
    _decorate(ax, STEP_TITLES[6], "input x1", "input x2")
    ax.text(
        0.02, 0.02, f"Disagreement: {disagreement.mean():.1%}",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )
    ax.legend(frameon=False, loc="best")
    figures.append(("07_disagreement.png", fig))

    return figures


def save_outputs(result: DemoResult, output_dir: Path, dpi: int) -> dict[str, float | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = draw_steps(result)
    for filename, fig in figures:
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")

    overview, axes = plt.subplots(2, 4, figsize=(18, 9))
    for target_ax, (_, source_fig) in zip(axes.flat[:7], figures):
        source_ax = source_fig.axes[0]
        # Render each finished step into the overview to guarantee identical content.
        source_fig.canvas.draw()
        rgba = np.asarray(source_fig.canvas.buffer_rgba())
        target_ax.imshow(rgba)
        target_ax.axis("off")
        plt.close(source_fig)
    axes.flat[7].axis("off")
    overview.suptitle("Minimal pipeline: K-means vs spectral K-means", fontsize=17, weight="bold")
    overview.tight_layout()
    overview.savefig(output_dir / "00_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(overview)

    metrics: dict[str, float | int] = {
        "n_samples": len(result.truth),
        "graph_connected_components": result.components,
        "kmeans_ari": float(adjusted_rand_score(result.truth, result.kmeans_labels)),
        "spectral_ari": float(adjusted_rand_score(result.truth, result.spectral_labels)),
        "method_disagreement_rate": float(
            np.mean(result.kmeans_labels != result.spectral_labels)
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args.n_samples, args.noise, args.neighbors, args.seed)
    metrics = save_outputs(result, args.output_dir, args.dpi)
    print(f"Saved step figures to {args.output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
