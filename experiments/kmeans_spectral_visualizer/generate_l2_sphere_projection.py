#!/usr/bin/env python3
"""Visualize how per-sample L2 normalization projects 3D points onto a sphere."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


COLORS = np.array(["#2878B5", "#D95319", "#3A923A"])


def make_radial_clusters(
    samples_per_cluster: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create directional clusters with deliberately large norm variation."""
    rng = np.random.default_rng(seed)
    directions = np.array(
        [
            [1.0, 0.22, 0.45],
            [-0.35, 1.0, 0.30],
            [-0.55, -0.45, 1.0],
        ]
    )
    directions = normalize(directions)
    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label, direction in enumerate(directions):
        radii = rng.uniform(0.45, 3.5, size=samples_per_cluster)
        # Broad spherical caps make the post-normalization centroid visibly
        # shorter than one while the three directions remain separable.
        angular_noise = rng.normal(scale=0.20, size=(samples_per_cluster, 3))
        noisy_directions = normalize(direction + angular_noise)
        points.append(radii[:, None] * noisy_directions)
        labels.append(np.full(samples_per_cluster, label, dtype=int))
    X = np.vstack(points)
    y = np.concatenate(labels)
    order = rng.permutation(len(X))
    return X[order], y[order]


def draw_unit_sphere(ax, alpha: float = 0.20) -> None:
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 18)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(
        x, y, z, rstride=2, cstride=2, color="#777777",
        linewidth=0.45, alpha=alpha,
    )


def scatter_clusters(ax, X: np.ndarray, labels: np.ndarray, size: int = 24) -> None:
    for label in range(3):
        mask = labels == label
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2], s=size,
            color=COLORS[label], edgecolor="white", linewidth=0.3,
            alpha=0.9, depthshade=False, label=f"direction {label + 1}",
        )


def decorate(ax, title: str, limit: float) -> None:
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=21, azim=-55)


def generate_figure(X: np.ndarray, labels: np.ndarray, output: Path, dpi: int) -> None:
    X_normalized = normalize(X)
    norms_before = np.linalg.norm(X, axis=1)
    norms_after = np.linalg.norm(X_normalized, axis=1)

    fig = plt.figure(figsize=(17, 5.6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    scatter_clusters(ax1, X, labels)
    draw_unit_sphere(ax1, alpha=0.12)
    decorate(ax1, "1. Raw 3D features: direction + varying norm", 3.7)
    ax1.legend(frameon=False, loc="upper left", fontsize=8)
    ax1.text2D(
        0.03, 0.03,
        f"norm range: {norms_before.min():.2f}–{norms_before.max():.2f}",
        transform=ax1.transAxes, color="#555555", fontsize=9,
    )

    draw_unit_sphere(ax2, alpha=0.26)
    # A subset of trajectories keeps the radial projection readable.
    selected = np.arange(0, len(X), max(1, len(X) // 36))
    for index in selected:
        color = COLORS[labels[index]]
        ax2.plot(
            [X[index, 0], X_normalized[index, 0]],
            [X[index, 1], X_normalized[index, 1]],
            [X[index, 2], X_normalized[index, 2]],
            color=color, linewidth=0.9, alpha=0.55,
        )
        ax2.scatter(*X[index], color=color, s=17, alpha=0.65, depthshade=False)
        ax2.scatter(
            *X_normalized[index], color=color, marker="^", s=28,
            edgecolor="black", linewidth=0.25, depthshade=False,
        )
    decorate(ax2, "2. Divide each vector by its own length", 3.7)
    ax2.text2D(
        0.03, 0.03, r"$\hat{x}=x/\Vert x\Vert_2$ (direction is preserved)",
        transform=ax2.transAxes, color="#555555", fontsize=9,
    )

    draw_unit_sphere(ax3, alpha=0.30)
    scatter_clusters(ax3, X_normalized, labels, size=28)
    decorate(ax3, "3. Every sample lies on the unit sphere", 1.15)
    ax3.text2D(
        0.03, 0.03,
        f"norm after: {norms_after.min():.3f}–{norms_after.max():.3f}",
        transform=ax3.transAxes, color="#555555", fontsize=9,
    )

    fig.suptitle(
        "L2 normalization removes magnitude and preserves direction",
        fontsize=18, weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs_3d" / "03_l2_sphere_projection.png",
    )
    parser.add_argument("--samples-per-cluster", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    X, labels = make_radial_clusters(args.samples_per_cluster, args.seed)
    generate_figure(X, labels, args.output, args.dpi)
    print(f"Saved L2 sphere projection figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
