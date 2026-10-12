"""Small observable Lloyd implementation for paired Euclidean/spherical experiments."""
import numpy as np


def unit_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero/near-zero vectors have no direction; exclude explicitly")
    return x / norms


def distances(x, centers):
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :] - 2 * x @ centers.T, 0,
    )


def fit_lloyd(x, initial, spherical=False, max_iter=300, tol=1e-5):
    """Input is normalized; both modes share initial centers and absolute tolerance."""
    centers = initial.copy()
    history = []
    previous = None
    for iteration in range(1, max_iter + 1):
        d = distances(x, centers)
        labels = d.argmin(axis=1)
        counts = np.bincount(labels, minlength=len(centers))
        means = np.zeros_like(centers)
        np.add.at(means, labels, x)
        means /= np.maximum(counts, 1)[:, None]
        raw_norms = np.linalg.norm(means, axis=1)
        invalid = (counts == 0) | (raw_norms < 1e-12)
        # Distinct, worst represented samples restore degenerate centers.
        candidates = np.argsort(d.min(axis=1))[::-1]
        means[invalid] = x[candidates[:invalid.sum()]]
        updated = unit_rows(means) if spherical else means
        movement = float(np.linalg.norm(updated - centers, axis=1).max())
        new_distances = distances(x, updated)
        new_labels = new_distances.argmin(axis=1)
        objective = float(new_distances[np.arange(len(x)), new_labels].sum())
        for cluster in range(len(centers)):
            history.append(dict(
                iteration=iteration, cluster=cluster, count=int(counts[cluster]),
                raw_mean_norm=float(raw_norms[cluster]),
                centroid_norm=float(np.linalg.norm(updated[cluster])),
                reinitialized=bool(invalid[cluster]), max_movement=movement,
                assignment_change=float(np.mean(labels != previous)) if previous is not None else 1.0,
                squared_distance_objective=objective,
            ))
        centers = updated
        if movement <= tol and not invalid.any():
            break
        previous = labels
    final_distances = distances(x, centers)
    labels = final_distances.argmin(axis=1)
    return dict(centers=centers, labels=labels, history=history,
                objective=float(final_distances[np.arange(len(x)), labels].sum()),
                iterations=iteration, converged=movement <= tol and not invalid.any())
